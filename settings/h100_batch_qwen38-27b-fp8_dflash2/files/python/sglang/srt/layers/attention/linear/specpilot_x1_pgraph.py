"""SPECPILOT_X1: length-bucketed CUDA graphs for the GDN prefill of single fresh requests.

One graph per (GDN layer, token bucket T) runs conv1d + qkv split + gating + the Triton chunk delta rule over a
single sequence of T tokens. A request of L <= T tokens is copied into the LAST L rows; the T - L leading rows are
zero inputs with beta = g = 0 and k = v = 0, so the recurrent state stays exactly 0 through them (the request starts
from a zero state) and the conv windows of the first real rows see zeros, as with a zero initial conv state. The
sequence ends on the last real token, so the conv kernel's final-state write is the eager one. Tail padding is not
used: the Triton chunk kernel's final state is wrong for some trailing zero-beta rows (x1 journal, 06:30).

Multi-request extends (SPECPILOT_X1_PGRAPH_MULTI=N > 1): one graph per (layer, n, T) for n = 2..N fresh requests,
each front-padded to the same T (request i occupies rows [i*T, (i+1)*T), real tokens last); a packed->padded row index
built once per forward scatters the inputs in and gathers the outputs out. n*T is capped at
SPECPILOT_X1_PGRAPH_MULTI_ROWS (default 2 x the largest single bucket: n=4 x T=1024 at the default 2048), T at the
largest single bucket. Every bucket's input/output rows are views of one shared buffer set sized for the largest bucket
(one bucket is active per forward), so a larger multi bucket costs graph-pool memory, not per-bucket buffers.

Knobs (read once at capture): SPECPILOT_X1_PGRAPH=1 enables; SPECPILOT_X1_PGRAPH_MAX caps the largest bucket;
SPECPILOT_X1_PGRAPH_MULTI (default 1) is the largest number of requests per captured extend.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096)


def enabled() -> bool:
    return os.environ.get("SPECPILOT_X1_PGRAPH", "0") == "1"


def _max_bucket() -> int:
    return int(os.environ.get("SPECPILOT_X1_PGRAPH_MAX", "2048"))


def _multi_rows() -> int:
    return int(os.environ.get("SPECPILOT_X1_PGRAPH_MULTI_ROWS", str(2 * _max_bucket())))


def _max_reqs() -> int:
    return max(1, int(os.environ.get("SPECPILOT_X1_PGRAPH_MULTI", "1")))


def _pin_fla_index_caches(keep: list) -> None:
    """The FLA chunk kernels read chunk-index tensors that tensor_cache (4 entries) holds outside the graph pool;
    a captured graph keeps their addresses, so keep every entry alive once it has been captured."""
    from sglang.kernels.ops.attention.fla import index as fla_index

    for fn in (fla_index.prepare_lens, fla_index.prepare_chunk_indices, fla_index.prepare_chunk_offsets):
        for cell in fn.__closure__ or ():
            entries = cell.cell_contents
            if isinstance(entries, list):
                keep.extend(e[2] for e in entries if isinstance(e, tuple) and len(e) == 3)


def _shared_buffers(R: int, *, D: int, HV: int, V: int, dt_ab, device) -> dict:
    return dict(
        mixed=torch.zeros(R, D, dtype=torch.bfloat16, device=device),
        a=torch.zeros(R, HV, dtype=dt_ab, device=device),
        b=torch.zeros(R, HV, dtype=dt_ab, device=device),
        out=torch.zeros(1, R, HV, V, dtype=torch.bfloat16, device=device),
        mask=torch.ones(R, dtype=torch.float32, device=device),
    )


class _Bucket:
    def __init__(self, *, T: int, D: int, HV: int, V: int, dt_ab, idx_dtype, qsl_dtype, device, n: int = 1,
                 shared: Optional[dict] = None):
        self.T = T
        self.n = n
        R = n * T
        if shared is None:
            shared = _shared_buffers(R, D=D, HV=HV, V=V, dt_ab=dt_ab, device=device)
        self.mixed = shared["mixed"][:R]
        self.a = shared["a"][:R]
        self.b = shared["b"][:R]
        self.out = shared["out"][:, :R]
        self.mask = shared["mask"][:R]
        self.arange = torch.arange(T, device=device)
        self.slot = torch.zeros(n, dtype=idx_dtype, device=device)
        self.qsl = torch.arange(0, R + 1, T, dtype=qsl_dtype, device=device)
        self.has_init = torch.zeros(n, dtype=torch.bool, device=device)
        self.lead = T - T  # leading pad rows of the current request (n == 1)
        if n > 1:  # packed -> padded row index of the current requests (n > 1)
            self.rows = torch.zeros(R, dtype=torch.int64, device=device)
            self.rows_host = torch.zeros(R, dtype=torch.int64, pin_memory=True)
            self.nrows = 0
            self.h2d_done = torch.cuda.Event()  # rows_host may be rewritten only after its last H2D copy ran
            self.h2d_done.record()
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.keep: list = []

    def set_meta(self, L, slot: torch.Tensor) -> None:
        if self.n == 1:
            self.lead = self.T - L
            torch.ge(self.arange, self.lead, out=self.mask)
            self.slot.copy_(slot[:1])
            return
        T, h, pos = self.T, self.rows_host, 0
        self.h2d_done.synchronize()
        for i, Li in enumerate(L):
            torch.arange(i * T + T - Li, (i + 1) * T, out=h[pos : pos + Li])
            pos += Li
        self.nrows = pos
        rows = self.rows[:pos]
        rows.copy_(h[:pos], non_blocking=True)
        self.h2d_done.record()
        self.mask.zero_()
        self.mask.index_fill_(0, rows, 1.0)
        self.slot.copy_(slot[: self.n])


class X1PrefillGraphs:
    """Owned by a GDNAttnBackend; captures at boot, replays in forward_extend."""

    def __init__(self, backend, layers, fns) -> None:
        self.backend = backend
        self.layers = {l.layer_id: l for l in layers}
        self.first_layer = min(self.layers)
        self.fns = fns
        self.buckets: list[_Bucket] = []
        self.pool = None
        self.active: Optional[_Bucket] = None
        self.hits = 0
        self.multi_hits = 0
        self.max_n = _max_reqs()

    # ---- graph body -------------------------------------------------------------------------------------------
    def _body(self, bk: _Bucket, layer) -> None:
        causal_conv1d_fn, fused_qkv_split_gdn_prefill, fused_gdn_gating = self.fns
        cache = self.backend.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states, ssm_states = cache.conv[0], cache.temporal
        T = bk.n * bk.T
        m2 = bk.mask.view(T, 1)
        x = causal_conv1d_fn(
            (bk.mixed * m2.to(bk.mixed.dtype)).transpose(0, 1),
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
            conv_states=conv_states,
            has_initial_state=bk.has_init,
            cache_indices=bk.slot,
            query_start_loc=bk.qsl,
            seq_lens_cpu=[bk.T] * bk.n,
        ).transpose(0, 1)[:T]
        q, k, v = fused_qkv_split_gdn_prefill(
            x, layer.num_q_heads, layer.num_k_heads, layer.num_v_heads,
            layer.head_q_dim, layer.head_k_dim, layer.head_v_dim,
        )
        g, beta = fused_gdn_gating(layer.A_log, bk.a, bk.b, layer.dt_bias)
        m = bk.mask.view(1, T, 1)
        g = g * m
        beta = beta * m
        m4 = bk.mask.view(1, T, 1, 1).to(k.dtype)
        k = k * m4
        v = v * m4
        o, _, _ = self.backend.kernel_dispatcher.extend(
            q=q, k=k, v=v, g=g, beta=beta,
            ssm_states=ssm_states, cache_indices=bk.slot, query_start_loc=bk.qsl,
        )
        bk.out.copy_(o)

    # ---- capture ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def capture(self, *, capture_slot: int) -> None:
        any_layer = self.layers[self.first_layer]
        D = any_layer.q_dim + any_layer.k_dim + any_layer.v_dim
        HV, V = any_layer.num_v_heads, any_layer.head_v_dim
        dev = any_layer.A_log.device
        self.pool = torch.cuda.graph_pool_handle()
        plan = [(1, T) for T in _BUCKETS if T <= _max_bucket()]
        plan += [(n, T) for n in range(2, _max_reqs() + 1) for T in _BUCKETS
                 if T <= _max_bucket() and n * T <= _multi_rows()]
        shared = _shared_buffers(max(n * T for n, T in plan), D=D, HV=HV, V=V, dt_ab=torch.bfloat16, device=dev)
        for n, T in plan:
            slot = torch.full((n,), capture_slot, dtype=torch.int32, device=dev)
            bk = _Bucket(T=T, D=D, HV=HV, V=V, dt_ab=torch.bfloat16, idx_dtype=torch.int32,
                         qsl_dtype=torch.int32, device=dev, n=n, shared=shared)
            bk.set_meta(T if n == 1 else [T] * n, slot)
            for lid in sorted(self.layers):
                layer = self.layers[lid]
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    self._body(bk, layer)  # warm-up: triton compile + chunk-index cache for bk.qsl
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool):
                    self._body(bk, layer)
                bk.graphs[lid] = g
                _pin_fla_index_caches(bk.keep)
            self.buckets.append(bk)
        torch.cuda.synchronize()
        for lid in self.layers:
            cache = self.backend.req_to_token_pool.mamba2_layer_cache(lid)
            cache.conv[0][capture_slot].zero_()
            cache.temporal[capture_slot].zero_()
        torch.cuda.synchronize()
        logger.info("SPECPILOT_X1 GDN prefill graphs captured: layers=%d buckets=%s slot=%d",
                    len(self.layers), [b.T if b.n == 1 else (b.n, b.T) for b in self.buckets], capture_slot)

    # ---- replay -----------------------------------------------------------------------------------------------
    def _run_multi(self, bk: _Bucket, layer, mixed_qkv, a, b, output, L) -> torch.Tensor:
        rows = bk.rows[: bk.nrows]
        bk.mixed.index_copy_(0, rows, mixed_qkv)
        bk.a.index_copy_(0, rows, a)
        bk.b.index_copy_(0, rows, b)
        bk.graphs[layer.layer_id].replay()
        self.hits += 1
        self.multi_hits += 1
        if self.multi_hits in (1, 24 * 100):
            logger.info("SPECPILOT_X1 GDN multi-request prefill graph replays: %d (n %d, bucket %d, L %d)",
                        self.multi_hits, bk.n, bk.T, L)
        if output is not None and tuple(output.shape) == (1, L, bk.out.shape[2], bk.out.shape[3]):
            torch.index_select(bk.out, 1, rows, out=output)
            return output
        return torch.index_select(bk.out, 1, rows)

    def try_run(self, *, layer, forward_batch, mixed_qkv, a, b, cache_indices, output) -> Optional[torch.Tensor]:
        L = mixed_qkv.shape[0]
        if layer.layer_id == self.first_layer:
            self.active = None
            n = forward_batch.batch_size
            if (forward_batch.batch_size == 1 and forward_batch.extend_prefix_lens_cpu is not None
                    and forward_batch.extend_prefix_lens_cpu[0] == 0
                    and forward_batch.extend_seq_lens_cpu is not None
                    and forward_batch.extend_seq_lens_cpu[0] == L
                    and not torch.cuda.is_current_stream_capturing()):
                bk = next((x for x in self.buckets if x.n == 1 and x.T >= L), None)
                if bk is not None and layer.layer_id in bk.graphs:
                    bk.set_meta(L, cache_indices)
                    self.active = bk
            elif (1 < n <= self.max_n and forward_batch.extend_prefix_lens_cpu is not None
                    and forward_batch.extend_seq_lens_cpu is not None
                    and len(forward_batch.extend_seq_lens_cpu) == n
                    and not any(forward_batch.extend_prefix_lens_cpu)
                    and sum(forward_batch.extend_seq_lens_cpu) == L
                    and cache_indices.shape[0] >= n
                    and not torch.cuda.is_current_stream_capturing()):
                Lmax = max(forward_batch.extend_seq_lens_cpu)
                bk = next((x for x in self.buckets if x.n == n and x.T >= Lmax), None)
                if bk is not None and layer.layer_id in bk.graphs:
                    bk.set_meta(list(forward_batch.extend_seq_lens_cpu), cache_indices)
                    self.active = bk
        bk = self.active
        if bk is None or mixed_qkv.shape[0] != L or layer.layer_id not in bk.graphs:
            return None
        if bk.n > 1:
            return self._run_multi(bk, layer, mixed_qkv, a, b, output, L)
        lead = bk.lead
        bk.mixed[lead:].copy_(mixed_qkv)
        bk.a[lead:].copy_(a)
        bk.b[lead:].copy_(b)
        bk.graphs[layer.layer_id].replay()
        self.hits += 1
        if self.hits in (1, 24 * 100):
            logger.info("SPECPILOT_X1 GDN prefill graph replays: %d (bucket %d, L %d)", self.hits, bk.T, L)
        if output is not None and tuple(output.shape) == (1, L, bk.out.shape[2], bk.out.shape[3]):
            output.copy_(bk.out[:, lead:])
            return output
        return bk.out[:, lead:].clone()


def maybe_capture(model_runner) -> None:
    """Boot hook (SPECPILOT_X1): capture the bucketed GDN prefill graphs on the target's GDN backend."""
    if not enabled():
        return
    from sglang.srt.layers.attention.linear import gdn_backend as gb
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
    from sglang.srt.layers.radix_linear_attention import RadixLinearAttention

    backend = model_runner.attn_backend
    backend = getattr(backend, "linear_attn_backend", backend)
    if not isinstance(backend, gb.GDNAttnBackend):
        logger.warning("SPECPILOT_X1_PGRAPH: no GDN backend found; disabled")
        return
    if not isinstance(backend.kernel_dispatcher.extend_kernel, TritonGDNKernel):
        logger.warning("SPECPILOT_X1_PGRAPH: needs --linear-attn-prefill-backend triton; disabled")
        return
    layers = [m for m in model_runner.model.modules() if isinstance(m, RadixLinearAttention)]
    if not layers:
        return
    ssm = backend.req_to_token_pool.mamba2_layer_cache(layers[0].layer_id).temporal
    pg = X1PrefillGraphs(backend, layers, (gb.causal_conv1d_fn, gb.fused_qkv_split_gdn_prefill, gb.fused_gdn_gating))
    pg.capture(capture_slot=ssm.shape[0] - 1)
    backend.x1_pgraph = pg
