"""SPECPILOT_DB_RV: ragged (packed) tree verify (BUDGET family, mechanism).

Each request i verifies only the first ``n_i`` nodes of its best-first N-node tree (``specpilot_tree``: parents
precede children, so nodes ``0..n_i-1`` are a valid ``n_i``-node tree). The target's residual stream runs PACKED on
``T`` rows (``T`` = the token tier of the batch, ``sum n_i <= T``): embedding, every token-wise op (GDN in/out
projections, attention qkv/o, MLP, norms, aux-hidden capture, lm_head, logits). The per-request structured mixers
(FA3 tree cascade, GDN conv windows, WY GDN tree verify + commit inputs, KV writes) keep the dense ``[bs, N]`` layout
of the verify batch: ``to_dense`` gathers packed rows into the dense layout before each mixer and ``to_packed`` gathers
the mixer output back. Nodes ``j >= n_i`` (and graph padding rows) are never ancestors of a real node, so a real node's
mixer output does not depend on them; accept masks them (``mask_candidates``).

Interface for budget policies (all device-side, no host sync):
  * ``state(device).n_buf``: int32 ``[MAX_REQS]``, the per-request node budget of the next target verify (request order
    = batch order). Written by a policy before the target verify replay (in the draft graph tail hook or eagerly).
    Read only when ``SPECPILOT_DB_RV_POLICY=external``; the static policy fills it in-graph.
  * ``state(device).num_real``: int32 ``[1]``, real requests of the batch (graph padding requests get n = 1). The worker
    writes it before every target verify (``set_num_real``).
  * ``rv_config().tokens_for_bs(bs, N)``: the host token tier T of a (captured) batch size; one graph per bs as in L.
    Budgets are clamped in-graph: ``1 <= n_i <= N`` and ``sum n_i <= T`` (requests keep >= 1 row; overflow is cut
    from the tail requests' extra nodes). ``state.n_eff`` holds the budgets actually verified.

Knobs (read once; SPECPILOT_DB_RV_MODE unset = L's dense verify, bit-for-bit):
  SPECPILOT_DB_RV_MODE    '' / 'off' | 'pack'
  SPECPILOT_DB_RV_N       static node budget per bs: 'n' or 'bsmax:n,bsmax:n,...' (bs above the last bsmax uses the
                        last n; n is clamped to [1, N]); default N
  SPECPILOT_DB_RV_POLICY  'static' (default: every request n(bs)) | 'external' (a policy writes n_buf)
  SPECPILOT_DB_RV_TOKENS  token tier per bs: 'bsmax:T' or 'bsmax:xF' (F = average nodes per request, T = ceil(bs*F));
                        default bs * n(bs). Clamped to [bs, bs*N + 64]: a tier above bs*N adds padding rows only
                        (e.g. '26:432;28:464' keeps the full tree but moves the GEMM M off the cuBLAS spikes at
                        416/448 rows).
  SPECPILOT_DB_RV_LOGITS_NOCOPY  1 = packed verify logits stay the lm_head output (no copy into the shared buffer)
  SPECPILOT_DB_RV_SUFFIX_SKIP   1 = the fused tree-suffix attention merge skips unverified node rows (suffix length 0)
  A full tier (T >= bs * N) always verifies all N nodes of every request (policy budgets are ignored there).
  SPECPILOT_DB_RV_FUSE    1 = gather-free GDN boundary: the q/k/v/z/b/a split reads packed rows and writes the dense mixer
                        inputs (``split_qkvzba_dense``), the WY tree verify stores verified nodes into packed rows
                        (``Pack.rows_out``); 0 / unset = index_select gathers around every mixer (db-101).
"""
from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

MAX_REQS = 1024
MAX_NODES = 64
PAD_MAX = 64  # extra (padding) rows a token tier may add above bs * N


def _parse_table(raw: str, name: str):
    raw = raw.strip()
    if not raw:
        return None
    out = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            k, v = part.split(":", 1)
            out.append((int(k), v.strip()))
        else:
            out.append((1 << 30, part))
    if not out:
        return None
    out.sort(key=lambda kv: kv[0])
    return tuple(out)


def _lookup(table, bs: int) -> str:
    for bmax, v in table:
        if bs <= bmax:
            return v
    return table[-1][1]


@dataclass(frozen=True)
class RvConfig:
    mode: str
    n_table: Optional[tuple]
    policy: str
    tokens_table: Optional[tuple]

    def n_for_bs(self, bs: int, n_nodes: int) -> int:
        if self.n_table is None:
            return int(n_nodes)
        n = int(_lookup(self.n_table, bs))
        return max(1, min(int(n_nodes), n))

    def tokens_for_bs(self, bs: int, n_nodes: int) -> int:
        bs = int(bs)
        if self.tokens_table is None:
            t = bs * self.n_for_bs(bs, n_nodes)
        else:
            v = _lookup(self.tokens_table, bs)
            t = math.ceil(bs * float(v[1:])) if v.startswith("x") else int(v)
        # T may exceed bs * N by up to PAD_MAX rows: padding rows that move a GEMM M off a slow cuBLAS size
        return max(bs, min(bs * int(n_nodes) + PAD_MAX, int(t)))


@functools.lru_cache(maxsize=None)
def rv_config() -> Optional[RvConfig]:
    mode = os.environ.get("SPECPILOT_DB_RV_MODE", "").strip().lower()
    if mode in ("", "0", "off", "none"):
        return None
    if mode != "pack":
        raise ValueError(f"SPECPILOT_DB_RV_MODE={mode!r}: expected 'pack' or unset")
    from sglang.srt.speculative.specpilot_tree import tree_enabled

    if not tree_enabled():
        raise ValueError("SPECPILOT_DB_RV_MODE=pack needs the tree verify (SPECPILOT_TREE_MODE)")
    policy = os.environ.get("SPECPILOT_DB_RV_POLICY", "static").strip().lower() or "static"
    if policy not in ("static", "external"):
        raise ValueError(f"SPECPILOT_DB_RV_POLICY={policy!r}: expected 'static' or 'external'")
    n_table = _parse_table(os.environ.get("SPECPILOT_DB_RV_N", ""), "SPECPILOT_DB_RV_N")
    tokens_table = _parse_table(os.environ.get("SPECPILOT_DB_RV_TOKENS", ""), "SPECPILOT_DB_RV_TOKENS")
    if n_table is not None:
        for _, v in n_table:
            if int(v) < 1:
                raise ValueError(f"SPECPILOT_DB_RV_N values must be >= 1, got {v!r}")
    return RvConfig(mode=mode, n_table=n_table, policy=policy, tokens_table=tokens_table)


def rv_enabled() -> bool:
    return rv_config() is not None


@functools.lru_cache(maxsize=None)
def rv_fuse() -> bool:
    return rv_enabled() and os.environ.get("SPECPILOT_DB_RV_FUSE", "0").strip() not in ("", "0")


class RvState:
    """Static device buffers (addresses baked into the captured target verify graphs)."""

    def __init__(self, device):
        self.device = device
        self.n_buf = torch.full((MAX_REQS,), MAX_NODES, dtype=torch.int32, device=device)
        self.num_real = torch.full((1,), MAX_REQS, dtype=torch.int32, device=device)
        self.n_eff = torch.zeros((MAX_REQS,), dtype=torch.int32, device=device)
        self.pack = torch.zeros((MAX_REQS * MAX_NODES,), dtype=torch.int64, device=device)
        self.unpack = torch.zeros((MAX_REQS * MAX_NODES,), dtype=torch.int64, device=device)
        self.rows_out = torch.full((MAX_REQS * MAX_NODES,), -1, dtype=torch.int64, device=device)

    def set_num_real(self, bs: int) -> None:
        self.num_real.fill_(int(bs))


_STATE: Optional[RvState] = None


def state(device=None) -> RvState:
    global _STATE
    if _STATE is None:
        assert device is not None, "specpilot_rv.state() first needs a device"
        _STATE = RvState(device)
    return _STATE


@triton.jit
def _rv_pack_kernel(
    n_ptr,  # [>= bs] int32 policy budgets
    num_real_ptr,  # [1] int32
    n_eff_ptr,  # [>= bs] int32 out
    pack_ptr,  # [>= T] int64 out: packed row -> dense row (b * N + j); pad rows -> 0
    unpack_ptr,  # [>= bs * N] int64 out: dense row -> packed row; unverified nodes -> 0
    rows_out_ptr,  # [>= bs * N] int64 out: dense row -> packed row; unverified nodes -> -1
    bs,
    T,
    STATIC_N: tl.constexpr,  # > 0: every request gets STATIC_N (policy 'static')
    N: tl.constexpr,
    BS: tl.constexpr,
    NP: tl.constexpr,
    TB: tl.constexpr,
):
    b = tl.arange(0, BS)
    live = b < bs
    if STATIC_N > 0:
        n = tl.full([BS], STATIC_N, dtype=tl.int32)
    else:
        n = tl.load(n_ptr + b, mask=live, other=1).to(tl.int32)
        num_real = tl.load(num_real_ptr)
        n = tl.where(b < num_real, n, 1)
    n = tl.minimum(tl.maximum(n, 1), N)
    extra = tl.where(live, n - 1, 0)
    excl = tl.cumsum(extra, 0) - extra
    budget = T - bs
    extra = tl.minimum(extra, tl.maximum(budget - excl, 0))
    n = tl.where(live, extra + 1, 0)
    off = tl.cumsum(n, 0) - n
    total = tl.sum(n, 0)
    tl.store(n_eff_ptr + b, n, mask=live)
    j = tl.arange(0, NP)
    valid = live[:, None] & (j[None, :] < n[:, None])
    dense = b[:, None].to(tl.int64) * N + j[None, :]
    packed = off[:, None].to(tl.int64) + j[None, :]
    tl.store(
        unpack_ptr + dense,
        tl.where(valid, packed, 0),
        mask=live[:, None] & (j[None, :] < N),
    )
    tl.store(
        rows_out_ptr + dense,
        tl.where(valid, packed, -1),
        mask=live[:, None] & (j[None, :] < N),
    )
    tl.store(pack_ptr + packed, dense, mask=valid)
    for t0 in range(0, T, TB):
        t = t0 + tl.arange(0, TB)
        tl.store(pack_ptr + t, tl.zeros([TB], dtype=tl.int64), mask=(t >= total) & (t < T))


def _np2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


def build_layout(st: RvState, bs: int, n_nodes: int, tokens: int, static_n: int = 0) -> None:
    """In-graph: budgets -> n_eff [bs], pack [tokens], unpack [bs * N] (static buffers of ``st``)."""
    assert 1 <= bs <= MAX_REQS and 1 <= n_nodes <= MAX_NODES and bs <= tokens <= bs * n_nodes + PAD_MAX
    _rv_pack_kernel[(1,)](
        st.n_buf,
        st.num_real,
        st.n_eff,
        st.pack,
        st.unpack,
        st.rows_out,
        int(bs),
        int(tokens),
        STATIC_N=int(static_n),
        N=int(n_nodes),
        BS=max(16, _np2(bs)),
        NP=_np2(n_nodes),
        TB=min(1024, max(16, _np2(tokens))),
    )


@dataclass
class Pack:
    bs: int
    n_nodes: int
    tokens: int
    pack: torch.Tensor  # [tokens]
    unpack: torch.Tensor  # [bs * N]
    rows_out: Optional[torch.Tensor] = None  # [bs * N] packed row of each verified node, -1 otherwise
    n_eff: Optional[torch.Tensor] = None  # [bs]
    gdn_out_packed: bool = False  # set by the GDN backend when the WY verify stored packed rows (SPECPILOT_DB_RV_FUSE)
    _sfx_src: Optional[torch.Tensor] = None
    _sfx: Optional[torch.Tensor] = None
    d4r3_meta: Optional[object] = None  # SPECPILOT_D4R3_PATTN: FA3 packed-row metadata shared by the attention layers

    def suffix_seqlens(self, seqlens: torch.Tensor) -> torch.Tensor:
        """SPECPILOT_DB_RV_SUFFIX_SKIP: per-query tree-suffix lengths with every unverified node row set to 0 (its
        attention output is never used); computed once per forward (all attention layers share the metadata)."""
        if not _suffix_skip_knob() or self.rows_out is None or seqlens.shape != self.rows_out.shape:
            return seqlens
        if self._sfx_src is not seqlens:
            self._sfx_src = seqlens
            self._sfx = torch.where(self.rows_out >= 0, seqlens, torch.zeros_like(seqlens))
        return self._sfx

    def to_dense(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return x.index_select(dim, self.unpack)

    def to_packed(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return x.index_select(dim, self.pack)


_CUR: Optional[Pack] = None
_LAST_PACKED = False  # the last target-verify forward ran the packed stream (read by the logits processor)
_LOGGED = False


def current() -> Optional[Pack]:
    return _CUR


def begin(forward_batch, num_rows: int, device) -> Optional[Pack]:
    """Called at the top of the target model forward. Returns the active Pack (and makes it current) for a
    target-verify forward when SPECPILOT_DB_RV_MODE=pack; None otherwise (L's dense path)."""
    global _CUR, _LAST_PACKED
    _CUR = None
    _LAST_PACKED = False
    cfg = rv_config()
    if cfg is None or not forward_batch.forward_mode.is_target_verify():
        return None
    bs = int(forward_batch.batch_size)
    if bs <= 0 or num_rows % bs != 0:
        return None
    n_nodes = num_rows // bs
    if n_nodes < 1 or n_nodes > MAX_NODES or bs > MAX_REQS:
        return None
    tokens = cfg.tokens_for_bs(bs, n_nodes)
    global _LOGGED
    if not _LOGGED:
        _LOGGED = True
        import logging

        logging.getLogger(__name__).info(
            "SPECPILOT_DB_RV packed verify: policy=%s n=%s tokens=%s fuse=%s (first layout: bs=%d N=%d T=%d)",
            cfg.policy, cfg.n_table, cfg.tokens_table, rv_fuse(), bs, n_nodes, tokens,
        )
    st = state(device)
    static_n = cfg.n_for_bs(bs, n_nodes) if cfg.policy == "static" else 0
    if tokens >= bs * n_nodes:  # a full tier verifies the whole tree whatever the policy wrote (no sync needed)
        static_n = n_nodes
    build_layout(st, bs, n_nodes, tokens, static_n=static_n)
    if static_n == n_nodes and tokens == bs * n_nodes:
        # identity layout (full budget): L's dense forward, no gathers; the worker reads the identity maps
        return None
    _LAST_PACKED = True
    _CUR = Pack(
        bs=bs,
        n_nodes=n_nodes,
        tokens=tokens,
        pack=st.pack[:tokens],
        unpack=st.unpack[: bs * n_nodes],
        rows_out=st.rows_out[: bs * n_nodes],
        n_eff=st.n_eff[:bs],
    )
    return _CUR


def logits_nocopy() -> bool:
    """SPECPILOT_DB_RV_LOGITS_NOCOPY=1: a packed verify returns the lm_head output (bf16, already in the graph pool)
    instead of copying it into the shared graph logits buffer."""
    return _LAST_PACKED and _logits_nocopy_knob()


@functools.lru_cache(maxsize=None)
def _logits_nocopy_knob() -> bool:
    return rv_enabled() and os.environ.get("SPECPILOT_DB_RV_LOGITS_NOCOPY", "0").strip() not in ("", "0")


@functools.lru_cache(maxsize=None)
def _suffix_skip_knob() -> bool:
    return rv_enabled() and os.environ.get("SPECPILOT_DB_RV_SUFFIX_SKIP", "0").strip() not in ("", "0")


def end() -> None:
    global _CUR
    _CUR = None


def pack_inputs(p: Pack, input_ids, positions, input_embeds):
    if input_embeds is not None:
        input_embeds = p.to_packed(input_embeds)
    elif input_ids is not None:
        input_ids = p.to_packed(input_ids)
    if positions is not None:
        positions = p.to_packed(positions, dim=positions.dim() - 1)
    return input_ids, positions, input_embeds


# ------------------------------------------------------------------------------------------------
# worker side (eager, after the target verify; no host sync)
# ------------------------------------------------------------------------------------------------


def graph_bs(bs: int, capture_bs) -> int:
    """The batch size the target verify graph runs at (raw bs padded to the captured list; eager above it)."""
    if capture_bs:
        for b in sorted(capture_bs):
            if b >= bs:
                return int(b)
    return int(bs)


def full_tier_step(bs: int, n_nodes: int, capture_bs) -> bool:
    """Host-side twin of ``begin``: True when this step's tier verifies all N nodes of every request (identity
    layout on the real rows), so the worker can take L's dense post-verify path."""
    cfg = rv_config()
    if cfg is None:
        return True
    gbs = graph_bs(bs, capture_bs)
    return cfg.tokens_for_bs(gbs, n_nodes) >= gbs * n_nodes


def worker_layout(bs: int, n_nodes: int):
    """(unpack [bs, N] int64 packed row of each dense node, n_eff [bs] int32) of the last target verify. Real requests
    come first in the packed layout, so their rows are unchanged by graph padding."""
    st = state()
    return st.unpack[: bs * n_nodes].view(bs, n_nodes), st.n_eff[:bs]


def mask_candidates(tokens: torch.Tensor, n_eff: torch.Tensor) -> torch.Tensor:
    """Tree tokens with every unverified node (j >= n_i) set to -1: it never matches a target prediction, so the
    greedy longest-path accept stays inside the verified prefix tree."""
    j = torch.arange(tokens.shape[1], device=tokens.device)
    return torch.where(j[None, :] < n_eff[:, None].to(j.dtype), tokens, torch.full_like(tokens, -1))


# ------------------------------------------------------------------------------------------------
# SPECPILOT_DB_RV_FUSE: packed in-projection rows -> dense GDN mixer inputs in the split kernel
# ------------------------------------------------------------------------------------------------


@triton.jit
def _rv_split_qkvzba_dense_kernel(
    mixed_qkv,  # [bs * N, 2 * H * Dk + HV * Dv] out (dense)
    z,  # [T, HV, Dv] out (packed; written by the dense row that owns each verified packed row)
    b,  # [bs * N, HV] out (dense)
    a,  # [bs * N, HV] out (dense)
    mixed_qkvz,  # [T, ...] packed rows: [all_q | all_k | all_v | all_z]
    mixed_ba,  # [T, ...] packed rows: [all_b | all_a]
    qkvz_row_stride,
    ba_row_stride,
    unpack_ptr,  # [bs * N] dense row -> packed source row (unverified nodes: 0)
    rows_out_ptr,  # [bs * N] dense row -> packed row, -1 if unverified
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
    V_POW2: tl.constexpr = True,  # SPECPILOT_TREE (tree-103): False walks a non-power-of-two v-group (27B: 3) per head
):
    # same element moves as fused_qkvzba_split_reshape_cat_contiguous_kernel, with a row map
    d, i_qk = tl.program_id(0).to(tl.int64), tl.program_id(1)
    src = tl.load(unpack_ptr + d)
    dst_z = tl.load(rows_out_ptr + d)
    V_PER_GROUP: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    TOTAL_Q: tl.constexpr = NUM_HEADS_QK * HEAD_QK
    TOTAL_V: tl.constexpr = NUM_HEADS_V * HEAD_V
    QKV_DIM_T: tl.constexpr = 2 * TOTAL_Q + TOTAL_V
    in_row = mixed_qkvz + src * qkvz_row_stride
    out_row = mixed_qkv + d * QKV_DIM_T
    oq = tl.arange(0, HEAD_QK)
    tl.store(out_row + i_qk * HEAD_QK + oq, tl.load(in_row + i_qk * HEAD_QK + oq))
    tl.store(out_row + TOTAL_Q + i_qk * HEAD_QK + oq, tl.load(in_row + TOTAL_Q + i_qk * HEAD_QK + oq))
    v_ld = in_row + 2 * TOTAL_Q + i_qk * V_PER_GROUP * HEAD_V
    if V_POW2:
        og = tl.arange(0, V_PER_GROUP * HEAD_V)
        tl.store(out_row + 2 * TOTAL_Q + i_qk * V_PER_GROUP * HEAD_V + og, tl.load(v_ld + og))
        zmask = (og < V_PER_GROUP * HEAD_V) & (dst_z >= 0)
        tl.store(
            z + tl.maximum(dst_z, 0) * TOTAL_V + i_qk * V_PER_GROUP * HEAD_V + og,
            tl.load(v_ld + TOTAL_V + og),
            mask=zmask,
        )
    else:  # SPECPILOT_TREE (tree-103): one HEAD_V-wide head at a time
        oh = tl.arange(0, HEAD_V)
        zm = (oh < HEAD_V) & (dst_z >= 0)
        for j in tl.static_range(V_PER_GROUP):
            off = i_qk * V_PER_GROUP * HEAD_V + j * HEAD_V + oh
            tl.store(out_row + 2 * TOTAL_Q + off, tl.load(in_row + 2 * TOTAL_Q + off))
            tl.store(z + tl.maximum(dst_z, 0) * TOTAL_V + off, tl.load(in_row + 2 * TOTAL_Q + TOTAL_V + off), mask=zm)
    ba_row = mixed_ba + src * ba_row_stride
    for i in tl.static_range(V_PER_GROUP):
        h = i_qk * V_PER_GROUP + i
        tl.store(b + d * NUM_HEADS_V + h, tl.load(ba_row + h))
        tl.store(a + d * NUM_HEADS_V + h, tl.load(ba_row + NUM_HEADS_V + h))


def split_qkvzba_dense(p: Pack, mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v):
    """Packed in-projection rows -> (mixed_qkv dense [bs*N, .], z packed [T, HV, Dv], b dense, a dense)."""
    from sglang.srt.layers import specpilot_d4_glue as _d4g  # SPECPILOT_D4

    if _d4g.SPLIT:  # SPECPILOT_D4: one program per dense row, same copies
        return _d4g.split_qkvzba_dense(p, mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v)
    v_per_group = num_heads_v // num_heads_qk
    assert v_per_group * num_heads_qk == num_heads_v
    v_pow2 = v_per_group & (v_per_group - 1) == 0  # SPECPILOT_TREE (tree-103): 27B has v_per_group 3
    rows = p.bs * p.n_nodes
    T = mixed_qkvz.shape[0]
    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkv = mixed_qkvz.new_empty((rows, qkv_dim_t))
    z = mixed_qkvz.new_empty((T, num_heads_v, head_v))
    b = mixed_ba.new_empty((rows, num_heads_v))
    a = torch.empty_like(b)
    _rv_split_qkvzba_dense_kernel[(rows, num_heads_qk)](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        mixed_qkvz.stride(0),
        mixed_ba.stride(0),
        p.unpack,
        p.rows_out,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        V_POW2=v_pow2,
        num_warps=1,
        num_stages=3,
    )
    return mixed_qkv, z, b, a
