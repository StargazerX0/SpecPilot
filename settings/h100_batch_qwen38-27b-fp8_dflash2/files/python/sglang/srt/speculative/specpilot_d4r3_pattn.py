"""SPECPILOT_D4R3_PATTN: tree-verify attention on the packed verify rows (d4r3, solo round 3).

ENGINE PATH = the hybrid at the end of this file (``build_fwd_meta`` + ``packed_tree_attention_fa3``): FA3 on the
packed rows + one tree-suffix merge / KV store / output-gate kernel per layer (microbench 83aa5da59e72: 16 layers
bs24 1272 -> 830 us, bs16 1026 -> 653 us). SPECPILOT_D4R3_PATTN_SFX='BM:warps' (default 16:4).
Engine hook: ``engine_forward`` from qwen3_5 ``self_attention`` when an rv Pack is active (replaces to_dense, the
backend's KV store + FA3 cascade + suffix merge, to_packed and fused_sigmoid_mul); the backend side is
``FlashAttentionBackend.specpilot_d4r3_packed_verify``. Only the verified nodes' K/V are stored (an unverified node is
never an ancestor of a verified one and never accepted). Needs SPECPILOT_DB_RA_WY_ANC=1 (ancestor bits).
The one-launch Triton kernel below (``packed_tree_attention``, variant ONE) is CLOSED (slower than FA3 at head_dim 256:
1.86 vs 1.62 ms) and kept only as a test reference together with ``reference_tree_attention``.

The seed's full-attention verify (16 layers) gathers the packed q/k/v rows into the dense ``[bs, N]`` layout, stores
the dense K/V, runs FA3 (scheduler metadata + split-KV forward + split combine) over the committed prefix, merges the
ancestor-masked tree suffix in a second kernel (per query row) and gathers the output back to the packed rows. This
kernel does all of it in one launch over the packed rows:

  grid (bs * MB [+1 pad-row program], KV heads, S splits); one program = one request's M-block of (node, q-head-of-the-
  KV-group) rows x one KV head x one prefix split:
    * queries: packed rows off_b + j (j < n_b, off_b = unpack[b * N]), the G query heads of the KV head as rows;
    * prefix: committed KV through the page table (keys [s * chunk, min(L_b, (s + 1) * chunk))), online softmax in
      fp32 (exp2), P in bf16 for the P.V tensor-core dot (as FA3);
    * suffix (split S - 1): the request's tree K/V straight from the packed k/v rows, masked by the per-node
      ancestor-or-self bits (``specpilot_db_conv.anc_buffer``, written in-graph by GDN layer 0); the same program
      stores those K/V rows into their cache slots (``out_cache_loc`` of the dense node), so no separate KV store and
      no K/V of unverified nodes is written (they are never ancestors of a verified node and never accepted);
    * S > 1: fp32 partials (unnormalised acc, running max, sum) go to a workspace; the last program of a (request,
      M-block, KV head) to arrive (atomic ticket, reset by it) merges the S partials in fixed order s = 0..S-1
      (deterministic) and writes bf16;
    * output written straight into the packed rows; graph padding rows (t >= sum n_b) are written as zeros.
Numerics: fp32 scores / softmax statistics / accumulation, bf16 P (FA3's choice for the prefix; the seed's suffix
merge used fp32 P with tf32x3), bf16 output.
Knob (read once at import): SPECPILOT_D4R3_PATTN=1 (unset = parent path); SPECPILOT_D4R3_PATTN_CFG=
'bsmax:S:BM:BN:warps:stages;...' overrides the per-batch-size launch table.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

_ENABLED = os.environ.get("SPECPILOT_D4R3_PATTN", "0") not in ("", "0")
# default launch table: bs <= bsmax -> (splits, BLOCK_M, BLOCK_N, warps, stages)
_DEFAULT_CFG = "8:6:64:64:4:2;16:3:64:64:4:2;1024:2:64:64:4:2"


def pattn_enabled() -> bool:
    return _ENABLED


def _parse_cfg(raw: str):
    out = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        bsmax, s, bm, bn, w, st = (int(x) for x in part.split(":"))
        out.append((bsmax, (s, bm, bn, w, st)))
    out.sort()
    return out


_CFG = _parse_cfg(os.environ.get("SPECPILOT_D4R3_PATTN_CFG", "") or _DEFAULT_CFG)


def launch_cfg(bs: int):
    for bsmax, cfg in _CFG:
        if bs <= bsmax:
            return cfg
    return _CFG[-1][1]


@triton.jit
def _pattn_kernel(
    q_ptr, k_ptr, v_ptr,  # packed rows: [T, HQ * D], [T, HK * D], [T, HK * D] (row strides below, unit inner)
    kc_ptr, vc_ptr,  # KV cache of the layer: [slots, HK, D] (slot strides below)
    pt_ptr,  # [bs, W] int32 prefix page table (page size 1)
    seqlen_ptr,  # [bs] int32 committed prefix length
    n_ptr,  # [bs] int32 verified nodes per request (n_eff)
    unpack_ptr,  # [bs * N] int64 dense row -> packed row (node 0 = packed offset of the request)
    anc_ptr,  # [bs * N] int32 ancestor-or-self bits per node
    loc_ptr,  # [bs * N] int64 cache slot of each dense node (out_cache_loc)
    o_ptr,  # packed output [T, HQ * D]
    part_ptr,  # fp32 [P, BM, D] split partials (S > 1)
    ml_ptr,  # fp32 [P, 2, BM] running max (log2 domain) / sum (S > 1)
    cnt_ptr,  # int32 [P // S] arrival tickets (zero between launches)
    scale_log2,  # softmax scale * log2(e)
    bs, T,
    q_stride, k_stride, v_stride, o_stride, pt_stride, kc_stride, vc_stride,
    N: tl.constexpr, G: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, MB: tl.constexpr, S: tl.constexpr,
    STORE_KV: tl.constexpr,
):
    pid0 = tl.program_id(0)
    kvh = tl.program_id(1)
    s = tl.program_id(2)
    d = tl.arange(0, D)
    if pid0 == bs * MB:
        # graph padding rows of the token tier: zeros (finite rows for o_proj and onwards)
        if s == 0:
            total = tl.load(unpack_ptr + (bs - 1) * N).to(tl.int32) + tl.load(n_ptr + bs - 1)
            r = tl.arange(0, BM)
            for t0 in range(total, T, BM // G):
                t = t0 + r // G
                g = r % G
                tl.store(o_ptr + t[:, None].to(tl.int64) * o_stride + (kvh * G + g)[:, None] * D + d[None, :],
                         tl.zeros([BM, D], dtype=o_ptr.dtype.element_ty),
                         mask=(t < T)[:, None] & ((r // G) < (BM // G))[:, None])
        return
    b = pid0 // MB
    mb = pid0 % MB
    n = tl.load(n_ptr + b)
    if mb * BM >= n * G:
        return
    off = tl.load(unpack_ptr + b * N).to(tl.int64)
    r = mb * BM + tl.arange(0, BM)
    j = r // G
    g = r % G
    rv = j < n
    head = kvh * G + g
    q = tl.load(q_ptr + (off + j)[:, None] * q_stride + head[:, None] * D + d[None, :], mask=rv[:, None], other=0.0)

    m_i = tl.full([BM], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, D], dtype=tl.float32)

    L = tl.load(seqlen_ptr + b)
    chunk = tl.cdiv(tl.cdiv(L, S), BN) * BN
    start = s * chunk
    end = tl.minimum(L, start + chunk)
    kn = tl.arange(0, BN)
    pt_row = pt_ptr + b.to(tl.int64) * pt_stride
    for k0 in range(start, end, BN):
        idx = k0 + kn
        km = idx < end
        slot = tl.load(pt_row + idx, mask=km, other=0).to(tl.int64)
        kt = tl.load(kc_ptr + slot[None, :] * kc_stride + kvh * D + d[:, None], mask=km[None, :], other=0.0)
        qk = tl.dot(q, kt) * scale_log2
        qk = tl.where(km[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(qk - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        vb = tl.load(vc_ptr + slot[:, None] * vc_stride + kvh * D + d[None, :], mask=km[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(vb.dtype), vb)
        m_i = m_new

    if s == S - 1:
        jj = tl.arange(0, N)
        tv = jj < n
        trow = off + jj
        kt = tl.load(k_ptr + trow[None, :] * k_stride + kvh * D + d[:, None], mask=tv[None, :], other=0.0)
        vt = tl.load(v_ptr + trow[:, None] * v_stride + kvh * D + d[None, :], mask=tv[:, None], other=0.0)
        if STORE_KV:
            if mb == 0:
                slot_t = tl.load(loc_ptr + b * N + jj, mask=tv, other=0).to(tl.int64)
                tl.store(kc_ptr + slot_t[None, :] * kc_stride + kvh * D + d[:, None], kt, mask=tv[None, :])
                tl.store(vc_ptr + slot_t[:, None] * vc_stride + kvh * D + d[None, :], vt, mask=tv[:, None])
        anc = tl.load(anc_ptr + b * N + j, mask=rv, other=0)
        vis = (((anc[:, None] >> jj[None, :]) & 1) != 0) & tv[None, :]
        qk = tl.dot(q, kt) * scale_log2
        qk = tl.where(vis, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(qk - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
        m_i = m_new

    o_ptrs = o_ptr + (off + j)[:, None] * o_stride + head[:, None] * D + d[None, :]
    if S == 1:
        out = acc / l_i[:, None]
        tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty), mask=rv[:, None])
    else:
        rb = (pid0 * HK + kvh)  # (request, M-block, KV head)
        pidx = rb * S + s
        rr = tl.arange(0, BM)
        tl.store(part_ptr + pidx.to(tl.int64) * (BM * D) + rr[:, None] * D + d[None, :], acc, cache_modifier=".cg")
        tl.store(ml_ptr + pidx.to(tl.int64) * (2 * BM) + rr, m_i, cache_modifier=".cg")
        tl.store(ml_ptr + pidx.to(tl.int64) * (2 * BM) + BM + rr, l_i, cache_modifier=".cg")
        tl.debug_barrier()
        ticket = tl.atomic_add(cnt_ptr + rb, 1, sem="acq_rel")
        if ticket == S - 1:
            tl.debug_barrier()
            base = rb.to(tl.int64) * S
            m_tot = tl.full([BM], float("-inf"), dtype=tl.float32)
            for t in range(S):
                m_t = tl.load(ml_ptr + (base + t) * (2 * BM) + rr, cache_modifier=".cg")
                m_tot = tl.maximum(m_tot, m_t)
            m_tot = tl.where(m_tot == float("-inf"), 0.0, m_tot)
            o_acc = tl.zeros([BM, D], dtype=tl.float32)
            l_tot = tl.zeros([BM], dtype=tl.float32)
            for t in range(S):
                m_t = tl.load(ml_ptr + (base + t) * (2 * BM) + rr, cache_modifier=".cg")
                l_t = tl.load(ml_ptr + (base + t) * (2 * BM) + BM + rr, cache_modifier=".cg")
                w = tl.exp2(m_t - m_tot)
                a_t = tl.load(part_ptr + (base + t) * (BM * D) + rr[:, None] * D + d[None, :], cache_modifier=".cg")
                o_acc += a_t * w[:, None]
                l_tot += l_t * w
            out = o_acc / l_tot[:, None]
            tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty), mask=rv[:, None])
            tl.atomic_xchg(cnt_ptr + rb, 0)


class Workspace:
    """Split partials + tickets, sized for the largest batch (static addresses for graph capture)."""

    def __init__(self, device, max_bs: int, n_nodes: int, hk: int, d: int, max_splits: int = 8, bm: int = 64):
        self.device = device
        g_max = 8  # query heads per KV head bound used for the M-block count
        mb = triton.cdiv(n_nodes * g_max, bm)
        p = (max_bs + 1) * mb * hk * max_splits
        self.part = torch.empty((p * bm * d,), dtype=torch.float32, device=device)
        self.ml = torch.empty((p * 2 * bm,), dtype=torch.float32, device=device)
        self.cnt = torch.zeros((p,), dtype=torch.int32, device=device)
        self.max_splits, self.bm_max, self.cap = max_splits, bm, p


_WS: Optional[Workspace] = None


def workspace(device, max_bs: int = 64, n_nodes: int = 16, hk: int = 4, d: int = 256) -> Workspace:
    global _WS
    if _WS is None:
        _WS = Workspace(device, max_bs, n_nodes, hk, d)
    return _WS


def packed_tree_attention(
    q: torch.Tensor,  # [T, HQ * D] packed (row-major heads)
    k: torch.Tensor,  # [T, HK * D] packed, unit inner stride
    v: torch.Tensor,  # [T, HK * D] packed, unit inner stride (row stride may be the qkv GEMM's)
    k_cache: torch.Tensor,  # [slots, HK, D] (or [slots, 1, HK, D])
    v_cache: torch.Tensor,
    page_table: torch.Tensor,  # [>= bs, W] int32
    seq_lens: torch.Tensor,  # [>= bs] int32 committed prefix lengths
    n_eff: torch.Tensor,  # [bs] int32
    unpack: torch.Tensor,  # [bs * N] int64
    anc_bits: torch.Tensor,  # [bs * N] int32
    out_cache_loc: Optional[torch.Tensor],  # [bs * N] int64 (None: K/V of the tree not stored)
    softmax_scale: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    n_nodes: int,
    out: Optional[torch.Tensor] = None,
    cfg=None,
    ws: Optional[Workspace] = None,
) -> torch.Tensor:
    T = q.shape[0]
    bs = n_eff.shape[0]
    HQ, HK, D, N = num_q_heads, num_kv_heads, head_dim, n_nodes
    G = HQ // HK
    S, BM, BN, warps, stages = cfg if cfg is not None else launch_cfg(bs)
    MB = triton.cdiv(N * G, BM)
    kc = k_cache.view(k_cache.shape[0] * (k_cache.shape[1] if k_cache.dim() == 4 else 1), HK, D)
    vc = v_cache.view(v_cache.shape[0] * (v_cache.shape[1] if v_cache.dim() == 4 else 1), HK, D)
    if out is None:
        out = torch.empty((T, HQ * D), dtype=q.dtype, device=q.device)
    if ws is None:
        ws = workspace(q.device, hk=HK, d=D)
    assert S <= ws.max_splits and BM <= ws.bm_max and (bs + 1) * MB * HK * S <= ws.cap
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1 and out.stride(-1) == 1
    assert kc.stride(2) == 1 and kc.stride(1) == D and vc.stride(2) == 1 and vc.stride(1) == D
    store = out_cache_loc is not None
    _pattn_kernel[(bs * MB + 1, HK, S)](
        q, k, v, kc, vc, page_table, seq_lens, n_eff, unpack, anc_bits,
        out_cache_loc if store else unpack, out, ws.part, ws.ml, ws.cnt,
        softmax_scale * 1.4426950408889634, bs, T,
        q.stride(0), k.stride(0), v.stride(0), out.stride(0), page_table.stride(0), kc.stride(0), vc.stride(0),
        N=N, G=G, HK=HK, D=D, BM=BM, BN=BN, MB=MB, S=S, STORE_KV=store,
        num_warps=warps, num_stages=stages,
    )
    return out


def reference_tree_attention(q, k, v, k_cache, v_cache, page_table, seq_lens, n_eff, unpack, anc_bits, softmax_scale,
                             num_q_heads, num_kv_heads, head_dim, n_nodes):
    """fp32 reference on the packed rows (tests): prefix keys from the cache, suffix keys from the packed rows."""
    HQ, HK, D, N = num_q_heads, num_kv_heads, head_dim, n_nodes
    G = HQ // HK
    kc = k_cache.reshape(-1, HK, D).float()
    vc = v_cache.reshape(-1, HK, D).float()
    out = torch.zeros(q.shape[0], HQ, D, dtype=torch.float32, device=q.device)
    qf = q.view(-1, HQ, D).float()
    kf = k.reshape(k.shape[0], HK, D).float()
    vf = v.reshape(v.shape[0], HK, D).float()
    for b in range(n_eff.shape[0]):
        n = int(n_eff[b])
        off = int(unpack[b * N])
        L = int(seq_lens[b])
        slots = page_table[b, :L].long()
        kp, vp = kc[slots], vc[slots]  # [L, HK, D]
        for jn in range(n):
            bits = int(anc_bits[b * N + jn])
            anc = [x for x in range(n) if (bits >> x) & 1]
            ks = torch.cat([kp, kf[off + torch.tensor(anc, device=q.device)]], 0)
            vs = torch.cat([vp, vf[off + torch.tensor(anc, device=q.device)]], 0)
            for h in range(HQ):
                kvh = h // G
                sc = (ks[:, kvh] @ qf[off + jn, h]) * softmax_scale
                p = torch.softmax(sc, 0)
                out[off + jn, h] = p @ vs[:, kvh]
    return out.view(q.shape[0], HQ * D)


# ---------------------------------------------------------------------------------------------------------------------
# Hybrid path (SPECPILOT_D4R3_PATTN=1): FA3 over the committed prefix straight on the packed rows (varlen q from n_eff,
# scheduler metadata computed once per forward and shared by the 16 layers, output written into the packed buffer),
# then ONE small kernel per layer that adds the ancestor-masked tree suffix from the packed k/v rows, merges it with
# FA3's (out, lse) in place, stores the verified nodes' K/V into their cache slots and zeroes the graph padding rows.
# Removes per layer: 3 dense q/k/v gathers, store_cache, prepare_varlen_num_blocks, the per-query suffix merge and the
# output gather; FA3 forward + split combine stay.
# ---------------------------------------------------------------------------------------------------------------------


@triton.jit
def _pattn_sfx_kernel(
    q_ptr, k_ptr, v_ptr,  # packed rows (unit inner stride)
    o_ptr,  # packed FA3 prefix output [T, HQ * D] (bf16), merged in place
    lse_ptr,  # FA3 prefix lse [HQ, T] fp32 (natural log)
    kc_ptr, vc_ptr,  # KV cache [slots, HK, D]
    n_ptr, unpack_ptr, anc_ptr, loc_ptr,
    gate_ptr,  # packed attention output gate (GATE): out = merged * sigmoid(gate)
    scale, bs, T,
    q_stride, k_stride, v_stride, o_stride, lse_stride_h, kc_stride, vc_stride, gate_stride_t, gate_stride_h,
    N: tl.constexpr, G: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, MB: tl.constexpr,
    STORE_KV: tl.constexpr, GATE: tl.constexpr, PV_TF32X3: tl.constexpr = True,
):
    pid0 = tl.program_id(0)
    kvh = tl.program_id(1)
    d = tl.arange(0, D)
    if pid0 == bs * MB:  # graph padding rows of the token tier: zeros
        total = tl.load(unpack_ptr + (bs - 1) * N).to(tl.int32) + tl.load(n_ptr + bs - 1)
        PR: tl.constexpr = 16
        tr = tl.arange(0, PR)
        zero = tl.zeros([PR, D], dtype=o_ptr.dtype.element_ty)
        for t0 in range(total, T, PR):
            t = (t0 + tr).to(tl.int64)
            for g in range(G):
                tl.store(o_ptr + t[:, None] * o_stride + (kvh * G + g) * D + d[None, :], zero,
                         mask=(t < T)[:, None])
        return
    b = pid0 // MB
    mb = pid0 % MB
    n = tl.load(n_ptr + b)
    off = tl.load(unpack_ptr + b * N).to(tl.int64)
    if mb * BM >= n * G:
        return
    r = mb * BM + tl.arange(0, BM)
    j = r // G
    rv = j < n
    head = kvh * G + r % G
    jj = tl.arange(0, N)
    tv = jj < n
    trow = off + jj
    # every load issued up front (one memory round trip per program; the kernel is latency-bound)
    o_ptrs = o_ptr + (off + j)[:, None] * o_stride + head[:, None] * D + d[None, :]
    q = tl.load(q_ptr + (off + j)[:, None] * q_stride + head[:, None] * D + d[None, :], mask=rv[:, None], other=0.0)
    kt = tl.load(k_ptr + trow[None, :] * k_stride + kvh * D + d[:, None], mask=tv[None, :], other=0.0)  # [D, N]
    vt = tl.load(v_ptr + trow[:, None] * v_stride + kvh * D + d[None, :], mask=tv[:, None], other=0.0)  # [N, D]
    anc = tl.load(anc_ptr + b * N + j, mask=rv, other=1)
    lse = tl.load(lse_ptr + head * lse_stride_h + (off + j), mask=rv, other=0.0)
    pre = tl.load(o_ptrs, mask=rv[:, None], other=0.0)
    if GATE:
        gt = tl.load(gate_ptr + (off + j)[:, None] * gate_stride_t + head[:, None] * gate_stride_h + d[None, :],
                     mask=rv[:, None], other=0.0)
    if STORE_KV:
        if mb == 0:
            slot_t = tl.load(loc_ptr + b * N + jj, mask=tv, other=0).to(tl.int64)
            tl.store(kc_ptr + slot_t[None, :] * kc_stride + kvh * D + d[:, None], kt, mask=tv[None, :])
            tl.store(vc_ptr + slot_t[:, None] * vc_stride + kvh * D + d[None, :], vt, mask=tv[:, None])
    sc = tl.dot(q, kt) * scale  # [BM, N] fp32
    vis = (((anc[:, None] >> jj[None, :]) & 1) != 0) & tv[None, :]
    sc = tl.where(vis, sc, float("-inf"))
    m = tl.maximum(lse, tl.max(sc, 1))
    wp = tl.exp(lse - m)
    p = tl.exp(sc - m[:, None])
    den = wp + tl.sum(p, 1)
    if PV_TF32X3:
        num = tl.dot(p, vt.to(tl.float32), input_precision="tf32x3")
    else:
        num = tl.dot(p.to(vt.dtype), vt)
    out = ((pre.to(tl.float32) * wp[:, None] + num) / den[:, None]).to(o_ptr.dtype.element_ty)
    if GATE:  # as the seed's fused_sigmoid_mul on the bf16 attention output (bitwise)
        out = (out.to(tl.float32) * tl.sigmoid(gt.to(tl.float32))).to(o_ptr.dtype.element_ty)
    tl.store(o_ptrs, out, mask=rv[:, None])


_SFX_CFG = tuple(int(x) for x in (os.environ.get("SPECPILOT_D4R3_PATTN_SFX", "") or "16:4").split(":"))


class FwdMeta:
    """Per-forward packed-attention metadata shared by the 16 attention layers (built once, in-graph)."""

    def __init__(self, cu_seqlens_q, scheduler_metadata):
        self.cu_seqlens_q = cu_seqlens_q
        self.scheduler_metadata = scheduler_metadata


def build_fwd_meta(n_eff, cache_seqlens, page_table, num_q_heads, num_kv_heads, head_dim, n_nodes,
                   use_scheduler_metadata=True, num_splits=0) -> FwdMeta:
    cuq = torch.nn.functional.pad(torch.cumsum(n_eff, 0, dtype=torch.int32), (1, 0))
    sm = None
    if use_scheduler_metadata:
        from sgl_kernel.flash_attn import get_scheduler_metadata

        sm = get_scheduler_metadata(
            batch_size=n_eff.shape[0], max_seqlen_q=n_nodes, max_seqlen_k=page_table.shape[1],
            num_heads=num_q_heads, num_heads_k=num_kv_heads, headdim=head_dim, cache_seqlens=cache_seqlens,
            qkv_dtype=torch.bfloat16, cu_seqlens_q=cuq, page_size=1, causal=False, num_splits=num_splits)
    return FwdMeta(cuq, sm)


def packed_tree_attention_fa3(
    q, k, v, k_cache, v_cache, page_table, seq_lens, n_eff, unpack, anc_bits, out_cache_loc, softmax_scale,
    num_q_heads, num_kv_heads, head_dim, n_nodes, meta: FwdMeta, out=None, num_splits=0, sfx_cfg=None, gate=None,
):
    from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache

    T = q.shape[0]
    bs = n_eff.shape[0]
    HQ, HK, D, N = num_q_heads, num_kv_heads, head_dim, n_nodes
    G = HQ // HK
    kc = k_cache.view(-1, HK, D)
    vc = v_cache.view(-1, HK, D)
    if out is None:
        out = torch.empty((T, HQ * D), dtype=q.dtype, device=q.device)
    o, lse = flash_attn_with_kvcache(
        q=q.view(T, HQ, D), k_cache=kc.view(-1, 1, HK, D), v_cache=vc.view(-1, 1, HK, D), page_table=page_table,
        cache_seqlens=seq_lens, cu_seqlens_q=meta.cu_seqlens_q, max_seqlen_q=N, softmax_scale=softmax_scale,
        causal=False, window_size=(-1, -1), softcap=0.0, return_softmax_lse=True, num_splits=num_splits,
        scheduler_metadata=meta.scheduler_metadata, out=out.view(T, HQ, D), ver=3)[:2]
    BM, warps = sfx_cfg if sfx_cfg is not None else _SFX_CFG
    MB = triton.cdiv(N * G, BM)
    store = out_cache_loc is not None
    if gate is not None:  # [T, HQ * D] or [T, HQ, D] view (unit inner stride)
        g3 = gate.view(T, HQ, D) if gate.dim() == 2 else gate
        assert g3.stride(2) == 1
        gst, gsh = g3.stride(0), g3.stride(1)
    else:
        g3, gst, gsh = q, 0, 0
    _pattn_sfx_kernel[(bs * MB + 1, HK)](
        q, k, v, out, lse, kc, vc, n_eff, unpack, anc_bits, out_cache_loc if store else unpack, g3,
        softmax_scale, bs, T,
        q.stride(0), k.stride(0), v.stride(0), out.stride(0), lse.stride(0), kc.stride(0), vc.stride(0), gst, gsh,
        N=N, G=G, HK=HK, D=D, BM=BM, MB=MB, STORE_KV=store, GATE=gate is not None, num_warps=warps, num_stages=1,
    )
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Engine hook (qwen3_5 full-attention layers, target verify with an active rv Pack). Everything below runs inside the
# captured target-verify graph: no host syncs, static buffers only (rv state, db_conv ancestor bits, backend metadata).
# ---------------------------------------------------------------------------------------------------------------------


def engine_forward(*, layer, q, k, v, gate, forward_batch, pack) -> Optional[torch.Tensor]:
    """Gated attention output [T, HQ * D] on the packed rows, or None (the caller runs the seed's dense path)."""
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import HybridLinearAttnBackend
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    if isinstance(backend, HybridLinearAttnBackend):
        backend = backend.full_attn_backend
    if not isinstance(backend, FlashAttentionBackend):
        return None
    return backend.specpilot_d4r3_packed_verify(
        q=q, k=k, v=v, gate=gate, layer=layer, forward_batch=forward_batch, pack=pack
    )


def packed_verify(*, q, k, v, gate, k_cache, v_cache, page_table, cache_seqlens, out_cache_loc, softmax_scale,
                  num_q_heads, num_kv_heads, head_dim, num_splits, pack) -> torch.Tensor:
    from sglang.srt.speculative import specpilot_db_conv as _dbconv

    bs, n_nodes = pack.bs, pack.n_nodes
    if pack.d4r3_meta is None:  # first attention layer of this forward; the later layers share it
        pack.d4r3_meta = build_fwd_meta(pack.n_eff, cache_seqlens, page_table, num_q_heads, num_kv_heads, head_dim,
                                        n_nodes, num_splits=num_splits)
    # ancestor-or-self bits of this verify's tree, written in-graph by the first GDN layer's tree conv
    anc = _dbconv.anc_view(q.device, bs, n_nodes).view(-1)
    T = q.shape[0]
    return packed_tree_attention_fa3(
        q.view(T, -1), k.reshape(T, -1), v.reshape(T, -1), k_cache, v_cache, page_table, cache_seqlens,
        pack.n_eff, pack.unpack, anc, out_cache_loc, softmax_scale, num_q_heads, num_kv_heads, head_dim, n_nodes,
        pack.d4r3_meta, num_splits=num_splits, gate=gate,
    )
