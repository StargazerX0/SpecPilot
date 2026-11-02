"""SPECPILOT_D8_ATTN=1: tree-verify attention for the full-attention layers at batch 1 in two PDL-chained kernels.

The parent runs per full-attention layer of a tree verify: FA3 prefix attention over the context (scheduler metadata
+ split-KV forward + combine), the Triton suffix merge (tree keys of every query row, LSE merge with the prefix) and the
sigmoid output gate: five launches, ~28 us after the qk-norm/rope and KV store. Here:

  * ``_d8_tree_attn_kernel``, grid (q block, split, kv head), the G query heads of a KV head packed as rows
    (row = token * G + head): splits 0..S-1 run the context prefix [0, L) in equal BN-aligned chunks (online softmax,
    bf16 tensor-core dots, fp32 statistics, next block's slots prefetched, K and V issued together); split S runs the
    N tree keys (node j's slot = last entry of its suffix row) masked per row by its ancestor-or-self set (the
    cascade's suffix page table rows; P as a hi + lo bf16 pair, ~16 bits); every program stores its partial
    (acc, m, l); early PDL trigger;
  * ``_d8_tree_attn_combine_kernel`` (PDL), grid (row block, kv head): loads the output gate before the dependency
    wait, merges the S + 1 partials, applies the sigmoid gate to the bf16-rounded output and stores bf16.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_D8_ATTN = os.environ.get("SPECPILOT_D8_ATTN", "0") not in ("", "0")
_D8_ATTN_SPLITS = int(os.environ.get("SPECPILOT_D8_ATTN_SPLITS", "8") or 8)
_D8_ATTN_BM = int(os.environ.get("SPECPILOT_D8_ATTN_BM", "64") or 64)
_D8_ATTN_BN = int(os.environ.get("SPECPILOT_D8_ATTN_BN", "64") or 64)
_D8_ATTN_WARPS = int(os.environ.get("SPECPILOT_D8_ATTN_WARPS", "8") or 8)
_D8_ATTN_RB = int(os.environ.get("SPECPILOT_D8_ATTN_RB", "8") or 8)
_D8_TRIGGER = os.environ.get("SPECPILOT_D8_TRIGGER", "0") not in ("", "0")  # combine triggers its dependents early


def enabled() -> bool:
    return _D8_ATTN


@triton.jit
def _d8_tree_attn_kernel(
    q_ptr, k_ptr, v_ptr, pt_ptr, plen_ptr, st_ptr, slen_ptr, pacc_ptr, pml_ptr, scale,
    sq_t, sq_h, sk_t, sk_h, spt, sst, P_W, NSLOT,
    N: tl.constexpr, NP: tl.constexpr, G: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, S: tl.constexpr, QB: tl.constexpr, PDL: tl.constexpr,
    SFX_F32: tl.constexpr = True,
):
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()  # the combine kernel launches and loads the gate meanwhile
    i_qb, i_s, i_kv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = i_qb * BM + tl.arange(0, BM)
    rvalid = rows < N * G
    tok = rows // G
    hq = i_kv * G + rows % G
    dd = tl.arange(0, D)
    q = tl.load(q_ptr + tok[:, None].to(tl.int64) * sq_t + hq[:, None] * sq_h + dd[None, :], mask=rvalid[:, None],
                other=0.0)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    qk_scale = scale * 1.44269504088896  # exp2 domain
    if i_s < S:
        L = tl.minimum(tl.maximum(tl.load(plen_ptr), 0), P_W)  # memory-safe for capture-time metadata
        chunk = tl.cdiv(tl.cdiv(L, S), BN) * BN
        start = i_s * chunk
        end = tl.minimum(L, start + chunk)
        kj = start + tl.arange(0, BN)
        slot = tl.load(pt_ptr + kj * spt, mask=kj < end, other=0)
        slot = tl.minimum(tl.maximum(slot, 0), NSLOT - 1).to(tl.int64)
        for j in tl.range(start, end, BN):
            kval = kj < end
            kb = tl.load(k_ptr + slot[:, None] * sk_t + i_kv * sk_h + dd[None, :], mask=kval[:, None], other=0.0)
            vb = tl.load(v_ptr + slot[:, None] * sk_t + i_kv * sk_h + dd[None, :], mask=kval[:, None], other=0.0)
            kn = kj + BN
            slot = tl.load(pt_ptr + kn * spt, mask=kn < end, other=0)  # next block's slots
            slot = tl.minimum(tl.maximum(slot, 0), NSLOT - 1).to(tl.int64)
            s = tl.dot(q, tl.trans(kb)) * qk_scale
            s = tl.where(kval[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(vb.dtype), vb)
            m_i = m_new
            kj = kn
    else:  # tree suffix: node slots + per-row ancestor-or-self mask from the suffix page table
        nn = tl.arange(0, NP)
        nval = nn < N
        nlen = tl.minimum(tl.maximum(tl.load(slen_ptr + nn, mask=nval, other=1), 1), N)
        nslot = tl.load(st_ptr + nn * sst + (nlen - 1), mask=nval, other=-1)
        nslot = tl.where(nval, tl.minimum(tl.maximum(nslot, 0), NSLOT - 1), -1)
        rlen = tl.minimum(tl.load(slen_ptr + tok, mask=rvalid, other=0), N)
        vis = tl.zeros([BM, NP], tl.int1)
        for kk in tl.static_range(N):
            e = tl.load(st_ptr + tok * sst + kk, mask=rvalid & (kk < rlen), other=-2)
            vis = vis | (e[:, None] == nslot[None, :])
        vis = vis & nval[None, :]
        ns64 = tl.maximum(nslot, 0).to(tl.int64)
        kb = tl.load(k_ptr + ns64[:, None] * sk_t + i_kv * sk_h + dd[None, :], mask=nval[:, None], other=0.0)
        s = tl.dot(q, tl.trans(kb)) * qk_scale
        s = tl.where(vis, s, float("-inf"))
        m_i = tl.max(s, 1)
        m_i = tl.where(rvalid, m_i, 0.0)
        p = tl.exp2(s - m_i[:, None])
        p = tl.where(vis, p, 0.0)
        l_i = tl.sum(p, 1)
        vt = tl.load(v_ptr + ns64[:, None] * sk_t + i_kv * sk_h + dd[None, :], mask=nval[:, None], other=0.0)
        if SFX_F32:  # P as a hi + lo bf16 pair: ~16 mantissa bits (V is bf16, products exact in fp32)
            p_hi = p.to(vt.dtype)
            p_lo = (p - p_hi.to(tl.float32)).to(vt.dtype)
            acc = tl.dot(p_hi, vt) + tl.dot(p_lo, vt)
        else:
            acc = tl.dot(p.to(vt.dtype), vt)
    # partial out
    pid = (i_kv * QB + i_qb) * (S + 1) + i_s
    rr = tl.arange(0, BM)
    tl.store(pacc_ptr + (pid * BM + rr[:, None]) * D + dd[None, :], acc)
    tl.store(pml_ptr + (pid * 2) * BM + rr, m_i)
    tl.store(pml_ptr + (pid * 2 + 1) * BM + rr, l_i)


@triton.jit
def _d8_tree_attn_combine_kernel(
    gate_ptr, out_ptr, pacc_ptr, pml_ptr, sg_t, sg_h, so_t,
    N: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, S: tl.constexpr, QB: tl.constexpr,
    RB: tl.constexpr, HAS_GATE: tl.constexpr, PDL: tl.constexpr, TRIGGER: tl.constexpr = False,
):
    if TRIGGER:  # a PDL-launched o_proj GEMM may pre-stage its weights now (it waits for this kernel's output)
        tl.extra.cuda.gdc_launch_dependents()
    # merge the S + 1 partials of RB rows of one KV head, sigmoid output gate, bf16 store
    i_r, i_kv = tl.program_id(0), tl.program_id(1)
    rows = i_r * RB + tl.arange(0, RB)
    rvalid = rows < N * G
    tok = rows // G
    hq = i_kv * G + rows % G
    dd = tl.arange(0, D)
    if HAS_GATE:
        gt = tl.load(gate_ptr + tok[:, None].to(tl.int64) * sg_t + hq[:, None] * sg_h + dd[None, :],
                     mask=rvalid[:, None], other=0.0).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
    qb = rows // BM
    rr = rows % BM
    base = (i_kv * QB + qb) * (S + 1)
    mx = tl.full([RB], float("-inf"), tl.float32)
    for t in tl.static_range(S + 1):
        mt = tl.load(pml_ptr + ((base + t) * 2) * BM + rr, mask=rvalid, other=float("-inf"))
        mx = tl.maximum(mx, mt)
    num = tl.zeros([RB, D], tl.float32)
    den = tl.zeros([RB], tl.float32)
    for t in tl.static_range(S + 1):
        mt = tl.load(pml_ptr + ((base + t) * 2) * BM + rr, mask=rvalid, other=float("-inf"))
        lt = tl.load(pml_ptr + ((base + t) * 2 + 1) * BM + rr, mask=rvalid, other=0.0)
        w = tl.where(mt == float("-inf"), 0.0, tl.exp2(mt - mx))
        at = tl.load(pacc_ptr + ((base + t) * BM + rr)[:, None] * D + dd[None, :], mask=rvalid[:, None], other=0.0)
        num += w[:, None] * at
        den += w * lt
    o = num / tl.where(rvalid, den, 1.0)[:, None]
    o = o.to(out_ptr.dtype.element_ty).to(tl.float32)
    if HAS_GATE:
        o = o * tl.sigmoid(gt)
    tl.store(out_ptr + tok[:, None].to(tl.int64) * so_t + hq[:, None] * D + dd[None, :],
             o.to(out_ptr.dtype.element_ty), mask=rvalid[:, None])


_SCRATCH: dict = {}


def _buf(name, shape, dtype, device, zero=False):
    key = (name, tuple(int(s) for s in shape), dtype, str(device))
    t = _SCRATCH.get(key)
    if t is None:
        t = (torch.zeros if zero else torch.empty)(key[1], dtype=dtype, device=device)
        _SCRATCH[key] = t
    return t


def tree_attention(q, k_cache, v_cache, page_table, prefix_len, sfx_table, sfx_len, gate, scale, n_nodes,
                   splits=None, bm=None, bn=None, num_warps=None, rb=None, pdl=True, sfx_f32=True):
    """bs 1. q [N, HQ, D] (any token / head stride, unit dim stride); k/v_cache [slots, HKV, D]; page_table [P] int32
    prefix slots (page size 1); prefix_len [1] int32; sfx_table [N, N] int32 suffix slots per row (valid first, the
    row's own slot last); sfx_len [N] int32; gate [N, HQ, D] view or None. Returns out [N, HQ * D] (gated)."""
    S = splits or _D8_ATTN_SPLITS
    BM = bm or _D8_ATTN_BM
    BN = bn or _D8_ATTN_BN
    RB = rb or _D8_ATTN_RB
    nw = num_warps or _D8_ATTN_WARPS
    N = int(n_nodes)
    T, HQ, D = q.shape
    assert T == N and q.stride(-1) == 1 and k_cache.stride(-1) == 1 and k_cache.stride() == v_cache.stride()
    HKV = k_cache.shape[1]
    G = HQ // HKV
    QB = triton.cdiv(N * G, BM)
    dev = q.device
    pacc = _buf("pacc", (HKV * QB * (S + 1), BM, D), torch.float32, dev)
    pml = _buf("pml", (HKV * QB * (S + 1), 2, BM), torch.float32, dev)
    out = q.new_empty(T, HQ * D)
    g = gate if gate is not None else q
    assert g.stride(-1) == 1
    _d8_tree_attn_kernel[(QB, S + 1, HKV)](
        q, k_cache, v_cache, page_table, prefix_len, sfx_table, sfx_len, pacc, pml, float(scale),
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), page_table.stride(-1), sfx_table.stride(0),
        page_table.shape[-1], k_cache.shape[0],
        N=N, NP=max(16, triton.next_power_of_2(N)), G=G, HKV=HKV, D=D, BM=BM, BN=BN, S=S, QB=QB, PDL=pdl, SFX_F32=sfx_f32,
        num_warps=nw, num_stages=2,
    )
    kw = {"launch_pdl": True} if pdl else {}
    _d8_tree_attn_combine_kernel[(triton.cdiv(N * G, RB), HKV)](
        g, out, pacc, pml, g.stride(0), g.stride(1), out.stride(0),
        N=N, G=G, D=D, BM=BM, S=S, QB=QB, RB=RB, HAS_GATE=gate is not None, PDL=pdl, TRIGGER=_D8_TRIGGER, num_warps=4,
        **kw,
    )
    return out


_LOGGED: set = set()


def _why(reason):
    """Log each fallback reason once (the hook runs at graph capture / eager steps only, never per replay)."""
    if reason not in _LOGGED:
        _LOGGED.add(reason)
        import logging

        logging.getLogger(__name__).warning("SPECPILOT_D8_ATTN: %s", reason)
    return None


def verify_layer(attn, q, k, v, gate, forward_batch):
    """SPECPILOT_D8_ATTN hook for Qwen3_5AttentionDecoderLayer.self_attention (tree target verify, batch 1): stores
    the step's K/V like the FA3 backend, runs the tree attention (+ PDL combine/gate kernel) and returns the gated
    output [T, HQ * D] (input of o_proj), or None to run the parent path."""
    if not _D8_ATTN or not forward_batch.forward_mode.is_target_verify():
        return None
    from sglang.srt.mem_cache.memory_pool import KVWriteLoc
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    fa = getattr(backend, "full_attn_backend", backend)
    expand = getattr(fa, "forward_metadata_spec_decode_expand", None)
    md = getattr(fa, "forward_metadata", None)
    if expand is None or md is None:
        return _why(f"fallback: no cascade metadata on {type(fa).__name__}")
    if not getattr(fa, "_specpilot_tree_fused_suffix", False) or getattr(fa, "page_size", 0) != 1 or getattr(fa, "use_mla", False):
        return _why(f"fallback: backend {type(fa).__name__} tree_suffix={getattr(fa, '_specpilot_tree_fused_suffix', None)} "
                    f"page_size={getattr(fa, 'page_size', None)}")
    if (attn.logit_cap or attn.is_cross_attention or attn.head_dim != attn.v_head_dim
            or (attn.sliding_window_size is not None and attn.sliding_window_size > -1)):
        return _why("fallback: layer attributes")
    N = int(forward_batch.spec_info.draft_token_num)
    T = q.shape[0]
    if T != N or N > 64 or forward_batch.batch_size != 1:
        return _why(f"fallback: T={T} N={N} bs={forward_batch.batch_size}")
    HQ, HKV, D = attn.tp_q_head_num, attn.tp_k_head_num, attn.head_dim
    key_cache, value_cache = fa.token_to_kv_pool.get_kv_buffer(attn.layer_id)
    if key_cache.dtype != q.dtype:
        return _why(f"fallback: kv dtype {key_cache.dtype}")
    k = k.view(-1, HKV, D)
    v = v.view(-1, attn.tp_v_head_num, attn.v_head_dim)
    fa.token_to_kv_pool.set_kv_buffer(  # as FlashAttentionBackend.forward_extend (save_kv_cache)
        attn, KVWriteLoc(forward_batch.out_cache_loc, md.swa_out_cache_loc), k, v, attn.k_scale, attn.v_scale
    )
    q3 = q.view(T, HQ, D)
    g3 = None
    if gate is not None:
        g3 = gate if gate.dim() == 3 else gate.view(T, HQ, D)
    _why(f"active: N={N} HQ={HQ} HKV={HKV} D={D}")
    return tree_attention(
        q3, key_cache.view(-1, HKV, D), value_cache.view(-1, HKV, D), md.page_table[0], md.cache_seqlens_int32[:1],
        expand.page_table[:N], expand.cache_seqlens_int32[:N], g3, attn.scaling, N,
    )
