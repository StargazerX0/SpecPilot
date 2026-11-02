"""SPECPILOT_D5_SPLITKV: split-KV tree-verify attention for the triton backend (verify forward, custom tree mask).

The parent's ``extend_attention_fwd`` runs one CTA per (request, q head) over the whole prefix (grid [bs, HQ, 1]:
24 CTAs on 188 SMs at batch 1), so the verify attention grows ~0.13 us per context token per layer (microbench
588d8660acfd). Here, two kernels (lib tree_attn2's mechanism on the triton backend's metadata):

  * ``_d5_tv_kernel``, grid (request x q block, split, kv head): the G query heads of a KV head are packed as rows
    (row = token * G + head), so each KV block is read once per KV head instead of once per q head. Splits 0..S-1 run
    the prefix [0, L) of kv_indices in equal BN-aligned chunks (online softmax, fp32 statistics, bf16 tensor-core
    dots); split S runs the N fresh draft keys (k_extend / v_extend) masked per row by the custom mask's suffix block
    (the prefix columns are all visible, as extend's skip_prefix_custom_mask). Each program stores (acc, m, l).
  * ``_d5_tv_combine_kernel``, grid (request x row block, kv head): merges the S + 1 partials and stores bf16.

Same inputs and semantics as ``extend_attention_fwd`` for page size 1, bf16 KV, no sinks / logit cap / score_mod,
N <= 64 (the extend kernel's single q block); a sliding window (the DFlash2 drafter) is applied as extend does.

Knobs (read once at import; unset = parent): SPECPILOT_D5_SPLITKV=1 target verify, SPECPILOT_D5_SPLITKV_DRAFT=1 the
drafter's verify-mode forward; SPECPILOT_D5_SPLITKV_CFG / _DRAFT_CFG = "splits,bm,bn,warps" launch overrides.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip() not in ("", "0")


def _cfg(name: str, default: tuple) -> tuple:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    vals = tuple(int(x) for x in raw.split(","))
    if len(vals) != 4:
        raise ValueError(f"{name}={raw!r}: expected splits,bm,bn,warps")
    return vals


_TARGET_ON = _flag("SPECPILOT_D5_SPLITKV")
_DRAFT_ON = _flag("SPECPILOT_D5_SPLITKV_DRAFT")
# Microbench 588d8660acfd (sm_120): target 24q/4kv x 256 best at S8 BM32 BN32 4 warps; drafter 32q/8kv x 128 at
# S16 BM64 BN64 (equal to S8 at L <= 512, 1.3-1.8x faster at 1-2k).
_TARGET_CFG = _cfg("SPECPILOT_D5_SPLITKV_CFG", (8, 32, 32, 4))
_DRAFT_CFG = _cfg("SPECPILOT_D5_SPLITKV_DRAFT_CFG", (16, 64, 64, 4))
MAX_TOKENS = 64


def launch_config(is_draft_worker: bool):
    """(splits, bm, bn, warps) for this worker's verify forward, or None to run the parent extend kernel."""
    if is_draft_worker:
        return _DRAFT_CFG if _DRAFT_ON else None
    return _TARGET_CFG if _TARGET_ON else None


@triton.jit
def _d5_tv_kernel(
    q_ptr, ke_ptr, ve_ptr, kb_ptr, vb_ptr, kv_indptr_ptr, kv_indices_ptr, mask_ptr, mask_indptr_ptr, woff_ptr,
    pacc_ptr, pml_ptr, scale,
    sq_t, sq_h, ske_t, ske_h, sve_t, sve_h, skb_t, skb_h, svb_t, svb_h,
    N: tl.constexpr, NP: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, S: tl.constexpr, QB: tl.constexpr,
    HAS_MASK: tl.constexpr, CAUSAL: tl.constexpr, WINDOW: tl.constexpr,
):
    i_bq, i_s, i_kv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    HKV = tl.num_programs(2)
    b = i_bq // QB
    i_qb = i_bq % QB
    rows = i_qb * BM + tl.arange(0, BM)
    rvalid = rows < N * G
    tok = rows // G
    hq = i_kv * G + rows % G
    dd = tl.arange(0, D)
    q = tl.load(q_ptr + (b * N + tok[:, None]).to(tl.int64) * sq_t + hq[:, None] * sq_h + dd[None, :],
                mask=rvalid[:, None], other=0.0)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    qk_scale = scale * 1.44269504088896  # exp2 domain
    kv_start = tl.load(kv_indptr_ptr + b)
    L = tl.load(kv_indptr_ptr + b + 1) - kv_start
    if i_s < S:
        chunk = tl.cdiv(tl.cdiv(L, S), BN) * BN
        start = i_s * chunk
        end = tl.minimum(L, start + chunk)
        kj = start + tl.arange(0, BN)
        slot = tl.load(kv_indices_ptr + kv_start + kj, mask=kj < end, other=0).to(tl.int64)
        for _ in tl.range(start, end, BN):
            kval = kj < end
            kb = tl.load(kb_ptr + slot[:, None] * skb_t + i_kv * skb_h + dd[None, :], mask=kval[:, None], other=0.0)
            vb = tl.load(vb_ptr + slot[:, None] * svb_t + i_kv * svb_h + dd[None, :], mask=kval[:, None], other=0.0)
            kn = kj + BN
            slot = tl.load(kv_indices_ptr + kv_start + kn, mask=kn < end, other=0).to(tl.int64)
            s = tl.dot(q, tl.trans(kb)) * qk_scale
            vis = kval[None, :]
            if WINDOW > 0:  # as extend's prefix loop: q_id = L + row token, kv_id = prefix position
                vis = vis & ((L + tok[:, None]) <= (kj[None, :] + WINDOW))
            s = tl.where(vis, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp2(m_i - m_safe)
            p = tl.exp2(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(vb.dtype), vb)
            m_i = m_new
            kj = kn
    else:  # draft suffix: fresh K/V rows of this request
        nn = tl.arange(0, NP)
        nval = nn < N
        vis = rvalid[:, None] & nval[None, :]
        if HAS_MASK:  # suffix block of the custom mask: row stride L + woff + N, columns from woff + L
            mstart = tl.load(mask_indptr_ptr + b).to(tl.int64)
            woff = 0
            if WINDOW > 0:
                woff = tl.load(woff_ptr + b)
            m = tl.load(mask_ptr + mstart + tok[:, None].to(tl.int64) * (L + woff + N) + woff + L + nn[None, :],
                        mask=vis, other=0)
            vis = vis & (m != 0)
        elif CAUSAL:
            vis = vis & (nn[None, :] <= tok[:, None])
        if WINDOW > 0:
            vis = vis & (tok[:, None] <= nn[None, :] + WINDOW)
        kb = tl.load(ke_ptr + (b * N + nn[:, None]).to(tl.int64) * ske_t + i_kv * ske_h + dd[None, :],
                     mask=nval[:, None], other=0.0)
        s = tl.dot(q, tl.trans(kb)) * qk_scale
        s = tl.where(vis, s, float("-inf"))
        m_i = tl.max(s, 1)
        m_safe = tl.where(m_i == float("-inf"), 0.0, m_i)
        p = tl.exp2(s - m_safe[:, None])
        l_i = tl.sum(p, 1)
        vt = tl.load(ve_ptr + (b * N + nn[:, None]).to(tl.int64) * sve_t + i_kv * sve_h + dd[None, :],
                     mask=nval[:, None], other=0.0)
        acc = tl.dot(p.to(vt.dtype), vt)
    pid = ((b * HKV + i_kv) * QB + i_qb) * (S + 1) + i_s
    rr = tl.arange(0, BM)
    tl.store(pacc_ptr + (pid * BM + rr[:, None]).to(tl.int64) * D + dd[None, :], acc)
    tl.store(pml_ptr + (pid * 2) * BM + rr, m_i)
    tl.store(pml_ptr + (pid * 2 + 1) * BM + rr, l_i)


@triton.jit
def _d5_tv_combine_kernel(
    out_ptr, pacc_ptr, pml_ptr, so_t, so_h,
    N: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, S: tl.constexpr, QB: tl.constexpr,
    RB: tl.constexpr, RBLK: tl.constexpr,
):
    i_br, i_kv = tl.program_id(0), tl.program_id(1)
    HKV = tl.num_programs(1)
    b = i_br // RBLK
    i_r = i_br % RBLK
    rows = i_r * RB + tl.arange(0, RB)
    rvalid = rows < N * G
    tok = rows // G
    hq = i_kv * G + rows % G
    dd = tl.arange(0, D)
    qb = rows // BM
    rr = rows % BM
    base = ((b * HKV + i_kv) * QB + qb) * (S + 1)
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
        at = tl.load(pacc_ptr + ((base + t) * BM + rr)[:, None].to(tl.int64) * D + dd[None, :], mask=rvalid[:, None],
                     other=0.0)
        num += w[:, None] * at
        den += w * lt
    o = num / tl.where(rvalid & (den > 0), den, 1.0)[:, None]
    tl.store(out_ptr + (b * N + tok[:, None]).to(tl.int64) * so_t + hq[:, None] * so_h + dd[None, :],
             o.to(out_ptr.dtype.element_ty), mask=rvalid[:, None])


# Partial buffers per (shape, device); a tensor first allocated during graph capture stays referenced here, so its
# memory is never reused by another graph.
_SCRATCH: dict = {}


def _buf(name, shape, dtype, device):
    key = (name, tuple(int(s) for s in shape), dtype, str(device))
    t = _SCRATCH.get(key)
    if t is None:
        t = _SCRATCH[key] = torch.empty(key[1], dtype=dtype, device=device)
    return t


def tree_verify_splitkv(q, k_extend, v_extend, o, k_buffer, v_buffer, kv_indptr, kv_indices, custom_mask,
                        mask_indptr, sm_scale, *, bs, n_tokens, splits=8, bm=32, bn=32, num_warps=4, rb=8,
                        num_stages=1, causal=True, sliding_window_size=-1, window_kv_offsets=None):
    """q / o [bs * N, HQ, D], k/v_extend [bs * N, HKV, D] (unit last stride), k/v_buffer [slots, HKV, D];
    kv_indptr [bs + 1], kv_indices prefix slots; custom_mask uint8/bool with N x (woff + L + N) rows per request at
    mask_indptr, or None (causal / full suffix). Writes o."""
    N = int(n_tokens)
    HQ, D = q.shape[1], q.shape[2]
    HKV = k_buffer.shape[1]
    G = HQ // HKV
    S, BM = int(splits), int(bm)
    QB = triton.cdiv(N * G, BM)
    window = int(sliding_window_size) if sliding_window_size is not None and sliding_window_size > 0 else 0
    has_mask = custom_mask is not None
    if has_mask and window:
        assert window_kv_offsets is not None
    dev = q.device
    pacc = _buf("pacc", (bs * HKV * QB * (S + 1), BM, D), torch.float32, dev)
    pml = _buf("pml", (bs * HKV * QB * (S + 1), 2, BM), torch.float32, dev)
    dummy = kv_indptr
    _d5_tv_kernel[(bs * QB, S + 1, HKV)](
        q, k_extend, v_extend, k_buffer, v_buffer, kv_indptr, kv_indices,
        custom_mask if has_mask else dummy, mask_indptr if has_mask else dummy,
        window_kv_offsets if window_kv_offsets is not None else dummy, pacc, pml, float(sm_scale),
        q.stride(0), q.stride(1), k_extend.stride(0), k_extend.stride(1), v_extend.stride(0), v_extend.stride(1),
        k_buffer.stride(0), k_buffer.stride(1), v_buffer.stride(0), v_buffer.stride(1),
        N=N, NP=max(16, triton.next_power_of_2(N)), G=G, D=D, BM=BM, BN=int(bn), S=S, QB=QB,
        HAS_MASK=has_mask, CAUSAL=bool(causal), WINDOW=window,
        num_warps=num_warps, num_stages=num_stages,
    )
    RBLK = triton.cdiv(N * G, rb)
    _d5_tv_combine_kernel[(bs * RBLK, HKV)](
        o, pacc, pml, o.stride(0), o.stride(1),
        N=N, G=G, D=D, BM=BM, S=S, QB=QB, RB=int(rb), RBLK=RBLK, num_warps=4,
    )
    return o
