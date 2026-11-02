"""SPECPILOT_D3_STATE="BV,warps,BKC,stages,TA,TQ": K-chunked split-WY GDN state kernel for the tree verify.

Replaces the seed's _gdn_tree_wy_state_kernel launch (SPECPILOT_K_WY_SPLIT): same inputs (W, e^gam from the split
prep kernel) and outputs, but the fp32 state tile is streamed along K in chunks of BKC rows with a pipelined loop
instead of being held whole in registers (the resident tile + three bf16 copies needs ~254 registers per thread,
2 CTAs/SM, and the seed kernel is latency-bound at ~1.5 TB/s). Kn h and Qn h use TA / TQ bf16 split terms of h
(3 = fp32-exact products, the seed's precision; 2 = 16 mantissa bits). The verify never writes the state: the
committed state is replayed in fp32 by the seed's commit, so the split only changes verify outputs (numerical).
"""
from __future__ import annotations

import os

import triton
import triton.language as tl

# Read once at import; unset = the seed's state launch.
_D3_STATE = tuple(int(x) for x in os.environ.get("SPECPILOT_D3_STATE", "").split(",") if x.strip())


def d3_state_enabled() -> bool:
    return bool(_D3_STATE)


@triton.jit
def _d3_vstate_kernel(
    q, k, v, o, h0_source, h0_indices, stride_h0, w_ptr, eg_ptr, scale, stride_q, stride_k, stride_v, out_rows,
    N: tl.constexpr, NP: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BV: tl.constexpr, BKC: tl.constexpr, STAGES: tl.constexpr, TA: tl.constexpr, TQ: tl.constexpr,
    PACKED_OUT: tl.constexpr, STACK: tl.constexpr = False,
):
    i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_hv // (HV // H)
    tt = tl.arange(0, NP)
    valid = tt < N
    toks = (i_n * N + tt).to(tl.int64)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V
    mv = valid[:, None] & mask_v[None, :]
    idx = tl.load(h0_indices + i_n).to(tl.int64)
    live = idx >= 0
    base = h0_source + tl.maximum(idx, 0) * stride_h0 + i_hv * K * V
    b_A = tl.zeros([NP, BV], dtype=tl.float32)
    b_QH = tl.zeros([NP, BV], dtype=tl.float32)
    ssq = tl.zeros([NP], dtype=tl.float32)
    ssk = tl.zeros([NP], dtype=tl.float32)
    for kc in tl.range(0, K, BKC, num_stages=STAGES):
        o_k = kc + tl.arange(0, BKC)
        mask_k = o_k < K
        h = tl.load(base + o_v[None, :] * K + o_k[:, None], mask=mask_k[:, None] & mask_v[None, :] & live, other=0)
        mk = valid[:, None] & mask_k[None, :]
        qb = tl.load(q + toks[:, None] * stride_q + i_h * K + o_k[None, :], mask=mk, other=0)
        kb = tl.load(k + toks[:, None] * stride_k + i_h * K + o_k[None, :], mask=mk, other=0)
        h_hi = h.to(tl.bfloat16)
        r1 = h - h_hi.to(tl.float32)
        h_mid = r1.to(tl.bfloat16)
        h_lo = (r1 - h_mid.to(tl.float32)).to(tl.bfloat16)
        if STACK:  # one [2NP, BKC] x [BKC, BV] dot per split term (TA == TQ == 3)
            qk = tl.reshape(tl.permute(tl.join(qb, kb), (2, 0, 1)), (2 * NP, BKC))
            res = tl.dot(qk, h_hi) + tl.dot(qk, h_mid) + tl.dot(qk, h_lo)
            r_q, r_k = tl.split(tl.permute(tl.reshape(res, (2, NP, BV)), (1, 2, 0)))
            b_QH += r_q
            b_A += r_k
        else:
            b_A += tl.dot(kb, h_hi)
            b_QH += tl.dot(qb, h_hi)
            if TA >= 2:
                b_A += tl.dot(kb, h_mid)
            if TA >= 3:
                b_A += tl.dot(kb, h_lo)
            if TQ >= 2:
                b_QH += tl.dot(qb, h_mid)
            if TQ >= 3:
                b_QH += tl.dot(qb, h_lo)
        qf = qb.to(tl.float32)
        kf = kb.to(tl.float32)
        ssq += tl.sum(qf * qf, 1)
        ssk += tl.sum(kf * kf, 1)
    rq = 1.0 / tl.sqrt(ssq + 1e-6)
    rk = 1.0 / tl.sqrt(ssk + 1e-6)
    b_vr = tl.load(v + toks[:, None] * stride_v + i_hv * V + o_v[None, :], mask=mv, other=0)
    wb = ((i_n * HV + i_hv) * NP + tt).to(tl.int64)
    W = tl.load(w_ptr + wb[:, None] * NP + tt[None, :])
    egam = tl.load(eg_ptr + wb)
    u = b_vr.to(tl.float32) - (egam * rk)[:, None] * b_A
    b_o = (egam * rq * scale)[:, None] * b_QH + tl.dot(W, u, input_precision="tf32x3")
    if PACKED_OUT:
        orow = tl.load(out_rows + toks, mask=valid, other=-1)
        tl.store(
            o + (tl.maximum(orow, 0)[:, None] * HV + i_hv) * V + o_v[None, :],
            b_o.to(o.dtype.element_ty),
            mask=mv & (orow[:, None] >= 0),
        )
    else:
        tl.store(o + (toks[:, None] * HV + i_hv) * V + o_v[None, :], b_o.to(o.dtype.element_ty), mask=mv)


def d3_state_launch(*, q, k, v, o, ssm_states, cache_indices, w, eg, scale, out_rows, packed_out, N, NP, H, HV, K, V,
                    cfg=None):
    """Same contract as the seed's _gdn_tree_wy_state_kernel launch (gdn_tree_wy_split_verify)."""
    bv, n_warps, bkc, stages, ta, tq = (cfg or _D3_STATE)[:6]
    stack = len(cfg or _D3_STATE) > 6 and (cfg or _D3_STATE)[6] == 1
    bs = cache_indices.shape[0]
    _d3_vstate_kernel[(triton.cdiv(V, bv), bs, HV)](
        q, k, v, o, ssm_states, cache_indices, ssm_states.stride(0), w, eg, scale,
        q.stride(1), k.stride(1), v.stride(1), out_rows if packed_out else cache_indices,
        N=N, NP=NP, H=H, HV=HV, K=K, V=V, BV=bv, BKC=bkc, STAGES=stages, TA=ta, TQ=tq, PACKED_OUT=packed_out,
        STACK=stack, num_warps=n_warps, num_stages=1,
    )
