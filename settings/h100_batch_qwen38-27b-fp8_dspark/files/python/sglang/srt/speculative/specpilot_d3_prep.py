"""SPECPILOT_D3_PREP32=warps: 2x2 block split-WY prep kernel for trees of 17..32 nodes (NP = 32).

The seed's prep (_gdn_tree_wy_prep_kernel) inverts (I + X) for X = diag(beta) Lr strictly lower triangular in node
order with the nilpotent product (I - X)(I + X^2)(I + X^4)(I + X^8)(I + X^16) on 32 x 32 tiles: a dependent chain of
nine 32^3 tf32x3 dots per (request, value head), 1 warp, ~10x the 16-node cost. Here the 32 nodes are two halves
(topological order is kept: node i < j whenever i is an ancestor of j), so (I + X) = [[I + X11, 0], [X21, I + X22]]
and its inverse is [[A, 0], [-B X21 A, B]] with A = (I + X11)^-1, B = (I + X22)^-1. Every 16 x 16 strictly lower
triangular block is nilpotent of index <= 16, so A and B take the 16-node chain (I - X)(I + X^2)(I + X^4)(I + X^8)
exactly, for any tree depth; the two chains are independent. M' = (I + X)^-1 diag(beta), W = P M' (P lower
block triangular). Same outputs as the seed's prep up to reduction order (contract numerical).
"""
from __future__ import annotations

import os

import triton
import triton.language as tl

# Read once at import; unset = the seed's prep for every tree size.
_D3_PREP32 = int(os.environ.get("SPECPILOT_D3_PREP32", "0") or 0)


def d3_prep32_warps() -> int:
    return _D3_PREP32


@triton.jit
def _inv_unit_lower16(X, eye):
    # (I + X)^-1 for a 16 x 16 strictly lower triangular X: (I - X)(I + X^2)(I + X^4)(I + X^8), exact (X^16 == 0)
    X2 = tl.dot(X, X, input_precision="tf32x3")
    X4 = tl.dot(X2, X2, input_precision="tf32x3")
    X8 = tl.dot(X4, X4, input_precision="tf32x3")
    r = tl.where(eye, 1.0, 0.0)
    r = r + tl.dot(X8, r, input_precision="tf32x3")
    r = r + tl.dot(X4, r, input_precision="tf32x3")
    r = r + tl.dot(X2, r, input_precision="tf32x3")
    return r - tl.dot(X, r, input_precision="tf32x3")


@triton.jit
def _d3_prep32_kernel(
    A_log, a, dt_bias, q, k, v, b, anc_ptr, row_ptr,
    kn_out, v_out, g_out, beta_out,
    w_out,  # [bs, HV, 32, 32] fp32: W = P M'
    eg_out,  # [bs, HV, 32] fp32: e^gam
    scale, softplus_beta, softplus_threshold, stride_q, stride_k, stride_v, stride_a, stride_b,
    N: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BVR: tl.constexpr,
):
    i_n, i_hv = tl.program_id(0), tl.program_id(1)
    i_h = i_hv // (HV // H)
    t16 = tl.arange(0, 16)
    eye = t16[:, None] == t16[None, :]
    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
    row = tl.load(row_ptr + i_n).to(tl.int64)
    o_vr = tl.arange(0, BVR)
    # per half (rows 0..15 = 1, rows 16..31 = 2): ancestor bits, inputs, gates, normalised q / k
    n1 = t16
    n2 = 16 + t16
    v1 = n1 < N
    v2 = n2 < N
    bits1 = tl.load(anc_ptr + i_n * N + n1, mask=v1, other=0)
    bits2 = tl.load(anc_ptr + i_n * N + n2, mask=v2, other=0)
    # strict-ancestor blocks: anc11[t, s] = s (half 1) ancestor of t (half 1); anc21: s in half 1, t in half 2
    anc11 = (((bits1[:, None] >> n1[None, :]) & 1) != 0) & v1[:, None] & (~eye)
    anc21 = (((bits2[:, None] >> n1[None, :]) & 1) != 0) & v2[:, None] & v1[None, :]
    anc22 = (((bits2[:, None] >> n2[None, :]) & 1) != 0) & v2[:, None] & v2[None, :] & (~eye)
    ae11 = anc11 | (eye & v1[:, None])
    ae22 = anc22 | (eye & v2[:, None])
    tok1 = (i_n * N + n1).to(tl.int64)
    tok2 = (i_n * N + n2).to(tl.int64)
    mk1 = v1[:, None] & mask_k[None, :]
    mk2 = v2[:, None] & mask_k[None, :]
    q1 = tl.load(q + tok1[:, None] * stride_q + i_h * K + o_k[None, :], mask=mk1, other=0)
    q2 = tl.load(q + tok2[:, None] * stride_q + i_h * K + o_k[None, :], mask=mk2, other=0)
    k1 = tl.load(k + tok1[:, None] * stride_k + i_h * K + o_k[None, :], mask=mk1, other=0)
    k2 = tl.load(k + tok2[:, None] * stride_k + i_h * K + o_k[None, :], mask=mk2, other=0)
    ba1 = tl.load(a + tok1 * stride_a + i_hv, mask=v1, other=0).to(tl.float32)
    ba2 = tl.load(a + tok2 * stride_a + i_hv, mask=v2, other=0).to(tl.float32)
    bb1 = tl.load(b + tok1 * stride_b + i_hv, mask=v1, other=0).to(tl.float32)
    bb2 = tl.load(b + tok2 * stride_b + i_hv, mask=v2, other=0).to(tl.float32)
    x1 = ba1 + b_dt_bias
    x2 = ba2 + b_dt_bias
    sp1 = tl.where(softplus_beta * x1 <= softplus_threshold,
                   (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(softplus_beta * x1)), x1)
    sp2 = tl.where(softplus_beta * x2 <= softplus_threshold,
                   (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(softplus_beta * x2)), x2)
    g1 = tl.where(v1, -tl.exp(b_A_log) * sp1, 0.0)
    g2 = tl.where(v2, -tl.exp(b_A_log) * sp2, 0.0)
    be1 = tl.where(v1, 1.0 / (1.0 + tl.exp(-bb1)), 0.0)
    be2 = tl.where(v2, 1.0 / (1.0 + tl.exp(-bb2)), 0.0)
    q1f = q1.to(tl.float32)
    q2f = q2.to(tl.float32)
    k1f = k1.to(tl.float32)
    k2f = k2.to(tl.float32)
    rq1 = 1.0 / tl.sqrt(tl.sum(q1f * q1f, 1) + 1e-6)
    rq2 = 1.0 / tl.sqrt(tl.sum(q2f * q2f, 1) + 1e-6)
    rk1 = 1.0 / tl.sqrt(tl.sum(k1f * k1f, 1) + 1e-6)
    rk2 = 1.0 / tl.sqrt(tl.sum(k2f * k2f, 1) + 1e-6)
    # gam_t = sum of g over t's root path (incl. t); a half-2 node's path may run through half 1
    gam1 = tl.sum(tl.where(ae11, g1[None, :], 0.0), 1)
    gam2 = tl.sum(tl.where(anc21, g1[None, :], 0.0), 1) + tl.sum(tl.where(ae22, g2[None, :], 0.0), 1)
    k1t = tl.trans(k1)
    k2t = tl.trans(k2)
    KK11 = tl.dot(k1, k1t) * (rk1[:, None] * rk1[None, :])
    KK21 = tl.dot(k2, k1t) * (rk2[:, None] * rk1[None, :])
    KK22 = tl.dot(k2, k2t) * (rk2[:, None] * rk2[None, :])
    sq1 = rq1 * scale
    sq2 = rq2 * scale
    QK11 = tl.dot(q1, k1t) * (sq1[:, None] * rk1[None, :])
    QK21 = tl.dot(q2, k1t) * (sq2[:, None] * rk1[None, :])
    QK22 = tl.dot(q2, k2t) * (sq2[:, None] * rk2[None, :])
    d11 = gam1[:, None] - gam1[None, :]
    d21 = gam2[:, None] - gam1[None, :]
    d22 = gam2[:, None] - gam2[None, :]
    X11 = tl.where(anc11, be1[:, None] * tl.exp(tl.where(anc11, d11, 0.0)) * KK11, 0.0)
    X21 = tl.where(anc21, be2[:, None] * tl.exp(tl.where(anc21, d21, 0.0)) * KK21, 0.0)
    X22 = tl.where(anc22, be2[:, None] * tl.exp(tl.where(anc22, d22, 0.0)) * KK22, 0.0)
    P11 = tl.where(ae11, tl.exp(tl.where(ae11, d11, 0.0)) * QK11, 0.0)
    P21 = tl.where(anc21, tl.exp(tl.where(anc21, d21, 0.0)) * QK21, 0.0)
    P22 = tl.where(ae22, tl.exp(tl.where(ae22, d22, 0.0)) * QK22, 0.0)
    Ai = _inv_unit_lower16(X11, eye)
    Bi = _inv_unit_lower16(X22, eye)
    M11 = Ai * be1[None, :]
    M22 = Bi * be2[None, :]
    M21 = -tl.dot(Bi, tl.dot(X21, M11, input_precision="tf32x3"), input_precision="tf32x3")
    W11 = tl.dot(P11, M11, input_precision="tf32x3")
    W21 = tl.dot(P21, M11, input_precision="tf32x3") + tl.dot(P22, M21, input_precision="tf32x3")
    W22 = tl.dot(P22, M22, input_precision="tf32x3")
    wb = (i_n * HV + i_hv) * 32
    tl.store(w_out + (wb + n1)[:, None] * 32 + n1[None, :], W11)
    tl.store(w_out + (wb + n1)[:, None] * 32 + n2[None, :], tl.zeros([16, 16], dtype=tl.float32))
    tl.store(w_out + (wb + n2)[:, None] * 32 + n1[None, :], W21)
    tl.store(w_out + (wb + n2)[:, None] * 32 + n2[None, :], W22)
    tl.store(eg_out + wb + n1, tl.exp(gam1))
    tl.store(eg_out + wb + n2, tl.exp(gam2))
    # commit inputs, as the seed's prep stores them
    sb1 = row * N + n1
    sb2 = row * N + n2
    mv1 = v1[:, None] & (o_vr < V)[None, :]
    mv2 = v2[:, None] & (o_vr < V)[None, :]
    tl.store(v_out + (sb1[:, None] * HV + i_hv) * V + o_vr[None, :],
             tl.load(v + tok1[:, None] * stride_v + i_hv * V + o_vr[None, :], mask=mv1, other=0), mask=mv1)
    tl.store(v_out + (sb2[:, None] * HV + i_hv) * V + o_vr[None, :],
             tl.load(v + tok2[:, None] * stride_v + i_hv * V + o_vr[None, :], mask=mv2, other=0), mask=mv2)
    if i_hv % (HV // H) == 0:  # kn is per key head: one of its value heads stores it
        tl.store(kn_out + (sb1[:, None] * H + i_h) * K + o_k[None, :], k1f * rk1[:, None], mask=mk1)
        tl.store(kn_out + (sb2[:, None] * H + i_h) * K + o_k[None, :], k2f * rk2[:, None], mask=mk2)
    tl.store(g_out + sb1 * HV + i_hv, g1, mask=v1)
    tl.store(g_out + sb2 * HV + i_hv, g2, mask=v2)
    tl.store(beta_out + sb1 * HV + i_hv, be1, mask=v1)
    tl.store(beta_out + sb2 * HV + i_hv, be2, mask=v2)


def d3_prep32_launch(*, A_log, a, dt_bias, q, k, v, b, anc_bits, scratch_rows, kn_buf, v_buf, g_buf, beta_buf, w, eg,
                     scale, softplus_beta, softplus_threshold, N, H, HV, K, V, warps=None):
    """Drop-in for the seed's prep launch when 16 < N <= 32 (w [bs, HV, 32, 32], eg [bs, HV, 32])."""
    assert 16 < N <= 32 and w.shape[-1] == 32, (N, w.shape)
    bs = w.shape[0]
    _d3_prep32_kernel[(bs, HV)](
        A_log, a, dt_bias, q, k, v, b, anc_bits, scratch_rows, kn_buf, v_buf, g_buf, beta_buf, w, eg,
        scale, softplus_beta, softplus_threshold, q.stride(1), k.stride(1), v.stride(1), a.stride(-2), b.stride(-2),
        N=N, H=H, HV=HV, K=K, V=V, BK=triton.next_power_of_2(K), BVR=triton.next_power_of_2(V),
        num_warps=warps or _D3_PREP32, num_stages=1,
    )
