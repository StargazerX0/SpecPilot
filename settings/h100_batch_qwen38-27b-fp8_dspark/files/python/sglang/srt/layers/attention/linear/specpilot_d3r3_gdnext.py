"""SPECPILOT_D3R3_GDNEXT: fused GDN extend (prefill) of fresh requests on the packed prompt rows.

Replaces, per GDN layer, the SPECPILOT_X1 bucketed prefill graph (copy-in into front-padded buckets, mask multiplies,
Triton causal conv1d, qkv split, gating, l2norm q/k, chunk-local cumsum, fused kkt+solve_tril, recompute_w_u,
chunk_fwd_h, chunk_fwd_o, copy-out) with three kernels (SPLIT=1, default):

  pro   grid (16-row blocks, 2*HK + H channel blocks + 1): conv1d+SiLU, split, l2norm, gating, conv state
  prep  grid (chunks, H):            chunk cumsum, kkt + diagonal-block inverse + block merge, recompute_w_u
  rec   grid (V / BV, requests * H): chunk_fwd_h recurrence with chunk_fwd_o fused per chunk (h stays in registers)
(SPLIT=0 runs pro's work inside prep, one program per chunk-head: slower, measured a5df239d3877 / 2a6b1e4cfcdf.)

Every rounding point is the seed's: conv products in bf16 and the conv output rounded to bf16, beta rounded to bf16,
A / w / u / v_new / h (for the output) in bf16, the recurrent state in fp32. Chunks are laid out exactly as in the
seed's front-padded bucket: request n gets (-L_n) mod 64 virtual zero rows before its first token, so the chunk
partition and every chunk-local quantity match; the zero rows (k = v = beta = g = 0) leave the state unchanged.
The kkt+solve_tril, recompute_w_u, chunk_fwd_h and chunk_fwd_o bodies are the seed's FLA kernels (same block
shapes); INV=1 (default) inverts the 16x16 diagonal blocks by doubling instead of the seed's forward substitution
(numerically distinct at the underflow end only: 22 of 3.1M A elements differ by <= 3e-17 at L1000, outputs and
states bitwise in every test). The kernels exchange data through an L2-resident scratch (ExtendMeta).

Knob (read once at import): SPECPILOT_D3R3_GDNEXT=1 enables the engine path (gdn_backend forward_extend).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.utils import is_tf32_supported

# fla.op's exp/safe_exp are tl.exp unless FLA_USE_FAST_OPS=1; the fused path is only enabled on that default.
ENABLED = os.environ.get("SPECPILOT_D3R3_GDNEXT", "0") == "1" and os.environ.get("FLA_USE_FAST_OPS", "0") != "1"
MAX_ROWS = int(os.environ.get("SPECPILOT_D3R3_GDNEXT_MAX", "2048"))

CHUNK = 64
_BV_REC = 32
_KKT_BK = 64
PREP_WARPS = int(os.environ.get("SPECPILOT_D3R3_GDNEXT_PREP_WARPS", "4"))
PREP_ROWS = int(os.environ.get("SPECPILOT_D3R3_GDNEXT_PREP_ROWS", "16"))
# 1 = doubling inverse of the diagonal blocks (measured 9665e656b2d4); 0 = the seed's forward substitution
INV = int(os.environ.get("SPECPILOT_D3R3_GDNEXT_INV", "1"))
SPLIT = int(os.environ.get("SPECPILOT_D3R3_GDNEXT_SPLIT", "1"))  # 1 = separate wide-grid prologue kernel
_PRO_ROWS = 16  # 0 = the seed's forward substitution (bitwise)
_MERGE_PREC = "tf32" if is_tf32_supported else "ieee"


@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


@triton.jit
def _conv_silu(x, stride_x, cw, stride_cw_d, stride_cw_w, cb, c0, bos, o_tok, m_row,
               BT: tl.constexpr, NC: tl.constexpr, HAS_BIAS: tl.constexpr):
    """Causal depthwise conv (width 4) + SiLU over channels [c0, c0 + NC) of rows o_tok; zero history before row 0.
    Same accumulation order and bf16 products as the seed's Triton causal_conv1d_fwd kernel."""
    o_c = c0 + tl.arange(0, NC)
    if HAS_BIAS:
        acc = tl.zeros([BT, NC], dtype=tl.float32) + tl.load(cb + o_c).to(tl.float32)[None, :]
    else:
        acc = tl.zeros([BT, NC], dtype=tl.float32)
    for j in tl.static_range(4):
        tt = o_tok - 3 + j
        m = m_row & (tt >= 0)
        xj = tl.load(x + (bos + tt).to(tl.int64)[:, None] * stride_x + o_c[None, :], mask=m[:, None], other=0.0)
        wj = tl.load(cw + o_c * stride_cw_d + j * stride_cw_w)
        acc += xj * wj[None, :]
    acc = acc / (1 + tl.exp(-acc))
    return acc


@triton.jit
def _neg_powsum(A, m_I):
    M = -A
    P = M + m_I
    Mp = M
    for _ in tl.static_range(3):
        Mp = tl.dot(Mp, Mp, input_precision="ieee")
        P = P + tl.dot(P, Mp, input_precision="ieee")
    return P - m_I


@triton.jit
def _l2norm_rows(xb):
    b_x = xb.to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    return b_x / tl.sqrt(b_var + 1e-6)[:, None]


@triton.jit
def _store_conv_state(x, stride_x, conv_st, stride_cs_d, stride_cs_tok, c0, bos, L, NC: tl.constexpr):
    o_c = c0 + tl.arange(0, NC)
    for j in tl.static_range(3):
        tt = L - 3 + j
        xj = tl.load(x + (bos + tt).to(tl.int64) * stride_x + o_c, mask=(tt >= 0) & (o_c >= 0), other=0.0)
        tl.store(conv_st + o_c * stride_cs_d + j * stride_cs_tok, xj)


@triton.jit(do_not_specialize=["n_chunks"])
def d3r3_gdnext_prep_kernel(
    x, stride_x, cw, stride_cw_d, stride_cw_w, cb,
    a, stride_a, b, stride_b, A_log, dt_bias,
    conv_st, stride_cs_slot, stride_cs_d, stride_cs_tok, slots,
    cu, cv, table,
    qn, kn, vc, gs, bs, Ab, w, u,
    n_chunks,
    HK: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
    HAS_BIAS: tl.constexpr, MERGE_PREC: tl.constexpr, INV: tl.constexpr = 0, DIAG: tl.constexpr = 0,
    SPLIT: tl.constexpr = 0,
):
    i_c, i_h = tl.program_id(0), tl.program_id(1)
    i_n = tl.load(table + i_c * 2).to(tl.int32)
    i_t = tl.load(table + i_c * 2 + 1).to(tl.int32)
    bos = tl.load(cu + i_n).to(tl.int32)
    L = tl.load(cu + i_n + 1).to(tl.int32) - bos
    vbos = tl.load(cv + i_n).to(tl.int32)
    T = tl.load(cv + i_n + 1).to(tl.int32) - vbos
    lead = T - L
    G: tl.constexpr = H // HK
    i_kh = i_h // G

    # ---- prologue: conv + split + l2norm + gating + cumsum for this chunk, into the scratch (virtual rows) ----
    o_i = tl.arange(0, BT)
    o_r = i_t * BT + o_i
    o_tok = o_r - lead
    m_row = o_tok >= 0
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    vrow = (vbos + o_r).to(tl.int64)

    if SPLIT:  # conv/split/l2norm/gating ran in d3r3_gdnext_pro_kernel; gs holds the raw g of this chunk
        b_gc = tl.cumsum(tl.load(gs + vrow * H + i_h), axis=0)
        tl.store(gs + vrow * H + i_h, b_gc)
    else:
        for i_r in tl.static_range(BT // BR):  # row sub-blocks: register footprint independent of BT
            o_rr = i_t * BT + i_r * BR + tl.arange(0, BR)
            o_tr = o_rr - lead
            m_rr = o_tr >= 0
            vrr = (vbos + o_rr).to(tl.int64)
            k_c = _conv_silu(x, stride_x, cw, stride_cw_d, stride_cw_w, cb, HK * K + i_kh * K, bos, o_tr, m_rr,
                             BR, K, HAS_BIAS).to(tl.bfloat16)
            k_n = tl.where(m_rr[:, None], _l2norm_rows(k_c), 0.0)
            tl.store(kn + (vrr * H + i_h)[:, None] * K + o_k[None, :], k_n.to(tl.bfloat16))
            v_c = _conv_silu(x, stride_x, cw, stride_cw_d, stride_cw_w, cb, 2 * HK * K + i_h * V, bos, o_tr, m_rr,
                             BR, V, HAS_BIAS).to(tl.bfloat16)
            v_c = tl.where(m_rr[:, None], v_c, 0.0)
            tl.store(vc + (vrr * H + i_h)[:, None] * V + o_v[None, :], v_c.to(tl.bfloat16))
            if i_h % G == 0:
                q_c = _conv_silu(x, stride_x, cw, stride_cw_d, stride_cw_w, cb, i_kh * K, bos, o_tr, m_rr,
                                 BR, K, HAS_BIAS).to(tl.bfloat16)
                q_n = tl.where(m_rr[:, None], _l2norm_rows(q_c), 0.0)
                tl.store(qn + (vrr * HK + i_kh)[:, None] * K + o_k[None, :], q_n.to(tl.bfloat16))

        prow = (bos + o_tok).to(tl.int64)
        b_a = tl.load(a + prow * stride_a + i_h, mask=m_row, other=0.0).to(tl.float32)
        b_b = tl.load(b + prow * stride_b + i_h, mask=m_row, other=0.0).to(tl.float32)
        xg = b_a + tl.load(dt_bias + i_h).to(tl.float32)
        softplus_x = tl.where(xg <= 20.0, tl.log(1 + tl.exp(xg)), xg)
        b_graw = -tl.exp(tl.load(A_log + i_h).to(tl.float32)) * softplus_x
        b_graw = tl.where(m_row, b_graw, 0.0)
        b_beta = tl.where(m_row, tl.sigmoid(b_b).to(tl.bfloat16).to(tl.float32), 0.0)
        b_gc = tl.cumsum(b_graw, axis=0)
        tl.store(gs + vrow * H + i_h, b_gc)
        tl.store(bs + vrow * H + i_h, b_beta)

        if (i_t + 1) * BT >= T:  # last chunk of the request: conv state = its last 3 raw rows
            slot = tl.load(slots + i_n).to(tl.int64)
            cst = conv_st + slot * stride_cs_slot
            _store_conv_state(x, stride_x, cst, stride_cs_d, stride_cs_tok, 2 * HK * K + i_h * V, bos, L, V)
            if i_h % G == 0:
                _store_conv_state(x, stride_x, cst, stride_cs_d, stride_cs_tok, i_kh * K, bos, L, K)
                _store_conv_state(x, stride_x, cst, stride_cs_d, stride_cs_tok, HK * K + i_kh * K, bos, L, K)

    if DIAG & 1:  # timing diagnostic: prologue only
        return
    tl.debug_barrier()

    # ---- the seed's chunk_gated_delta_rule_fwd_kkt_solve_kernel body (Hg = H: per-v-head key copy) ----
    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC
    KH: tl.constexpr = HK if SPLIT else H
    k_s = kn + (vbos * KH + (i_kh if SPLIT else i_h)) * K
    A_s = Ab + (vbos * H + i_h) * BT
    beta_s = bs + vbos * H + i_h
    g_s = gs + vbos * H + i_h

    o_j = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_j) < T
    m_tc1 = (i_tc1 + o_j) < T
    m_tc2 = (i_tc2 + o_j) < T
    m_tc3 = (i_tc3 + o_j) < T

    b_b0 = tl.load(tl.make_block_ptr(beta_s, (T,), (H,), (i_tc0,), (BC,), (0,)), boundary_check=(0,)).to(tl.float32)
    b_b1 = tl.load(tl.make_block_ptr(beta_s, (T,), (H,), (i_tc1,), (BC,), (0,)), boundary_check=(0,)).to(tl.float32)
    b_b2 = tl.load(tl.make_block_ptr(beta_s, (T,), (H,), (i_tc2,), (BC,), (0,)), boundary_check=(0,)).to(tl.float32)
    b_b3 = tl.load(tl.make_block_ptr(beta_s, (T,), (H,), (i_tc3,), (BC,), (0,)), boundary_check=(0,)).to(tl.float32)
    b_g0 = tl.load(tl.make_block_ptr(g_s, (T,), (H,), (i_tc0,), (BC,), (0,)), boundary_check=(0,)).to(tl.float32)
    b_g1 = tl.load(tl.make_block_ptr(g_s, (T,), (H,), (i_tc1,), (BC,), (0,)), boundary_check=(0,)).to(tl.float32)
    b_g2 = tl.load(tl.make_block_ptr(g_s, (T,), (H,), (i_tc2,), (BC,), (0,)), boundary_check=(0,)).to(tl.float32)
    b_g3 = tl.load(tl.make_block_ptr(g_s, (T,), (H,), (i_tc3,), (BC,), (0,)), boundary_check=(0,)).to(tl.float32)

    b_A00 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A11 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A22 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A33 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k0 = tl.make_block_ptr(k_s, (T, K), (KH * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_A00 += tl.dot(b_k0, tl.trans(b_k0))
        if i_tc1 < T:
            p_k1 = tl.make_block_ptr(k_s, (T, K), (KH * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            b_k1 = tl.load(p_k1, boundary_check=(0, 1))
            b_A11 += tl.dot(b_k1, tl.trans(b_k1))
            b_A10 += tl.dot(b_k1, tl.trans(b_k0))
            if i_tc2 < T:
                p_k2 = tl.make_block_ptr(k_s, (T, K), (KH * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                b_k2 = tl.load(p_k2, boundary_check=(0, 1))
                b_A22 += tl.dot(b_k2, tl.trans(b_k2))
                b_A20 += tl.dot(b_k2, tl.trans(b_k0))
                b_A21 += tl.dot(b_k2, tl.trans(b_k1))
                if i_tc3 < T:
                    p_k3 = tl.make_block_ptr(k_s, (T, K), (KH * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1))
                    b_A33 += tl.dot(b_k3, tl.trans(b_k3))
                    b_A30 += tl.dot(b_k3, tl.trans(b_k0))
                    b_A31 += tl.dot(b_k3, tl.trans(b_k1))
                    b_A32 += tl.dot(b_k3, tl.trans(b_k2))

    b_A00 *= safe_exp(b_g0[:, None] - b_g0[None, :])
    b_A11 *= safe_exp(b_g1[:, None] - b_g1[None, :])
    b_A22 *= safe_exp(b_g2[:, None] - b_g2[None, :])
    b_A33 *= safe_exp(b_g3[:, None] - b_g3[None, :])
    b_A10 *= safe_exp(b_g1[:, None] - b_g0[None, :])
    b_A20 *= safe_exp(b_g2[:, None] - b_g0[None, :])
    b_A21 *= safe_exp(b_g2[:, None] - b_g1[None, :])
    b_A30 *= safe_exp(b_g3[:, None] - b_g0[None, :])
    b_A31 *= safe_exp(b_g3[:, None] - b_g1[None, :])
    b_A32 *= safe_exp(b_g3[:, None] - b_g2[None, :])

    m_d = o_j[:, None] > o_j[None, :]
    m_I = o_j[:, None] == o_j[None, :]
    b_A00 = tl.where(m_d & (m_tc0[:, None] & m_tc0[None, :]), b_A00, 0.0) * b_b0[:, None]
    b_A11 = tl.where(m_d & (m_tc1[:, None] & m_tc1[None, :]), b_A11, 0.0) * b_b1[:, None]
    b_A22 = tl.where(m_d & (m_tc2[:, None] & m_tc2[None, :]), b_A22, 0.0) * b_b2[:, None]
    b_A33 = tl.where(m_d & (m_tc3[:, None] & m_tc3[None, :]), b_A33, 0.0) * b_b3[:, None]
    b_A10 = b_A10 * b_b1[:, None]
    b_A20 = b_A20 * b_b2[:, None]
    b_A21 = b_A21 * b_b2[:, None]
    b_A30 = b_A30 * b_b3[:, None]
    b_A31 = b_A31 * b_b3[:, None]
    b_A32 = b_A32 * b_b3[:, None]

    if INV == 0:
        b_Ai00 = -b_A00
        b_Ai11 = -b_A11
        b_Ai22 = -b_A22
        b_Ai33 = -b_A33
        if DIAG & 2 == 0:
            for i in range(2, min(BC, T - i_tc0)):
                b_a00 = tl.sum(tl.where((o_j == i)[:, None], -b_A00, 0.0), 0)
                b_a00 = tl.where(o_j < i, b_a00, 0.0)
                b_a00 = b_a00 + tl.sum(b_a00[:, None] * b_Ai00, 0)
                b_Ai00 = tl.where((o_j == i)[:, None], b_a00, b_Ai00)
            for i in range(2, min(BC, T - i_tc1)):
                b_a11 = tl.sum(tl.where((o_j == i)[:, None], -b_A11, 0.0), 0)
                b_a11 = tl.where(o_j < i, b_a11, 0.0)
                b_a11 = b_a11 + tl.sum(b_a11[:, None] * b_Ai11, 0)
                b_Ai11 = tl.where((o_j == i)[:, None], b_a11, b_Ai11)
            for i in range(2, min(BC, T - i_tc2)):
                b_a22 = tl.sum(tl.where((o_j == i)[:, None], -b_A22, 0.0), 0)
                b_a22 = tl.where(o_j < i, b_a22, 0.0)
                b_a22 = b_a22 + tl.sum(b_a22[:, None] * b_Ai22, 0)
                b_Ai22 = tl.where((o_j == i)[:, None], b_a22, b_Ai22)
            for i in range(2, min(BC, T - i_tc3)):
                b_a33 = tl.sum(tl.where((o_j == i)[:, None], -b_A33, 0.0), 0)
                b_a33 = tl.where(o_j < i, b_a33, 0.0)
                b_a33 = b_a33 + tl.sum(b_a33[:, None] * b_Ai33, 0)
                b_Ai33 = tl.where((o_j == i)[:, None], b_a33, b_Ai33)

    else:  # (I + A)^-1 - I for strictly lower A by doubling: sum_{j=1..15} (-A)^j, fp32 ieee dots
        b_Ai00 = _neg_powsum(b_A00, m_I)
        b_Ai11 = _neg_powsum(b_A11, m_I)
        b_Ai22 = _neg_powsum(b_A22, m_I)
        b_Ai33 = _neg_powsum(b_A33, m_I)
    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    b_Ai10 = -tl.dot(tl.dot(b_Ai11, b_A10, input_precision=MERGE_PREC), b_Ai00, input_precision=MERGE_PREC)
    b_Ai21 = -tl.dot(tl.dot(b_Ai22, b_A21, input_precision=MERGE_PREC), b_Ai11, input_precision=MERGE_PREC)
    b_Ai32 = -tl.dot(tl.dot(b_Ai33, b_A32, input_precision=MERGE_PREC), b_Ai22, input_precision=MERGE_PREC)
    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_A20, b_Ai00, input_precision=MERGE_PREC) + tl.dot(b_A21, b_Ai10, input_precision=MERGE_PREC),
        input_precision=MERGE_PREC,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_A31, b_Ai11, input_precision=MERGE_PREC) + tl.dot(b_A32, b_Ai21, input_precision=MERGE_PREC),
        input_precision=MERGE_PREC,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_A30, b_Ai00, input_precision=MERGE_PREC)
        + tl.dot(b_A31, b_Ai10, input_precision=MERGE_PREC)
        + tl.dot(b_A32, b_Ai20, input_precision=MERGE_PREC),
        input_precision=MERGE_PREC,
    )

    ad = Ab.dtype.element_ty
    tl.store(tl.make_block_ptr(A_s, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0)), b_Ai00.to(ad), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(A_s, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0)), b_Ai10.to(ad), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(A_s, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0)), b_Ai11.to(ad), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(A_s, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0)), b_Ai20.to(ad), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(A_s, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0)), b_Ai21.to(ad), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(A_s, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0)), b_Ai22.to(ad), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(A_s, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0)), b_Ai30.to(ad), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(A_s, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0)), b_Ai31.to(ad), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(A_s, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0)), b_Ai32.to(ad), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(A_s, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0)), b_Ai33.to(ad), boundary_check=(0, 1))
    # the seed's A is zero-initialised (torch.zeros) and recompute_w_u reads the full 64 x 64 tile: zero the upper blocks
    b_zero = tl.zeros([BC, BC], dtype=ad)
    tl.store(tl.make_block_ptr(A_s, (T, BT), (H * BT, 1), (i_tc0, BC), (BC, BC), (1, 0)), b_zero, boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(A_s, (T, BT), (H * BT, 1), (i_tc0, 2 * BC), (BC, BC), (1, 0)), b_zero, boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(A_s, (T, BT), (H * BT, 1), (i_tc0, 3 * BC), (BC, BC), (1, 0)), b_zero, boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(A_s, (T, BT), (H * BT, 1), (i_tc1, 2 * BC), (BC, BC), (1, 0)), b_zero, boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(A_s, (T, BT), (H * BT, 1), (i_tc1, 3 * BC), (BC, BC), (1, 0)), b_zero, boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(A_s, (T, BT), (H * BT, 1), (i_tc2, 3 * BC), (BC, BC), (1, 0)), b_zero, boundary_check=(0, 1))

    tl.debug_barrier()

    if DIAG & 4:  # timing diagnostic: no recompute_w_u
        return
    # ---- the seed's recompute_w_u_fwd_kernel body (BK = BV = 64) ----
    p_beta = tl.make_block_ptr(beta_s, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_g = tl.make_block_ptr(g_s, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(A_s, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_ge = tl.exp(tl.load(p_g, boundary_check=(0,)))
    v_s = vc + (vbos * H + i_h) * V
    u_s = u + (vbos * H + i_h) * V
    w_s = w + (vbos * H + i_h) * K
    for i_v in range(tl.cdiv(V, 64)):
        p_v = tl.make_block_ptr(v_s, (T, V), (H * V, 1), (i_t * BT, i_v * 64), (BT, 64), (1, 0))
        p_u = tl.make_block_ptr(u_s, (T, V), (H * V, 1), (i_t * BT, i_v * 64), (BT, 64), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))
    for i_k in range(tl.cdiv(K, 64)):
        p_k = tl.make_block_ptr(k_s, (T, K), (KH * K, 1), (i_t * BT, i_k * 64), (BT, 64), (1, 0))
        p_w = tl.make_block_ptr(w_s, (T, K), (H * K, 1), (i_t * BT, i_k * 64), (BT, 64), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_ge[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def d3r3_gdnext_pro_kernel(
    x, stride_x, cw, stride_cw_d, stride_cw_w, cb,
    a, stride_a, b, stride_b, A_log, dt_bias,
    conv_st, stride_cs_slot, stride_cs_d, stride_cs_tok, slots,
    cu, cv, rtable,
    qn, kn, vc, graw, bs,
    HK: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BR: tl.constexpr, BH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Grid (row blocks of BR virtual rows, 2*HK + H channel blocks of 128 + 1 gating block): conv1d+SiLU, split,
    l2norm of q/k, zero virtual lead rows, conv state of the request's last row block; the gating block writes the
    raw g (cumsum happens per chunk in prep) and beta of all heads."""
    i_r, i_c = tl.program_id(0), tl.program_id(1)
    i_n = tl.load(rtable + i_r * 2).to(tl.int32)
    i_rb = tl.load(rtable + i_r * 2 + 1).to(tl.int32)
    bos = tl.load(cu + i_n).to(tl.int32)
    L = tl.load(cu + i_n + 1).to(tl.int32) - bos
    vbos = tl.load(cv + i_n).to(tl.int32)
    T = tl.load(cv + i_n + 1).to(tl.int32) - vbos
    lead = T - L
    o_rr = i_rb * BR + tl.arange(0, BR)
    o_tr = o_rr - lead
    m_rr = o_tr >= 0
    vrr = (vbos + o_rr).to(tl.int64)
    if i_c < 2 * HK + H:
        c0 = i_c * K
        y = _conv_silu(x, stride_x, cw, stride_cw_d, stride_cw_w, cb, c0, bos, o_tr, m_rr, BR, K, HAS_BIAS).to(tl.bfloat16)
        o_k = tl.arange(0, K)
        if i_c < 2 * HK:
            y_n = tl.where(m_rr[:, None], _l2norm_rows(y), 0.0).to(tl.bfloat16)
            if i_c < HK:
                tl.store(qn + (vrr * HK + i_c)[:, None] * K + o_k[None, :], y_n)
            else:
                tl.store(kn + (vrr * HK + (i_c - HK))[:, None] * K + o_k[None, :], y_n)
        else:
            y = tl.where(m_rr[:, None], y, 0.0)
            tl.store(vc + (vrr * H + (i_c - 2 * HK))[:, None] * V + o_k[None, :], y.to(tl.bfloat16))
        if (i_rb + 1) * BR >= T:  # the request's last row block: conv state = its last 3 raw rows
            slot = tl.load(slots + i_n).to(tl.int64)
            _store_conv_state(x, stride_x, conv_st + slot * stride_cs_slot, stride_cs_d, stride_cs_tok, c0, bos, L, K)
    else:
        o_h = tl.arange(0, BH)
        m_h = o_h < H
        prow = (bos + o_tr).to(tl.int64)
        m2 = m_rr[:, None] & m_h[None, :]
        b_a = tl.load(a + prow[:, None] * stride_a + o_h[None, :], mask=m2, other=0.0).to(tl.float32)
        b_b = tl.load(b + prow[:, None] * stride_b + o_h[None, :], mask=m2, other=0.0).to(tl.float32)
        xg = b_a + tl.load(dt_bias + o_h, mask=m_h, other=0.0).to(tl.float32)[None, :]
        softplus_x = tl.where(xg <= 20.0, tl.log(1 + tl.exp(xg)), xg)
        b_graw = -tl.exp(tl.load(A_log + o_h, mask=m_h, other=0.0).to(tl.float32))[None, :] * softplus_x
        b_graw = tl.where(m2, b_graw, 0.0)
        b_beta = tl.where(m2, tl.sigmoid(b_b).to(tl.bfloat16).to(tl.float32), 0.0)
        tl.store(graw + vrr[:, None] * H + o_h[None, :], b_graw, mask=m_h[None, :])
        tl.store(bs + vrr[:, None] * H + o_h[None, :], b_beta, mask=m_h[None, :])


@triton.jit(do_not_specialize=["scale"])
def d3r3_gdnext_rec_kernel(
    qn, kn, w, u, gs, o, stride_o_t, ssm, stride_ssm_slot, slots, cu, cv, scale,
    HK: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
    KH: tl.constexpr = 48,
):
    """The seed's chunk_fwd_h (blockdim64, K = 128) with chunk_fwd_o of each chunk computed from the in-register h."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    G: tl.constexpr = H // HK
    i_kh = i_h // G
    bos = tl.load(cu + i_n).to(tl.int32)
    L = tl.load(cu + i_n + 1).to(tl.int32) - bos
    vbos = tl.load(cv + i_n).to(tl.int32)
    T = tl.load(cv + i_n + 1).to(tl.int32) - vbos
    lead = T - L
    NT = T // BT
    slot = tl.load(slots + i_n).to(tl.int64)
    h0 = ssm + slot * stride_ssm_slot + i_h * V * K

    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    b_h1 += tl.load(tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)), boundary_check=(0, 1)).to(tl.float32)
    b_h2 += tl.load(tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)), boundary_check=(0, 1)).to(tl.float32)

    w_s = w + (vbos * H + i_h) * K
    u_s = u + (vbos * H + i_h) * V
    k_s = kn + (vbos * KH + (i_h if KH == H else i_kh)) * K
    q_s = qn + (vbos * HK + i_kh) * K
    g_s = gs + vbos * H + i_h
    o_i = tl.arange(0, BT)
    o_c = i_v * BV + tl.arange(0, BV)
    m_A = o_i[:, None] >= o_i[None, :]
    for i_t in range(NT):
        b_hb1 = b_h1.to(tl.bfloat16)
        b_hb2 = b_h2.to(tl.bfloat16)
        b_w = tl.load(tl.make_block_ptr(w_s, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0)), boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_hb1))
        b_w = tl.load(tl.make_block_ptr(w_s, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0)), boundary_check=(0, 1))
        b_v += tl.dot(b_w, tl.trans(b_hb2))
        p_u = tl.make_block_ptr(u_s, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_u, boundary_check=(0, 1)) - b_v
        b_vn = b_v.to(tl.bfloat16)

        # chunk_fwd_o for this chunk (BK = 128 as two chained halves)
        b_q1 = tl.load(tl.make_block_ptr(q_s, (T, K), (HK * K, 1), (i_t * BT, 0), (BT, 64), (1, 0)), boundary_check=(0, 1))
        b_q2 = tl.load(tl.make_block_ptr(q_s, (T, K), (HK * K, 1), (i_t * BT, 64), (BT, 64), (1, 0)), boundary_check=(0, 1))
        b_k1 = tl.load(tl.make_block_ptr(k_s, (K, T), (1, KH * K), (0, i_t * BT), (64, BT), (0, 1)), boundary_check=(0, 1))
        b_k2 = tl.load(tl.make_block_ptr(k_s, (K, T), (1, KH * K), (64, i_t * BT), (64, BT), (0, 1)), boundary_check=(0, 1))
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        b_o += tl.dot(b_q1, tl.trans(b_hb1))
        b_o += tl.dot(b_q2, tl.trans(b_hb2))
        b_A += tl.dot(b_q1, b_k1)
        b_A += tl.dot(b_q2, b_k2)
        b_g = tl.load(tl.make_block_ptr(g_s, (T,), (H,), (i_t * BT,), (BT,), (0,)), boundary_check=(0,))
        b_o = b_o * tl.exp(b_g)[:, None]
        b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])
        b_A = tl.where(m_A, b_A, 0)
        b_o = b_o * scale + tl.dot(b_A.to(b_vn.dtype), b_vn) * scale
        o_tok = i_t * BT + o_i - lead
        p_o = o + (bos + o_tok).to(tl.int64)[:, None] * stride_o_t + i_h * V + o_c[None, :]
        tl.store(p_o, b_o.to(o.dtype.element_ty), mask=(o_tok >= 0)[:, None])

        # state update (chunk_fwd_h)
        b_g_last = tl.load(g_s + ((i_t + 1) * BT - 1) * H)
        b_v = b_v * safe_exp(b_g_last - b_g)[:, None]
        b_g_last = tl.exp(b_g_last)
        b_h1 = b_h1 * b_g_last
        b_h2 = b_h2 * b_g_last
        b_v = b_v.to(tl.bfloat16)
        b_h1 += tl.trans(tl.dot(b_k1, b_v))
        b_h2 += tl.trans(tl.dot(b_k2, b_v))

    tl.store(tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)), b_h1.to(ssm.dtype.element_ty), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)), b_h2.to(ssm.dtype.element_ty), boundary_check=(0, 1))


class ExtendMeta:
    """Per-forward layout of the fresh requests (built once at the first GDN layer, shared by all layers)."""

    def __init__(self, seq_lens: list, *, HK: int, H: int, K: int, V: int, device) -> None:
        cu, cv, table, rtable = [0], [0], [], []
        for n, L in enumerate(seq_lens):
            Tv = L + (-L) % CHUNK
            table += [x for i_t in range(Tv // CHUNK) for x in (n, i_t)]
            rtable += [x for i_r in range(Tv // _PRO_ROWS) for x in (n, i_r)]
            cu.append(cu[-1] + L)
            cv.append(cv[-1] + Tv)
        host = torch.tensor(cu + cv + table + rtable, dtype=torch.int32, pin_memory=torch.device(device).type == "cuda")
        dev = host.to(device, non_blocking=True)
        N = len(seq_lens)
        self.n_req = N
        self.n_rows = cu[-1]
        self.cu = dev[: N + 1]
        self.cv = dev[N + 1 : 2 * N + 2]
        self.table = dev[2 * N + 2 : 2 * N + 2 + len(table)]
        self.rtable = dev[2 * N + 2 + len(table) :]
        self.n_chunks = len(table) // 2
        self.n_rowblocks = len(rtable) // 2
        R = cv[-1]
        self.qn = torch.empty(R, HK, K, dtype=torch.bfloat16, device=device)
        self.kn = torch.empty(R, H, K, dtype=torch.bfloat16, device=device)
        self.vc = torch.empty(R, H, V, dtype=torch.bfloat16, device=device)
        self.w = torch.empty(R, H, K, dtype=torch.bfloat16, device=device)
        self.u = torch.empty(R, H, V, dtype=torch.bfloat16, device=device)
        self.gs = torch.empty(R, H, dtype=torch.float32, device=device)
        self.bs = torch.empty(R, H, dtype=torch.float32, device=device)
        self.A = torch.empty(R, H, CHUNK, dtype=torch.bfloat16, device=device)
        self._host = host  # keep the pinned source alive until the copy ran


def gdn_extend(*, meta: ExtendMeta, mixed_qkv, a, b, conv_weight, conv_bias, A_log, dt_bias, conv_states,
               ssm_states, slots, HK: int, H: int, K: int, V: int, out: torch.Tensor | None = None,
               prep_warps: int | None = None, prep_rows: int | None = None, rec_warps: int = 4,
               launch: str = "both", inv: int | None = None, diag: int = 0, split: int | None = None) -> torch.Tensor:
    """mixed_qkv [rows, HK*K*2 + H*V] (unit channel stride), a/b [rows, H]; conv_weight [D, 4];
    conv_states [slots, D, 3] (any strides), ssm_states [slots, H, V, K] fp32 (slot contiguous); slots int32 [N].
    Writes the final recurrent/conv state of each request in place; returns o [1, rows, H, V] bf16."""
    assert K == 128 and V % _BV_REC == 0 and mixed_qkv.stride(-1) == 1
    assert ssm_states.stride(-1) == 1 and ssm_states[0].is_contiguous()
    if out is None:
        out = torch.empty(1, meta.n_rows, H, V, dtype=torch.bfloat16, device=mixed_qkv.device)
    split = SPLIT if split is None else split
    if split and launch in ("both", "prep", "pro"):
      d3r3_gdnext_pro_kernel[(meta.n_rowblocks, 2 * HK + H + 1)](
        mixed_qkv, mixed_qkv.stride(0), conv_weight, conv_weight.stride(0), conv_weight.stride(1), conv_bias,
        a, a.stride(0), b, b.stride(0), A_log, dt_bias,
        conv_states, conv_states.stride(0), conv_states.stride(1), conv_states.stride(2), slots,
        meta.cu, meta.cv, meta.rtable,
        meta.qn, meta.kn, meta.vc, meta.gs, meta.bs,
        HK=HK, H=H, K=K, V=V, BR=_PRO_ROWS, BH=triton.next_power_of_2(H), HAS_BIAS=conv_bias is not None,
        num_warps=4, num_stages=2,
      )
    if launch in ("both", "prep"):
      d3r3_gdnext_prep_kernel[(meta.n_chunks, H)](
        mixed_qkv, mixed_qkv.stride(0), conv_weight, conv_weight.stride(0), conv_weight.stride(1), conv_bias,
        a, a.stride(0), b, b.stride(0), A_log, dt_bias,
        conv_states, conv_states.stride(0), conv_states.stride(1), conv_states.stride(2), slots,
        meta.cu, meta.cv, meta.table,
        meta.qn, meta.kn, meta.vc, meta.gs, meta.bs, meta.A, meta.w, meta.u,
        meta.n_chunks,
        HK=HK, H=H, K=K, V=V, BT=CHUNK, BC=16, BK=_KKT_BK, BR=prep_rows or PREP_ROWS,
        HAS_BIAS=conv_bias is not None, MERGE_PREC=_MERGE_PREC, INV=INV if inv is None else inv, DIAG=diag, SPLIT=split,
        num_warps=prep_warps or PREP_WARPS, num_stages=2,
      )
    if launch in ("both", "rec"):
      d3r3_gdnext_rec_kernel[(V // _BV_REC, meta.n_req * H)](
        meta.qn, meta.kn, meta.w, meta.u, meta.gs, out, out.stride(1), ssm_states, ssm_states.stride(0), slots,
        meta.cu, meta.cv, K ** -0.5,
        HK=HK, H=H, K=K, V=V, BT=CHUNK, BV=_BV_REC, KH=HK if split else H,
        num_warps=rec_warps, num_stages=2,
      )
    return out


class EngineHook:
    """gdn_backend.forward_extend hook (SPECPILOT_D3R3_GDNEXT=1): fresh extends (every request without prefix, the
    whole batch is the packed prompt rows, <= MAX_ROWS rows) run gdn_extend instead of the SPECPILOT_X1 pgraph / the
    eager path. The layout (ExtendMeta) is built once per forward at its first GDN layer, from host lists only (one
    pinned non-blocking H2D copy, no host sync). Verify, MIS, mamba tracking, state checkpoints and gathered pools
    keep their paths (the caller checks those)."""

    def __init__(self) -> None:
        self._fb = None  # weakref to the forward batch the cached meta belongs to
        self._meta = None
        self._last_lid = -1
        self.hits = 0

    @staticmethod
    def _layout(forward_batch, rows: int, cache_indices) -> list | None:
        n = forward_batch.batch_size
        pre, lens = forward_batch.extend_prefix_lens_cpu, forward_batch.extend_seq_lens_cpu
        if (pre is None or lens is None or len(lens) != n or len(pre) != n or any(pre)
                or not 0 < rows <= MAX_ROWS or sum(lens) != rows or min(lens) < 1
                or cache_indices.shape[0] < n or torch.cuda.is_current_stream_capturing()):
            return None
        return list(lens)

    def try_run(self, *, layer, forward_batch, mixed_qkv, a, b, cache_indices, conv_states, ssm_states,
                output) -> torch.Tensor | None:
        import weakref

        rows = mixed_qkv.shape[0]
        HK, H, K, V = layer.num_k_heads, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim
        if (layer.num_q_heads != HK or layer.head_q_dim != K or K != 128 or V % _BV_REC or H % HK
                or mixed_qkv.dtype != torch.bfloat16 or mixed_qkv.stride(-1) != 1
                or a.stride(-1) != 1 or b.stride(-1) != 1 or a.shape[0] != rows or b.shape[0] != rows
                or ssm_states.dtype != torch.float32 or ssm_states.stride(-1) != 1 or not ssm_states[0].is_contiguous()
                or conv_states.dtype != mixed_qkv.dtype or layer.activation != "silu"
                or not isinstance(layer.conv_weights, torch.Tensor) or layer.conv_weights.dim() != 2
                or tuple(layer.conv_weights.shape) != (mixed_qkv.shape[1], 4)
                or tuple(conv_states.shape[1:]) != (mixed_qkv.shape[1], 3)
                or mixed_qkv.shape[1] != 2 * HK * K + H * V):
            return None
        fb_ref = self._fb() if self._fb is not None else None
        new_pass = layer.layer_id <= self._last_lid  # the same batch object run through the layers again
        self._last_lid = layer.layer_id
        if fb_ref is not forward_batch or new_pass:
            try:
                self._fb, self._meta = weakref.ref(forward_batch), None
            except TypeError:
                self._fb, self._meta = None, None
                return None
            lens = self._layout(forward_batch, rows, cache_indices)
            if lens is not None:
                self._meta = ExtendMeta(lens, HK=HK, H=H, K=K, V=V, device=mixed_qkv.device)
                self._meta.dims = (HK, H, K, V)
        meta = self._meta
        if meta is None or meta.n_rows != rows or meta.dims != (HK, H, K, V):
            return None
        out = None
        if (output is not None and tuple(output.shape) == (1, rows, H, V) and output.dtype == torch.bfloat16
                and output.stride(3) == 1 and output.stride(2) == V):
            out = output
        o = gdn_extend(meta=meta, mixed_qkv=mixed_qkv, a=a, b=b, conv_weight=layer.conv_weights,
                       conv_bias=layer.bias, A_log=layer.A_log, dt_bias=layer.dt_bias, conv_states=conv_states,
                       ssm_states=ssm_states, slots=cache_indices[: meta.n_req], HK=HK, H=H, K=K, V=V, out=out)
        self.hits += 1
        if self.hits in (1, 48 * 100):
            import logging

            logging.getLogger(__name__).info("SPECPILOT_D3R3_GDNEXT fused extends: %d layer calls (rows %d, reqs %d)",
                                             self.hits, rows, meta.n_req)
        return o


def make_hook() -> EngineHook | None:
    return EngineHook() if ENABLED else None
