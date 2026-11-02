"""SPECPILOT_D3W_GDN=1: wide GDN tree verify (33 <= N <= 48 tree nodes, batch-1 path) without the parent WY fallback.

d8's 3-kernel PDL chain (specpilot_d8_gdn: conv -> prep -> state) handles N <= 32 with one 32-row tile. Here the
nodes are split in node order into block A (rows 0..31) and block B (rows 32..N-1, one 16-row tile) instead of one
64-row tile. Parents precede children, so the WY matrix X (X[t, s] = beta_t e^(gam_t - gam_s) k_t.k_s for s a strict
ancestor of t) is strictly lower triangular in node order and block-lower-triangular in (A, B):

    (I + X)^-1 = [[A^-1, 0], [-D^-1 X_BA A^-1, D^-1]],   A = I + X_AA, D = I + X_BB
    Pf = P (I + X)^-1:  Pf_AA = P_AA A^-1,  Pf_BB = P_BB D^-1,  Pf_BA = (P_BA - Pf_BB X_BA) A^-1.

When block B holds only leaves (no node >= 32 is an ancestor: X_BB = 0, P_BB diagonal; uniform in-kernel branch on
the tree meta) Pf_BB = P_BB and Pf_BA = (P_BA - diag(P_BB) X_BA) A^-1: the leaf rows add no solve level (the leaf
factorization of a wide tree). Otherwise D^-1 is a 16x16 power series (X_BB^16 = 0). A^-1 is d8's power series
(I - X)(I + X^2)(I + X^4)(I + X^8)[(I + X^16) if the depth of block A >= 16] with exact fp32 dots, applied to the
stacked rows [P_AA; W] as two interleaved chains.

  * ``tree_meta_wide``: parents rebuilt from first-child / next-sibling by pointer jumping (6 rounds, no serial
    N-step loop), conv ancestors (a1, a2, a3) and the strict-ancestor set as 64-bit masks, stored as d8's [bs, 4, N]
    meta (a1, a2, a3, low word) + a [bs, N] high word;
  * conv: d8's ``_d8_tree_conv_kernel`` at NP = 64 (bitwise the parent's conv arithmetic);
  * ``_d3w_prep_kernel``: d8's prep arithmetic per block (commit inputs bitwise equal to the parent's), block solve;
  * ``_d3w_state_kernel``: o_A = e^gam Qn h0 + Pf_AA rhs_A, o_B = e^gam Qn h0 + Pf_BA rhs_A + Pf_BB rhs_B, gated RMSNorm.

Knobs (read once at import): SPECPILOT_D3W_GDN (1 = on; the d8 hook calls ``gdn_layer_verify_wide`` for
32 < N <= 48), SPECPILOT_D3W_PREP_WARPS (8), SPECPILOT_D3W_STATE_WARPS (8). N <= 32 stays on d8's kernels.
SPECPILOT_INT_D3W32 (int-012; independent of SPECPILOT_D3W_GDN): the register-gather tree accept and tree meta
kernels also for N <= 32 (``d7_tree_accept_commit`` / d8's ``verify_layer`` dispatch; same integer outputs as d7's
accept and d8's meta, the low ancestor word is the whole set; d8's N <= 32 GDN kernels unchanged).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_D3W_GDN = os.environ.get("SPECPILOT_D3W_GDN", "0") not in ("", "0")
_D3W_PREP_WARPS = int(os.environ.get("SPECPILOT_D3W_PREP_WARPS", "8") or 8)
_D3W_STATE_WARPS = int(os.environ.get("SPECPILOT_D3W_STATE_WARPS", "8") or 8)
_D3W_LEAF_WARPS = int(os.environ.get("SPECPILOT_D3W_LEAF_WARPS", "8") or 8)
_D3W_META_WARPS = int(os.environ.get("SPECPILOT_D3W_META_WARPS", "1") or 1)
_D3W_ACC_WARPS = int(os.environ.get("SPECPILOT_D3W_ACC_WARPS", "1") or 1)
_INT_D3W32 = os.environ.get("SPECPILOT_INT_D3W32", "0") not in ("", "0")
_D3W_FUSED_B = os.environ.get("SPECPILOT_D3W_FUSED_B", "0") not in ("", "0")
_D3W_TWO_CHAIN = os.environ.get("SPECPILOT_D3W_TWO_CHAIN", "0") not in ("", "0")

N_MIN, N_MAX = 33, 48
BA, BB = 32, 16  # row blocks


def enabled() -> bool:
    return _D3W_GDN


def small_enabled() -> bool:
    """SPECPILOT_INT_D3W32: register-gather accept / tree meta also for N <= 32."""
    return _INT_D3W32


def supports(n_nodes: int) -> bool:
    return N_MIN <= int(n_nodes) <= N_MAX


@triton.jit
def _gat(v, idx):
    # v[idx] for 0 <= idx < NP, else 0 (register gather; idx clamped for the hardware gather)
    NP: tl.constexpr = v.shape[0]
    ok = (idx >= 0) & (idx < NP)
    g = tl.gather(v, tl.minimum(tl.maximum(idx, 0), NP - 1).to(tl.int32), 0)
    return tl.where(ok, g, tl.zeros_like(g))


@triton.jit
def _d3w_tree_meta_kernel(
    nt_ptr,  # [bs, N] first child
    ns_ptr,  # [bs, N] next sibling
    par_ptr,  # [bs, N] parent (out unless given; root = 0 as the conv kernel writes it)
    meta_ptr,  # [bs, 4, N] int32 (out): a1, a2, a3 (conv ancestors; <= 0 -> state column), strict-ancestor bits 0..31
    hi_ptr,  # [bs, N] int32 (out): strict-ancestor bits 32..63
    snt, sns, spr,
    N: tl.constexpr, NP: tl.constexpr, PARENTS_GIVEN: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    tt = tl.arange(0, NP)
    vt = tt < N
    if PARENTS_GIVEN:
        par = tl.load(par_ptr + b * spr + tt, mask=vt, other=0).to(tl.int32)
    else:
        # par[nt[i]] = i; par[ns[i]] = par[i] (the conv kernel's serial rebuild): inverse links by one compare each,
        # then pointer jumping along previous-sibling links to the first child of the sibling run
        jj = tt[None, :]
        nt = tl.load(nt_ptr + b * snt + tt, mask=vt, other=-1).to(tl.int32)
        ns = tl.load(ns_ptr + b * sns + tt, mask=vt, other=-1).to(tl.int32)
        hit_fc = (nt[None, :] == tt[:, None]) & vt[None, :]  # [j, i]: nt[i] == j
        hit_pv = (ns[None, :] == tt[:, None]) & vt[None, :]  # [j, i]: ns[i] == j
        has_fc = tl.max(hit_fc.to(tl.int32), 1) > 0
        has_pv = tl.max(hit_pv.to(tl.int32), 1) > 0
        fc = tl.max(tl.where(hit_fc, jj, 0), 1)
        pv = tl.max(tl.where(hit_pv, jj, 0), 1)
        known = (has_fc | (tt == 0) | (~has_pv)).to(tl.int32)
        val = tl.where(has_fc & (tt > 0), fc, 0)
        jmp = tl.where(known > 0, -1, pv)
        for _ in tl.static_range(6):
            act = jmp >= 0
            v_t = _gat(val, jmp)
            k_t = _gat(known, jmp) > 0
            j_t = _gat(jmp, jmp)
            newly = act & k_t
            val = tl.where(newly, v_t, val)
            jmp = tl.where(act & (~k_t), j_t, -1)
            known = known | newly.to(tl.int32)
        par = tl.where(vt, val, 0)
        tl.store(par_ptr + b * spr + tt, par.to(par_ptr.dtype.element_ty), mask=vt)
    a1 = tl.where(tt > 0, par, tt - 1)
    a2 = tl.where(a1 > 0, _gat(par, a1), a1 - 1)
    a3 = tl.where(a2 > 0, _gat(par, a2), a2 - 1)
    # strict ancestors by pointer jumping (6 rounds cover 63 hops), 64-bit sets
    wpar = tl.where((tt > 0) & vt, par, -1)
    bits = tl.where(wpar >= 0, tl.full([NP], 1, tl.int64) << tl.maximum(wpar, 0).to(tl.int64), 0)
    jmp2 = wpar
    for _ in tl.static_range(6):
        act = jmp2 >= 0
        bits = bits | tl.where(act, _gat(bits, jmp2), 0)
        jmp2 = tl.where(act, _gat(jmp2, jmp2), -1)
        jmp2 = tl.where(act & (jmp2 >= 0), jmp2, -1)
    base = meta_ptr + b * 4 * N
    tl.store(base + tt, a1, mask=vt)
    tl.store(base + N + tt, a2, mask=vt)
    tl.store(base + 2 * N + tt, a3, mask=vt)
    tl.store(base + 3 * N + tt, (bits & 0xFFFFFFFF).to(tl.int32), mask=vt)
    tl.store(hi_ptr + b * N + tt, (bits >> 32).to(tl.int32), mask=vt)


@triton.jit
def _d3w_tree_accept_kernel(
    tokens_ptr, parents_ptr, pred_ptr, prefix_ptr, req_pool_ptr, req_to_token_ptr, stride_rtt,
    accept_len_out, commit_out, bonus_out, out_tokens_out, new_seq_out, leaf_out, path_out, slots_out,
    N: tl.constexpr, NP: tl.constexpr, LOGN: tl.constexpr, PERMUTE: tl.constexpr,
):
    # specpilot_d7_accept._d7_tree_accept_kernel with register gathers (tl.gather) instead of [NP, NP] select-sums
    # (same integer results); for 32 < N <= 64 trees.
    b = tl.program_id(0).to(tl.int64)
    nidx = tl.arange(0, NP)
    live = nidx < N
    base = b * N
    tok = tl.load(tokens_ptr + base + nidx, mask=live, other=-1)
    par = tl.load(parents_ptr + base + nidx, mask=live, other=-2)
    pred = tl.load(pred_ptr + base + nidx, mask=live, other=-1)
    ii = nidx[:, None]
    jj = nidx[None, :]
    ppred = _gat(pred, par)
    ok = live & (par >= 0) & (tok == ppred)
    jmp = par
    dep = tl.where(par >= 0, 1, 0).to(tl.int64)
    okp = ((nidx == 0) | ok).to(tl.int32)
    anc = tl.where(live, tl.full([NP], 1, tl.int64) << nidx.to(tl.int64), 0)
    for _ in tl.static_range(LOGN):
        has = jmp >= 0
        dep = tl.where(has, dep + _gat(dep, jmp), dep)
        okp = tl.where(has, okp & _gat(okp, jmp), okp)
        anc = tl.where(has, anc | _gat(anc, jmp), anc)
        jmp = tl.where(has, _gat(jmp, jmp), jmp)
    acc = okp > 0
    key = tl.where(acc, dep * (NP + 1) + (NP - nidx), -1)
    best = tl.max(key, 0)
    leaf = tl.min(tl.where(key == best, nidx, NP), 0).to(tl.int64)
    a = tl.sum(tl.where(nidx == leaf, dep, 0), 0)
    lmask = tl.sum(tl.where(nidx == leaf, anc, 0), 0)
    onpath = ((lmask >> nidx.to(tl.int64)) & 1) > 0
    sel = (dep[None, :] == ii) & onpath[None, :]
    path = tl.where(tl.max(sel.to(tl.int32), 1) > 0, tl.max(tl.where(sel, jj, 0), 1).to(tl.int64), -1)
    path_next = _gat(path, nidx + 1)
    tok_next = _gat(tok, path_next)
    bonus = tl.sum(tl.where(nidx == leaf, pred, 0), 0)
    out = tl.where(nidx < a, tok_next, tl.where(nidx == a, bonus, 0))

    prefix = tl.load(prefix_ptr + b).to(tl.int64)
    tl.store(accept_len_out + b, a.to(tl.int32))
    tl.store(commit_out + b, (a + 1).to(tl.int32))
    tl.store(bonus_out + b, bonus)
    tl.store(new_seq_out + b, prefix + a + 1)
    tl.store(leaf_out + b, leaf)
    tl.store(out_tokens_out + base + nidx, out, mask=live)
    tl.store(path_out + base + nidx, path, mask=live)

    r = tl.load(req_pool_ptr + b).to(tl.int64)
    row = req_to_token_ptr + r * stride_rtt + prefix
    slots = tl.load(row + nidx, mask=live, other=0).to(tl.int64)
    if PERMUTE:
        for d in range(1, a + 1):
            nd = tl.sum(tl.where(nidx == d, path, 0), 0)
            do = (nd != d) & (nd > 0)
            vd = tl.sum(tl.where(nidx == d, slots, 0), 0)
            vn = tl.sum(tl.where(nidx == nd, slots, 0), 0)
            slots = tl.where(do & (nidx == d), vn, tl.where(do & (nidx == nd), vd, slots))
        tl.store(row + nidx, slots.to(tl.int32), mask=live)
    tl.store(slots_out + base + nidx, slots, mask=live)


@triton.jit
def _gating(a_ptr, b_ptr, toks, valid, i_hv, stride_a, stride_b, b_A_log, b_dt_bias, softplus_beta,
            softplus_threshold):
    b_b = tl.load(b_ptr + toks * stride_b + i_hv, mask=valid, other=0).to(tl.float32)
    b_a = tl.load(a_ptr + toks * stride_a + i_hv, mask=valid, other=0).to(tl.float32)
    x = b_a + b_dt_bias
    beta_x = softplus_beta * x
    softplus_x = tl.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
        x,
    )
    b_g = -tl.exp(b_A_log) * softplus_x
    b_g = tl.where(valid, b_g, 0.0)
    b_beta = 1.0 / (1.0 + tl.exp(-b_b))
    b_beta = tl.where(valid, b_beta, 0.0)
    return b_g, b_beta


@triton.jit
def _d3w_prep_a_kernel(
    c_ptr,  # conv output [T, 2HK + HV V] (q | k | v)
    a_ptr, b_ptr, cs_idx, row_ptr, meta_ptr, A_log, dt_bias, kn_out, v_out, g_out, beta_out, qk_ptr, pf_ptr,
    ainv_ptr, fac_ptr, scale, softplus_beta, softplus_threshold, sct, stride_a, stride_b,
    N: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    PDL_IN: tl.constexpr, PDL_OUT: tl.constexpr, TWO_CHAIN: tl.constexpr = False,
):
    # d8's prep on block A (rows 0..31) with the solve kept explicit: A^-1 = (I - X)(I + X^2)(I + X^4)(I + X^8)
    # [(I + X^16)], Pf_AA = P_AA A^-1 (block B needs A^-1 rhs_A).
    if PDL_OUT:
        tl.extra.cuda.gdc_launch_dependents()
    i_n, i_hv = tl.program_id(0), tl.program_id(1)
    G: tl.constexpr = HV // H
    i_h = i_hv // G
    slot = tl.load(cs_idx + i_n).to(tl.int64)
    if slot < 0:
        return
    tt = tl.arange(0, 32)
    valid = tt < N
    mv = valid[:, None]
    bits = tl.load(meta_ptr + i_n * 4 * N + 3 * N + tt, mask=valid, other=0)
    anc = (((bits[:, None] >> tt[None, :]) & 1) != 0) & valid[None, :]
    eye = (tt[:, None] == tt[None, :]) & valid[:, None]
    anc_eq = anc | eye
    maxd = tl.max(tl.sum(anc.to(tl.int32), 1), 0)
    toks = (i_n * N + tt).to(tl.int64)
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
    b_g, b_beta = _gating(a_ptr, b_ptr, toks, valid, i_hv, stride_a, stride_b, b_A_log, b_dt_bias, softplus_beta,
                          softplus_threshold)
    row = tl.load(row_ptr + i_n).to(tl.int64)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    sbase = row * N + tt
    gam = tl.sum(tl.where(anc_eq, b_g[None, :], 0.0), 1)
    if PDL_IN:
        tl.extra.cuda.gdc_wait()  # conv output ready
    cb = c_ptr + toks[:, None] * sct
    qb = tl.load(cb + i_h * K + o_k[None, :], mask=mv, other=0.0)
    kb = tl.load(cb + H * K + i_h * K + o_k[None, :], mask=mv, other=0.0)
    vb = tl.load(cb + 2 * H * K + i_hv * V + o_v[None, :], mask=mv, other=0.0)
    tl.store(v_out + (sbase[:, None] * HV + i_hv) * V + o_v[None, :], vb, mask=mv)
    qf = qb.to(tl.float32)
    kf = kb.to(tl.float32)
    rq = 1.0 / tl.sqrt(tl.sum(qf * qf, 1) + 1e-6)
    rk = 1.0 / tl.sqrt(tl.sum(kf * kf, 1) + 1e-6)
    tl.store(g_out + sbase * HV + i_hv, b_g, mask=valid)
    tl.store(beta_out + sbase * HV + i_hv, b_beta, mask=valid)
    if i_hv % G == 0:
        tl.store(kn_out + (sbase[:, None] * H + i_h) * K + o_k[None, :], kf * rk[:, None], mask=mv)
    kbt = tl.trans(kb)
    b_KK = tl.dot(kb, kbt) * (rk[:, None] * rk[None, :])
    b_QK = tl.dot(qb, kbt) * ((rq * scale)[:, None] * rk[None, :])
    dgam = gam[:, None] - gam[None, :]
    X = tl.where(anc, b_beta[:, None] * tl.exp(tl.where(anc, dgam, 0.0)) * b_KK, 0.0)
    P = tl.where(anc_eq, tl.exp(tl.where(anc_eq, dgam, 0.0)) * b_QK, 0.0)
    X2 = tl.dot(X, X, input_precision="ieee")
    M1 = tl.where(eye, 1.0, 0.0) - X
    if TWO_CHAIN:  # Pf by d8's own chain next to the A^-1 chain (no extra level, more live tiles)
        P1 = P - tl.dot(P, X, input_precision="ieee")
    X4 = tl.dot(X2, X2, input_precision="ieee")
    M2 = M1 + tl.dot(M1, X2, input_precision="ieee")
    if TWO_CHAIN:
        P2 = P1 + tl.dot(P1, X2, input_precision="ieee")
    X8 = tl.dot(X4, X4, input_precision="ieee")
    M3 = M2 + tl.dot(M2, X4, input_precision="ieee")
    Ainv = M3 + tl.dot(M3, X8, input_precision="ieee")
    if TWO_CHAIN:
        P3 = P2 + tl.dot(P2, X4, input_precision="ieee")
        Pf = P3 + tl.dot(P3, X8, input_precision="ieee")
    if maxd >= 16:  # else X^16 == 0 exactly
        X16 = tl.dot(X8, X8, input_precision="ieee")
        Ainv = Ainv + tl.dot(Ainv, X16, input_precision="ieee")
        if TWO_CHAIN:
            Pf = Pf + tl.dot(Pf, X16, input_precision="ieee")
    if not TWO_CHAIN:
        Pf = tl.dot(P, Ainv, input_precision="ieee")
    hb = i_n * HV + i_hv
    tl.store(pf_ptr + hb * 32 * 32 + tt[:, None] * 32 + tt[None, :], Pf)
    tl.store(ainv_ptr + hb * 32 * 32 + tt[:, None] * 32 + tt[None, :], Ainv)
    fb = fac_ptr + hb * 4 * 32
    tl.store(fb + tt, tl.exp(gam))
    tl.store(fb + 32 + tt, b_beta)
    tl.store(fb + 64 + tt, rk)
    tl.store(fb + 96 + tt, rq * scale)
    if i_hv % G == 0:
        qkb = qk_ptr + toks[:, None] * (2 * H * K) + i_h * K + o_k[None, :]
        tl.store(qkb, qb, mask=mv)
        tl.store(qkb + H * K, kb, mask=mv)


@triton.jit
def _gated_norm_store(b_o, zz, w, o_ptr, toks, mv, i_hv, o_v, sot, eps, V: tl.constexpr):
    xo = tl.where(mv, b_o.to(o_ptr.dtype.element_ty).to(tl.float32), 0.0)
    var = tl.sum(xo * xo, axis=1) / V
    rstd = tl.rsqrt(var + eps)
    y = xo * rstd[:, None] * w[None, :]
    y *= zz * tl.sigmoid(zz)
    tl.store(o_ptr + toks[:, None] * sot + i_hv * V + o_v[None, :], y.to(o_ptr.dtype.element_ty), mask=mv)


@triton.jit
def _hsplit(h0_source, slot, stride_h0, i_hv, o_k, o_v, K: tl.constexpr, V: tl.constexpr):
    b_h = tl.load(h0_source + tl.maximum(slot, 0) * stride_h0 + i_hv * K * V + o_v[None, :] * K + o_k[:, None])
    h_hi = b_h.to(tl.bfloat16)
    r1 = b_h - h_hi.to(tl.float32)
    h_mid = r1.to(tl.bfloat16)
    h_lo = (r1 - h_mid.to(tl.float32)).to(tl.bfloat16)
    return h_hi, h_mid, h_lo


@triton.jit
def _hdot(x, h_hi, h_mid, h_lo):
    return tl.dot(x, h_hi) + tl.dot(x, h_mid) + tl.dot(x, h_lo)


@triton.jit
def _d3w_state_a_kernel(
    z_ptr, o_ptr, cs_idx, h0_source, stride_h0, row_ptr, norm_w, v_out, qk_ptr, pf_ptr, ainv_ptr, fac_ptr, ua_ptr,
    eps, szt, szh, sot,
    N: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    PDL_IN: tl.constexpr, TRIGGER: tl.constexpr,
):
    # d8's state kernel on block A, plus u_A = A^-1 rhs_A for block B (same dot level as Pf_AA rhs_A)
    if TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()  # the block-B kernel launches and loads its h0 meanwhile
    i_n, i_hv = tl.program_id(0), tl.program_id(1)
    G: tl.constexpr = HV // H
    i_h = i_hv // G
    slot = tl.load(cs_idx + i_n).to(tl.int64)
    tt = tl.arange(0, 32)
    valid = tt < N
    mv = valid[:, None]
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    toks = (i_n * N + tt).to(tl.int64)
    h_hi, h_mid, h_lo = _hsplit(h0_source, slot, stride_h0, i_hv, o_k, o_v, K, V)
    zz = tl.load(z_ptr + toks[:, None] * szt + i_hv * szh + o_v[None, :], mask=mv, other=0.0).to(tl.float32)
    w = tl.load(norm_w + o_v).to(tl.float32)
    if PDL_IN:
        tl.extra.cuda.gdc_wait()
    if slot < 0:
        return
    row = tl.load(row_ptr + i_n).to(tl.int64)
    qkb = qk_ptr + toks[:, None] * (2 * H * K) + i_h * K + o_k[None, :]
    qb = tl.load(qkb, mask=mv, other=0.0)
    kb = tl.load(qkb + H * K, mask=mv, other=0.0)
    sbase = row * N + tt
    b_vr = tl.load(v_out + (sbase[:, None] * HV + i_hv) * V + o_v[None, :], mask=mv, other=0.0)
    hb = i_n * HV + i_hv
    Pf = tl.load(pf_ptr + hb * 32 * 32 + tt[:, None] * 32 + tt[None, :])
    Ainv = tl.load(ainv_ptr + hb * 32 * 32 + tt[:, None] * 32 + tt[None, :])
    fb = fac_ptr + hb * 4 * 32
    egam = tl.load(fb + tt)
    b_beta = tl.load(fb + 32 + tt)
    rk = tl.load(fb + 64 + tt)
    qs = tl.load(fb + 96 + tt)
    b_A = _hdot(kb, h_hi, h_mid, h_lo) * rk[:, None]
    b_QH = _hdot(qb, h_hi, h_mid, h_lo) * qs[:, None]
    rhs = b_beta[:, None] * (b_vr.to(tl.float32) - egam[:, None] * b_A)
    u = tl.dot(Ainv, rhs, input_precision="tf32x3")
    tl.store(ua_ptr + (hb * 32 + tt[:, None]) * V + o_v[None, :], u)
    b_o = egam[:, None] * b_QH + tl.dot(Pf, rhs, input_precision="tf32x3")
    _gated_norm_store(b_o, zz, w, o_ptr, toks, mv, i_hv, o_v, sot, eps, V)


@triton.jit
def _d3w_leaf_b_kernel(
    c_ptr, a_ptr, b_ptr, z_ptr, o_ptr, cs_idx, h0_source, stride_h0, row_ptr, meta_ptr, hi_ptr, A_log, dt_bias,
    norm_w, kn_out, v_out, g_out, beta_out, ua_ptr, eps, scale, softplus_beta, softplus_threshold, sct, stride_a,
    stride_b, szt, szh, sot,
    N: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    PDL_IN: tl.constexpr, TRIGGER: tl.constexpr,
):
    # block B (rows 32..N-1) after the block-A state kernel: o_B = e^gam Qn_B h0 + W u_A + Pf_BB rhs_B with
    # W = P_BA - Pf_BB X_BA (Pf_BA = W A^-1), Pf_BB = P_BB (leaves) or P_BB (I + X_BB)^-1 (general).
    # Before the wait: h0 tile, gate, gating, ancestor sets (inputs of the layer, complete before the conv).
    if TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    i_n, i_hv = tl.program_id(0), tl.program_id(1)
    G: tl.constexpr = HV // H
    i_h = i_hv // G
    slot = tl.load(cs_idx + i_n).to(tl.int64)
    ca = tl.arange(0, 32)
    cb = tl.arange(0, 16)
    ta = ca
    tb = 32 + cb
    va = ta < N
    vb = tb < N
    mva = va[:, None]
    mvb = vb[:, None]
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    toka = (i_n * N + ta).to(tl.int64)
    tokb = (i_n * N + tb).to(tl.int64)
    h_hi, h_mid, h_lo = _hsplit(h0_source, slot, stride_h0, i_hv, o_k, o_v, K, V)
    zz = tl.load(z_ptr + tokb[:, None] * szt + i_hv * szh + o_v[None, :], mask=mvb, other=0.0).to(tl.float32)
    w = tl.load(norm_w + o_v).to(tl.float32)
    mb = meta_ptr + i_n * 4 * N
    loA = tl.load(mb + 3 * N + ta, mask=va, other=0)
    loB = tl.load(mb + 3 * N + tb, mask=vb, other=0)
    hiB = tl.load(hi_ptr + i_n * N + tb, mask=vb, other=0)
    ancAA = (((loA[:, None] >> ca[None, :]) & 1) != 0) & va[None, :]
    eqAA = ancAA | ((ca[:, None] == ca[None, :]) & va[:, None])
    ancBA = (((loB[:, None] >> ca[None, :]) & 1) != 0) & va[None, :]
    ancBB = (((hiB[:, None] >> cb[None, :]) & 1) != 0) & vb[None, :]
    eyeB = (cb[:, None] == cb[None, :]) & vb[:, None]
    eqBB = ancBB | eyeB
    has_bb = tl.max(tl.sum(ancBB.to(tl.int32), 1), 0) > 0  # some node >= 32 is an ancestor (uniform branch)
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
    gA, _bA = _gating(a_ptr, b_ptr, toka, va, i_hv, stride_a, stride_b, b_A_log, b_dt_bias, softplus_beta,
                      softplus_threshold)
    gB, betaB = _gating(a_ptr, b_ptr, tokb, vb, i_hv, stride_a, stride_b, b_A_log, b_dt_bias, softplus_beta,
                        softplus_threshold)
    gamA = tl.sum(tl.where(eqAA, gA[None, :], 0.0), 1)
    gamB = tl.sum(tl.where(ancBA, gA[None, :], 0.0), 1) + tl.sum(tl.where(eqBB, gB[None, :], 0.0), 1)
    egB = tl.exp(gamB)
    dBA = gamB[:, None] - gamA[None, :]
    dBB = gamB[:, None] - gamB[None, :]
    eBA = tl.where(ancBA, tl.exp(tl.where(ancBA, dBA, 0.0)), 0.0)
    eBB = tl.where(eqBB, tl.exp(tl.where(eqBB, dBB, 0.0)), 0.0)
    if PDL_IN:
        tl.extra.cuda.gdc_wait()  # block-A state done (=> prep and conv done)
    if slot < 0:
        return
    row = tl.load(row_ptr + i_n).to(tl.int64)
    sb = row * N + tb
    tl.store(g_out + sb * HV + i_hv, gB, mask=vb)
    tl.store(beta_out + sb * HV + i_hv, betaB, mask=vb)
    cba = c_ptr + toka[:, None] * sct
    cbb = c_ptr + tokb[:, None] * sct
    kA = tl.load(cba + H * K + i_h * K + o_k[None, :], mask=mva, other=0.0)
    qB = tl.load(cbb + i_h * K + o_k[None, :], mask=mvb, other=0.0)
    kB = tl.load(cbb + H * K + i_h * K + o_k[None, :], mask=mvb, other=0.0)
    vB = tl.load(cbb + 2 * H * K + i_hv * V + o_v[None, :], mask=mvb, other=0.0)
    tl.store(v_out + (sb[:, None] * HV + i_hv) * V + o_v[None, :], vB, mask=mvb)
    kAf = kA.to(tl.float32)
    qBf = qB.to(tl.float32)
    kBf = kB.to(tl.float32)
    rkA = 1.0 / tl.sqrt(tl.sum(kAf * kAf, 1) + 1e-6)
    rqB = 1.0 / tl.sqrt(tl.sum(qBf * qBf, 1) + 1e-6)
    rkB = 1.0 / tl.sqrt(tl.sum(kBf * kBf, 1) + 1e-6)
    if i_hv % G == 0:
        tl.store(kn_out + (sb[:, None] * H + i_h) * K + o_k[None, :], kBf * rkB[:, None], mask=mvb)
    kAt = tl.trans(kA)
    kBt = tl.trans(kB)
    X_BA = betaB[:, None] * eBA * (tl.dot(kB, kAt) * (rkB[:, None] * rkA[None, :]))
    P_BA = eBA * (tl.dot(qB, kAt) * ((rqB * scale)[:, None] * rkA[None, :]))
    P_BB = eBB * (tl.dot(qB, kBt) * ((rqB * scale)[:, None] * rkB[None, :]))
    uA = tl.load(ua_ptr + ((i_n * HV + i_hv) * 32 + ca[:, None]) * V + o_v[None, :])
    aB = _hdot(kB, h_hi, h_mid, h_lo) * rkB[:, None]
    QB = _hdot(qB, h_hi, h_mid, h_lo) * (rqB * scale)[:, None]
    rhsB = betaB[:, None] * (vB.to(tl.float32) - egB[:, None] * aB)
    if has_bb:  # general tree: D^-1 = (I - Y)(I + Y^2)(I + Y^4)(I + Y^8), Y = X_BB (16 x 16, Y^16 = 0)
        Y = tl.where(ancBB, betaB[:, None] * eBB * (tl.dot(kB, kBt) * (rkB[:, None] * rkB[None, :])), 0.0)
        Y2 = tl.dot(Y, Y, input_precision="ieee")
        Q1 = P_BB - tl.dot(P_BB, Y, input_precision="ieee")
        Y4 = tl.dot(Y2, Y2, input_precision="ieee")
        Q2 = Q1 + tl.dot(Q1, Y2, input_precision="ieee")
        Y8 = tl.dot(Y4, Y4, input_precision="ieee")
        Q3 = Q2 + tl.dot(Q2, Y4, input_precision="ieee")
        Pf_BB = Q3 + tl.dot(Q3, Y8, input_precision="ieee")
        Wm = P_BA - tl.dot(Pf_BB, X_BA, input_precision="ieee")
        oB = egB[:, None] * QB + tl.dot(Wm, uA, input_precision="tf32x3") + tl.dot(Pf_BB, rhsB,
                                                                                  input_precision="tf32x3")
    else:  # block B = leaves: D = I, P_BB diagonal (leaf factorization)
        pd = tl.sum(tl.where(eyeB, P_BB, 0.0), 1)
        Wm = P_BA - pd[:, None] * X_BA
        oB = egB[:, None] * QB + tl.dot(Wm, uA, input_precision="tf32x3") + pd[:, None] * rhsB
    _gated_norm_store(oB, zz, w, o_ptr, tokb, mvb, i_hv, o_v, sot, eps, V)


@triton.jit
def _d3w_state_ab_kernel(
    c_ptr, a_ptr, b_ptr, z_ptr, o_ptr, cs_idx, h0_source, stride_h0, row_ptr, meta_ptr, hi_ptr, A_log, dt_bias,
    norm_w, kn_out, v_out, g_out, beta_out, qk_ptr, pf_ptr, ainv_ptr, fac_ptr, eps, scale, softplus_beta,
    softplus_threshold, sct, stride_a, stride_b, szt, szh, sot,
    N: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    PDL_IN: tl.constexpr, TRIGGER: tl.constexpr,
):
    # one program per (request, value head) after prepA: block A (d8 state + u_A = A^-1 rhs_A in registers), then
    # block B (o_B = e^gam Qn_B h0 + W u_A + Pf_BB rhs_B, W and Pf_BB built here); one h0 tile for both blocks.
    if TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    i_n, i_hv = tl.program_id(0), tl.program_id(1)
    G: tl.constexpr = HV // H
    i_h = i_hv // G
    slot = tl.load(cs_idx + i_n).to(tl.int64)
    ca = tl.arange(0, 32)
    cb = tl.arange(0, 16)
    ta = ca
    tb = 32 + cb
    va = ta < N
    vb = tb < N
    mva = va[:, None]
    mvb = vb[:, None]
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    toka = (i_n * N + ta).to(tl.int64)
    tokb = (i_n * N + tb).to(tl.int64)
    h_hi, h_mid, h_lo = _hsplit(h0_source, slot, stride_h0, i_hv, o_k, o_v, K, V)
    zA = tl.load(z_ptr + toka[:, None] * szt + i_hv * szh + o_v[None, :], mask=mva, other=0.0).to(tl.float32)
    w = tl.load(norm_w + o_v).to(tl.float32)
    mb = meta_ptr + i_n * 4 * N
    loA = tl.load(mb + 3 * N + ta, mask=va, other=0)
    loB = tl.load(mb + 3 * N + tb, mask=vb, other=0)
    hiB = tl.load(hi_ptr + i_n * N + tb, mask=vb, other=0)
    ancAA = (((loA[:, None] >> ca[None, :]) & 1) != 0) & va[None, :]
    eqAA = ancAA | ((ca[:, None] == ca[None, :]) & va[:, None])
    ancBA = (((loB[:, None] >> ca[None, :]) & 1) != 0) & va[None, :]
    ancBB = (((hiB[:, None] >> cb[None, :]) & 1) != 0) & vb[None, :]
    eyeB = (cb[:, None] == cb[None, :]) & vb[:, None]
    eqBB = ancBB | eyeB
    has_bb = tl.max(tl.sum(ancBB.to(tl.int32), 1), 0) > 0
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
    gA, _bA = _gating(a_ptr, b_ptr, toka, va, i_hv, stride_a, stride_b, b_A_log, b_dt_bias, softplus_beta,
                      softplus_threshold)
    gB, betaB = _gating(a_ptr, b_ptr, tokb, vb, i_hv, stride_a, stride_b, b_A_log, b_dt_bias, softplus_beta,
                        softplus_threshold)
    gamA = tl.sum(tl.where(eqAA, gA[None, :], 0.0), 1)
    gamB = tl.sum(tl.where(ancBA, gA[None, :], 0.0), 1) + tl.sum(tl.where(eqBB, gB[None, :], 0.0), 1)
    egB = tl.exp(gamB)
    dBA = gamB[:, None] - gamA[None, :]
    dBB = gamB[:, None] - gamB[None, :]
    eBA = tl.where(ancBA, tl.exp(tl.where(ancBA, dBA, 0.0)), 0.0)
    eBB = tl.where(eqBB, tl.exp(tl.where(eqBB, dBB, 0.0)), 0.0)
    if PDL_IN:
        tl.extra.cuda.gdc_wait()
    if slot < 0:
        return
    row = tl.load(row_ptr + i_n).to(tl.int64)
    hb = i_n * HV + i_hv
    # block A
    qka = qk_ptr + toka[:, None] * (2 * H * K) + i_h * K + o_k[None, :]
    qA = tl.load(qka, mask=mva, other=0.0)
    kA = tl.load(qka + H * K, mask=mva, other=0.0)
    sa = row * N + ta
    vA = tl.load(v_out + (sa[:, None] * HV + i_hv) * V + o_v[None, :], mask=mva, other=0.0)
    Pf = tl.load(pf_ptr + hb * 32 * 32 + ca[:, None] * 32 + ca[None, :])
    Ainv = tl.load(ainv_ptr + hb * 32 * 32 + ca[:, None] * 32 + ca[None, :])
    fb = fac_ptr + hb * 4 * 32
    egA = tl.load(fb + ta)
    btA = tl.load(fb + 32 + ta)
    rkA = tl.load(fb + 64 + ta)
    qsA = tl.load(fb + 96 + ta)
    aA = _hdot(kA, h_hi, h_mid, h_lo) * rkA[:, None]
    QA = _hdot(qA, h_hi, h_mid, h_lo) * qsA[:, None]
    rhsA = btA[:, None] * (vA.to(tl.float32) - egA[:, None] * aA)
    uA = tl.dot(Ainv, rhsA, input_precision="tf32x3")
    oA = egA[:, None] * QA + tl.dot(Pf, rhsA, input_precision="tf32x3")
    _gated_norm_store(oA, zA, w, o_ptr, toka, mva, i_hv, o_v, sot, eps, V)
    # block B
    zB = tl.load(z_ptr + tokb[:, None] * szt + i_hv * szh + o_v[None, :], mask=mvb, other=0.0).to(tl.float32)
    sb = row * N + tb
    tl.store(g_out + sb * HV + i_hv, gB, mask=vb)
    tl.store(beta_out + sb * HV + i_hv, betaB, mask=vb)
    cbb = c_ptr + tokb[:, None] * sct
    qB = tl.load(cbb + i_h * K + o_k[None, :], mask=mvb, other=0.0)
    kB = tl.load(cbb + H * K + i_h * K + o_k[None, :], mask=mvb, other=0.0)
    vB = tl.load(cbb + 2 * H * K + i_hv * V + o_v[None, :], mask=mvb, other=0.0)
    tl.store(v_out + (sb[:, None] * HV + i_hv) * V + o_v[None, :], vB, mask=mvb)
    qBf = qB.to(tl.float32)
    kBf = kB.to(tl.float32)
    rqB = 1.0 / tl.sqrt(tl.sum(qBf * qBf, 1) + 1e-6)
    rkB = 1.0 / tl.sqrt(tl.sum(kBf * kBf, 1) + 1e-6)
    if i_hv % G == 0:
        tl.store(kn_out + (sb[:, None] * H + i_h) * K + o_k[None, :], kBf * rkB[:, None], mask=mvb)
    kAt = tl.trans(kA)
    kBt = tl.trans(kB)
    X_BA = betaB[:, None] * eBA * (tl.dot(kB, kAt) * (rkB[:, None] * rkA[None, :]))
    P_BA = eBA * (tl.dot(qB, kAt) * ((rqB * scale)[:, None] * rkA[None, :]))
    P_BB = eBB * (tl.dot(qB, kBt) * ((rqB * scale)[:, None] * rkB[None, :]))
    aB = _hdot(kB, h_hi, h_mid, h_lo) * rkB[:, None]
    QB = _hdot(qB, h_hi, h_mid, h_lo) * (rqB * scale)[:, None]
    rhsB = betaB[:, None] * (vB.to(tl.float32) - egB[:, None] * aB)
    if has_bb:
        Y = tl.where(ancBB, betaB[:, None] * eBB * (tl.dot(kB, kBt) * (rkB[:, None] * rkB[None, :])), 0.0)
        Y2 = tl.dot(Y, Y, input_precision="ieee")
        Q1 = P_BB - tl.dot(P_BB, Y, input_precision="ieee")
        Y4 = tl.dot(Y2, Y2, input_precision="ieee")
        Q2 = Q1 + tl.dot(Q1, Y2, input_precision="ieee")
        Y8 = tl.dot(Y4, Y4, input_precision="ieee")
        Q3 = Q2 + tl.dot(Q2, Y4, input_precision="ieee")
        Pf_BB = Q3 + tl.dot(Q3, Y8, input_precision="ieee")
        Wm = P_BA - tl.dot(Pf_BB, X_BA, input_precision="ieee")
        oB = egB[:, None] * QB + tl.dot(Wm, uA, input_precision="tf32x3") + tl.dot(Pf_BB, rhsB,
                                                                                  input_precision="tf32x3")
    else:
        pd = tl.sum(tl.where(eyeB, P_BB, 0.0), 1)
        Wm = P_BA - pd[:, None] * X_BA
        oB = egB[:, None] * QB + tl.dot(Wm, uA, input_precision="tf32x3") + pd[:, None] * rhsB
    _gated_norm_store(oB, zB, w, o_ptr, tokb, mvb, i_hv, o_v, sot, eps, V)


_SCRATCH: dict = {}


def _scratch(name, shape, dtype, device):
    """Persistent per-(name, shape) buffers: a captured graph keeps pointers into them (never freed/reallocated)."""
    key = (name, tuple(int(s) for s in shape), dtype, str(device))
    t = _SCRATCH.get(key)
    if t is None:
        t = torch.empty(key[1], dtype=dtype, device=device)
        _SCRATCH[key] = t
    return t


def meta_buffers(bs, n_nodes, device):
    N = int(n_nodes)
    return _scratch("meta", (bs, 4, N), torch.int32, device), _scratch("hi", (bs, N), torch.int32, device)


def tree_meta_wide(retrieve_next_token, retrieve_next_sibling, retrieve_parent_token, n_nodes, bs,
                   parents_given=False):
    """Writes the parents (unless given), the [bs, 4, N] meta (d8 layout, low ancestor word) and the [bs, N] high
    ancestor word; returns (meta, hi)."""
    N = int(n_nodes)
    assert N <= 64
    meta, hi = meta_buffers(bs, N, retrieve_parent_token.device)
    _d3w_tree_meta_kernel[(bs,)](
        retrieve_next_token, retrieve_next_sibling, retrieve_parent_token, meta, hi,
        retrieve_next_token.stride(0), retrieve_next_sibling.stride(0), retrieve_parent_token.stride(0),
        N=N, NP=max(16, triton.next_power_of_2(N)), PARENTS_GIVEN=parents_given, num_warps=_D3W_META_WARPS,
    )
    return meta, hi


def tree_meta_into(meta, retrieve_next_token, retrieve_next_sibling, retrieve_parent_token, n_nodes, bs,
                   parents_given=False):
    """N <= 32: writes the parents (unless given) and d8's [bs, 4, N] meta into ``meta`` (d8's own scratch buffer;
    the low ancestor word is the whole set, the high word goes to a private scratch)."""
    N = int(n_nodes)
    assert N <= 32 and tuple(meta.shape) == (bs, 4, N) and meta.dtype == torch.int32
    hi = _scratch("hi", (bs, N), torch.int32, meta.device)
    _d3w_tree_meta_kernel[(bs,)](
        retrieve_next_token, retrieve_next_sibling, retrieve_parent_token, meta, hi,
        retrieve_next_token.stride(0), retrieve_next_sibling.stride(0), retrieve_parent_token.stride(0),
        N=N, NP=max(16, triton.next_power_of_2(N)), PARENTS_GIVEN=parents_given, num_warps=_D3W_META_WARPS,
    )
    return meta


def d3w_tree_accept_commit(*, tokens, parents, pred, prefix_lens, req_pool_indices, req_to_token, out, permute=True):
    """Drop-in for specpilot_d7_accept.d7_tree_accept_commit for n <= 63 (same buffers, same outputs)."""
    bs, n = tokens.shape
    assert req_to_token.dtype == torch.int32
    assert n <= 63, n  # int64 ancestor bitmask
    _d3w_tree_accept_kernel[(bs,)](
        tokens, parents, pred, prefix_lens, req_pool_indices, req_to_token, req_to_token.stride(0),
        out["accept_len"], out["commit_lens"], out["bonus"], out["out_tokens"], out["new_seq_lens"], out["leaf"],
        out["path"], out["slots"], N=n, NP=triton.next_power_of_2(n), LOGN=max(1, (n - 1).bit_length()),
        PERMUTE=permute, num_warps=_D3W_ACC_WARPS)


def gdn_layer_verify_wide(
    *, x, a, b, z, conv_weight, conv_bias, conv_state, ssm_states, cache_indices, xin, scratch_rows, meta, hi,
    A_log, dt_bias, norm_weight, norm_eps, kn_buf, v_buf, g_buf, beta_buf, H, HV, K, V, n_nodes, silu=True,
    pdl=None, conv_bn=None, conv_warps=None, prep_warps=None, state_warps=None, trigger=None,
):
    """Same contract as specpilot_d8_gdn.gdn_layer_verify for 33 <= N <= 48 (meta, hi from ``tree_meta_wide``)."""
    from sglang.srt.speculative import specpilot_d8_gdn as D8

    pdl = D8._D8_PDL if pdl is None else pdl
    trigger = D8._D8_TRIGGER if trigger is None else trigger
    conv_bn = conv_bn or D8._D8_CONV_BN
    conv_warps = conv_warps or D8._D8_CONV_WARPS
    prep_warps = prep_warps or _D3W_PREP_WARPS
    state_warps = state_warps or _D3W_STATE_WARPS
    T = x.shape[0]
    N = int(n_nodes)
    bs = T // N
    NP = triton.next_power_of_2(N)
    dim = 2 * H * K + HV * V
    assert supports(N) and T == bs * N
    assert x.stride(-1) == 1 and conv_weight.shape == (dim, 4) and conv_state.shape[-1] == 3
    assert xin.shape[1] == N and xin.shape[2] == dim and xin.stride(2) == 1
    assert z.stride(-1) == 1 and ssm_states.stride(-1) == 1 and ssm_states.stride(-2) == K
    assert ssm_states.stride(-3) == K * V
    dev = x.device
    conv_out = _scratch("conv", (T, dim), x.dtype, dev)
    qk = _scratch("qk", (T, 2 * H * K), x.dtype, dev)
    pf = _scratch("pf", (bs, HV, 32, 32), torch.float32, dev)
    ainv = _scratch("ainv", (bs, HV, 32, 32), torch.float32, dev)
    fac = _scratch("fac", (bs, HV, 4, 32), torch.float32, dev)
    ua = _scratch("ua", (bs, HV, 32, V), torch.float32, dev)
    out = x.new_empty(T, HV * V)
    bias = conv_bias if conv_bias is not None else conv_weight
    D8._d8_tree_conv_kernel[(bs, triton.cdiv(dim, conv_bn))](
        x, conv_weight, bias, conv_state, cache_indices, xin, scratch_rows, meta, conv_out, dim,
        N * x.stride(0), x.stride(0), conv_weight.stride(0), conv_weight.stride(1),
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2), xin.stride(0), xin.stride(1),
        N * conv_out.stride(0), conv_out.stride(0),
        N=N, NP=NP, BLOCK_N=conv_bn, HAS_BIAS=conv_bias is not None, SILU=silu, PDL=pdl, num_warps=conv_warps,
    )
    kw = {"launch_pdl": True} if pdl else {}
    _d3w_prep_a_kernel[(bs, HV)](
        conv_out, a, b, cache_indices, scratch_rows, meta, A_log, dt_bias, kn_buf, v_buf, g_buf, beta_buf, qk, pf,
        ainv, fac, K ** -0.5, 1.0, 20.0, conv_out.stride(0), a.stride(-2), b.stride(-2),
        N=N, H=H, HV=HV, K=K, V=V, PDL_IN=pdl, PDL_OUT=pdl, TWO_CHAIN=_D3W_TWO_CHAIN, num_warps=prep_warps,
        num_stages=1, **kw,
    )
    if _D3W_FUSED_B:
        _d3w_state_ab_kernel[(bs, HV)](
            conv_out, a, b, z, out, cache_indices, ssm_states, ssm_states.stride(0), scratch_rows, meta, hi, A_log,
            dt_bias, norm_weight, kn_buf, v_buf, g_buf, beta_buf, qk, pf, ainv, fac, float(norm_eps), K ** -0.5, 1.0,
            20.0, conv_out.stride(0), a.stride(-2), b.stride(-2), z.stride(0), z.stride(1), out.stride(0),
            N=N, H=H, HV=HV, K=K, V=V, PDL_IN=pdl, TRIGGER=trigger, num_warps=state_warps, num_stages=1, **kw,
        )
        return out
    _d3w_state_a_kernel[(bs, HV)](
        z, out, cache_indices, ssm_states, ssm_states.stride(0), scratch_rows, norm_weight, v_buf, qk, pf, ainv, fac,
        ua, float(norm_eps), z.stride(0), z.stride(1), out.stride(0),
        N=N, H=H, HV=HV, K=K, V=V, PDL_IN=pdl, TRIGGER=pdl, num_warps=state_warps, num_stages=1, **kw,
    )
    _d3w_leaf_b_kernel[(bs, HV)](
        conv_out, a, b, z, out, cache_indices, ssm_states, ssm_states.stride(0), scratch_rows, meta, hi, A_log, dt_bias,
        norm_weight, kn_buf, v_buf, g_buf, beta_buf, ua, float(norm_eps), K ** -0.5, 1.0, 20.0, conv_out.stride(0),
        a.stride(-2), b.stride(-2), z.stride(0), z.stride(1), out.stride(0),
        N=N, H=H, HV=HV, K=K, V=V, PDL_IN=pdl, TRIGGER=trigger, num_warps=_D3W_LEAF_WARPS, num_stages=1, **kw,
    )
    return out
