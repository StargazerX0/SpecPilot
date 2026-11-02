"""SPECPILOT_D8_GDN=1: batch-1 GDN tree-verify layer as a 3-kernel PDL chain (conv -> prep -> state).

The parent runs, per GDN layer of a tree verify, conv_tm -> whole-tree WY verify -> gated RMSNorm: three
latency-bound launches; the WY program (255 regs + spills) holds the fp32 state tile while it walks the parent
pointers (N-1 hops) and runs the dependent nilpotent solve. Here:

  * ``tree_meta`` (first GDN layer of a forward): parents rebuilt from first-child / next-sibling exactly as the conv
    kernel does (stored to ``retrieve_parent_token``), the three conv ancestors of every node and the strict-ancestor
    bitmask (N <= 32), once per forward instead of per program per layer;
  * ``_d8_tree_conv_kernel``: ``_tree_conv_tm_kernel`` arithmetic (bitwise equal output and node-input store), conv
    ancestors from the meta, early PDL trigger;
  * ``_d8_prep_kernel`` per (request, value head): gating, q/k L2 norms, commit inputs (bitwise equal to the WY
    kernel's), and the V-independent solve folded into the output operand
    ``Pf = P (I - X)(I + X^2)(I + X^4)(I + X^8) [(I + X^16) only if the tree depth >= 16]`` with exact fp32 (FMA) dots:
    dependent depth 5 instead of 8 tf32x3 dots; then ``o_tree = Pf rhs``;
  * ``_d8_state_kernel`` per (request, value head), PDL-launched: loads and 3-term-splits the fp32 state tile and the
    gate while the prep runs, then ``o = e^gam (Qn h0) + Pf beta (v - e^gam Kn h0)`` and the gated RMSNorm
    (``norm_before_gate``, silu/swish gate) of the bf16-rounded output, in registers.
The replay commits (GDN state and conv) read the same commit inputs and are unchanged. Outputs equal the parent's up to
reassociation of the solve (within 1 bf16 ulp in the tests). Knobs read once at import; unset = parent path.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_D8_GDN = os.environ.get("SPECPILOT_D8_GDN", "0") not in ("", "0")
_D8_CONV_BN = int(os.environ.get("SPECPILOT_D8_CONV_BN", "64") or 64)
_D8_CONV_WARPS = int(os.environ.get("SPECPILOT_D8_CONV_WARPS", "4") or 4)
_D8_PREP_WARPS = int(os.environ.get("SPECPILOT_D8_PREP_WARPS", "8") or 8)
_D8_PDL = os.environ.get("SPECPILOT_D8_PDL", "1") not in ("", "0")
# SPECPILOT_D8_TRIGGER=1: the state kernel triggers its dependents at its start (a PDL-launched out_proj GEMM may
# pre-stage weights meanwhile; it still waits for the state kernel to complete before reading the output)
_D8_TRIGGER = os.environ.get("SPECPILOT_D8_TRIGGER", "0") not in ("", "0")


def enabled() -> bool:
    return _D8_GDN


@triton.jit
def _d8_tree_meta_kernel(
    nt_ptr,  # [bs, N] first child
    ns_ptr,  # [bs, N] next sibling
    par_ptr,  # [bs, N] parent (out; root = 0 as the conv kernel writes it)
    meta_ptr,  # [bs, 4, N] int32 (out): a1, a2, a3 (conv ancestors; <= 0 -> state column), strict-ancestor bits
    snt,
    sns,
    spr,
    N: tl.constexpr,
    NP: tl.constexpr,
    PARENTS_GIVEN: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    tt = tl.arange(0, NP)
    vt = tt < N
    if PARENTS_GIVEN:
        par = tl.load(par_ptr + b * spr + tt, mask=vt, other=0).to(tl.int32)
    else:  # the conv kernel's rebuild (first-child / next-sibling walk)
        nt = tl.load(nt_ptr + b * snt + tt, mask=vt, other=-1).to(tl.int32)
        ns = tl.load(ns_ptr + b * sns + tt, mask=vt, other=-1).to(tl.int32)
        par = tl.zeros([NP], dtype=tl.int32)
        for i in tl.static_range(N):
            nti = tl.sum(tl.where(tt == i, nt, 0), 0)
            par = tl.where((tt == nti) & (nti != -1), i, par)
            nsi = tl.sum(tl.where(tt == i, ns, 0), 0)
            pi = tl.sum(tl.where(tt == i, par, 0), 0)
            par = tl.where((tt == nsi) & (nsi != -1), pi, par)
        tl.store(par_ptr + b * spr + tt, par.to(par_ptr.dtype.element_ty), mask=vt)
    pm = tt[None, :]
    a1 = tl.where(tt > 0, par, tt - 1)
    p_a1 = tl.sum(tl.where(pm == a1[:, None], par[None, :], 0), 1)
    a2 = tl.where(a1 > 0, p_a1, a1 - 1)
    p_a2 = tl.sum(tl.where(pm == a2[:, None], par[None, :], 0), 1)
    a3 = tl.where(a2 > 0, p_a2, a2 - 1)
    # strict ancestors by pointer jumping: bits_t |= bits_jump(t), jump_t = jump_jump(t) (5 rounds cover 31 hops)
    wpar = tl.where((tt > 0) & vt, par, -1)
    bits = tl.where(wpar >= 0, (1 << tl.maximum(wpar, 0)).to(tl.int32), 0)
    jmp = wpar
    for _ in tl.static_range(5):
        sel = pm == jmp[:, None]
        bits = bits | tl.sum(tl.where(sel, bits[None, :], 0), 1)
        jmp = tl.where(jmp >= 0, tl.sum(tl.where(sel, jmp[None, :], 0), 1), -1)
        jmp = tl.where(tl.sum(sel.to(tl.int32), 1) > 0, jmp, -1)
    base = meta_ptr + b * 4 * N
    tl.store(base + tt, a1, mask=vt)
    tl.store(base + N + tt, a2, mask=vt)
    tl.store(base + 2 * N + tt, a3, mask=vt)
    tl.store(base + 3 * N + tt, bits, mask=vt)


@triton.jit
def _d8_tree_conv_kernel(
    x_ptr, w_ptr, bias_ptr, cs_ptr, cs_idx_ptr, xin_ptr, xin_idx_ptr, meta_ptr, o_ptr, dim,
    sxb, sxt, swd, sww, scb, scd, scw, sib, sit, sob, sot,
    N: tl.constexpr, NP: tl.constexpr, BLOCK_N: tl.constexpr, HAS_BIAS: tl.constexpr, SILU: tl.constexpr,
    PDL: tl.constexpr,
):
    # _tree_conv_tm_kernel (specpilot_db_conv) with the conv ancestors from the tree meta (same values, same
    # accumulation order and dtypes: bitwise equal) and an early PDL trigger so the prep kernel launches meanwhile.
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    b = tl.program_id(0).to(tl.int64)
    fb = tl.program_id(1)
    slot = tl.load(cs_idx_ptr + b).to(tl.int64)
    if slot < 0:
        return
    feats = fb * BLOCK_N + tl.arange(0, BLOCK_N)
    mf = feats < dim
    tt = tl.arange(0, NP)
    vt = tt < N
    mb = meta_ptr + b * 4 * N
    a1 = tl.load(mb + tt, mask=vt, other=0)
    a2 = tl.load(mb + N + tt, mask=vt, other=0)
    a3 = tl.load(mb + 2 * N + tt, mask=vt, other=0)
    cs = cs_ptr + slot * scb + feats * scd
    col0 = tl.load(cs + 0 * scw, mask=mf, other=0.0)
    col1 = tl.load(cs + 1 * scw, mask=mf, other=0.0)
    col2 = tl.load(cs + 2 * scw, mask=mf, other=0.0)
    wb = w_ptr + feats * swd
    w0 = tl.load(wb + 0 * sww, mask=mf, other=0.0)
    w1 = tl.load(wb + 1 * sww, mask=mf, other=0.0)
    w2 = tl.load(wb + 2 * sww, mask=mf, other=0.0)
    w3 = tl.load(wb + 3 * sww, mask=mf, other=0.0)
    xb = x_ptr + b * sxb + feats[None, :]
    m2 = vt[:, None] & mf[None, :]
    x0 = tl.load(xb + tt[:, None] * sxt, mask=m2, other=0.0)
    x1 = tl.load(xb + tl.maximum(a1, 0)[:, None] * sxt, mask=m2 & (a1 >= 0)[:, None], other=0.0)
    x1 = tl.where((a1 == -1)[:, None], col2[None, :], x1)
    x2 = tl.load(xb + tl.maximum(a2, 0)[:, None] * sxt, mask=m2 & (a2 >= 0)[:, None], other=0.0)
    x2 = tl.where((a2 == -1)[:, None], col2[None, :], tl.where((a2 == -2)[:, None], col1[None, :], x2))
    x3 = tl.load(xb + tl.maximum(a3, 0)[:, None] * sxt, mask=m2 & (a3 >= 0)[:, None], other=0.0)
    x3 = tl.where(
        (a3 == -1)[:, None],
        col2[None, :],
        tl.where((a3 == -2)[:, None], col1[None, :], tl.where((a3 == -3)[:, None], col0[None, :], x3)),
    )
    if HAS_BIAS:
        acc = tl.load(bias_ptr + feats, mask=mf, other=0.0).to(tl.float32)[None, :] + tl.zeros([NP, BLOCK_N], tl.float32)
    else:
        acc = tl.zeros([NP, BLOCK_N], dtype=tl.float32)
    acc += x0 * w3[None, :]
    acc += x1 * w2[None, :]
    acc += x2 * w1[None, :]
    acc += x3 * w0[None, :]
    if SILU:
        acc = acc / (1 + tl.exp(-acc))
    tl.store(o_ptr + b * sob + feats[None, :] + tt[:, None] * sot, acc, mask=m2)
    row = tl.load(xin_idx_ptr + b).to(tl.int64)
    tl.store(xin_ptr + row * sib + tt[:, None] * sit + feats[None, :], x0, mask=m2)


@triton.jit
def _d8_prep_kernel(
    c_ptr,  # conv output [T, 2HK + HV V] (q | k | v)
    a_ptr, b_ptr, cs_idx, row_ptr, meta_ptr, A_log, dt_bias, kn_out, v_out, g_out, beta_out, qk_ptr, pf_ptr, fac_ptr,
    scale, softplus_beta, softplus_threshold, sct, stride_a, stride_b,
    N: tl.constexpr, NP: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    PDL_IN: tl.constexpr, PDL_OUT: tl.constexpr,
):
    if PDL_OUT:
        tl.extra.cuda.gdc_launch_dependents()  # the state kernel may start loading h0 now
    i_n, i_hv = tl.program_id(0), tl.program_id(1)
    G: tl.constexpr = HV // H
    i_h = i_hv // G
    slot = tl.load(cs_idx + i_n).to(tl.int64)
    if slot < 0:
        return
    tt = tl.arange(0, NP)
    valid = tt < N
    mv = valid[:, None]
    mb = meta_ptr + i_n * 4 * N
    bits = tl.load(mb + 3 * N + tt, mask=valid, other=0)
    anc = (((bits[:, None] >> tt[None, :]) & 1) != 0) & valid[None, :]
    eye = tt[:, None] == tt[None, :]
    anc_eq = anc | (eye & valid[:, None])
    maxd = tl.max(tl.sum(anc.to(tl.int32), 1), 0)  # tree depth (uniform branch below)
    toks = (i_n * N + tt).to(tl.int64)
    b_b = tl.load(b_ptr + toks * stride_b + i_hv, mask=valid, other=0).to(tl.float32)
    b_a = tl.load(a_ptr + toks * stride_a + i_hv, mask=valid, other=0).to(tl.float32)
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
    row = tl.load(row_ptr + i_n).to(tl.int64)
    o_k = tl.arange(0, K)
    o_vv = tl.arange(0, V)
    if PDL_IN:
        tl.extra.cuda.gdc_wait()  # conv output ready
    cb = c_ptr + toks[:, None] * sct
    qb = tl.load(cb + i_h * K + o_k[None, :], mask=mv, other=0.0)
    kb = tl.load(cb + H * K + i_h * K + o_k[None, :], mask=mv, other=0.0)
    vb = tl.load(cb + 2 * H * K + i_hv * V + o_vv[None, :], mask=mv, other=0.0)
    sbase = row * N + tt
    tl.store(v_out + (sbase[:, None] * HV + i_hv) * V + o_vv[None, :], vb, mask=mv)
    # gating and norms: _gdn_tree_wy_kernel arithmetic (commit inputs bitwise equal)
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
    qf = qb.to(tl.float32)
    kf = kb.to(tl.float32)
    rq = 1.0 / tl.sqrt(tl.sum(qf * qf, 1) + 1e-6)
    rk = 1.0 / tl.sqrt(tl.sum(kf * kf, 1) + 1e-6)
    tl.store(g_out + sbase * HV + i_hv, b_g, mask=valid)
    tl.store(beta_out + sbase * HV + i_hv, b_beta, mask=valid)
    if i_hv % G == 0:
        tl.store(kn_out + (sbase[:, None] * H + i_h) * K + o_k[None, :], kf * rk[:, None], mask=mv)
    gam = tl.sum(tl.where(anc_eq, b_g[None, :], 0.0), 1)
    kbt = tl.trans(kb)
    b_KK = tl.dot(kb, kbt) * (rk[:, None] * rk[None, :])
    b_QK = tl.dot(qb, kbt) * ((rq * scale)[:, None] * rk[None, :])
    dgam = gam[:, None] - gam[None, :]
    X = tl.where(anc, b_beta[:, None] * tl.exp(tl.where(anc, dgam, 0.0)) * b_KK, 0.0)
    P = tl.where(anc_eq, tl.exp(tl.where(anc_eq, dgam, 0.0)) * b_QK, 0.0)
    # Pf = P (I + X)^-1 = P (I - X)(I + X^2)(I + X^4)(I + X^8)[(I + X^16)]: X strictly lower (nilpotent, depth < 32)
    X2 = tl.dot(X, X, input_precision="ieee")
    P1 = P - tl.dot(P, X, input_precision="ieee")
    X4 = tl.dot(X2, X2, input_precision="ieee")
    P2 = P1 + tl.dot(P1, X2, input_precision="ieee")
    X8 = tl.dot(X4, X4, input_precision="ieee")
    P3 = P2 + tl.dot(P2, X4, input_precision="ieee")
    Pf = P3 + tl.dot(P3, X8, input_precision="ieee")
    if maxd >= 16:  # else X^16 == 0 exactly
        X16 = tl.dot(X8, X8, input_precision="ieee")
        Pf = Pf + tl.dot(Pf, X16, input_precision="ieee")
    hb = i_n * HV + i_hv
    tl.store(pf_ptr + hb * NP * NP + tt[:, None] * NP + tt[None, :], Pf)
    fb = fac_ptr + hb * 4 * NP
    tl.store(fb + tt, tl.exp(gam))
    tl.store(fb + NP + tt, b_beta)
    tl.store(fb + 2 * NP + tt, rk)
    tl.store(fb + 3 * NP + tt, rq * scale)
    if i_hv % G == 0:
        qkb = qk_ptr + toks[:, None] * (2 * H * K) + i_h * K + o_k[None, :]
        tl.store(qkb, qb, mask=mv)
        tl.store(qkb + H * K, kb, mask=mv)


@triton.jit
def _d8_state_kernel(
    z_ptr, o_ptr, cs_idx, h0_source, stride_h0, row_ptr, norm_w, v_out, qk_ptr, pf_ptr, fac_ptr, eps, szt, szh, sot,
    N: tl.constexpr, NP: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    PDL_IN: tl.constexpr, TRIGGER: tl.constexpr = False,
):
    if TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    # one program per (request, value head), full V row: state tile + gate loaded before the dependency wait
    # (overlapping the prep kernel), then o = e^gam (Qn h0) + Pf beta (v - e^gam Kn h0) and the gated RMSNorm.
    i_n, i_hv = tl.program_id(0), tl.program_id(1)
    G: tl.constexpr = HV // H
    i_h = i_hv // G
    slot = tl.load(cs_idx + i_n).to(tl.int64)
    tt = tl.arange(0, NP)
    valid = tt < N
    mv = valid[:, None]
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    toks = (i_n * N + tt).to(tl.int64)
    b_h = tl.load(h0_source + tl.maximum(slot, 0) * stride_h0 + i_hv * K * V + o_v[None, :] * K + o_k[:, None])
    zz = tl.load(z_ptr + toks[:, None] * szt + i_hv * szh + o_v[None, :], mask=mv, other=0.0).to(tl.float32)
    w = tl.load(norm_w + o_v).to(tl.float32)
    h_hi = b_h.to(tl.bfloat16)
    r1 = b_h - h_hi.to(tl.float32)
    h_mid = r1.to(tl.bfloat16)
    h_lo = (r1 - h_mid.to(tl.float32)).to(tl.bfloat16)
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
    Pf = tl.load(pf_ptr + hb * NP * NP + tt[:, None] * NP + tt[None, :])
    fb = fac_ptr + hb * 4 * NP
    egam = tl.load(fb + tt)
    b_beta = tl.load(fb + NP + tt)
    rk = tl.load(fb + 2 * NP + tt)
    qs = tl.load(fb + 3 * NP + tt)
    b_A = tl.dot(kb, h_hi) + tl.dot(kb, h_mid) + tl.dot(kb, h_lo)
    b_QH = tl.dot(qb, h_hi) + tl.dot(qb, h_mid) + tl.dot(qb, h_lo)
    b_A = b_A * rk[:, None]
    b_QH = b_QH * qs[:, None]
    rhs = b_beta[:, None] * (b_vr.to(tl.float32) - egam[:, None] * b_A)
    b_o = egam[:, None] * b_QH + tl.dot(Pf, rhs, input_precision="tf32x3")
    # gated RMSNorm (layernorm_gated 1-pass arithmetic) of the bf16-rounded output
    xo = tl.where(mv, b_o.to(o_ptr.dtype.element_ty).to(tl.float32), 0.0)
    var = tl.sum(xo * xo, axis=1) / V
    rstd = tl.rsqrt(var + eps)
    y = xo * rstd[:, None] * w[None, :]
    y *= zz * tl.sigmoid(zz)
    tl.store(o_ptr + toks[:, None] * sot + i_hv * V + o_v[None, :], y.to(o_ptr.dtype.element_ty), mask=mv)


_SCRATCH: dict = {}


def _scratch(name, shape, dtype, device):
    """Persistent per-(name, shape) buffers: a captured graph keeps pointers into them, so they are never freed or
    reallocated (every graph batch size gets its own)."""
    key = (name, tuple(int(s) for s in shape), dtype, str(device))
    t = _SCRATCH.get(key)
    if t is None:
        t = torch.empty(key[1], dtype=dtype, device=device)
        _SCRATCH[key] = t
    return t


def tree_meta(retrieve_next_token, retrieve_next_sibling, retrieve_parent_token, n_nodes, bs, parents_given=False):
    """Writes the parents (unless given) and the [bs, 4, N] tree meta; returns the meta."""
    N = int(n_nodes)
    assert N <= 32
    meta = _scratch("meta", (bs, 4, N), torch.int32, retrieve_parent_token.device)
    _d8_tree_meta_kernel[(bs,)](
        retrieve_next_token, retrieve_next_sibling, retrieve_parent_token, meta,
        retrieve_next_token.stride(0), retrieve_next_sibling.stride(0), retrieve_parent_token.stride(0),
        N=N, NP=max(16, triton.next_power_of_2(N)), PARENTS_GIVEN=parents_given, num_warps=1,
    )
    return meta


def gdn_layer_verify(
    *, x, a, b, z, conv_weight, conv_bias, conv_state, ssm_states, cache_indices, xin, scratch_rows, meta,
    A_log, dt_bias, norm_weight, norm_eps, kn_buf, v_buf, g_buf, beta_buf, H, HV, K, V, n_nodes, silu=True,
    pdl=None, conv_bn=None, conv_warps=None, prep_warps=None,
):
    """x [T, >= 2HK + HV V] pre-conv node inputs (q | k | v, unit channel stride, any row stride); a, b [T, HV] and
    z [T, HV, V] (views); conv_state [slots, dim, 3]; ssm_states [slots, HV, V, K] fp32 (read only);
    xin [rows, N, dim] time-major node inputs (written); meta from ``tree_meta``; commit bufs [rows, N, ...] of this
    layer (written). Returns the gated-normed output [T, HV * V]."""
    pdl = _D8_PDL if pdl is None else pdl
    conv_bn = conv_bn or _D8_CONV_BN
    conv_warps = conv_warps or _D8_CONV_WARPS
    prep_warps = prep_warps or _D8_PREP_WARPS
    T = x.shape[0]
    N = int(n_nodes)
    bs = T // N
    NP = max(16, triton.next_power_of_2(N))
    dim = 2 * H * K + HV * V
    assert N <= 32 and x.stride(-1) == 1 and conv_weight.shape == (dim, 4) and conv_state.shape[-1] == 3
    assert xin.shape[1] == N and xin.shape[2] == dim and xin.stride(2) == 1
    assert z.stride(-1) == 1 and ssm_states.stride(-1) == 1 and ssm_states.stride(-2) == K
    assert ssm_states.stride(-3) == K * V
    dev = x.device
    conv_out = _scratch("conv", (T, dim), x.dtype, dev)
    qk = _scratch("qk", (T, 2 * H * K), x.dtype, dev)
    pf = _scratch("pf", (bs, HV, NP, NP), torch.float32, dev)
    fac = _scratch("fac", (bs, HV, 4, NP), torch.float32, dev)
    out = x.new_empty(T, HV * V)
    bias = conv_bias if conv_bias is not None else conv_weight
    _d8_tree_conv_kernel[(bs, triton.cdiv(dim, conv_bn))](
        x, conv_weight, bias, conv_state, cache_indices, xin, scratch_rows, meta, conv_out, dim,
        N * x.stride(0), x.stride(0), conv_weight.stride(0), conv_weight.stride(1),
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2), xin.stride(0), xin.stride(1),
        N * conv_out.stride(0), conv_out.stride(0),
        N=N, NP=NP, BLOCK_N=conv_bn, HAS_BIAS=conv_bias is not None, SILU=silu, PDL=pdl, num_warps=conv_warps,
    )
    kw = {"launch_pdl": True} if pdl else {}
    _d8_prep_kernel[(bs, HV)](
        conv_out, a, b, cache_indices, scratch_rows, meta, A_log, dt_bias, kn_buf, v_buf, g_buf, beta_buf, qk, pf,
        fac, K ** -0.5, 1.0, 20.0, conv_out.stride(0), a.stride(-2), b.stride(-2),
        N=N, NP=NP, H=H, HV=HV, K=K, V=V, PDL_IN=pdl, PDL_OUT=pdl, num_warps=prep_warps, num_stages=1, **kw,
    )
    _d8_state_kernel[(bs, HV)](
        z, out, cache_indices, ssm_states, ssm_states.stride(0), scratch_rows, norm_weight, v_buf, qk, pf, fac,
        float(norm_eps), z.stride(0), z.stride(1), out.stride(0),
        N=N, NP=NP, H=H, HV=HV, K=K, V=V, PDL_IN=pdl, TRIGGER=_D8_TRIGGER, num_warps=8, num_stages=1, **kw,
    )
    return out


def verify_layer(gdn, forward_batch, mixed_qkv, z, a, b):
    """SPECPILOT_D8_GDN hook for Qwen3_5GatedDeltaNet.forward (target verify of a tree, batch-1 path): returns the
    gated-normed core attention output [T, HV * V] (input of out_proj), or None to run the parent path."""
    if not _D8_GDN or not forward_batch.forward_mode.is_target_verify():
        return None
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.speculative import specpilot_db_conv as _dbconv

    backend = get_attn_backend()
    lin = getattr(backend, "linear_attn_backend", backend)
    md = getattr(lin, "forward_metadata", None)
    if md is None or getattr(md, "retrieve_next_token", None) is None or not _dbconv.conv_tm_enabled():
        return None
    layer = gdn.attn
    cache = lin.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    if getattr(cache, "specpilot_tree_kn", None) is None:
        return None
    N = int(forward_batch.spec_info.draft_token_num)
    T = mixed_qkv.shape[0]
    from sglang.srt.speculative import specpilot_d3w_gdn as _d3w  # SPECPILOT_D3W_GDN: 33 <= N <= 48

    wide = N > 32 and _d3w.enabled() and _d3w.supports(N)
    if (N > 32 and not wide) or T % N:
        return None
    bs = T // N
    norm = gdn.norm
    if not (norm.norm_before_gate and norm.group_size is None and norm.activation in ("swish", "silu")
            and norm.bias is None):
        return None
    first = not _dbconv.parents_given_for_layer(layer.layer_id)  # first GDN layer of the forward (PAR1 convention)
    if wide:
        if first:
            _d3w.tree_meta_wide(md.retrieve_next_token, md.retrieve_next_sibling, md.retrieve_parent_token, N, bs)
        meta, hi = _d3w.meta_buffers(bs, N, mixed_qkv.device)
        verify, extra = _d3w.gdn_layer_verify_wide, {"hi": hi}
    else:
        meta = _scratch("meta", (bs, 4, N), torch.int32, mixed_qkv.device)
        if first and _d3w.small_enabled():  # SPECPILOT_INT_D3W32: register-gather meta (same parents and meta)
            _d3w.tree_meta_into(meta, md.retrieve_next_token, md.retrieve_next_sibling, md.retrieve_parent_token, N, bs)
        elif first:
            tree_meta(md.retrieve_next_token, md.retrieve_next_sibling, md.retrieve_parent_token, N, bs)
        verify, extra = gdn_layer_verify, {}
    return verify(
        x=mixed_qkv, a=a, b=b, z=z, conv_weight=layer.conv_weights, conv_bias=layer.bias,
        conv_state=cache.conv[0], ssm_states=cache.temporal, cache_indices=md.mamba_cache_indices[:bs],
        xin=_dbconv.xin_view(cache.intermediate_conv_window[0]),
        scratch_rows=lin.verify_intermediate_state_indices[:bs], meta=meta, A_log=layer.A_log,
        dt_bias=layer.dt_bias, norm_weight=norm.weight, norm_eps=norm.eps, kn_buf=cache.specpilot_tree_kn,
        v_buf=cache.specpilot_tree_v, g_buf=cache.specpilot_tree_g, beta_buf=cache.specpilot_tree_beta,
        H=layer.num_k_heads, HV=layer.num_v_heads, K=layer.head_k_dim, V=layer.head_v_dim, n_nodes=N,
        silu=layer.activation in ("silu", "swish", True), **extra,
    )
