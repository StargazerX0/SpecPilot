"""SPECPILOT_TREE_GDN=paths: snapshot-free GDN tree verify (one program per (leaf, head, V-block) walks the leaf's
root path with the state tile in registers) and a replay commit of the accepted path into ``temporal``.

Verify writes no per-node fp32 state (stock writes HV*K*V*4 B = 3 MiB per node per layer); it stores each node's
commit inputs (L2-normalised k, raw v, gate g, beta) instead, ~33 KiB per node per layer. The commit replays the
accepted path from the unchanged ``temporal`` state with the same arithmetic, so the committed state equals the
state verify used for the accepted leaf. Per-node outputs equal the sequential recurrence along the node's root
path (branch_oracle.gdn_branch_reference); every program that covers a node writes identical values.
"""
from __future__ import annotations

import os

import triton
import triton.language as tl

# SPECPILOT_K: tree-WY verify (SPECPILOT_K_GDN_WY=1) and commit launch width (SPECPILOT_K_COMMIT_BV); read once at import.
_K_GDN_WY = os.environ.get("SPECPILOT_K_GDN_WY", "0") not in ("", "0")
_K_COMMIT_BV = int(os.environ.get("SPECPILOT_K_COMMIT_BV", "32"))
# SPECPILOT_X4: WY launch (V block width, warps) for wide trees; 0/unset = parent launch (64, 8).
_X4_WY_BV = int(os.environ.get("SPECPILOT_X4_WY_BV", "0") or 0)
_X4_WY_WARPS = int(os.environ.get("SPECPILOT_X4_WY_WARPS", "0") or 0)
# SPECPILOT_D2: SPECPILOT_D2_EARLY bitmask (1 add-norm, 2 gated norm, 4 WY, 8 attention)
_D2_EARLY = bool(int(__import__("os").environ.get("SPECPILOT_D2_EARLY", "0").strip() or 0) & 4)


@triton.jit
def _gdn_tree_paths_kernel(
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    stride_h0,
    parent_ptr,  # [bs, N] int32 parent node (root entry ignored)
    row_ptr,  # [bs] int32 scratch row
    kn_out,  # [rows, N, H, K] fp32
    v_out,  # [rows, N, HV, V] v dtype
    g_out,  # [rows, N, HV] fp32
    beta_out,  # [rows, N, HV] fp32
    scale,
    softplus_beta,
    softplus_threshold,
    stride_q,
    stride_k,
    stride_v,
    stride_a,
    stride_b,
    N: tl.constexpr,
    NP: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_nl, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n = i_nl // N
    leaf = i_nl % N
    i_h = i_hv // (HV // H)
    pidx = tl.arange(0, NP)
    par = tl.load(parent_ptr + i_n * N + pidx, mask=pidx < N, other=-2).to(tl.int32)
    par = tl.where(pidx == 0, -1, par)
    n_kids = tl.sum(tl.where((par == leaf) & (pidx < N), 1, 0), 0)
    if n_kids > 0:
        return
    # ancestors of the leaf, leaf first
    path_rev = tl.full([NP], -1, dtype=tl.int32)
    cur = leaf + 0 * i_n
    for s in tl.static_range(N):
        path_rev = tl.where((pidx == s) & (cur >= 0), cur, path_rev)
        nxt = tl.sum(tl.where(pidx == cur, par, 0), 0)
        cur = tl.where(cur > 0, nxt, -1)
    plen = tl.sum(tl.where(path_rev >= 0, 1, 0), 0)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    idx = tl.load(h0_indices + i_n).to(tl.int64)
    if idx >= 0:
        p_h0 = h0_source + idx * stride_h0 + i_hv * K * V + o_v[None, :] * K + o_k[:, None]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    row = tl.load(row_ptr + i_n).to(tl.int64)
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
    bos = i_n * N
    for t in range(0, plen):
        node = tl.sum(tl.where(pidx == plen - 1 - t, path_rev, 0), 0).to(tl.int64)
        tok = bos + node
        b_q = tl.load(q + tok * stride_q + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(k + tok * stride_k + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
        b_vr = tl.load(v + tok * stride_v + i_hv * V + o_v, mask=mask_v, other=0)
        b_v = b_vr.to(tl.float32)
        b_b = tl.load(b + tok * stride_b + i_hv).to(tl.float32)
        b_a = tl.load(a + tok * stride_a + i_hv).to(tl.float32)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))
        b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
        b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(o + (tok * HV + i_hv) * V + o_v, b_o.to(o.dtype.element_ty), mask=mask_v)
        sbase = row * N + node
        tl.store(v_out + (sbase * HV + i_hv) * V + o_v, b_vr, mask=mask_v)
        if i_v == 0:
            tl.store(kn_out + (sbase * H + i_h) * K + o_k, b_k, mask=mask_k)
            tl.store(g_out + sbase * HV + i_hv, b_g)
            tl.store(beta_out + sbase * HV + i_hv, b_beta)


@triton.jit
def _gdn_tree_wy_kernel(
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    stride_h0,
    parent_ptr,  # [bs, N] int32 parent node (root entry ignored); parents precede children (topological order)
    row_ptr,  # [bs] int32 scratch row
    kn_out,  # [rows, N, H, K] fp32
    v_out,  # [rows, N, HV, V] v dtype
    g_out,  # [rows, N, HV] fp32
    beta_out,  # [rows, N, HV] fp32
    scale,
    softplus_beta,
    softplus_threshold,
    stride_q,
    stride_k,
    stride_v,
    stride_a,
    stride_b,
    N: tl.constexpr,
    NP: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    MAXD: tl.constexpr,  # SPECPILOT_X4: max tree depth (<= N - 1); bounds the ancestor walk and the X^16 factor
    D2_EARLY: tl.constexpr = False,
):
    if D2_EARLY:  # SPECPILOT_D2: let the gated norm (and through it out_proj's weight pre-staging) launch now
        tl.extra.cuda.gdc_launch_dependents()
    # SPECPILOT_K_GDN_WY tree-WY: the gated delta rule of every node along its own root path, for the whole tree at once.
    # With Anc[t,s] = s is an ancestor of t (strict), gam_t = sum of g over t's root path (incl. t), A = Kn h0:
    #   v' = (I + diag(beta) Lr)^-1 beta (v - e^gam A),   Lr[t,s] = Anc[t,s] e^(gam_t - gam_s) (kn_s . kn_t)
    #   o  = e^gam (Qn h0) + (AncEq o e^(gam_t - gam_s) o (Qn Kn^T)) v'
    # equal to the per-path sequential recurrence up to reduction order. Lr is strictly lower triangular in
    # topological node order (nilpotent, depth < 16), so the inverse is (I - X)(I + X^2)(I + X^4)(I + X^8),
    # X = diag(beta) Lr: no sequential loop, one program per (request, head, V block) instead of per leaf.
    i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_hv // (HV // H)
    tt = tl.arange(0, NP)
    valid = tt < N
    par = tl.load(parent_ptr + i_n * N + tt, mask=valid, other=-1).to(tl.int32)
    par = tl.where(tt == 0, -1, par)
    # ancestor matrix by walking parent pointers (<= N-1 hops)
    cur = tl.where(valid, par, -1)
    anc = tt[:, None] < 0  # all False
    for _ in tl.static_range(MAXD):
        anc = anc | ((tt[None, :] == cur[:, None]) & (cur[:, None] >= 0))
        nxt = tl.sum(tl.where(tt[None, :] == cur[:, None], par[None, :], 0), 1)
        cur = tl.where(cur > 0, nxt, -1)
    eye = tt[:, None] == tt[None, :]
    anc_eq = anc | (eye & valid[:, None])

    toks = (i_n * N + tt).to(tl.int64)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mk = valid[:, None] & mask_k[None, :]
    mv = valid[:, None] & mask_v[None, :]
    qb = tl.load(q + toks[:, None] * stride_q + i_h * K + o_k[None, :], mask=mk, other=0)  # bf16
    kb = tl.load(k + toks[:, None] * stride_k + i_h * K + o_k[None, :], mask=mk, other=0)
    b_vr = tl.load(v + toks[:, None] * stride_v + i_hv * V + o_v[None, :], mask=mv, other=0)
    b_b = tl.load(b + toks * stride_b + i_hv, mask=valid, other=0).to(tl.float32)
    b_a = tl.load(a + toks * stride_a + i_hv, mask=valid, other=0).to(tl.float32)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    idx = tl.load(h0_indices + i_n).to(tl.int64)
    if idx >= 0:
        p_h0 = h0_source + idx * stride_h0 + i_hv * K * V + o_v[None, :] * K + o_k[:, None]
        b_h += tl.load(p_h0, mask=mask_k[:, None] & mask_v[None, :], other=0).to(tl.float32)
    row = tl.load(row_ptr + i_n).to(tl.int64)
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)

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
    gam = tl.sum(tl.where(anc_eq, b_g[None, :], 0.0), 1)
    # fp32 state as hi + mid + lo bf16 terms (24 mantissa bits): bf16 x bf16 products are exact in fp32
    h_hi = b_h.to(tl.bfloat16)
    r1 = b_h - h_hi.to(tl.float32)
    h_mid = r1.to(tl.bfloat16)
    h_lo = (r1 - h_mid.to(tl.float32)).to(tl.bfloat16)
    b_A = tl.dot(kb, h_hi) + tl.dot(kb, h_mid) + tl.dot(kb, h_lo)
    b_QH = tl.dot(qb, h_hi) + tl.dot(qb, h_mid) + tl.dot(qb, h_lo)
    b_A = b_A * rk[:, None]
    b_QH = b_QH * (rq * scale)[:, None]
    kbt = tl.trans(kb)
    b_KK = tl.dot(kb, kbt) * (rk[:, None] * rk[None, :])
    b_QK = tl.dot(qb, kbt) * ((rq * scale)[:, None] * rk[None, :])
    dgam = gam[:, None] - gam[None, :]
    X = tl.where(anc, b_beta[:, None] * tl.exp(tl.where(anc, dgam, 0.0)) * b_KK, 0.0)
    P = tl.where(anc_eq, tl.exp(tl.where(anc_eq, dgam, 0.0)) * b_QK, 0.0)
    egam = tl.exp(gam)
    rhs = b_beta[:, None] * (b_vr.to(tl.float32) - egam[:, None] * b_A)
    # (I + X)^-1 rhs = (I - X)(I + X^2)(I + X^4)(I + X^8) rhs, applied right to left to the [NP, BV] rhs
    X2 = tl.dot(X, X, input_precision="tf32x3")
    X4 = tl.dot(X2, X2, input_precision="tf32x3")
    X8 = tl.dot(X4, X4, input_precision="tf32x3")
    r = rhs
    if NP > 16 and MAXD >= 16:  # depth >= 16 possible: one more factor (I + X^16); else X^16 == 0 exactly
        X16 = tl.dot(X8, X8, input_precision="tf32x3")
        r = r + tl.dot(X16, r, input_precision="tf32x3")
    r = r + tl.dot(X8, r, input_precision="tf32x3")
    r = r + tl.dot(X4, r, input_precision="tf32x3")
    r = r + tl.dot(X2, r, input_precision="tf32x3")
    vp = r - tl.dot(X, r, input_precision="tf32x3")
    b_o = egam[:, None] * b_QH + tl.dot(P, vp, input_precision="tf32x3")
    tl.store(o + (toks[:, None] * HV + i_hv) * V + o_v[None, :], b_o.to(o.dtype.element_ty), mask=mv)
    sbase = row * N + tt
    tl.store(v_out + (sbase[:, None] * HV + i_hv) * V + o_v[None, :], b_vr, mask=mv)
    if i_v == 0:
        tl.store(kn_out + (sbase[:, None] * H + i_h) * K + o_k[None, :], kf * rk[:, None], mask=mk)
        tl.store(g_out + sbase * HV + i_hv, b_g, mask=valid)
        tl.store(beta_out + sbase * HV + i_hv, b_beta, mask=valid)



def _x4_max_depth(n_nodes: int) -> int:
    """SPECPILOT_X4: WY ancestor-walk bound; N - 1 (any tree). The B - 1 bound measured no gain (e970236ca058)."""
    return int(n_nodes) - 1


# SPECPILOT_K_WY_TUNE=1: batch-dependent WY launch, BV 128 / 4 warps for bs >= 4, else 64 / 8 (the gdn_wy launch).
# Bit-identical to gdn_wy (BV only partitions independent output columns); read once at import.
_K_WY_TUNE = os.environ.get("SPECPILOT_K_WY_TUNE", "0") not in ("", "0")


def gdn_tree_paths_verify(
    *,
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    ssm_states,
    cache_indices,
    parents,
    scratch_rows,
    kn_buf,
    v_buf,
    g_buf,
    beta_buf,
    n_nodes: int,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    scale=None,
):
    """q, k [1, T, H, K]; v [1, T, HV, V]; a, b [T, HV] (T = bs * N, uniform N per request);
    ssm_states [slots, HV, V, K] fp32 (read only); parents [bs, N] int32; scratch_rows [bs] int32;
    kn/v/g/beta bufs [rows, N, ...] of this layer. Returns o [1, T, HV, V]."""
    _, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    N = int(n_nodes)
    bs = T // N
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NV = triton.cdiv(V, BV)
    if scale is None:
        scale = K ** -0.5
    o = q.new_empty(1, T, HV, V)
    if _K_GDN_WY:  # SPECPILOT_K: whole-tree WY form, one program per (request, head, 64-wide V block)
        bvw, n_warps = (  # bs is static per captured graph
            (128, 4) if _K_WY_TUNE and bs >= 4 else (_X4_WY_BV or 64, _X4_WY_WARPS or 8)
        )
        BVW = min(triton.next_power_of_2(V), bvw)
        _gdn_tree_wy_kernel[(triton.cdiv(V, BVW), bs, HV)](
            A_log, a, dt_bias, q, k, v, b, o, ssm_states, cache_indices, ssm_states.stride(0), parents,
            scratch_rows, kn_buf, v_buf, g_buf, beta_buf, scale, softplus_beta, softplus_threshold,
            q.stride(1), k.stride(1), v.stride(1), a.stride(-2), b.stride(-2),
            N=N, NP=max(16, triton.next_power_of_2(N)), H=H, HV=HV, K=K, V=V, BK=BK, BV=BVW,
            MAXD=_x4_max_depth(N), num_warps=n_warps, num_stages=1, D2_EARLY=_D2_EARLY,
        )
        return o
    _gdn_tree_paths_kernel[(NV, bs * N, HV)](
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        ssm_states,
        cache_indices,
        ssm_states.stride(0),
        parents,
        scratch_rows,
        kn_buf,
        v_buf,
        g_buf,
        beta_buf,
        scale,
        softplus_beta,
        softplus_threshold,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        a.stride(-2),
        b.stride(-2),
        N=N,
        NP=triton.next_power_of_2(N),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        num_warps=1,
        num_stages=3,
    )
    return o


@triton.jit
def _gdn_tree_commit_kernel(
    temporal,  # [L, slots, HV, V, K] (strided by layer / slot)
    stride_tl,
    stride_ts,
    slot_ptr,  # [bs] state slot
    row_ptr,  # [bs] scratch row
    path_ptr,  # [bs, N] int64 accepted node per depth (-1 past the leaf)
    alen_ptr,  # [bs] int32 accepted drafts
    kn_buf,  # [L, rows, N, H, K]
    v_buf,  # [L, rows, N, HV, V]
    g_buf,  # [L, rows, N, HV]
    beta_buf,
    stride_knl,
    stride_vl,
    stride_gl,
    N: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_n, i_lh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    lay = (i_lh // HV).to(tl.int64)
    i_hv = i_lh % HV
    i_h = i_hv // (HV // H)
    slot = tl.load(slot_ptr + i_n).to(tl.int64)
    if slot < 0:
        return
    row = tl.load(row_ptr + i_n).to(tl.int64)
    a_len = tl.load(alen_ptr + i_n).to(tl.int32)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    p_h = temporal + lay * stride_tl + slot * stride_ts + i_hv * K * V + o_v[None, :] * K + o_k[:, None]
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    for d in range(0, a_len + 1):
        node = tl.load(path_ptr + i_n * N + d).to(tl.int64)
        sbase = row * N + node
        b_k = tl.load(kn_buf + lay * stride_knl + (sbase * H + i_h) * K + o_k, mask=mask_k, other=0)
        b_v = tl.load(v_buf + lay * stride_vl + (sbase * HV + i_hv) * V + o_v, mask=mask_v, other=0).to(tl.float32)
        b_g = tl.load(g_buf + lay * stride_gl + sbase * HV + i_hv)
        b_beta = tl.load(beta_buf + lay * stride_gl + sbase * HV + i_hv)
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
    tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)


def gdn_tree_commit(
    *, temporal, state_slots, scratch_rows, path, accept_len, kn_buf, v_buf, g_buf, beta_buf
):
    """Replay the accepted path of every request into ``temporal`` for all GDN layers at once.
    temporal [L, slots, HV, V, K]; bufs [L, rows, N, ...]; path [bs, N] int64; accept_len [bs]."""
    L = temporal.shape[0]
    HV, V, K = temporal.shape[2], temporal.shape[3], temporal.shape[4]
    H = kn_buf.shape[3]
    N = path.shape[1]
    bs = path.shape[0]
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), _K_COMMIT_BV)  # SPECPILOT_K_COMMIT_BV (default 32 = tree-004)
    NV = triton.cdiv(V, BV)
    assert temporal.stride(-1) == 1 and temporal.stride(-2) == K and temporal.stride(-3) == K * V
    _gdn_tree_commit_kernel[(NV, bs, L * HV)](
        temporal,
        temporal.stride(0),
        temporal.stride(1),
        state_slots,
        scratch_rows,
        path,
        accept_len,
        kn_buf,
        v_buf,
        g_buf,
        beta_buf,
        kn_buf.stride(0),
        v_buf.stride(0),
        g_buf.stride(0),
        N=N,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        num_warps=1,
    )
