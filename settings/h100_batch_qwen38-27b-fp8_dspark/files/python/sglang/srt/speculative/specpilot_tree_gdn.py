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
# SPECPILOT_K_WY_NILP=1: drop the (I + X^8) factor of the nilpotent inverse when N <= 8 (depth <= 7 => X^8 == 0
# exactly: X is ancestor-masked with exact zeros), bit-identical outputs, shorter dependent dot chain.
_K_WY_NILP = os.environ.get("SPECPILOT_K_WY_NILP", "0") not in ("", "0")


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
    out_rows,  # SPECPILOT_DB_RV_FUSE: [bs * N] int64 packed output row per node (-1: not stored); unused otherwise
    N: tl.constexpr,
    NP: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    PACKED_OUT: tl.constexpr = False,
    anc_ptr=None,  # SPECPILOT_DB_RA_WY_ANC: [bs, N] int32 ancestor-or-self bits per node (computed once per verify)
    ANC_GIVEN: tl.constexpr = False,
    SKIP8: tl.constexpr = False,
):
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
    eye = tt[:, None] == tt[None, :]
    if ANC_GIVEN:  # SPECPILOT_DB_RA_WY_ANC: same strict-ancestor matrix from the precomputed bits
        bits = tl.load(anc_ptr + i_n * N + tt, mask=valid, other=0)
        anc = (((bits[:, None] >> tt[None, :]) & 1) != 0) & valid[:, None] & (tt[None, :] < N) & (~eye)
    else:
        # ancestor matrix by walking parent pointers (<= N-1 hops)
        cur = tl.where(valid, par, -1)
        anc = tt[:, None] < 0  # all False
        for _ in tl.static_range(N - 1):
            anc = anc | ((tt[None, :] == cur[:, None]) & (cur[:, None] >= 0))
            nxt = tl.sum(tl.where(tt[None, :] == cur[:, None], par[None, :], 0), 1)
            cur = tl.where(cur > 0, nxt, -1)
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
    r = rhs
    if not SKIP8:  # SPECPILOT_K_WY_NILP: X^8 == 0 exactly when N <= 8
        X8 = tl.dot(X4, X4, input_precision="tf32x3")
        if NP > 16:  # depth up to NP-1 >= 16: one more factor (I + X^16)
            X16 = tl.dot(X8, X8, input_precision="tf32x3")
            r = r + tl.dot(X16, r, input_precision="tf32x3")
        r = r + tl.dot(X8, r, input_precision="tf32x3")
    r = r + tl.dot(X4, r, input_precision="tf32x3")
    r = r + tl.dot(X2, r, input_precision="tf32x3")
    vp = r - tl.dot(X, r, input_precision="tf32x3")
    b_o = egam[:, None] * b_QH + tl.dot(P, vp, input_precision="tf32x3")
    if PACKED_OUT:  # SPECPILOT_DB_RV_FUSE: verified nodes straight into their packed rows
        orow = tl.load(out_rows + toks, mask=valid, other=-1)
        tl.store(
            o + (tl.maximum(orow, 0)[:, None] * HV + i_hv) * V + o_v[None, :],
            b_o.to(o.dtype.element_ty),
            mask=mv & (orow[:, None] >= 0),
        )
    else:
        tl.store(o + (toks[:, None] * HV + i_hv) * V + o_v[None, :], b_o.to(o.dtype.element_ty), mask=mv)
    sbase = row * N + tt
    tl.store(v_out + (sbase[:, None] * HV + i_hv) * V + o_v[None, :], b_vr, mask=mv)
    if i_v == 0:
        tl.store(kn_out + (sbase[:, None] * H + i_h) * K + o_k[None, :], kf * rk[:, None], mask=mk)
        tl.store(g_out + sbase * HV + i_hv, b_g, mask=valid)
        tl.store(beta_out + sbase * HV + i_hv, b_beta, mask=valid)


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
    out_rows=None,
    out_tokens: int = 0,
    anc_bits=None,  # SPECPILOT_DB_RA_WY_ANC
    d8r3_ragged=None,  # SPECPILOT_D8R3_RPREP: (tiles, n_eff) of the packed verify
):
    """SPECPILOT_DB_RV_FUSE: with ``out_rows`` ([T] int64, -1 = skip) the WY path returns o [1, out_tokens, HV, V] with
    each node's output in row out_rows[node] (rows nobody writes are left uninitialized).
    q, k [1, T, H, K]; v [1, T, HV, V]; a, b [T, HV] (T = bs * N, uniform N per request);
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
    if _K_WY_SPLIT and bs >= _K_WY_SPLIT_MINBS:  # SPECPILOT_K_WY_SPLIT: prep + state-streaming (bs static per graph)
        return gdn_tree_wy_split_verify(
            A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=b, ssm_states=ssm_states,
            cache_indices=cache_indices, parents=parents, scratch_rows=scratch_rows, kn_buf=kn_buf, v_buf=v_buf,
            g_buf=g_buf, beta_buf=beta_buf, n_nodes=n_nodes, softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold, scale=scale, out_rows=out_rows, out_tokens=out_tokens,
            anc_bits=anc_bits, d8r3_ragged=d8r3_ragged,
        )
    packed_out = out_rows is not None and _K_GDN_WY
    o = q.new_empty(1, int(out_tokens) if packed_out else T, HV, V)
    if _K_GDN_WY:  # SPECPILOT_K: whole-tree WY form, one program per (request, head, 64-wide V block)
        bvw, n_warps = (128, 4) if _K_WY_TUNE and bs >= 4 else (64, 8)  # bs is static per captured graph
        BVW = min(triton.next_power_of_2(V), bvw)
        _gdn_tree_wy_kernel[(triton.cdiv(V, BVW), bs, HV)](
            A_log, a, dt_bias, q, k, v, b, o, ssm_states, cache_indices, ssm_states.stride(0), parents,
            scratch_rows, kn_buf, v_buf, g_buf, beta_buf, scale, softplus_beta, softplus_threshold,
            q.stride(1), k.stride(1), v.stride(1), a.stride(-2), b.stride(-2),
            out_rows if packed_out else cache_indices,
            N=N, NP=max(16, triton.next_power_of_2(N)), H=H, HV=HV, K=K, V=V, BK=BK, BV=BVW,
            PACKED_OUT=packed_out,
            anc_ptr=anc_bits if anc_bits is not None else parents, ANC_GIVEN=anc_bits is not None,  # SPECPILOT_DB_RA_WY_ANC
            SKIP8=_K_WY_NILP and N <= 8, num_warps=n_warps, num_stages=1,
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


# ------------------------------------------------------------------------------------------------
# SPECPILOT_K_WY_LAZY=1 (with SPECPILOT_K_GDN_WY=1): deferred closed-form commit fused into the next WY tree verify.
# The commit step only records, per mamba slot, the accepted leaf and the on-path node mask and flips the slot's
# double-buffer parity (wy_lazy_mark, one tiny kernel; the separate 48-layer replay commit is gone). The next WY
# verify of that slot loads `temporal` once, applies the pending commit in WY form,
#   h' = e^{gam_leaf} h0 + Kn^T (onpath . e^{gam_leaf - gam_s} . v'_s)     (v'_s = solved delta of node s)
# with the v', normalised k and cumulative gate gam the previous verify stored for every node, writes h' back, and
# verifies the new tree from h'. Buffers (per layer): kn [2S+1, N, H, K] fp32, v (holds v') [2S+1, N, HV, V] fp32,
# g (holds gam) [2S+1, N, HV] fp32; row = parity * S + slot, row 2S = padding sink. Extend resets the slot's leaf
# to -1 (nothing pending). One state read + one write per step instead of verify read + commit read + write.
# ------------------------------------------------------------------------------------------------

_K_WY_LAZY = _K_GDN_WY and os.environ.get("SPECPILOT_K_WY_LAZY", "0") not in ("", "0")
_WYL = {"leaf": None, "mask": None, "parity": None}
# SPECPILOT_K_WYL_CFG="BV,warps[,stages]" launch of the lazy verify (default: the gdn_wy_tune launch); read once at import
_K_WYL_CFG = tuple(int(x) for x in os.environ.get("SPECPILOT_K_WYL_CFG", "").split(",") if x.strip())


# SPECPILOT_D3W_GDNF=1 (N9b): CUDA lazy GDN tree verify (specpilot_d3w_gdnf_cuda) for trees of <= 16 nodes with the
# ancestor bits; it turns the lazy commit machinery above on (wy_lazy_*: mark instead of the replay commit); other shapes
# take the Triton lazy kernel. Read once at import; unset = the seed's split verify + replay commit.
_D3W_GDNF = _K_GDN_WY and os.environ.get("SPECPILOT_D3W_GDNF", "0") not in ("", "0")


def d3w_gdnf_enabled() -> bool:
    return _D3W_GDNF


def d3w_gdnf_supported(n_nodes: int) -> bool:
    return _D3W_GDNF and 1 <= int(n_nodes) <= 16


def wy_lazy_enabled() -> bool:
    return _K_WY_LAZY or _D3W_GDNF  # SPECPILOT_D3W_GDNF


def wy_lazy_register(num_slots: int, device):
    """Per-slot pending-commit state (allocated with the mamba pool, before graph capture)."""
    import torch

    _WYL["leaf"] = torch.full((num_slots,), -1, dtype=torch.int32, device=device)
    _WYL["mask"] = torch.zeros((num_slots,), dtype=torch.int32, device=device)
    _WYL["parity"] = torch.zeros((num_slots,), dtype=torch.int32, device=device)
    if _D3W_GDNF:  # SPECPILOT_D3W_GDNF: build / load the CUDA extension before any graph capture
        from sglang.srt.speculative import specpilot_d3w_gdnf_cuda as _cu

        _cu.ensure_built()
    return _WYL


def wy_lazy_state():
    return _WYL["leaf"], _WYL["mask"], _WYL["parity"]


@triton.jit
def _gdn_tree_wy_lazy_kernel(
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    o,
    h0_source,  # [S, HV, V, K] fp32 temporal of this layer (pending commit written back in place)
    h0_indices,  # [bs] slot (-1 = padding)
    stride_h0,
    parent_ptr,  # [bs, N] int32 parent node (topological order)
    leaf_ptr,  # [S] int32 pending accepted leaf (-1 none)
    mask_ptr,  # [S] int32 pending on-path node bitmask
    par_ptr,  # [S] int32 read half of the double buffer
    n_slots,
    kn_buf,  # [2S+1, N, H, K] fp32
    vp_buf,  # [2S+1, N, HV, V] fp32 (solved v')
    gam_buf,  # [2S+1, N, HV] fp32 (cumulative log decay along the root path)
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
    SKIP8: tl.constexpr = False,
):
    i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_hv // (HV // H)
    tt = tl.arange(0, NP)
    valid = tt < N
    prt = tl.load(parent_ptr + i_n * N + tt, mask=valid, other=-1).to(tl.int32)
    prt = tl.where(tt == 0, -1, prt)
    cur = tl.where(valid, prt, -1)
    anc = tt[:, None] < 0
    for _ in tl.static_range(N - 1):
        anc = anc | ((tt[None, :] == cur[:, None]) & (cur[:, None] >= 0))
        nxt = tl.sum(tl.where(tt[None, :] == cur[:, None], prt[None, :], 0), 1)
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
    qb = tl.load(q + toks[:, None] * stride_q + i_h * K + o_k[None, :], mask=mk, other=0)
    kb = tl.load(k + toks[:, None] * stride_k + i_h * K + o_k[None, :], mask=mk, other=0)
    b_vr = tl.load(v + toks[:, None] * stride_v + i_hv * V + o_v[None, :], mask=mv, other=0)
    b_b = tl.load(b + toks * stride_b + i_hv, mask=valid, other=0).to(tl.float32)
    b_a = tl.load(a + toks * stride_a + i_hv, mask=valid, other=0).to(tl.float32)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    idx = tl.load(h0_indices + i_n).to(tl.int64)
    wrow = idx * 0 + 2 * n_slots  # padding rows write into the sink row (never read)
    if idx >= 0:
        p_h0 = h0_source + idx * stride_h0 + i_hv * K * V + o_v[None, :] * K + o_k[:, None]
        b_h += tl.load(p_h0, mask=mask_k[:, None] & mask_v[None, :], other=0).to(tl.float32)
        leaf = tl.load(leaf_ptr + idx).to(tl.int32)
        pp = tl.load(par_ptr + idx).to(tl.int64)
        wrow = (1 - pp) * n_slots + idx
        if leaf >= 0:
            # pending commit of the previous step: h' = e^{gam_l} h0 + Kn^T (w . v'), w_s = onpath_s e^{gam_l - gam_s}
            rrow = pp * n_slots + idx
            bits = tl.load(mask_ptr + idx).to(tl.int32)
            onp = (((bits >> tt) & 1) != 0) & valid
            rb = rrow * N + tt
            c_kn = tl.load(kn_buf + (rb[:, None] * H + i_h) * K + o_k[None, :], mask=onp[:, None] & mask_k[None, :],
                           other=0)
            c_vp = tl.load(vp_buf + (rb[:, None] * HV + i_hv) * V + o_v[None, :], mask=onp[:, None] & mask_v[None, :],
                           other=0)
            c_gam = tl.load(gam_buf + rb * HV + i_hv, mask=onp, other=0)
            g_l = tl.load(gam_buf + (rrow * N + leaf) * HV + i_hv)
            c_w = tl.where(onp, tl.exp(g_l - c_gam), 0.0)
            b_h = b_h * tl.exp(g_l) + tl.dot(tl.trans(c_kn), c_w[:, None] * c_vp, input_precision="tf32x3")
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_k[:, None] & mask_v[None, :])
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
    X2 = tl.dot(X, X, input_precision="tf32x3")
    X4 = tl.dot(X2, X2, input_precision="tf32x3")
    r = rhs
    if not SKIP8:
        X8 = tl.dot(X4, X4, input_precision="tf32x3")
        if NP > 16:
            X16 = tl.dot(X8, X8, input_precision="tf32x3")
            r = r + tl.dot(X16, r, input_precision="tf32x3")
        r = r + tl.dot(X8, r, input_precision="tf32x3")
    r = r + tl.dot(X4, r, input_precision="tf32x3")
    r = r + tl.dot(X2, r, input_precision="tf32x3")
    vp = r - tl.dot(X, r, input_precision="tf32x3")
    b_o = egam[:, None] * b_QH + tl.dot(P, vp, input_precision="tf32x3")
    tl.store(o + (toks[:, None] * HV + i_hv) * V + o_v[None, :], b_o.to(o.dtype.element_ty), mask=mv)
    sbase = wrow * N + tt
    tl.store(vp_buf + (sbase[:, None] * HV + i_hv) * V + o_v[None, :], vp, mask=mv)
    if i_v == 0:
        tl.store(kn_buf + (sbase[:, None] * H + i_h) * K + o_k[None, :], kf * rk[:, None], mask=mk)
        tl.store(gam_buf + sbase * HV + i_hv, gam, mask=valid)


def gdn_tree_wy_lazy_verify(
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
    kn_buf,
    v_buf,
    g_buf,
    n_nodes: int,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    scale=None,
):
    """Lazy WY tree verify (SPECPILOT_K_WY_LAZY). ssm_states [S, HV, V, K] fp32 (pending commit written back);
    bufs of this layer [2S+1, N, ...] (v_buf fp32 holds v', g_buf holds gam). Returns o [1, T, HV, V]."""
    _, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    N = int(n_nodes)
    bs = T // N
    n_slots = ssm_states.shape[0]
    assert kn_buf.shape[0] >= 2 * n_slots + 1 and v_buf.dtype == kn_buf.dtype, (kn_buf.shape, n_slots, v_buf.dtype)
    leaf, mask, parity = wy_lazy_state()
    if scale is None:
        scale = K ** -0.5
    o = q.new_empty(1, T, HV, V)
    bvw, n_warps = (128, 4) if _K_WY_TUNE and bs >= 4 else (64, 8)
    n_stages = 1
    if _K_WYL_CFG:
        bvw, n_warps = _K_WYL_CFG[0], _K_WYL_CFG[1]
        n_stages = _K_WYL_CFG[2] if len(_K_WYL_CFG) > 2 else 1
    BVW = min(triton.next_power_of_2(V), bvw)
    _gdn_tree_wy_lazy_kernel[(triton.cdiv(V, BVW), bs, HV)](
        A_log, a, dt_bias, q, k, v, b, o, ssm_states, cache_indices, ssm_states.stride(0), parents,
        leaf, mask, parity, n_slots, kn_buf, v_buf, g_buf, scale, softplus_beta, softplus_threshold,
        q.stride(1), k.stride(1), v.stride(1), a.stride(-2), b.stride(-2),
        N=N, NP=max(16, triton.next_power_of_2(N)), H=H, HV=HV, K=K, V=V, BK=triton.next_power_of_2(K), BV=BVW,
        SKIP8=_K_WY_NILP and N <= 8, num_warps=n_warps, num_stages=n_stages,
    )
    return o


def d3w_gdnf_verify(
    *, A_log, a, dt_bias, q, k, v, b, ssm_states, cache_indices, anc_bits, kn_buf, v_buf, g_buf, n_nodes: int,
    softplus_beta: float = 1.0, softplus_threshold: float = 20.0, scale=None, out_rows=None, out_tokens: int = 0,
):
    """SPECPILOT_D3W_GDNF: CUDA lazy WY tree verify, same state / buffer contract as gdn_tree_wy_lazy_verify (pending
    commit applied in closed form and written back, then the new tree verified). q, k [1, T, H, K], v [1, T, HV, V];
    anc_bits [bs, N] int32 ancestor-or-self bits. With out_rows ([T] int64, -1 = skip) o is [1, out_tokens, HV, V] in
    packed rows (SPECPILOT_DB_RV_FUSE), else [1, T, HV, V]."""
    import torch

    from sglang.srt.speculative import specpilot_d3w_gdnf_cuda as _cu

    _, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    leaf, mask, parity = wy_lazy_state()
    packed = out_rows is not None
    o = q.new_empty(1, int(out_tokens) if packed else T, HV, V)
    if cache_indices.dtype != torch.int32:
        cache_indices = cache_indices.to(torch.int32)
    a2 = a.reshape(-1, a.shape[-1]) if a.dim() != 2 else a
    b2 = b.reshape(-1, b.shape[-1]) if b.dim() != 2 else b
    _cu.gdnf_cuda_verify(
        q=q[0], k=k[0], v=v[0], a=a2, b=b2, A_log=A_log, dt_bias=dt_bias, anc=anc_bits, ssm_states=ssm_states,
        cache_indices=cache_indices, leaf=leaf, mask=mask, parity=parity, kn_buf=kn_buf, vp_buf=v_buf, gam_buf=g_buf,
        o=o[0], out_rows=out_rows if packed else None, N=int(n_nodes), scale=scale, softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
    )
    return o


@triton.jit
def _wy_lazy_mark_kernel(slot_ptr, path_ptr, alen_ptr, leaf_ptr, mask_ptr, par_ptr, bs,
                         N: tl.constexpr, NP: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < bs
    slot = tl.load(slot_ptr + offs, mask=m, other=-1).to(tl.int64)
    ok = m & (slot >= 0)
    alen = tl.load(alen_ptr + offs, mask=m, other=0).to(tl.int32)
    dd = tl.arange(0, NP)
    on = ok[:, None] & (dd[None, :] <= alen[:, None]) & (dd[None, :] < N)
    node = tl.load(path_ptr + offs[:, None] * N + dd[None, :], mask=on, other=0).to(tl.int32)
    bits = tl.sum(tl.where(on, (tl.zeros_like(node) + 1) << node, 0), 1)
    leaf = tl.sum(tl.where(on & (dd[None, :] == alen[:, None]), node, 0), 1)
    par = tl.load(par_ptr + slot, mask=ok, other=0)
    tl.store(leaf_ptr + slot, leaf, mask=ok)
    tl.store(mask_ptr + slot, bits, mask=ok)
    tl.store(par_ptr + slot, 1 - par, mask=ok)


def wy_lazy_mark(*, state_slots, path, accept_len):
    """Commit step of the lazy path: record the accepted leaf + on-path mask per slot, flip its parity.
    path [bs, N] int64 accepted node per depth (-1 past the leaf); accept_len [bs] accepted drafts."""
    bs = state_slots.shape[0]
    if bs == 0:
        return
    leaf, mask, parity = wy_lazy_state()
    N = path.shape[1]
    assert N <= 31 and path.stride(1) == 1 and path.stride(0) == N
    _wy_lazy_mark_kernel[(1,)](state_slots, path, accept_len, leaf, mask, parity, bs, N=N,
                               NP=triton.next_power_of_2(N), BLOCK=triton.next_power_of_2(max(bs, 16)))


def wy_lazy_reset(cache_indices):
    """Extend/prefill writes a fresh state into these slots: nothing pending (negative indices are padding)."""
    import torch

    leaf = _WYL["leaf"]
    idx = cache_indices.long()
    leaf.index_fill_(0, torch.where(idx >= 0, idx, leaf.shape[0] - 1), -1)


def wy_lazy_flush(*, temporal, kn_buf, v_buf, g_buf, slots):
    """Test/debug helper (eager torch): apply the pending commit of `slots` to every layer of `temporal`, as the
    next lazy verify does first, and clear it. temporal [L, S, HV, V, K]; bufs [L, 2S+1, N, ...]."""
    import torch

    leaf, mask, parity = wy_lazy_state()
    S = temporal.shape[1]
    N = kn_buf.shape[2]
    H, HV = kn_buf.shape[3], g_buf.shape[3]
    for s in slots.tolist():
        lf = int(leaf[s])
        if lf < 0:
            continue
        row = int(parity[s]) * S + s
        on = torch.tensor([(int(mask[s]) >> t) & 1 for t in range(N)], device=temporal.device, dtype=torch.bool)
        kn = kn_buf[:, row][:, on]  # [L, P, H, K]
        vp = v_buf[:, row][:, on].float()  # [L, P, HV, V]
        gam = g_buf[:, row][:, on]  # [L, P, HV]
        gl = g_buf[:, row, lf]  # [L, HV]
        w = torch.exp(gl[:, None] - gam)  # [L, P, HV]
        kn = kn.repeat_interleave(HV // H, dim=2)  # [L, P, HV, K]
        upd = torch.einsum("lphk,lphv,lph->lhvk", kn, vp, w)
        h = temporal[:, s].float() * torch.exp(gl)[:, :, None, None] + upd
        temporal[:, s] = h.to(temporal.dtype)
        leaf[s] = -1


# ------------------------------------------------------------------------------------------------
# SPECPILOT_K_WY_SPLIT=1 (with SPECPILOT_K_GDN_WY=1): the WY tree verify split into two kernels.
# The WY kernel is latency-bound (73 us/layer at bs 24 vs a ~22 us state-read floor): one register-heavy program per
# (request, head) runs the whole dependent dot chain (KK/QK, X^2, X^4, X^8 and four r-updates) while holding the fp32
# state tile. Here
#   (A) one light program per (request, value head) computes everything that does not need the state:
#       gam, M' = (I + X)^-1 diag(beta) = (I - X)(I + X^2)(I + X^4)(I + X^8) diag(beta), W = P M' ([NP, NP] each),
#       and stores the commit inputs (kn, raw v, g, beta) exactly as the WY kernel does;
#   (B) one program per (request, head, V block) streams the state: A = Kn h0, QH = Qn h0 (3-term bf16 split as
#       before), u = v - e^gam A, o = e^gam QH + W u.
# Same math as the WY kernel with the inverse applied as an explicit matrix (reduction order differs).
# ------------------------------------------------------------------------------------------------

_K_WY_SPLIT = _K_GDN_WY and os.environ.get("SPECPILOT_K_WY_SPLIT", "0") not in ("", "0")
# SPECPILOT_K_WY_SPLIT_MINBS (db-109): smallest batch that takes the split kernels (microbench ea304620b787: the split
# pays at bs 32, equal at 16, worse at 8 on Qwen3.5-4B); smaller batches keep the whole-tree WY. Default 16; 0 = always.
_K_WY_SPLIT_MINBS = int(os.environ.get("SPECPILOT_K_WY_SPLIT_MINBS", "16") or 0)
_K_WYS_CFG = tuple(int(x) for x in os.environ.get("SPECPILOT_K_WYS_CFG", "").split(",") if x.strip())  # BV,warps
# SPECPILOT_K_WYS_PREP_WARPS (prep kernel warps, default 4), SPECPILOT_K_WYS_STACK=1 (stacked [Q;K] state dots)
_K_WYS_PREP_WARPS = int(os.environ.get("SPECPILOT_K_WYS_PREP_WARPS", "4"))
_K_WYS_STACK = os.environ.get("SPECPILOT_K_WYS_STACK", "0") not in ("", "0")
# SPECPILOT_D3_STATE="BV,warps,BKC,stages,TA,TQ[,STACK]": K-chunked state kernel (specpilot_d3_state);
# SPECPILOT_D3_PREP32=warps: 2x2 block prep for 17..32-node trees (specpilot_d3_prep). Read once at import.
_D3_STATE = tuple(int(x) for x in os.environ.get("SPECPILOT_D3_STATE", "").split(",") if x.strip())
assert not _D3_STATE or len(_D3_STATE) in (6, 7), "SPECPILOT_D3_STATE=BV,warps,BKC,stages,TA,TQ[,STACK]"
_D3_PREP32 = int(os.environ.get("SPECPILOT_D3_PREP32", "0") or 0)


@triton.jit
def _gdn_tree_wy_prep_kernel(
    A_log, a, dt_bias, q, k, v, b, parent_ptr, row_ptr,
    kn_out, v_out, g_out, beta_out,
    w_out,  # [bs, HV, NP, NP] fp32: W = P M'
    m_out,  # [bs, HV, NP, NP] fp32: M' = (I + X)^-1 diag(beta)
    eg_out,  # [bs, HV, NP] fp32: e^gam
    scale, softplus_beta, softplus_threshold, stride_q, stride_k, stride_v, stride_a, stride_b,
    N: tl.constexpr, NP: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BVR: tl.constexpr, STORE_M: tl.constexpr, SKIP8: tl.constexpr = False,
    anc_ptr=None,  # SPECPILOT_DB_RA_WY_ANC (db-111): [bs, N] int32 ancestor-or-self bits per node
    ANC_GIVEN: tl.constexpr = False,
):
    i_n, i_hv = tl.program_id(0), tl.program_id(1)
    i_h = i_hv // (HV // H)
    tt = tl.arange(0, NP)
    valid = tt < N
    eye = tt[:, None] == tt[None, :]
    if ANC_GIVEN:  # SPECPILOT_DB_RA_WY_ANC: same strict-ancestor matrix from the precomputed bits (no parent walk)
        bits = tl.load(anc_ptr + i_n * N + tt, mask=valid, other=0)
        anc = (((bits[:, None] >> tt[None, :]) & 1) != 0) & valid[:, None] & (tt[None, :] < N) & (~eye)
    else:
        prt = tl.load(parent_ptr + i_n * N + tt, mask=valid, other=-1).to(tl.int32)
        prt = tl.where(tt == 0, -1, prt)
        cur = tl.where(valid, prt, -1)
        anc = tt[:, None] < 0
        for _ in tl.static_range(N - 1):
            anc = anc | ((tt[None, :] == cur[:, None]) & (cur[:, None] >= 0))
            nxt = tl.sum(tl.where(tt[None, :] == cur[:, None], prt[None, :], 0), 1)
            cur = tl.where(cur > 0, nxt, -1)
    anc_eq = anc | (eye & valid[:, None])
    toks = (i_n * N + tt).to(tl.int64)
    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    mk = valid[:, None] & mask_k[None, :]
    qb = tl.load(q + toks[:, None] * stride_q + i_h * K + o_k[None, :], mask=mk, other=0)
    kb = tl.load(k + toks[:, None] * stride_k + i_h * K + o_k[None, :], mask=mk, other=0)
    b_b = tl.load(b + toks * stride_b + i_hv, mask=valid, other=0).to(tl.float32)
    b_a = tl.load(a + toks * stride_a + i_hv, mask=valid, other=0).to(tl.float32)
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
    kbt = tl.trans(kb)
    b_KK = tl.dot(kb, kbt) * (rk[:, None] * rk[None, :])
    b_QK = tl.dot(qb, kbt) * ((rq * scale)[:, None] * rk[None, :])
    dgam = gam[:, None] - gam[None, :]
    X = tl.where(anc, b_beta[:, None] * tl.exp(tl.where(anc, dgam, 0.0)) * b_KK, 0.0)
    P = tl.where(anc_eq, tl.exp(tl.where(anc_eq, dgam, 0.0)) * b_QK, 0.0)
    X2 = tl.dot(X, X, input_precision="tf32x3")
    X4 = tl.dot(X2, X2, input_precision="tf32x3")
    r = tl.where(eye, b_beta[None, :], 0.0)  # diag(beta): M' = (I - X)(I + X^2)(I + X^4)(I + X^8) diag(beta)
    if not SKIP8:
        X8 = tl.dot(X4, X4, input_precision="tf32x3")
        if NP > 16:
            X16 = tl.dot(X8, X8, input_precision="tf32x3")
            r = r + tl.dot(X16, r, input_precision="tf32x3")
        r = r + tl.dot(X8, r, input_precision="tf32x3")
    r = r + tl.dot(X4, r, input_precision="tf32x3")
    r = r + tl.dot(X2, r, input_precision="tf32x3")
    Mp = r - tl.dot(X, r, input_precision="tf32x3")
    W = tl.dot(P, Mp, input_precision="tf32x3")
    wb = ((i_n * HV + i_hv) * NP + tt).to(tl.int64)
    tl.store(w_out + wb[:, None] * NP + tt[None, :], W)
    if STORE_M:
        tl.store(m_out + wb[:, None] * NP + tt[None, :], Mp)
    tl.store(eg_out + wb, tl.exp(gam))
    # commit inputs, as the WY kernel stores them
    row = tl.load(row_ptr + i_n).to(tl.int64)
    sbase = row * N + tt
    o_vr = tl.arange(0, BVR)
    mvr = valid[:, None] & (o_vr < V)[None, :]
    b_vr = tl.load(v + toks[:, None] * stride_v + i_hv * V + o_vr[None, :], mask=mvr, other=0)
    tl.store(v_out + (sbase[:, None] * HV + i_hv) * V + o_vr[None, :], b_vr, mask=mvr)
    tl.store(kn_out + (sbase[:, None] * H + i_h) * K + o_k[None, :], kf * rk[:, None], mask=mk)
    tl.store(g_out + sbase * HV + i_hv, b_g, mask=valid)
    tl.store(beta_out + sbase * HV + i_hv, b_beta, mask=valid)


@triton.jit
def _gdn_tree_wy_state_kernel(
    q, k, v, o, h0_source, h0_indices, stride_h0, w_ptr, eg_ptr, scale, stride_q, stride_k, stride_v,
    out_rows,  # SPECPILOT_DB_RV_FUSE: [bs * N] int64 packed output row per node (-1: not stored); unused otherwise
    N: tl.constexpr, NP: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, STACK: tl.constexpr = False, PACKED_OUT: tl.constexpr = False,
):
    i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_hv // (HV // H)
    tt = tl.arange(0, NP)
    valid = tt < N
    toks = (i_n * N + tt).to(tl.int64)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mk = valid[:, None] & mask_k[None, :]
    mv = valid[:, None] & mask_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    idx = tl.load(h0_indices + i_n).to(tl.int64)
    if idx >= 0:
        p_h0 = h0_source + idx * stride_h0 + i_hv * K * V + o_v[None, :] * K + o_k[:, None]
        b_h += tl.load(p_h0, mask=mask_k[:, None] & mask_v[None, :], other=0).to(tl.float32)
    qb = tl.load(q + toks[:, None] * stride_q + i_h * K + o_k[None, :], mask=mk, other=0)
    kb = tl.load(k + toks[:, None] * stride_k + i_h * K + o_k[None, :], mask=mk, other=0)
    b_vr = tl.load(v + toks[:, None] * stride_v + i_hv * V + o_v[None, :], mask=mv, other=0)
    wb = ((i_n * HV + i_hv) * NP + tt).to(tl.int64)
    W = tl.load(w_ptr + wb[:, None] * NP + tt[None, :])
    egam = tl.load(eg_ptr + wb)
    qf = qb.to(tl.float32)
    kf = kb.to(tl.float32)
    rq = 1.0 / tl.sqrt(tl.sum(qf * qf, 1) + 1e-6)
    rk = 1.0 / tl.sqrt(tl.sum(kf * kf, 1) + 1e-6)
    h_hi = b_h.to(tl.bfloat16)
    r1 = b_h - h_hi.to(tl.float32)
    h_mid = r1.to(tl.bfloat16)
    h_lo = (r1 - h_mid.to(tl.float32)).to(tl.bfloat16)
    if STACK:  # one [2NP, K] x [K, BV] dot per split term instead of two [NP, K] ones
        qk = tl.reshape(tl.permute(tl.join(qb, kb), (2, 0, 1)), (2 * NP, BK))
        res = tl.dot(qk, h_hi) + tl.dot(qk, h_mid) + tl.dot(qk, h_lo)
        b_QH, b_A = tl.split(tl.permute(tl.reshape(res, (2, NP, BV)), (1, 2, 0)))
    else:
        b_A = tl.dot(kb, h_hi) + tl.dot(kb, h_mid) + tl.dot(kb, h_lo)
        b_QH = tl.dot(qb, h_hi) + tl.dot(qb, h_mid) + tl.dot(qb, h_lo)
    u = b_vr.to(tl.float32) - (egam * rk)[:, None] * b_A
    b_o = (egam * rq * scale)[:, None] * b_QH + tl.dot(W, u, input_precision="tf32x3")
    if PACKED_OUT:  # SPECPILOT_DB_RV_FUSE: verified nodes straight into their packed rows
        orow = tl.load(out_rows + toks, mask=valid, other=-1)
        tl.store(
            o + (tl.maximum(orow, 0)[:, None] * HV + i_hv) * V + o_v[None, :],
            b_o.to(o.dtype.element_ty),
            mask=mv & (orow[:, None] >= 0),
        )
    else:
        tl.store(o + (toks[:, None] * HV + i_hv) * V + o_v[None, :], b_o.to(o.dtype.element_ty), mask=mv)


def gdn_tree_wy_split_verify(
    *, A_log, a, dt_bias, q, k, v, b, ssm_states, cache_indices, parents, scratch_rows, kn_buf, v_buf, g_buf,
    beta_buf, n_nodes: int, softplus_beta: float = 1.0, softplus_threshold: float = 20.0, scale=None,
    out_rows=None, out_tokens: int = 0, anc_bits=None, d8r3_ragged=None,
):
    """SPECPILOT_K_WY_SPLIT: same interface and commit inputs as gdn_tree_paths_verify (WY), two kernels."""
    import torch

    _, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    N = int(n_nodes)
    bs = T // N
    NP = max(16, triton.next_power_of_2(N))
    if scale is None:
        scale = K ** -0.5
    packed_out = out_rows is not None  # SPECPILOT_DB_RV_FUSE
    o = q.new_empty(1, int(out_tokens) if packed_out else T, HV, V)
    w = torch.empty(bs, HV, NP, NP, dtype=torch.float32, device=q.device)
    eg = torch.empty(bs, HV, NP, dtype=torch.float32, device=q.device)
    BK = triton.next_power_of_2(K)
    if d8r3_ragged is not None and NP == 16 and anc_bits is not None:  # SPECPILOT_D8R3_RPREP: packed-tile prep
        from sglang.srt.speculative import specpilot_d8r3_ragmix as _d8r3

        _d8r3.ragged_prep_launch(
            A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=b, tiles=d8r3_ragged[0], n_eff=d8r3_ragged[1],
            anc_bits=anc_bits, scratch_rows=scratch_rows, kn_buf=kn_buf, v_buf=v_buf, g_buf=g_buf,
            beta_buf=beta_buf, w=w, eg=eg, scale=scale, softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold, N=N, NP=NP, H=H, HV=HV, K=K, V=V,
        )
    elif _D3_PREP32 and NP == 32 and anc_bits is not None:  # SPECPILOT_D3_PREP32: 2x2 block prep, 17..32 nodes
        from sglang.srt.speculative import specpilot_d3_prep as _d3p

        _d3p.d3_prep32_launch(
            A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=b, anc_bits=anc_bits, scratch_rows=scratch_rows,
            kn_buf=kn_buf, v_buf=v_buf, g_buf=g_buf, beta_buf=beta_buf, w=w, eg=eg, scale=scale,
            softplus_beta=softplus_beta, softplus_threshold=softplus_threshold, N=N, H=H, HV=HV, K=K, V=V,
            warps=_D3_PREP32,
        )
    else:
        _gdn_tree_wy_prep_kernel[(bs, HV)](
            A_log, a, dt_bias, q, k, v, b, parents, scratch_rows, kn_buf, v_buf, g_buf, beta_buf, w, w, eg,
            scale, softplus_beta, softplus_threshold, q.stride(1), k.stride(1), v.stride(1), a.stride(-2),
            b.stride(-2), N=N, NP=NP, H=H, HV=HV, K=K, V=V, BK=BK, BVR=triton.next_power_of_2(V), STORE_M=False,
            SKIP8=_K_WY_NILP and N <= 8,
            anc_ptr=anc_bits if anc_bits is not None else parents, ANC_GIVEN=anc_bits is not None,  # SPECPILOT_DB_RA_WY_ANC
            num_warps=_K_WYS_PREP_WARPS, num_stages=1,
        )
    if _D3_STATE:  # SPECPILOT_D3_STATE: K-chunked pipelined state kernel
        from sglang.srt.speculative import specpilot_d3_state as _d3s

        _d3s.d3_state_launch(q=q, k=k, v=v, o=o, ssm_states=ssm_states, cache_indices=cache_indices, w=w, eg=eg,
                             scale=scale, out_rows=out_rows, packed_out=packed_out, N=N, NP=NP, H=H, HV=HV, K=K, V=V,
                             cfg=_D3_STATE)
        return o
    bvw, n_warps = _K_WYS_CFG[:2] if _K_WYS_CFG else ((128, 4) if bs >= 4 else (64, 4))
    BVW = min(triton.next_power_of_2(V), bvw)
    _gdn_tree_wy_state_kernel[(triton.cdiv(V, BVW), bs, HV)](
        q, k, v, o, ssm_states, cache_indices, ssm_states.stride(0), w, eg, scale,
        q.stride(1), k.stride(1), v.stride(1), out_rows if packed_out else cache_indices,
        N=N, NP=NP, H=H, HV=HV, K=K, V=V, BK=BK, BV=BVW, STACK=_K_WYS_STACK, PACKED_OUT=packed_out,
        num_warps=n_warps, num_stages=1,
    )
    return o
