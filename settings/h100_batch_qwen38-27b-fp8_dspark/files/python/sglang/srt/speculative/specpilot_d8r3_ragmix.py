"""SPECPILOT_D8R3: ragged GDN mixers for the packed tree verify (SPECPILOT_DB_RV_MODE=pack).

The packed verify runs every token-wise op on the T = sum n_i verified rows, but the seed's tree conv and split-WY prep
still run over the dense [bs, N] mixer layout. Request i's verified nodes are its first n_i nodes (parents precede
children), an ancestor-closed set, so a verified node's conv output and WY prep entries depend on verified rows only.

  SPECPILOT_D8R3_RCONV=1  tm tree conv on the verified rows only (loads, conv output, time-major node inputs). The
                          output is a static per-device buffer shared by the GDN layers (one stream): unverified rows
                          keep stale finite values (zeros at allocation, else an earlier layer's conv output), which the
                          state kernel multiplies only by exact zeros of W.
  SPECPILOT_D8R3_RPREP=1  split-WY prep over 16-row tiles packing consecutive requests (sum n_i <= 16; built once per
                          forward from n_eff). The tile's ancestor matrix is block diagonal, so (I + X)^-1 is block
                          diagonal and one tile computes every packed request exactly; every dot result is re-masked to
                          the blocks (a padding request's garbage rows cannot leak through 0 * NaN). The X^2/X^4/X^8
                          factors are skipped at run time when the tile's longest root path has <= 2/4/8 nodes (the
                          skipped factor is exactly zero). W/eg are stored in the seed layout [bs, HV, 16, 16]:
                          verified rows with zeros in the unverified columns, eg = 1 for unverified rows; W rows of
                          unverified nodes are left undefined (row-local in the state kernel, never stored). Commit
                          inputs only for verified rows; kn once per key head.
  SPECPILOT_D8R3_MINBS=b  RPREP / RSPLIT only for batches of >= b requests (below, a T=192 tier leaves ~1 request
                          per tile: the seed's prep is faster there); RCONV at every packed batch.
  SPECPILOT_D8R3_RSPLIT=1 (needs RCONV): the q/k/v/z/b/a split writes only z (packed) and b/a; the ragged conv
                          reads q/k/v straight from the packed in-projection rows through the unpack map (no dense
                          mixed_qkv copy). b/a of every dense row come through the unpack map as in the seed's split
                          (unverified rows: packed row 0), so a dense prep (the seed's, or SPECPILOT_D3W_GDNF's in-kernel
                          prep warps, int-201) never reads undefined gates. With SPECPILOT_D7R3_ZVIEW z is not copied
                          either (token-strided view of the qkvz rows; int-201).
Unset = the seed's kernels (no new launch, same graphs).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

RCONV = os.environ.get("SPECPILOT_D8R3_RCONV", "").strip() not in ("", "0")
RPREP = os.environ.get("SPECPILOT_D8R3_RPREP", "").strip() not in ("", "0")
RSPLIT = os.environ.get("SPECPILOT_D8R3_RSPLIT", "").strip() not in ("", "0")
if RSPLIT and not RCONV:
    raise ValueError("SPECPILOT_D8R3_RSPLIT needs SPECPILOT_D8R3_RCONV=1")
_MINBS = int(os.environ.get("SPECPILOT_D8R3_MINBS", "0") or 0)  # RPREP/RSPLIT only for graph bs >= MINBS


def rprep_on(bs: int) -> bool:
    return RPREP and bs >= _MINBS


def rsplit_on(bs: int) -> bool:
    return RSPLIT and bs >= _MINBS


_PREP_WARPS = int(os.environ.get("SPECPILOT_D8R3_PREP_WARPS", "1") or 1)
_BKC = int(os.environ.get("SPECPILOT_D8R3_PREP_BKC", "32") or 0)  # K chunk of the prep's q/k pass (0 = whole K)
_MAXNREG = int(os.environ.get("SPECPILOT_D8R3_PREP_MAXNREG", "0") or 0)
_PER_KH = os.environ.get("SPECPILOT_D8R3_PREP_KH", "").strip() not in ("", "0")  # one program per key head

TILE = 16
MAX_REQS = 1024


class _Static:
    tiles = None  # int32 [MAX_REQS * TILE]: dense row (b * N + j) of each tile row, -1 = empty
    conv_out = []  # bf16 flat buffers, newest last (older ones stay alive for the graphs that captured them)
    packed_src = None  # RSPLIT: this layer's packed in-projection rows, set by split_zba, read by the next conv


def tile_buffer(device) -> torch.Tensor:
    if _Static.tiles is None:
        _Static.tiles = torch.full((MAX_REQS * TILE,), -1, dtype=torch.int32, device=device)
    return _Static.tiles


def conv_out_buffer(numel: int, dtype, device) -> torch.Tensor:
    bufs = _Static.conv_out
    if not bufs or bufs[-1].numel() < numel or bufs[-1].dtype != dtype or bufs[-1].device != torch.device(device):
        if torch.device(device).type == "cuda":
            assert not torch.cuda.is_current_stream_capturing(), "SPECPILOT_D8R3_RCONV: buffer must exist before capture"
        bufs.append(torch.zeros(max(numel, 32 * 16 * 10240), dtype=dtype, device=device))
    return bufs[-1]


# ----------------------------------------------------------------------------------------------------------------
# tiles
# ----------------------------------------------------------------------------------------------------------------


@triton.jit
def _d8r3_tiles_kernel(n_ptr, tile_ptr, bs, N: tl.constexpr, TS: tl.constexpr, BT: tl.constexpr):
    o = tl.arange(0, BT)
    for t0 in range(0, bs * TS, BT):
        tl.store(tile_ptr + t0 + o, tl.full([BT], -1, tl.int32), mask=t0 + o < bs * TS)
    tl.debug_barrier()
    jj = tl.arange(0, TS)
    tile = 0
    fill = 0
    for r in range(0, bs):
        n = tl.load(n_ptr + r)
        if fill + n > TS:
            tile += 1
            fill = 0
        tl.store(tile_ptr + tile * TS + fill + jj, (r * N + jj).to(tl.int32), mask=jj < n)
        fill += n


def build_tiles(n_eff: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """In-graph, once per forward: consecutive requests packed first-fit into TILE-row tiles."""
    bs = n_eff.shape[0]
    assert n_nodes <= TILE and bs <= MAX_REQS
    tiles = tile_buffer(n_eff.device)
    _d8r3_tiles_kernel[(1,)](n_eff, tiles, bs, N=n_nodes, TS=TILE, BT=256, num_warps=1)
    return tiles


# ----------------------------------------------------------------------------------------------------------------
# ragged tm conv (copy of specpilot_db_conv._tree_conv_tm_kernel with the verified-row mask)
# ----------------------------------------------------------------------------------------------------------------


@triton.jit
def _d8r3_conv_kernel(
    x_ptr, w_ptr, bias_ptr, cs_ptr, cs_idx_ptr, xin_ptr, xin_idx_ptr, nt_ptr, ns_ptr, par_ptr, o_ptr, n_ptr,
    unpack_ptr,  # PACKED_SRC: [bs * N] dense node -> packed row of x_ptr (row stride sxt); rows j < n_b are off_b + j
    dim, sxb, sxd, sxt, swd, sww, scb, scd, scw, sib, sit, snt, sns, spr, sob, sod, sot,
    N: tl.constexpr, NP: tl.constexpr, BLOCK_N: tl.constexpr, HAS_BIAS: tl.constexpr, SILU: tl.constexpr,
    PARENTS_GIVEN: tl.constexpr = False, anc_ptr=None, STORE_ANC: tl.constexpr = False,
    PACKED_SRC: tl.constexpr = False,
):
    b = tl.program_id(0).to(tl.int64)
    fb = tl.program_id(1)
    slot = tl.load(cs_idx_ptr + b).to(tl.int64)
    if slot < 0:
        return
    feats = fb * BLOCK_N + tl.arange(0, BLOCK_N)
    mf = feats < dim
    tt = tl.arange(0, NP)
    vt = tt < N
    if PARENTS_GIVEN:
        par = tl.load(par_ptr + b * spr + tt, mask=vt, other=0).to(tl.int32)
    else:
        nt = tl.load(nt_ptr + b * snt + tt, mask=vt, other=-1).to(tl.int32)
        ns = tl.load(ns_ptr + b * sns + tt, mask=vt, other=-1).to(tl.int32)
        par = tl.zeros([NP], dtype=tl.int32)
        for i in tl.static_range(N):
            nti = tl.sum(tl.where(tt == i, nt, 0), 0)
            par = tl.where((tt == nti) & (nti != -1), i, par)
            nsi = tl.sum(tl.where(tt == i, ns, 0), 0)
            pi = tl.sum(tl.where(tt == i, par, 0), 0)
            par = tl.where((tt == nsi) & (nsi != -1), pi, par)
        if fb == 0:
            tl.store(par_ptr + b * spr + tt, par.to(par_ptr.dtype.element_ty), mask=vt)
            if STORE_ANC:
                bits = tl.where(tt == 0, 1, 0).to(tl.int32)
                for t in tl.static_range(1, N):
                    pt = tl.sum(tl.where(tt == t, par, 0), 0)
                    bpt = tl.sum(tl.where(tt == pt, bits, 0), 0)
                    bits = tl.where(tt == t, bpt | (1 << t), bits)
                tl.store(anc_ptr + b * N + tt, bits, mask=vt)
    nb = tl.load(n_ptr + b)
    vr = vt & (tt < nb)  # verified rows (ancestor-closed)

    pm = tt[None, :]
    a1 = tl.where(tt > 0, par, tt - 1)
    p_a1 = tl.sum(tl.where(pm == a1[:, None], par[None, :], 0), 1)
    a2 = tl.where(a1 > 0, p_a1, a1 - 1)
    p_a2 = tl.sum(tl.where(pm == a2[:, None], par[None, :], 0), 1)
    a3 = tl.where(a2 > 0, p_a2, a2 - 1)

    cs = cs_ptr + slot * scb + feats * scd
    col0 = tl.load(cs + 0 * scw, mask=mf, other=0.0)
    col1 = tl.load(cs + 1 * scw, mask=mf, other=0.0)
    col2 = tl.load(cs + 2 * scw, mask=mf, other=0.0)
    wb = w_ptr + feats * swd
    w0 = tl.load(wb + 0 * sww, mask=mf, other=0.0)
    w1 = tl.load(wb + 1 * sww, mask=mf, other=0.0)
    w2 = tl.load(wb + 2 * sww, mask=mf, other=0.0)
    w3 = tl.load(wb + 3 * sww, mask=mf, other=0.0)

    m2 = vr[:, None] & mf[None, :]
    if PACKED_SRC:
        # rv_pack stores a request's verified rows consecutively (packed = off_b + j): one scalar load, no gathers
        off = tl.load(unpack_ptr + b * N)
        xb = x_ptr + off * sxt + feats[None, :] * sxd
        r0 = tt
        r1 = tl.maximum(a1, 0)
        r2 = tl.maximum(a2, 0)
        r3 = tl.maximum(a3, 0)
    else:
        xb = x_ptr + b * sxb + feats[None, :] * sxd
        r0 = tt
        r1 = tl.maximum(a1, 0)
        r2 = tl.maximum(a2, 0)
        r3 = tl.maximum(a3, 0)
    x0 = tl.load(xb + r0[:, None] * sxt, mask=m2, other=0.0)
    x1 = tl.load(xb + r1[:, None] * sxt, mask=m2 & (a1 >= 0)[:, None], other=0.0)
    x1 = tl.where((a1 == -1)[:, None], col2[None, :], x1)
    x2 = tl.load(xb + r2[:, None] * sxt, mask=m2 & (a2 >= 0)[:, None], other=0.0)
    x2 = tl.where((a2 == -1)[:, None], col2[None, :], tl.where((a2 == -2)[:, None], col1[None, :], x2))
    x3 = tl.load(xb + r3[:, None] * sxt, mask=m2 & (a3 >= 0)[:, None], other=0.0)
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
    tl.store(o_ptr + b * sob + feats[None, :] * sod + tt[:, None] * sot, acc, mask=m2)
    row = tl.load(xin_idx_ptr + b).to(tl.int64)
    tl.store(xin_ptr + row * sib + tt[:, None] * sit + feats[None, :], x0, mask=m2)


def ragged_conv_verify(
    x, conv_state, weight, bias, activation, *, conv_state_indices, xin, xin_indices, retrieve_next_token,
    retrieve_next_sibling, retrieve_parent_token, n_eff, block_n: int = 128, num_warps: int = 4,
    parents_given: bool = False, anc_out=None, packed_src=None, unpack=None,
):
    """specpilot_db_conv.tree_conv_verify on the verified rows (j < n_eff[b]); returns a (bs, dim, N) view of the static
    output buffer with the strides of torch.empty_like(x). With ``packed_src`` ([T, >= dim] rows, unit column stride)
    and ``unpack`` ([bs * N] dense node -> packed row) the node inputs are read from the packed rows; x gives the
    shape only."""
    bs, dim, N = x.shape
    assert weight.shape == (dim, 4) and conv_state.shape[-1] == 3 and xin.stride(2) == 1
    packed = packed_src is not None
    if packed:
        assert unpack is not None and packed_src.stride(1) == 1 and packed_src.shape[1] >= dim
        src, sxb, sxd, sxt = packed_src, 0, 1, packed_src.stride(0)
    else:
        assert x.stride(1) == 1 and x.stride(2) == dim and x.stride(0) == N * dim, "time-major mixer rows"
        src, sxb, sxd, sxt = x, x.stride(0), x.stride(1), x.stride(2)
    if isinstance(activation, bool):
        activation = "silu" if activation else None
    flat = conv_out_buffer(bs * N * dim, x.dtype, x.device)
    out = flat[: bs * N * dim].as_strided((bs, dim, N), (N * dim, 1, dim))
    grid = (bs, triton.cdiv(dim, block_n))
    _d8r3_conv_kernel[grid](
        src, weight, bias if bias is not None else weight, conv_state, conv_state_indices, xin, xin_indices,
        retrieve_next_token, retrieve_next_sibling, retrieve_parent_token, out, n_eff,
        unpack if packed else n_eff,
        dim,
        sxb, sxd, sxt,
        weight.stride(0), weight.stride(1),
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        xin.stride(0), xin.stride(1),
        retrieve_next_token.stride(0), retrieve_next_sibling.stride(0), retrieve_parent_token.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        N=N, NP=triton.next_power_of_2(N), BLOCK_N=block_n, HAS_BIAS=bias is not None,
        SILU=activation in ("silu", "swish"), PARENTS_GIVEN=parents_given,
        anc_ptr=anc_out if anc_out is not None else retrieve_parent_token,
        STORE_ANC=anc_out is not None and not parents_given, PACKED_SRC=packed,
        num_warps=num_warps,
    )
    return out


@triton.jit
def _d8r3_split_zba_kernel(z, b, a, mixed_qkvz, mixed_ba, qkvz_row_stride, ba_row_stride, rows_out_ptr, unpack_ptr,
                           QKV: tl.constexpr, TOTAL_V: tl.constexpr, HV: tl.constexpr, HVP: tl.constexpr,
                           BLOCK: tl.constexpr, COPY_Z: tl.constexpr = True):
    d = tl.program_id(0).to(tl.int64)
    if COPY_Z:
        dst = tl.load(rows_out_ptr + d)  # verified node: its packed row (written to z at the same row)
        if dst >= 0:
            in_row = mixed_qkvz + dst * qkvz_row_stride
            for c0 in tl.static_range(0, TOTAL_V, BLOCK):
                o = c0 + tl.arange(0, BLOCK)
                m = o < TOTAL_V
                tl.store(z + dst * TOTAL_V + o, tl.load(in_row + QKV + o, mask=m), mask=m)
    src = tl.load(unpack_ptr + d)  # int-201: b/a of every dense row as the seed's split (unverified -> packed row 0)
    oh = tl.arange(0, HVP)
    mh = oh < HV
    ba_row = mixed_ba + src * ba_row_stride
    tl.store(b + d * HV + oh, tl.load(ba_row + oh, mask=mh), mask=mh)
    tl.store(a + d * HV + oh, tl.load(ba_row + HV + oh, mask=mh), mask=mh)


def take_packed_src() -> torch.Tensor:
    src, _Static.packed_src = _Static.packed_src, None
    assert src is not None, "SPECPILOT_D8R3_RSPLIT: the layer's split did not run (rv_fuse split path off?)"
    return src


def split_zba(p, mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v, zview: bool = False):
    """RSPLIT: specpilot_rv.split_qkvzba_dense without the dense q/k/v copy (the returned mixed_qkv is a placeholder
    of the dense shape that nothing writes or reads); z and b/a as the seed writes them. ``zview`` (SPECPILOT_D7R3_ZVIEW):
    z is not copied, the token-strided view mixed_qkvz[:, QKV:] is returned instead (equal on every verified row)."""
    rows = p.bs * p.n_nodes
    T = mixed_qkvz.shape[0]
    qkv = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    tv = num_heads_v * head_v
    assert mixed_qkvz.stride(1) == 1
    _Static.packed_src = mixed_qkvz
    mixed_qkv = mixed_qkvz.new_empty((rows, qkv))
    if zview:
        assert mixed_qkvz.shape[1] == qkv + tv
        z = mixed_qkvz[:, qkv:].view(T, num_heads_v, head_v)
    else:
        z = mixed_qkvz.new_empty((T, num_heads_v, head_v))
    b = mixed_ba.new_empty((rows, num_heads_v))
    a = torch.empty_like(b)
    _d8r3_split_zba_kernel[(rows,)](z, b, a, mixed_qkvz, mixed_ba, mixed_qkvz.stride(0), mixed_ba.stride(0),
                                    p.rows_out, p.unpack, QKV=qkv, TOTAL_V=tv, HV=num_heads_v,
                                    HVP=triton.next_power_of_2(num_heads_v), BLOCK=2048, COPY_Z=not zview,
                                    num_warps=1 if zview else 4)
    return mixed_qkv, z, b, a


# ----------------------------------------------------------------------------------------------------------------
# ragged split-WY prep
# ----------------------------------------------------------------------------------------------------------------


@triton.jit
def _blk_dot(x, y, blk, MASKDOT: tl.constexpr = True):
    if MASKDOT:
        return tl.where(blk, tl.dot(x, y, input_precision="tf32x3"), 0.0)
    return tl.dot(x, y, input_precision="tf32x3")


@triton.jit
def _d8r3_prep_kernel(
    A_log, a, dt_bias, q, k, v, b, tile_ptr, n_ptr, anc_ptr, row_ptr,
    kn_out, v_out, g_out, beta_out, w_out, eg_out,
    scale, softplus_beta, softplus_threshold, stride_q, stride_k, stride_v, stride_a, stride_b,
    N: tl.constexpr, NP: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BVR: tl.constexpr, NILP: tl.constexpr, KN_ONCE: tl.constexpr, PER_KH: tl.constexpr,
    MASKDOT: tl.constexpr = True, BKC: tl.constexpr = 0,
):
    # PER_KH: one program per (tile, key head) over its HV // H value heads (q/k loads and KK/QK shared);
    # else one program per (tile, value head) as the seed's prep.
    i_t = tl.program_id(0)
    first = tl.load(tile_ptr + i_t * NP)
    if first < 0:
        return
    GROUP: tl.constexpr = HV // H
    if PER_KH:
        i_h = tl.program_id(1)
    else:
        i_h = tl.program_id(1) // GROUP
    tt = tl.arange(0, NP)
    trow = tl.load(tile_ptr + i_t * NP + tt)
    valid = trow >= 0
    trow = tl.maximum(trow, 0)
    rq_i = trow // N  # request of each tile row
    jn = trow % N  # node index inside the request
    eye = tt[:, None] == tt[None, :]
    blk = (rq_i[:, None] == rq_i[None, :]) & valid[:, None] & valid[None, :]
    bits = tl.load(anc_ptr + trow, mask=valid, other=0)
    anc_eq = blk & (((bits[:, None] >> jn[None, :]) & 1) != 0)
    anc = anc_eq & (~eye)
    toks = trow.to(tl.int64)
    if NILP:
        depth = tl.max(tl.sum(anc_eq.to(tl.int32), 1), 0)  # nodes on the longest root path: X^depth = 0
    nreq = tl.load(n_ptr + rq_i, mask=valid, other=0)
    row = tl.load(row_ptr + rq_i, mask=valid, other=0).to(tl.int64)
    sbase = row * N + jn
    kn_row = valid
    if not PER_KH:
        if KN_ONCE:
            kn_row = valid & (tl.program_id(1) % GROUP == 0)
    if BKC > 0:  # K-chunked: [NP, BKC] operands only (register pressure at 1 warp), same fp32 MMA accumulation
        b_KK = tl.zeros([NP, NP], dtype=tl.float32)
        b_QK = tl.zeros([NP, NP], dtype=tl.float32)
        ssq = tl.zeros([NP], dtype=tl.float32)
        ssk = tl.zeros([NP], dtype=tl.float32)
        for kc in tl.static_range(0, K, BKC):
            o_k = kc + tl.arange(0, BKC)
            mk = valid[:, None] & (o_k < K)[None, :]
            qb = tl.load(q + toks[:, None] * stride_q + i_h * K + o_k[None, :], mask=mk, other=0)
            kb = tl.load(k + toks[:, None] * stride_k + i_h * K + o_k[None, :], mask=mk, other=0)
            kbt = tl.trans(kb)
            b_KK = tl.dot(kb, kbt, b_KK)
            b_QK = tl.dot(qb, kbt, b_QK)
            qf = qb.to(tl.float32)
            kf = kb.to(tl.float32)
            ssq += tl.sum(qf * qf, 1)
            ssk += tl.sum(kf * kf, 1)
        rq = 1.0 / tl.sqrt(ssq + 1e-6)
        rk = 1.0 / tl.sqrt(ssk + 1e-6)
        for kc in tl.static_range(0, K, BKC):
            o_k = kc + tl.arange(0, BKC)
            mk = valid[:, None] & (o_k < K)[None, :]
            kf = tl.load(k + toks[:, None] * stride_k + i_h * K + o_k[None, :], mask=mk, other=0).to(tl.float32)
            tl.store(kn_out + (sbase[:, None] * H + i_h) * K + o_k[None, :], kf * rk[:, None],
                     mask=kn_row[:, None] & (o_k < K)[None, :])
        b_KK = b_KK * (rk[:, None] * rk[None, :])
        b_QK = b_QK * ((rq * scale)[:, None] * rk[None, :])
    else:
        o_k = tl.arange(0, BK)
        mask_k = o_k < K
        mk = valid[:, None] & mask_k[None, :]
        qb = tl.load(q + toks[:, None] * stride_q + i_h * K + o_k[None, :], mask=mk, other=0)
        kb = tl.load(k + toks[:, None] * stride_k + i_h * K + o_k[None, :], mask=mk, other=0)
        qf = qb.to(tl.float32)
        kf = kb.to(tl.float32)
        rq = 1.0 / tl.sqrt(tl.sum(qf * qf, 1) + 1e-6)
        rk = 1.0 / tl.sqrt(tl.sum(kf * kf, 1) + 1e-6)
        tl.store(kn_out + (sbase[:, None] * H + i_h) * K + o_k[None, :], kf * rk[:, None],
                 mask=kn_row[:, None] & mask_k[None, :])
        kbt = tl.trans(kb)
        b_KK = tl.dot(kb, kbt) * (rk[:, None] * rk[None, :])
        b_QK = tl.dot(qb, kbt) * ((rq * scale)[:, None] * rk[None, :])
    o_vr = tl.arange(0, BVR)
    mvr = valid[:, None] & (o_vr < V)[None, :]
    for gi in tl.static_range(GROUP if PER_KH else 1):
        if PER_KH:
            i_hv = i_h * GROUP + gi
        else:
            i_hv = tl.program_id(1)
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
        gam = tl.sum(tl.where(anc_eq, b_g[None, :], 0.0), 1)
        dgam = gam[:, None] - gam[None, :]
        X = tl.where(anc, b_beta[:, None] * tl.exp(tl.where(anc, dgam, 0.0)) * b_KK, 0.0)
        P = tl.where(anc_eq, tl.exp(tl.where(anc_eq, dgam, 0.0)) * b_QK, 0.0)
        if not MASKDOT:  # finite operands: a padding request's NaN rows cannot leak across blocks (0 * x = 0)
            X = tl.where(tl.abs(X) < 1e30, X, 0.0)
            P = tl.where(tl.abs(P) < 1e30, P, 0.0)
        r = tl.where(eye & (tl.abs(b_beta) < 1e30)[None, :], b_beta[None, :], 0.0)  # M' = (I - X)(I + X^2)(I + X^4)(I + X^8) diag(beta)
        if NILP:
            if depth > 2:
                X2 = _blk_dot(X, X, blk, MASKDOT)
                if depth > 4:
                    X4 = _blk_dot(X2, X2, blk, MASKDOT)
                    if depth > 8:
                        X8 = _blk_dot(X4, X4, blk, MASKDOT)
                        r = r + _blk_dot(X8, r, blk, MASKDOT)
                    r = r + _blk_dot(X4, r, blk, MASKDOT)
                r = r + _blk_dot(X2, r, blk, MASKDOT)
        else:
            X2 = _blk_dot(X, X, blk, MASKDOT)
            X4 = _blk_dot(X2, X2, blk, MASKDOT)
            X8 = _blk_dot(X4, X4, blk, MASKDOT)
            r = r + _blk_dot(X8, r, blk, MASKDOT)
            r = r + _blk_dot(X4, r, blk, MASKDOT)
            r = r + _blk_dot(X2, r, blk, MASKDOT)
        Mp = r - _blk_dot(X, r, blk, MASKDOT)
        W = _blk_dot(P, Mp, blk, MASKDOT)
        wrow = ((rq_i * HV + i_hv) * NP + jn).to(tl.int64)
        tl.store(w_out + wrow[:, None] * NP + jn[None, :], W, mask=blk)
        tl.store(eg_out + wrow, tl.exp(gam), mask=valid)
        tl.store(w_out + wrow[:, None] * NP + tt[None, :], tl.zeros([NP, NP], tl.float32),
                 mask=valid[:, None] & (tt[None, :] >= nreq[:, None]))
        ebase = ((rq_i * HV + i_hv) * NP).to(tl.int64)
        tl.store(eg_out + ebase[:, None] + tt[None, :], tl.full([NP, NP], 1.0, tl.float32),
                 mask=(valid & (jn == 0))[:, None] & (tt[None, :] >= nreq[:, None]))
        # commit inputs of the verified rows, as the seed's prep stores them
        b_vr = tl.load(v + toks[:, None] * stride_v + i_hv * V + o_vr[None, :], mask=mvr, other=0)
        tl.store(v_out + (sbase[:, None] * HV + i_hv) * V + o_vr[None, :], b_vr, mask=mvr)
        tl.store(g_out + sbase * HV + i_hv, b_g, mask=valid)
        tl.store(beta_out + sbase * HV + i_hv, b_beta, mask=valid)


def ragged_prep_launch(
    *, A_log, a, dt_bias, q, k, v, b, tiles, n_eff, anc_bits, scratch_rows, kn_buf, v_buf, g_buf, beta_buf, w, eg,
    scale, softplus_beta, softplus_threshold, N, NP, H, HV, K, V, nilp: bool = True, kn_once: bool = True,
    warps: int = 0, per_kh=None, maskdot: bool = True, bkc: int = -1, maxnreg: int = -1,
):
    """Drop-in for the seed's _gdn_tree_wy_prep_kernel launch (gdn_tree_wy_split_verify) on the packed tiles."""
    assert NP == TILE and N <= TILE
    bs = n_eff.shape[0]
    per_kh = _PER_KH if per_kh is None else per_kh
    _d8r3_prep_kernel[(bs, H if per_kh else HV)](
        A_log, a, dt_bias, q, k, v, b, tiles, n_eff, anc_bits, scratch_rows,
        kn_buf, v_buf, g_buf, beta_buf, w, eg,
        scale, softplus_beta, softplus_threshold, q.stride(1), k.stride(1), v.stride(1), a.stride(-2), b.stride(-2),
        N=N, NP=NP, H=H, HV=HV, K=K, V=V, BK=triton.next_power_of_2(K), BVR=triton.next_power_of_2(V),
        NILP=nilp, KN_ONCE=kn_once, PER_KH=per_kh, MASKDOT=maskdot, BKC=_BKC if bkc < 0 else bkc, num_warps=warps or _PREP_WARPS, num_stages=1,
        **({"maxnreg": mr} if (mr := (_MAXNREG if maxnreg < 0 else maxnreg)) > 0 else {}),
    )
