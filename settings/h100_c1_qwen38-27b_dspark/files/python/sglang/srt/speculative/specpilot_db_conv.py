"""SPECPILOT_DB_RA_CONV: time-major tree conv for the GDN verify (budget family, db-3).

L's tree conv (``causal_conv1d_update`` with the EAGLE-tree retrieve buffers) walks the N nodes of a request
sequentially per feature block, re-loads every ancestor row per tap, rewrites the dense conv state in place, and stores
a per-node conv window ``[slot, N, dim, K-1]`` (dim-major: three strided stores per node and feature block, 3x the
bytes of the node input). The commit then copies the accepted node's window into ``conv``.

Here, per (request, feature block):
  * parents are rebuilt exactly as the stock kernel does (first-child / next-sibling walk; root parent = 0) and stored
    to ``retrieve_parent_token`` (the WY verify reads them);
  * the three ancestors of every node are computed as [NP] vectors, the node inputs and their ancestors are loaded as
    [NP, BLOCK_N] tiles and the 4-tap conv + SiLU is evaluated for all nodes at once, in the stock kernel's
    accumulation order (bias, then taps newest -> oldest), so the output is bitwise equal;
  * instead of the per-node windows, the node INPUT rows are stored time-major ``xin[row, node, dim]`` (one contiguous
    store, 1/3 of the window bytes), reusing the start of the window buffer's storage (no new allocation);
  * the dense conv-state rewrite is skipped (the commit overwrites the state of every verified request).
Commit (``tree_conv_commit``): new state = the last K-1 inputs along the accepted root path
``[.., x[path[a-1]], x[path[a]]]``, falling back to the old state columns above the root: bitwise equal to the window
copy. One launch over (request, feature block, layer).
"""
from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl


@functools.lru_cache(maxsize=None)
def conv_tm_enabled() -> bool:
    mode = os.environ.get("SPECPILOT_DB_RA_CONV", "").strip().lower()
    if mode in ("", "0", "off", "none"):
        return False
    if mode != "tm":
        raise ValueError(f"SPECPILOT_DB_RA_CONV={mode!r}: expected 'tm' or unset")
    return True


@functools.lru_cache(maxsize=None)
def nosplit_enabled() -> bool:
    """SPECPILOT_DB_RA_NOSPLIT=1: the tree WY verify takes q/k/v as views of the post-conv mixed_qkv (no split copy)."""
    return os.environ.get("SPECPILOT_DB_RA_NOSPLIT", "").strip() not in ("", "0", "off")


@functools.lru_cache(maxsize=None)
def par1_enabled() -> bool:
    """SPECPILOT_DB_RA_CONV_PAR1=1: rebuild the tree parents in the first GDN layer of a verify only; later layers load
    them from retrieve_parent_token (identical for every layer of the forward)."""
    return os.environ.get("SPECPILOT_DB_RA_CONV_PAR1", "").strip() not in ("", "0", "off")


_FIRST_GDN_LAYER = [None]


def parents_given_for_layer(layer_id: int) -> bool:
    """True for every GDN layer but the first one of the forward (layers run in order; the first call of the first
    forward, before any graph capture, records the lowest GDN layer id)."""
    if not par1_enabled():
        return False
    if _FIRST_GDN_LAYER[0] is None:
        _FIRST_GDN_LAYER[0] = int(layer_id)
    return int(layer_id) != _FIRST_GDN_LAYER[0]


def xin_view(win: torch.Tensor) -> torch.Tensor:
    """[S, N, dim, K-1] per-layer window buffer -> [S, N, dim] time-major node-input view of its leading storage."""
    S, N, dim, _ = win.shape
    assert win.is_contiguous()
    return win.view(-1)[: S * N * dim].view(S, N, dim)


@triton.jit
def _tree_conv_tm_kernel(
    x_ptr,  # node inputs, element (b, d, t) at b * sxb + d * sxd + t * sxt
    w_ptr,  # [dim, 4]
    bias_ptr,
    cs_ptr,  # conv state [slots, dim, 3]
    cs_idx_ptr,  # [bs] conv state slot
    xin_ptr,  # [rows, N, dim] time-major node inputs (out)
    xin_idx_ptr,  # [bs] row
    nt_ptr,  # [bs, N] first child
    ns_ptr,  # [bs, N] next sibling
    par_ptr,  # [bs, N] parent (out; root = 0 as in the stock kernel)
    o_ptr,  # out, element (b, d, t) at b * sob + d * sod + t * sot
    dim,
    sxb,
    sxd,
    sxt,
    swd,
    sww,
    scb,
    scd,
    scw,
    sib,
    sit,
    snt,
    sns,
    spr,
    sob,
    sod,
    sot,
    N: tl.constexpr,
    NP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    PARENTS_GIVEN: tl.constexpr = False,  # par_ptr already holds this step's parents (root = 0): skip the rebuild
):
    b = tl.program_id(0).to(tl.int64)
    fb = tl.program_id(1)
    slot = tl.load(cs_idx_ptr + b).to(tl.int64)
    if slot < 0:  # graph padding (PAD_SLOT_ID): skipped like the stock kernel (no parents, no output)
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

    # ancestors (stock walk semantics): a > 0 -> parent(a); a <= 0 -> a - 1 (0 = root input, -1/-2/-3 = state cols)
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

    xb = x_ptr + b * sxb + feats[None, :] * sxd
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
    tl.store(o_ptr + b * sob + feats[None, :] * sod + tt[:, None] * sot, acc, mask=m2)
    row = tl.load(xin_idx_ptr + b).to(tl.int64)
    tl.store(xin_ptr + row * sib + tt[:, None] * sit + feats[None, :], x0, mask=m2)


def tree_conv_verify(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias,
    activation,
    *,
    conv_state_indices: torch.Tensor,
    xin: torch.Tensor,
    xin_indices: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    retrieve_parent_token: torch.Tensor,
    block_n: int = 128,
    num_warps: int = 4,
    parents_given: bool = False,
) -> torch.Tensor:
    """Drop-in for the tree branch of ``causal_conv1d_update`` (x: (bs, dim, N) view; width 4; returns (bs, dim, N)
    with the same strides as ``torch.empty_like(x)``). Writes ``xin[xin_indices[b], :N]`` and the parents."""
    bs, dim, N = x.shape
    assert weight.shape == (dim, 4), "width-4 conv only"
    assert conv_state.shape[-1] == 3 and xin.shape[1] == N and xin.shape[2] == dim and xin.stride(2) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation else None
    out = torch.empty_like(x)
    grid = (bs, triton.cdiv(dim, block_n))
    _tree_conv_tm_kernel[grid](
        x, weight, bias if bias is not None else weight, conv_state, conv_state_indices, xin, xin_indices,
        retrieve_next_token, retrieve_next_sibling, retrieve_parent_token, out,
        dim,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(0), weight.stride(1),
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        xin.stride(0), xin.stride(1),
        retrieve_next_token.stride(0), retrieve_next_sibling.stride(0), retrieve_parent_token.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        N=N, NP=triton.next_power_of_2(N), BLOCK_N=block_n, HAS_BIAS=bias is not None,
        SILU=activation in ("silu", "swish"),
        PARENTS_GIVEN=parents_given,
        num_warps=num_warps,
    )
    return out


@triton.jit
def _tree_conv_commit_kernel(
    cs_ptr,  # conv state [L, slots, dim, 3]
    xin_ptr,  # window storage [L, rows*N*dim*3] (time-major node inputs at its start)
    slot_ptr,  # [bs] conv state slot
    path_ptr,  # [bs, N] node per depth
    alen_ptr,  # [bs] accepted depth a (leaf = path[a])
    dim,
    scl,
    scb,
    scd,
    scw,
    sxl,
    sxr,
    sxt,
    spb,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    layer = tl.program_id(2).to(tl.int64)
    mf = feats < dim
    slot = tl.load(slot_ptr + b).to(tl.int64)
    if slot < 0:
        return
    a = tl.load(alen_ptr + b).to(tl.int32)
    cs = cs_ptr + layer * scl + slot * scb + feats * scd
    o0 = tl.load(cs + 0 * scw, mask=mf, other=0.0)
    o1 = tl.load(cs + 1 * scw, mask=mf, other=0.0)
    o2 = tl.load(cs + 2 * scw, mask=mf, other=0.0)
    xr = xin_ptr + layer * sxl + b * sxr + feats
    n_a = tl.load(path_ptr + b * spb + a).to(tl.int64)
    n_a1 = tl.load(path_ptr + b * spb + tl.maximum(a - 1, 0)).to(tl.int64)
    n_a2 = tl.load(path_ptr + b * spb + tl.maximum(a - 2, 0)).to(tl.int64)
    v2 = tl.load(xr + n_a * sxt, mask=mf, other=0.0)
    v1 = tl.load(xr + n_a1 * sxt, mask=mf & (a >= 1), other=0.0)
    v1 = tl.where(a >= 1, v1, o2)
    v0 = tl.load(xr + n_a2 * sxt, mask=mf & (a >= 2), other=0.0)
    v0 = tl.where(a >= 2, v0, tl.where(a == 1, o2, o1))
    tl.store(cs + 0 * scw, v0, mask=mf)
    tl.store(cs + 1 * scw, v1, mask=mf)
    tl.store(cs + 2 * scw, v2, mask=mf)


def tree_conv_commit(
    conv: torch.Tensor,
    win: torch.Tensor,
    state_indices: torch.Tensor,
    path: torch.Tensor,
    accept_len: torch.Tensor,
    block_n: int = 256,
) -> None:
    """conv [L, slots, dim, 3]; win [L, rows, N, dim, 3] (the window buffer whose leading storage holds the time-major
    node inputs of the last verify, row = batch position); path [bs, N]; accept_len [bs]."""
    L, _, dim, K1 = conv.shape
    assert K1 == 3 and win.shape[0] == L and win.is_contiguous()
    rows, N = win.shape[1], win.shape[2]
    bs = path.shape[0]
    if bs == 0:
        return
    grid = (bs, triton.cdiv(dim, block_n), L)
    _tree_conv_commit_kernel[grid](
        conv, win, state_indices, path, accept_len,
        dim,
        conv.stride(0), conv.stride(1), conv.stride(2), conv.stride(3),
        win.stride(0), N * dim, dim,
        path.stride(0),
        N=N, BLOCK_N=block_n,
        num_warps=4,
    )
