# SPECPILOT_X3: token-parallel causal conv1d for the target-verify pass (chain and tree).
"""Target-verify conv1d (kernel width 4, state_len 3) without the per-token loop of
``_causal_conv1d_update_kernel``: each program owns BLOCK_N channels of one request
and computes every verify token at once from gathered tiles. In tree mode the
ancestors come from the EAGLE-style next_token / next_sibling links, resolved with
parents resolved by
pointer jumping (same result as the stock kernel's serial walk). The input may be a strided view of the
in_proj output (any token / sequence stride, unit channel stride), so the
split/reshape/cat copy is not needed. Products and accumulation order match the
stock kernel in both modes, so the outputs, intermediate conv windows, in-place
conv_state write and retrieve_parent_token are bitwise equal to stock."""

import os

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl

# 0 (unset) keeps the stock causal_conv1d_update path.
X3_CONV_BLOCK_N = int(os.environ.get("SPECPILOT_X3_CONV_BN", "0") or 0)
X3_CONV_WARPS = int(os.environ.get("SPECPILOT_X3_CONV_WARPS", "2"))


@triton.jit
def _x3_gather(vals, idx, positions):
    # vals[idx[i]] for small register vectors (idx entries outside positions give 0).
    hit = positions[None, :] == idx[:, None]
    return tl.sum(tl.where(hit, vals[None, :], 0), axis=1)


@triton.jit
def _x3_rows(x_base, stride_x_token, a, m_tok, mask_f, c0, c1, c2):
    # Value of "token" a per row: a >= 0 -> x[a]; -1 -> state col2; -2 -> col1; -3 -> col0.
    v = tl.load(
        x_base + tl.maximum(a, 0)[:, None] * stride_x_token,
        (m_tok & (a >= 0))[:, None] & mask_f[None, :],
        0.0,
    )
    aa = a[:, None]
    return tl.where(
        aa >= 0,
        v,
        tl.where(aa == -1, c2[None, :], tl.where(aa == -2, c1[None, :], c0[None, :])),
    )


@triton.jit
def _x3_conv_verify_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    inter_ptr,
    inter_indices_ptr,
    next_token_ptr,
    next_sibling_ptr,
    parent_ptr,
    o_ptr,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    stride_x_seq,
    stride_x_token,
    stride_o_seq,
    stride_o_token,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_cs_seq,
    stride_cs_dim: tl.constexpr,
    stride_cs_tok: tl.constexpr,
    stride_inter_seq,
    stride_inter_step: tl.constexpr,
    stride_inter_dim: tl.constexpr,
    stride_inter_win: tl.constexpr,
    stride_tree_seq,
    stride_tree_tok,
    stride_par_seq,
    stride_par_tok,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_TREE: tl.constexpr,
    NP2_SEQLEN: tl.constexpr,
    LOG2_SEQLEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    idx_seq = tl.program_id(0)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_f = idx_feats < dim

    slot = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    if slot == pad_slot_id:
        return
    inter_slot = tl.load(inter_indices_ptr + idx_seq).to(tl.int64)

    cs_base = conv_state_ptr + slot * stride_cs_seq + idx_feats * stride_cs_dim
    col0 = tl.load(cs_base, mask_f, 0.0)
    col1 = tl.load(cs_base + 1 * stride_cs_tok, mask_f, 0.0)
    col2 = tl.load(cs_base + 2 * stride_cs_tok, mask_f, 0.0)

    t = tl.arange(0, NP2_SEQLEN)
    mt = t < seqlen
    m2 = mt[:, None] & mask_f[None, :]
    x_base = x_ptr + idx_seq * stride_x_seq + idx_feats[None, :]

    if IS_TREE:
        nt = tl.load(
            next_token_ptr + idx_seq * stride_tree_seq + t * stride_tree_tok, mt, -1
        )
        ns = tl.load(
            next_sibling_ptr + idx_seq * stride_tree_seq + t * stride_tree_tok, mt, -1
        )
        # Parents by pointer jumping (no serial walk): a first child's parent is the
        # node linking to it; any other node inherits its first sibling's parent.
        hit_fc = (nt[:, None] == t[None, :]) & mt[:, None]
        fc_par = tl.sum(tl.where(hit_fc, t[:, None] + 1, 0), axis=0) - 1
        hit_ps = (ns[:, None] == t[None, :]) & mt[:, None]
        prev_sib = tl.sum(tl.where(hit_ps, t[:, None] + 1, 0), axis=0) - 1
        ptr = tl.where((fc_par >= 0) | (prev_sib < 0), t, prev_sib)
        for _ in tl.static_range(LOG2_SEQLEN):
            ptr = _x3_gather(ptr, ptr, t)
        par = tl.maximum(_x3_gather(fc_par, ptr, t), 0).to(tl.int32)
        if tl.program_id(1) == 0:
            tl.store(
                parent_ptr + idx_seq * stride_par_seq + t * stride_par_tok, par, mt
            )
        a1 = tl.where(t > 0, _x3_gather(par, t, t), -1)
        a2 = tl.where(a1 > 0, _x3_gather(par, a1, t), a1 - 1)
        a3 = tl.where(a2 > 0, _x3_gather(par, a2, t), a2 - 1)
    else:
        a1 = t - 1
        a2 = t - 2
        a3 = t - 3
    x0 = tl.load(x_base + t[:, None] * stride_x_token, m2, 0.0)
    xm1 = _x3_rows(x_base, stride_x_token, a1, mt, mask_f, col0, col1, col2)
    xm2 = _x3_rows(x_base, stride_x_token, a2, mt, mask_f, col0, col1, col2)
    xm3 = _x3_rows(x_base, stride_x_token, a3, mt, mask_f, col0, col1, col2)

    # In-place conv_state write of the stock kernel (state <- last 3 inputs by index).
    tl.debug_barrier()
    for i in tl.static_range(3):
        if i + seqlen >= 3:
            src = tl.load(
                x_ptr
                + idx_seq * stride_x_seq
                + (i + seqlen - 3) * stride_x_token
                + idx_feats,
                mask_f,
                0.0,
            )
            tl.store(cs_base + i * stride_cs_tok, src, mask_f)
        else:
            if i + seqlen == 1:
                tl.store(cs_base + i * stride_cs_tok, col1, mask_f)
            else:
                tl.store(cs_base + i * stride_cs_tok, col2, mask_f)

    w_base = w_ptr + idx_feats * stride_w_dim
    w0 = tl.load(w_base, mask_f, 0.0)[None, :]
    w1 = tl.load(w_base + 1 * stride_w_width, mask_f, 0.0)[None, :]
    w2 = tl.load(w_base + 2 * stride_w_width, mask_f, 0.0)[None, :]
    w3 = tl.load(w_base + 3 * stride_w_width, mask_f, 0.0)[None, :]
    acc = tl.zeros((NP2_SEQLEN, BLOCK_N), dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + idx_feats, mask_f, 0.0).to(tl.float32)[None, :]
    if IS_TREE:
        # Stock tree walk order: self, parent, grand-parent, great-grand-parent.
        acc += x0 * w3
        acc += xm1 * w2
        acc += xm2 * w1
        acc += xm3 * w0
    else:
        acc += xm3 * w0
        acc += xm2 * w1
        acc += xm1 * w2
        acc += x0 * w3
    if SILU_ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))

    o_ptrs = (
        o_ptr
        + idx_seq * stride_o_seq
        + t[:, None] * stride_o_token
        + idx_feats[None, :]
    )
    tl.store(o_ptrs, acc, m2)

    # Window after token t: (x[a2], x[a1], x[t]); layout [seq, step, dim, win].
    i_base = (
        inter_ptr
        + inter_slot * stride_inter_seq
        + t[:, None] * stride_inter_step
        + idx_feats[None, :] * stride_inter_dim
    )
    tl.store(i_base, xm2, m2)
    tl.store(i_base + 1 * stride_inter_win, xm1, m2)
    tl.store(i_base + 2 * stride_inter_win, x0, m2)


def x3_conv_verify_eligible(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    cache_seqlens,
    conv_state_indices,
    num_accept_tokens,
    intermediate_conv_window,
    intermediate_state_indices,
    retrieve_next_token,
    retrieve_parent_token,
) -> bool:
    return (
        X3_CONV_BLOCK_N > 0
        and x.dim() == 3
        and x.stride(1) == 1
        and weight.shape[1] == 4
        and conv_state.shape[-1] == 3
        and cache_seqlens is None
        and num_accept_tokens is None
        and conv_state_indices is not None
        and intermediate_conv_window is not None
        and intermediate_state_indices is not None
        and (retrieve_next_token is None or retrieve_parent_token is not None)
    )


def x3_conv_verify(
    *,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias,
    silu: bool,
    conv_state_indices: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    retrieve_next_token,
    retrieve_next_sibling,
    retrieve_parent_token,
    pad_slot_id: int,
) -> torch.Tensor:
    """x: (batch, dim, seqlen) with unit channel stride. Returns (batch, dim, seqlen)
    as a transposed view of a contiguous (batch, seqlen, dim) buffer."""
    batch, dim, seqlen = x.shape
    out = torch.empty((batch, seqlen, dim), dtype=x.dtype, device=x.device)
    out = out.transpose(1, 2)
    is_tree = retrieve_next_token is not None
    tree_ref = retrieve_next_token if is_tree else conv_state_indices
    sib_ref = retrieve_next_sibling if is_tree else conv_state_indices
    par_ref = retrieve_parent_token if is_tree else conv_state_indices
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    grid = (batch, triton.cdiv(dim, X3_CONV_BLOCK_N))
    _x3_conv_verify_kernel[grid](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        tree_ref,
        sib_ref,
        par_ref,
        out,
        dim,
        seqlen,
        x.stride(0),
        x.stride(2),
        out.stride(0),
        out.stride(2),
        weight.stride(0),
        weight.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        intermediate_conv_window.stride(0),
        intermediate_conv_window.stride(1),
        intermediate_conv_window.stride(2),
        intermediate_conv_window.stride(3),
        tree_ref.stride(0) if is_tree else 0,
        tree_ref.stride(1) if is_tree else 0,
        par_ref.stride(0) if is_tree else 0,
        par_ref.stride(1) if is_tree else 0,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=silu,
        IS_TREE=is_tree,
        NP2_SEQLEN=triton.next_power_of_2(seqlen),
        LOG2_SEQLEN=max(1, (triton.next_power_of_2(seqlen) - 1).bit_length()),
        BLOCK_N=X3_CONV_BLOCK_N,
        num_warps=X3_CONV_WARPS,
        **pdl_kwargs,
    )
    return out
