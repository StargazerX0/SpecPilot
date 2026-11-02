"""SPECPILOT_D4R3_HOST: one-launch FA3 topk>1 target-verify replay metadata.

The seed builds the topk-normal (prefix) cache_seqlens / cu_seqlens_k and the expand (tree suffix) page_table /
cache_seqlens / cu_seqlens_k with ~30 eager torch launches between the drafter and the verify graph replays
(arange, repeat_interleave, cumsum, tree_mask_rows, where, sort, gathers, sum, copies). This kernel writes the same
values in one launch. The sort keys (j for an ancestor column, j + N otherwise) are distinct, so the sorted order is
exactly "ancestor columns ascending, then the rest ascending": a per-row prefix count gives each column's rank.
Graph padding rows (request >= tree_mask.shape[0]) see only themselves, as in tree_mask_rows.
"""

import os

import torch
import triton
import triton.language as tl


def host_enabled() -> bool:
    return os.environ.get("SPECPILOT_D4R3_HOST", "0") == "1"


@triton.jit
def _verify_metadata_kernel(
    seq_lens_ptr,
    pool_ptr,
    r2t_ptr,
    r2t_stride,
    mask_ptr,
    mask_sb,
    mask_si,
    num_tree_reqs,
    normal_cache_ptr,
    normal_cu_ptr,
    expand_pt_ptr,
    expand_pt_stride,
    expand_cache_ptr,
    expand_cu_ptr,
    bs,
    N: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    carry = tl.zeros((), dtype=tl.int32)
    for start in range(0, bs, BLOCK_B):
        offs = start + tl.arange(0, BLOCK_B)
        valid = offs < bs
        sl = tl.load(seq_lens_ptr + offs, mask=valid, other=0).to(tl.int32)
        tl.store(normal_cache_ptr + offs, sl, mask=valid)
        tl.store(normal_cu_ptr + 1 + offs, tl.cumsum(sl, 0) + carry, mask=valid)
        carry += tl.sum(sl, 0)

    j = tl.arange(0, N)
    carry = tl.zeros((), dtype=tl.int32)
    for start in range(0, bs * N, BLOCK_R):
        r = start + tl.arange(0, BLOCK_R)
        rv = r < bs * N
        b = r // N
        i = r % N
        in_tree = rv & (b < num_tree_reqs)
        m_tree = tl.load(
            mask_ptr + b[:, None].to(tl.int64) * mask_sb + i[:, None] * mask_si + j[None, :],
            mask=in_tree[:, None],
            other=0,
        )
        m = tl.where(in_tree[:, None], m_tree != 0, j[None, :] == i[:, None])
        mi = m.to(tl.int32)
        cnt = tl.sum(mi, 1)
        before = tl.cumsum(mi, 1) - mi
        rank = tl.where(m, before, cnt[:, None] + j[None, :] - before)
        sl = tl.load(seq_lens_ptr + b, mask=rv, other=0).to(tl.int64)
        pool = tl.load(pool_ptr + b, mask=rv, other=0).to(tl.int64)
        tok = tl.load(
            r2t_ptr + pool[:, None] * r2t_stride + sl[:, None] + j[None, :],
            mask=rv[:, None],
            other=0,
        )
        tl.store(
            expand_pt_ptr + r[:, None].to(tl.int64) * expand_pt_stride + rank,
            tok,
            mask=rv[:, None],
        )
        tl.store(expand_cache_ptr + r, cnt, mask=rv)
        tl.store(expand_cu_ptr + 1 + r, tl.cumsum(cnt, 0) + carry, mask=rv)
        carry += tl.sum(cnt, 0)


def write_verify_metadata(
    *,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    tree_mask: torch.Tensor,
    normal_cache_seqlens: torch.Tensor,
    normal_cu_seqlens_k: torch.Tensor,
    expand_page_table: torch.Tensor,
    expand_cache_seqlens: torch.Tensor,
    expand_cu_seqlens_k: torch.Tensor,
) -> None:
    """tree_mask: [b, N, N] bool ancestor-or-self, b <= bs = seq_lens.numel(); expand_page_table [bs * N, N] int32.
    Writes normal cache_seqlens [bs], cu_seqlens_k[1:bs + 1], expand cache_seqlens [bs * N], cu_seqlens_k[1:]."""
    bs = seq_lens.numel()
    num_tree_reqs, n, _ = tree_mask.shape
    assert expand_page_table.shape == (bs * n, n) and expand_page_table.stride(1) == 1
    assert req_to_token.stride(1) == 1 and tree_mask.stride(2) == 1
    mask = tree_mask.view(torch.uint8) if tree_mask.dtype == torch.bool else tree_mask
    _verify_metadata_kernel[(1,)](
        seq_lens,
        req_pool_indices,
        req_to_token,
        req_to_token.stride(0),
        mask,
        mask.stride(0),
        mask.stride(1),
        num_tree_reqs,
        normal_cache_seqlens,
        normal_cu_seqlens_k,
        expand_page_table,
        expand_page_table.stride(0),
        expand_cache_seqlens,
        expand_cu_seqlens_k,
        bs,
        N=n,
        BLOCK_B=64,
        BLOCK_R=128,
        num_warps=4,
    )
