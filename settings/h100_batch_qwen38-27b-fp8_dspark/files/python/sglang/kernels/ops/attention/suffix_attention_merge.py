"""Fused attention over a short sparse suffix and merge with a prefix state."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def can_use_fused_suffix_attention_merge(
    *,
    layer,
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    extra_kwargs: dict,
) -> bool:
    """Whether attention can use the specialized suffix merge."""
    return bool(
        q.dtype in (torch.float16, torch.bfloat16)
        and key_cache.dtype == q.dtype
        and value_cache.dtype == q.dtype
        and layer.head_dim == layer.v_head_dim
        and not layer.is_cross_attention
        and not layer.logit_cap
        and not extra_kwargs
    )


@triton.jit
def _fused_suffix_attention_merge_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr,
    suffix_seqlens_ptr,
    prefix_ptr,
    prefix_lse_ptr,
    scale,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    k_stride_t,
    k_stride_h,
    k_stride_d,
    v_stride_t,
    v_stride_h,
    v_stride_d,
    page_stride_t,
    page_stride_s,
    prefix_stride_t,
    prefix_stride_h,
    prefix_stride_d,
    prefix_lse_stride_h,
    prefix_lse_stride_t,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SUFFIX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // (NUM_Q_HEADS // NUM_KV_HEADS)

    suffix_offsets = tl.arange(0, BLOCK_SUFFIX)
    suffix_length = tl.load(suffix_seqlens_ptr + token)
    if suffix_length == 0:
        # SPECPILOT_DB_RV_SUFFIX_SKIP: no visible suffix -> the merge leaves the prefix output as is (exact)
        return
    suffix_valid = suffix_offsets < suffix_length
    slots = tl.load(
        page_table_ptr + token * page_stride_t + suffix_offsets * page_stride_s,
        mask=suffix_valid,
        other=0,
    ).to(tl.int64)

    dims = tl.arange(0, BLOCK_D)
    dim_valid = dims < HEAD_DIM
    q = tl.load(
        q_ptr + token * q_stride_t + q_head * q_stride_h + dims * q_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        k_cache_ptr
        + slots[:, None] * k_stride_t
        + kv_head * k_stride_h
        + dims[None, :] * k_stride_d,
        mask=suffix_valid[:, None] & dim_valid[None, :],
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(k * q[None, :], axis=1) * scale
    scores = tl.where(suffix_valid, scores, -float("inf"))
    suffix_max = tl.max(scores, axis=0)

    prefix_lse = tl.load(
        prefix_lse_ptr + q_head * prefix_lse_stride_h + token * prefix_lse_stride_t
    ).to(tl.float32)
    global_max = tl.maximum(prefix_lse, suffix_max)
    prefix_weight = tl.exp(prefix_lse - global_max)
    suffix_weights = tl.exp(scores - global_max)
    denominator = prefix_weight + tl.sum(suffix_weights, axis=0)

    v = tl.load(
        v_cache_ptr
        + slots[:, None] * v_stride_t
        + kv_head * v_stride_h
        + dims[None, :] * v_stride_d,
        mask=suffix_valid[:, None] & dim_valid[None, :],
        other=0.0,
    ).to(tl.float32)
    suffix_numerator = tl.sum(suffix_weights[:, None] * v, axis=0)
    prefix = tl.load(
        prefix_ptr
        + token * prefix_stride_t
        + q_head * prefix_stride_h
        + dims * prefix_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    output = (prefix * prefix_weight + suffix_numerator) / denominator
    tl.store(
        prefix_ptr
        + token * prefix_stride_t
        + q_head * prefix_stride_h
        + dims * prefix_stride_d,
        output,
        mask=dim_valid,
    )


# SPECPILOT_K_SUFFIX_GROUP=1: one program per (query, kv head) covering the KV group's q heads (one K/V gather per
# group instead of one per q head); read once at import.
_K_SUFFIX_GROUP = os.environ.get("SPECPILOT_K_SUFFIX_GROUP", "0") not in ("", "0")
# SPECPILOT_K_SFX_DOT=1 (with the group kernel): q.K^T and P.V as tensor-core dots (reduction order differs).
_K_SFX_DOT = os.environ.get("SPECPILOT_K_SFX_DOT", "0") not in ("", "0")


@triton.jit
def _grouped_suffix_attention_merge_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, page_table_ptr, suffix_seqlens_ptr, prefix_ptr, prefix_lse_ptr, scale,
    q_stride_t, q_stride_h, q_stride_d, k_stride_t, k_stride_h, k_stride_d, v_stride_t, v_stride_h, v_stride_d,
    page_stride_t, page_stride_s, prefix_stride_t, prefix_stride_h, prefix_stride_d, prefix_lse_stride_h,
    prefix_lse_stride_t,
    GROUP: tl.constexpr, BLOCK_G: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_SUFFIX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    kv_head = tl.program_id(1)
    gg = tl.arange(0, BLOCK_G)
    g_valid = gg < GROUP
    q_head = kv_head * GROUP + tl.where(g_valid, gg, 0)
    suffix_offsets = tl.arange(0, BLOCK_SUFFIX)
    suffix_length = tl.load(suffix_seqlens_ptr + token)
    if suffix_length == 0:
        # SPECPILOT_DB_RV_SUFFIX_SKIP: no visible suffix -> the merge leaves the prefix output as is (exact)
        return
    suffix_valid = suffix_offsets < suffix_length
    slots = tl.load(page_table_ptr + token * page_stride_t + suffix_offsets * page_stride_s, mask=suffix_valid,
                    other=0).to(tl.int64)
    dims = tl.arange(0, BLOCK_D)
    dim_valid = dims < HEAD_DIM
    qm = g_valid[:, None] & dim_valid[None, :]
    q = tl.load(q_ptr + token * q_stride_t + q_head[:, None] * q_stride_h + dims[None, :] * q_stride_d, mask=qm,
                other=0.0).to(tl.float32)  # [G, D]
    kvm = suffix_valid[:, None] & dim_valid[None, :]
    k = tl.load(k_cache_ptr + slots[:, None] * k_stride_t + kv_head * k_stride_h + dims[None, :] * k_stride_d,
                mask=kvm, other=0.0).to(tl.float32)  # [S, D]
    scores = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale  # [G, S]
    scores = tl.where(suffix_valid[None, :], scores, -float("inf"))
    suffix_max = tl.max(scores, axis=1)
    prefix_lse = tl.load(prefix_lse_ptr + q_head * prefix_lse_stride_h + token * prefix_lse_stride_t).to(tl.float32)
    global_max = tl.maximum(prefix_lse, suffix_max)
    prefix_weight = tl.exp(prefix_lse - global_max)
    suffix_weights = tl.exp(scores - global_max[:, None])
    denominator = prefix_weight + tl.sum(suffix_weights, axis=1)
    v = tl.load(v_cache_ptr + slots[:, None] * v_stride_t + kv_head * v_stride_h + dims[None, :] * v_stride_d,
                mask=kvm, other=0.0).to(tl.float32)
    suffix_numerator = tl.sum(suffix_weights[:, :, None] * v[None, :, :], axis=1)  # [G, D]
    p_pre = prefix_ptr + token * prefix_stride_t + q_head[:, None] * prefix_stride_h + dims[None, :] * prefix_stride_d
    prefix = tl.load(p_pre, mask=qm, other=0.0).to(tl.float32)
    output = (prefix * prefix_weight[:, None] + suffix_numerator) / denominator[:, None]
    tl.store(p_pre, output, mask=qm)


@triton.jit
def _grouped_suffix_attention_merge_dot_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, page_table_ptr, suffix_seqlens_ptr, prefix_ptr, prefix_lse_ptr, scale,
    q_stride_t, q_stride_h, q_stride_d, k_stride_t, k_stride_h, k_stride_d, v_stride_t, v_stride_h, v_stride_d,
    page_stride_t, page_stride_s, prefix_stride_t, prefix_stride_h, prefix_stride_d, prefix_lse_stride_h,
    prefix_lse_stride_t,
    GROUP: tl.constexpr, BLOCK_G: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_SUFFIX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # SPECPILOT_K_SFX_DOT: as _grouped_suffix_attention_merge_kernel, with q.K^T and P.V as tensor-core dots on a
    # head group padded to BLOCK_G >= 16 rows (fp32 accumulation; P.V in tf32x3)
    token = tl.program_id(0)
    kv_head = tl.program_id(1)
    gg = tl.arange(0, BLOCK_G)
    g_valid = gg < GROUP
    q_head = kv_head * GROUP + tl.where(g_valid, gg, 0)
    suffix_offsets = tl.arange(0, BLOCK_SUFFIX)
    suffix_length = tl.load(suffix_seqlens_ptr + token)
    if suffix_length == 0:
        return
    suffix_valid = suffix_offsets < suffix_length
    slots = tl.load(page_table_ptr + token * page_stride_t + suffix_offsets * page_stride_s, mask=suffix_valid,
                    other=0).to(tl.int64)
    dims = tl.arange(0, BLOCK_D)
    dim_valid = dims < HEAD_DIM
    qm = g_valid[:, None] & dim_valid[None, :]
    q = tl.load(q_ptr + token * q_stride_t + q_head[:, None] * q_stride_h + dims[None, :] * q_stride_d, mask=qm,
                other=0.0)  # [G, D]
    kvm = suffix_valid[:, None] & dim_valid[None, :]
    k = tl.load(k_cache_ptr + slots[:, None] * k_stride_t + kv_head * k_stride_h + dims[None, :] * k_stride_d,
                mask=kvm, other=0.0)  # [S, D]
    scores = tl.dot(q, tl.trans(k)) * scale  # [G, S] fp32
    scores = tl.where(suffix_valid[None, :], scores, -float("inf"))
    suffix_max = tl.max(scores, axis=1)
    prefix_lse = tl.load(prefix_lse_ptr + q_head * prefix_lse_stride_h + token * prefix_lse_stride_t).to(tl.float32)
    global_max = tl.maximum(prefix_lse, suffix_max)
    prefix_weight = tl.exp(prefix_lse - global_max)
    suffix_weights = tl.exp(scores - global_max[:, None])
    denominator = prefix_weight + tl.sum(suffix_weights, axis=1)
    v = tl.load(v_cache_ptr + slots[:, None] * v_stride_t + kv_head * v_stride_h + dims[None, :] * v_stride_d,
                mask=kvm, other=0.0).to(tl.float32)
    suffix_numerator = tl.dot(suffix_weights, v, input_precision="tf32x3")  # [G, D]
    p_pre = prefix_ptr + token * prefix_stride_t + q_head[:, None] * prefix_stride_h + dims[None, :] * prefix_stride_d
    prefix = tl.load(p_pre, mask=qm, other=0.0).to(tl.float32)
    output = (prefix * prefix_weight[:, None] + suffix_numerator) / denominator[:, None]
    tl.store(p_pre, output, mask=qm)


def merge_suffix_attention_in_place(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    suffix_page_table: torch.Tensor,
    suffix_cache_seqlens: torch.Tensor,
    prefix: torch.Tensor,
    prefix_lse: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Compute a short sparse suffix and merge it into ``prefix`` in place.

    ``prefix_lse`` uses FlashAttention's varlen layout ``[num_q_heads,
    num_queries]``. The suffix page table contains physical token slots, one
    row per query, and only its first ``suffix_cache_seqlens[row]`` entries are
    visible.
    """
    if q.ndim != 3:
        raise ValueError("q must have shape [num_queries, num_q_heads, head_dim]")
    num_queries, num_q_heads, head_dim = q.shape
    if prefix.shape != q.shape:
        raise ValueError("prefix output must have the same shape as q")
    if prefix_lse.shape != (num_q_heads, num_queries):
        raise ValueError("prefix_lse must have shape [num_q_heads, num_queries]")
    if k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("flattened KV caches must have shape [slots, heads, dim]")
    if k_cache.shape != v_cache.shape:
        raise ValueError("K and V caches must have matching shapes")
    num_kv_heads = k_cache.shape[1]
    if k_cache.shape[2] != head_dim:
        raise ValueError("K/V and query head dimensions must match")
    if num_q_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if suffix_page_table.ndim != 2 or suffix_page_table.shape[0] != num_queries:
        raise ValueError("suffix page table must have one row per query")
    if suffix_cache_seqlens.numel() != num_queries:
        raise ValueError("suffix cache lengths must have one value per query")
    if suffix_page_table.shape[1] == 0 or num_queries == 0:
        return prefix

    if _K_SUFFIX_GROUP and _K_SFX_DOT and q.dtype == k_cache.dtype and q.dtype in (torch.bfloat16, torch.float16):
        group = num_q_heads // num_kv_heads
        _grouped_suffix_attention_merge_dot_kernel[(num_queries, num_kv_heads)](
            q, k_cache, v_cache, suffix_page_table, suffix_cache_seqlens, prefix, prefix_lse, softmax_scale,
            q.stride(0), q.stride(1), q.stride(2), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), suffix_page_table.stride(0),
            suffix_page_table.stride(1), prefix.stride(0), prefix.stride(1), prefix.stride(2), prefix_lse.stride(0),
            prefix_lse.stride(1),
            GROUP=group, BLOCK_G=max(16, triton.next_power_of_2(group)), HEAD_DIM=head_dim,
            BLOCK_SUFFIX=max(16, triton.next_power_of_2(suffix_page_table.shape[1])),
            BLOCK_D=triton.next_power_of_2(head_dim), num_warps=4, num_stages=1,
        )
        return prefix
    if _K_SUFFIX_GROUP:  # SPECPILOT_K_SUFFIX_GROUP
        group = num_q_heads // num_kv_heads
        _grouped_suffix_attention_merge_kernel[(num_queries, num_kv_heads)](
            q, k_cache, v_cache, suffix_page_table, suffix_cache_seqlens, prefix, prefix_lse, softmax_scale,
            q.stride(0), q.stride(1), q.stride(2), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), suffix_page_table.stride(0),
            suffix_page_table.stride(1), prefix.stride(0), prefix.stride(1), prefix.stride(2), prefix_lse.stride(0),
            prefix_lse.stride(1),
            GROUP=group, BLOCK_G=triton.next_power_of_2(group), HEAD_DIM=head_dim,
            BLOCK_SUFFIX=triton.next_power_of_2(suffix_page_table.shape[1]), BLOCK_D=triton.next_power_of_2(head_dim),
            num_warps=4, num_stages=1,
        )
        return prefix
    _fused_suffix_attention_merge_kernel[(num_queries, num_q_heads)](
        q,
        k_cache,
        v_cache,
        suffix_page_table,
        suffix_cache_seqlens,
        prefix,
        prefix_lse,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        suffix_page_table.stride(0),
        suffix_page_table.stride(1),
        prefix.stride(0),
        prefix.stride(1),
        prefix.stride(2),
        prefix_lse.stride(0),
        prefix_lse.stride(1),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SUFFIX=triton.next_power_of_2(suffix_page_table.shape[1]),
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
        num_stages=1,
    )
    return prefix
