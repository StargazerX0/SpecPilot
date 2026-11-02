# SPECPILOT_X3: snapshot-free chain GDN verify (ring of raw inputs + replay commit).
"""The chain target-verify runs the seed's fused_sigmoid_gating recurrent kernel
without per-step fp32 state snapshots; it stores each step's raw inputs (k, v in
the activation dtype; g, beta in fp32) to a per-slot ring through the kernel's
existing CACHE_RING path. After accept, ``x3_gdn_replay_commit`` replays the
accepted prefix of every GDN layer into ``temporal`` in one launch with the
verify kernel's arithmetic, which reproduces the snapshot the stock commit would
have copied bit for bit."""

import os

import torch
import triton
import triton.language as tl

X3_GDN_REPLAY = os.environ.get("SPECPILOT_X3_GDN_REPLAY", "0") == "1"

_rings = {}


def x3_gdn_rings(*, num_layers, slots, h, hv, k, v, steps, dtype, device):
    """Per-layer ring tensors [L, slots, heads, steps, dim]; allocated once."""
    key = (num_layers, slots, h, hv, k, v, steps, dtype, str(device))
    if key not in _rings:
        _rings.clear()
        _rings[key] = (
            torch.zeros(num_layers, slots, hv, steps, v, dtype=dtype, device=device),
            torch.zeros(num_layers, slots, h, steps, k, dtype=dtype, device=device),
            torch.zeros(num_layers, slots, hv, steps, dtype=torch.float32, device=device),
            torch.zeros(num_layers, slots, hv, steps, dtype=torch.float32, device=device),
        )
    return _rings[key]


def x3_gdn_current_rings():
    return next(iter(_rings.values())) if _rings else None


@triton.jit
def _x3_gdn_replay_kernel(
    h_ptr,
    rawv_ptr,
    rawk_ptr,
    g_ptr,
    beta_ptr,
    slot_ptr,
    last_ptr,
    stride_h_layer,
    stride_h_slot,
    stride_rv_layer,
    stride_rv_slot,
    stride_rk_layer,
    stride_rk_slot,
    stride_g_layer,
    stride_g_slot,
    STEPS: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_n, i_lh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_l = i_lh // HV
    i_hv = i_lh % HV
    i_h = i_hv // (HV // H)
    slot = tl.load(slot_ptr + i_n).to(tl.int64)
    if slot < 0:
        return
    n_steps = tl.load(last_ptr + i_n).to(tl.int32) + 1
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    p_h = (
        h_ptr
        + i_l.to(tl.int64) * stride_h_layer
        + slot * stride_h_slot
        + i_hv * K * V
        + o_v[None, :] * K
        + o_k[:, None]
    )
    b_h = tl.load(p_h).to(tl.float32)
    rv = rawv_ptr + i_l.to(tl.int64) * stride_rv_layer + slot * stride_rv_slot + i_hv * STEPS * V + o_v
    rk = rawk_ptr + i_l.to(tl.int64) * stride_rk_layer + slot * stride_rk_slot + i_h * STEPS * K + o_k
    pg = g_ptr + i_l.to(tl.int64) * stride_g_layer + slot * stride_g_slot + i_hv * STEPS
    pb = beta_ptr + i_l.to(tl.int64) * stride_g_layer + slot * stride_g_slot + i_hv * STEPS
    for t in range(0, n_steps):
        b_k = tl.load(rk + t * K).to(tl.float32)
        b_v = tl.load(rv + t * V).to(tl.float32)
        b_g = tl.load(pg + t)
        b_beta = tl.load(pb + t)
        # Same op sequence as fused_sigmoid_gating_delta_rule_update_kernel.
        b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
    tl.store(p_h, b_h.to(p_h.dtype.element_ty))


def x3_gdn_replay_commit(
    *,
    temporal: torch.Tensor,
    state_indices: torch.Tensor,
    last_correct_step_indices: torch.Tensor,
    bv: int,
    num_warps: int,
) -> None:
    """temporal: [L, slots, HV, V, K] fp32 (K contiguous, as the verify kernel reads it)."""
    rawv, rawk, g, beta = x3_gdn_current_rings()
    num_layers, _, hv, steps, v = rawv.shape
    h, k = rawk.shape[2], rawk.shape[4]
    n = state_indices.shape[0]
    grid = (triton.cdiv(v, bv), n, num_layers * hv)
    _x3_gdn_replay_kernel[grid](
        temporal,
        rawv,
        rawk,
        g,
        beta,
        state_indices,
        last_correct_step_indices,
        temporal.stride(0),
        temporal.stride(1),
        rawv.stride(0),
        rawv.stride(1),
        rawk.stride(0),
        rawk.stride(1),
        g.stride(0),
        g.stride(1),
        STEPS=steps,
        H=h,
        HV=hv,
        K=k,
        V=v,
        BV=bv,
        num_warps=num_warps,
        num_stages=3,
    )


def x3_gdn_replay_active() -> bool:
    return X3_GDN_REPLAY and bool(_rings)


def x3_gdn_commit_after_verify(*, mamba_caches, state_indices, last_correct_step_indices) -> None:
    """Replay commit of the SSM state for all GDN layers, then the stock conv-window scatter."""
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        x3_verify_launch_config,
    )
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        _conv_multi_eligible,
        fused_conv_window_scatter_multi,
        fused_conv_window_scatter_with_mask,
    )

    bv, warps = x3_verify_launch_config()
    x3_gdn_replay_commit(
        temporal=mamba_caches.temporal,
        state_indices=state_indices,
        last_correct_step_indices=last_correct_step_indices,
        bv=bv,
        num_warps=warps,
    )
    pairs = list(zip(mamba_caches.conv, mamba_caches.intermediate_conv_window))
    if not pairs:
        return
    if _conv_multi_eligible(pairs):
        fused_conv_window_scatter_multi(pairs, state_indices, last_correct_step_indices, None, None)
        return
    for conv_states, inter in pairs:
        fused_conv_window_scatter_with_mask(conv_states, inter, state_indices, last_correct_step_indices)
