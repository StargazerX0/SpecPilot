"""SPECPILOT_D6 (stream G): Muse layer glue provider for the specpilot_muse hook table.

``SPECPILOT_D6_GLUE=1`` binds ``decoder_stack``: the layer loop with each residual join
(``post_attn_norm`` + add + ``post_attention_layernorm``; ``post_ffn_norm`` + add + the next
layer's ``input_layernorm``) done by one fused kernel (kernels/ops/layernorm/specpilot_d6_norm.py),
bitwise equal to stock's rmsnorm -> bf16 add -> rmsnorm. Attention and MLP still run through
``layer.self_attn`` / ``layer.mlp`` (and so through the other streams' hooks). The last layer
fuses post_ffn_norm + add only; the model applies the final norm.

``SPECPILOT_D6_QKR=1`` (with ``SPECPILOT_D6_GLUE=1``) also runs the attention block here: qk_norm (+ RoPE on
sliding layers) + the backend's ``q.contiguous()`` copy become one kernel
(kernels/ops/attention/specpilot_d6_qknorm_rope.py, bitwise); the qkv_gate (W) and attend (A) hooks bound on
the attention module are honoured exactly as in MuseGlimmerAttention.forward. With the knobs unset
``MUSE_HOOKS`` is empty (strict no-op).
"""

from __future__ import annotations

import os

import torch

_GLUE = os.environ.get("SPECPILOT_D6_GLUE", "").strip() not in ("", "0")
_QKR = os.environ.get("SPECPILOT_D6_QKR", "").strip() not in ("", "0")


def _stock_uses_fused_qknorm(attn, q) -> bool:
    """The conditions under which stock apply_qk_norm takes the JIT fused in-place path."""
    from sglang.kernels.ops.layernorm.norm import can_use_fused_inplace_qknorm
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_exec

    return (
        not envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
        and get_exec().graph.cuda_graph_config.prefill.tc_compiler != "inductor"
        and can_use_fused_inplace_qknorm(attn.head_dim, q.dtype)
    )


def _qkr_ok(attn, q, k) -> bool:
    from sglang.kernels.ops.attention import specpilot_d6_qknorm_rope as d6q
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

    rope = attn.rotary_emb
    return (
        attn.qk_norm is not None
        and d6q.eligible(q, k, attn.head_dim)
        and (
            rope is None
            or (
                type(rope) is RotaryEmbedding
                and not rope.use_fallback_kernel
                and rope.head_size == rope.rotary_dim == attn.head_dim
                and rope.cos_sin_cache.dtype == torch.float32
            )
        )
        and _stock_uses_fused_qknorm(attn, q)
    )


def _attention(attn, positions, hidden_states, forward_batch):
    """MuseGlimmerAttention.forward with qk_norm + RoPE + q copy fused (stock fallback when ineligible)."""
    from sglang.kernels.ops.attention import specpilot_d6_qknorm_rope as d6q

    if attn.unfused_qkv:
        return attn(positions, hidden_states, forward_batch)
    projected = None
    if attn._sp_qkv_gate is not None:
        projected = attn._sp_qkv_gate(attn, hidden_states)
    if projected is not None:
        q, k, v, gate = projected
    else:
        gate = None
        qkv, _ = attn.qkv_proj(hidden_states)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)

    if _qkr_ok(attn, q, k):
        rope = attn.rotary_emb
        q = d6q.qknorm_rope(
            q=q,
            k=k,
            q_weight=attn.qk_norm.weight,
            k_weight=attn.qk_norm.weight,
            eps=attn.qk_norm.variance_epsilon,
            positions=positions,
            cos_sin_cache=None if rope is None else rope.cos_sin_cache,
            is_neox=False if rope is None else rope.is_neox_style,
        )
    else:
        from sglang.srt.models.utils import apply_qk_norm

        if attn.qk_norm is not None:
            q, k = apply_qk_norm(q, k, q_norm=attn.qk_norm, k_norm=attn.qk_norm, head_dim=attn.head_dim)
        if attn.rotary_emb is not None:
            q, k = attn.rotary_emb(positions, q, k)

    if attn._sp_attend is not None:
        if attn.use_output_gate and gate is None:
            gate, _ = attn.output_gate_proj(hidden_states)
        gated = attn._sp_attend(attn, q, k, v, gate, forward_batch)
        if gated is not None:
            out, _ = attn.o_proj(gated)
            return out

    attn_out = attn.attn(q, k, v, forward_batch)
    if attn.use_output_gate:
        from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

        if gate is None:
            gate, _ = attn.output_gate_proj(hidden_states)
        attn_out = fused_sigmoid_mul(attn_out, gate, inplace=True)
    out, _ = attn.o_proj(attn_out)
    return out


def _decoder_stack(model, positions, hidden_states, forward_batch):
    from sglang.kernels.ops.layernorm import specpilot_d6_norm as d6n

    layers = model.layers
    if (
        hidden_states.dim() != 2
        or hidden_states.shape[0] == 0
        or not d6n.eligible(hidden_states, hidden_states)
        or any(layer._sp_layer_forward is not None for layer in layers)
    ):
        return None

    capture = set(model.layers_to_capture)
    aux_hidden_states = []
    residual = hidden_states
    x = layers[0].input_layernorm(residual)
    n = len(layers)
    for i, layer in enumerate(layers):
        if i in capture:
            aux_hidden_states.append(residual)
        if _QKR:
            attn_out = _attention(layer.self_attn, positions, x, forward_batch)
        else:
            attn_out = layer.self_attn(positions, x, forward_batch)
        residual, x = d6n.norm_add_norm(
            x=attn_out,
            residual=residual,
            w1=layer.post_attn_norm.weight,
            eps1=layer.post_attn_norm.variance_epsilon,
            w2=layer.post_attention_layernorm.weight,
            eps2=layer.post_attention_layernorm.variance_epsilon,
        )
        mlp_out = layer.mlp(x)
        nxt = layers[i + 1].input_layernorm if i + 1 < n else None
        residual, x = d6n.norm_add_norm(
            x=mlp_out,
            residual=residual,
            w1=layer.post_ffn_norm.weight,
            eps1=layer.post_ffn_norm.variance_epsilon,
            w2=None if nxt is None else nxt.weight,
            eps2=0.0 if nxt is None else nxt.variance_epsilon,
        )
    return residual, aux_hidden_states


MUSE_HOOKS = {"decoder_stack": _decoder_stack} if _GLUE else {}
