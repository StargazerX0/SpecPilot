import logging
from typing import TYPE_CHECKING

from sglang.srt.configs.hybrid_arch import hybrid_gdn_config, mambaish_config
from sglang.srt.configs.linear_attn_model_registry import (
    get_linear_attn_config,
    import_backend_class,
)
from sglang.srt.runtime_context import get_spec
from sglang.srt.utils import get_device_capability

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    # evade circular imports
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.model_runner import ModelRunner

ATTENTION_BACKENDS = {}


def register_attention_backend(name):
    def decorator(fn):
        ATTENTION_BACKENDS[name] = fn
        return fn

    return decorator


# Backends kept in this checkout (NVIDIA): fa3 (default for Qwen3.8 + DFlash2 on
# SM90), flashinfer and triton as alternatives, torch_native for debugging.
# Out-of-tree backends can still be added with @register_attention_backend.


@register_attention_backend("flashinfer")
def create_flashinfer_backend(runner):
    import torch

    if runner.use_mla_backend:
        raise ValueError("MLA attention backends were pruned from this checkout.")
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

    # Init streams
    if get_spec().speculative_algorithm == "EAGLE":
        if (
            not hasattr(runner, "plan_stream_for_flashinfer")
            or not runner.plan_stream_for_flashinfer
        ):
            runner.plan_stream_for_flashinfer = torch.cuda.Stream()
    return FlashInferAttnBackend(runner, init_new_workspace=runner.init_new_workspace)


@register_attention_backend("triton")
def create_triton_backend(runner):
    assert not runner.model_config.is_encoder_decoder, (
        "Cross attention is not supported in the triton attention backend. "
        "Please use `--attention-backend flashinfer`."
    )
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    return TritonAttnBackend(runner)


@register_attention_backend("torch_native")
def create_torch_native_backend(runner):
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

    return TorchNativeAttnBackend(runner)


@register_attention_backend("fa3")
def create_flashattention_v3_backend(runner):
    major, _ = get_device_capability()
    assert (major == 8 and not runner.use_mla_backend) or major == 9, (
        "FlashAttention v3 Backend requires SM>=80 and SM<=90. "
        "Please use `--attention-backend flashinfer`."
    )
    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionBackend,
    )

    return FlashAttentionBackend(runner)


def attn_backend_wrapper_for_draft_extend(
    runner: "ModelRunner", full_attn_backend: "AttentionBackend"
):
    """Apply the model's attention wrapper to a DRAFT-EXTEND backend, if it needs one.

    ``DraftBackendFactory`` skips :func:`attn_backend_wrapper`, which is right for
    hybrids whose draft is all softmax attention (the DFlash2 drafter included).
    """
    return full_attn_backend


def attn_backend_wrapper_for_draft_decode(runner: "ModelRunner", backend):
    """Per-step draft backends need no model-specific wrapper in this checkout."""
    return backend


def attn_backend_wrapper(runner: "ModelRunner", full_attn_backend: "AttentionBackend"):
    """
    Wrapper for hybrid linear-attention models (Qwen3.5/3.8 hybrid GDN), so we
    don't need to change the code of the original attention backend.
    """
    assert not (
        hybrid_gdn_config(runner.model_config) is not None and runner.use_mla_backend
    ), "hybrid_gdn can only be used with non-MLA models."

    if cfg := mambaish_config(runner.model_config):
        from sglang.kernels.ops.attention.fla.utils import check_environments
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )
        from sglang.srt.layers.attention.linear.gdn_backend import (
            GDNAttnBackend,
            flashinfer_gdn_prefill_default,
        )
        from sglang.srt.layers.attention.linear.utils import (
            resolve_linear_attn_backends,
        )

        check_environments()
        prefill_default = None
        if hybrid_gdn_config(runner.model_config) is not None:
            prefill_default = flashinfer_gdn_prefill_default(runner)
        runner.linear_attn_backends = resolve_linear_attn_backends(
            prefill_default=prefill_default
        )
        hybrid_backend_cls = HybridLinearAttnBackend
        if hybrid_gdn_config(runner.model_config) is not None:
            logger.info(f"Using hybrid linear attention backend for hybrid GDN models.")
            linear_attn_backend = GDNAttnBackend(runner)
        else:
            spec_result = get_linear_attn_config(runner.model_config.hf_config)
            if spec_result is not None:
                spec, _ = spec_result
                cfg = runner.model_config
                BackendClass = import_backend_class(spec.backend_class_name)
                linear_attn_backend = BackendClass(runner)
                if spec.hybrid_backend_class_name is not None:
                    hybrid_backend_cls = import_backend_class(
                        spec.hybrid_backend_class_name
                    )
            else:
                raise ValueError(
                    "Only hybrid GDN linear-attention models are kept in this checkout. "
                    "For a custom hybrid model, use register_linear_attn_model() "
                    "from sglang.srt.configs.linear_attn_model_registry."
                )
        if runner.is_draft_worker:
            # FIXME: we assume that MTP/NEXTN always use full-attention.
            full_attn_layers = [0]
        else:
            full_attn_layers = cfg.full_attention_layer_ids
        return hybrid_backend_cls(
            full_attn_backend, linear_attn_backend, full_attn_layers
        )

    return full_attn_backend
