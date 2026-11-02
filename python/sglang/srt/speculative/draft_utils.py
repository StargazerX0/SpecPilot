from typing import Optional

from sglang.srt.layers.attention.qsa.config import QSAProfile
from sglang.srt.runtime_context import attention_backends, get_spec
from sglang.srt.utils.common import is_blackwell


def _assert_draft_needs_no_conv_sidecar(draft_model_runner) -> None:
    """Refuse a multi-step draft decode backend for a draft with conv layers."""
    from sglang.srt.configs.inkling import InklingMMConfig, InklingModelConfig

    if isinstance(
        draft_model_runner.model_config.hf_config,
        (InklingModelConfig, InklingMMConfig),
    ):
        raise NotImplementedError(
            "Inkling's draft model runs its own short convs, which need the "
            "conv-state sidecar the multi-step draft decode backend cannot carry. "
            "Use --enable-multi-layer-eagle."
        )


class DraftBackendFactory:
    def __init__(
        self,
        draft_model_runner,
        topk: int,
        speculative_num_steps: int,
        seed_dsa_topk_from_draft_extend: bool = False,
        qsa_profile: Optional[QSAProfile] = None,
    ):
        self.draft_model_runner = draft_model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.seed_dsa_topk_from_draft_extend = seed_dsa_topk_from_draft_extend
        if qsa_profile is not None:
            raise NotImplementedError(
                "Qwen sparse attention (QSA) drafts were pruned from this checkout."
            )
        # The draft runner's own backend, not the process-wide config.
        self.draft_attn_backend = draft_model_runner.draft_attention_backend

    def _create_backend(
        self,
        backend_name: str,
        backend_map: dict,
        error_template: str,
        stamps_children: bool = False,
    ):
        # The split pair with the base-backend fallback already applied.
        prefill_backend, decode_backend = attention_backends()
        configured = (
            decode_backend
            if backend_name == "decode_attention_backend"
            else prefill_backend
        )
        backend_type = self.draft_attn_backend or configured

        if backend_type not in backend_map:
            raise ValueError(error_template.format(backend_type=backend_type))

        stamp, backend = backend_map[backend_type]()
        if backend is not None:
            if stamps_children:
                from sglang.srt.layers.attention.attention_registry import (
                    attn_backend_wrapper_for_draft_decode,
                )

                backend = attn_backend_wrapper_for_draft_decode(
                    self.draft_model_runner, backend
                )
            backend.prefill_attention_backend_str = stamp
            backend.decode_attention_backend_str = stamp
            if stamps_children:
                for child in backend.attn_backends:
                    child.prefill_attention_backend_str = stamp
                    child.decode_attention_backend_str = stamp
        return backend

    def create_decode_backend(self):
        # No multi-step draft backend for steps=0 (nospec) or steps=1.
        if self.speculative_num_steps <= 1:
            return None

        # Returns a per-step CONTAINER, not an AttentionBackend, so
        # attn_backend_wrapper_for_draft_extend cannot give it a conv sidecar.
        _assert_draft_needs_no_conv_sidecar(self.draft_model_runner)

        backend_map = {
            "flashinfer": self._create_flashinfer_decode_backend,
            "triton": self._create_triton_decode_backend,
            "fa3": self._create_fa3_decode_backend,
            "hybrid_linear_attn": self._create_hybrid_linear_attn_decode_backend,
        }

        return self._create_backend(
            "decode_attention_backend",
            backend_map,
            "EAGLE is not supported in decode attention backend {backend_type}",
            stamps_children=True,
        )

    def create_draft_extend_backend(self):
        backend_map = {
            "flashinfer": self._create_flashinfer_prefill_backend,
            "triton": self._create_triton_prefill_backend,
            "fa3": self._create_fa3_prefill_backend,
            "hybrid_linear_attn": self._create_hybrid_linear_attn_prefill_backend,
        }
        backend_name = (
            "decode_attention_backend"
            if get_spec().speculative_attention_mode == "decode"
            else "prefill_attention_backend"
        )
        backend = self._create_backend(
            backend_name,
            backend_map,
            "EAGLE is not supported in attention backend {backend_type}",
        )
        # A draft with conv layers of its own (Inkling) needs its sidecar here too.
        from sglang.srt.layers.attention.attention_registry import (
            attn_backend_wrapper_for_draft_extend,
        )

        wrapped = attn_backend_wrapper_for_draft_extend(
            self.draft_model_runner, backend
        )
        if wrapped is not backend and wrapped is not None and backend is not None:
            wrapped.prefill_attention_backend_str = (
                backend.prefill_attention_backend_str
            )
            wrapped.decode_attention_backend_str = backend.decode_attention_backend_str
        return wrapped

    def _create_flashinfer_decode_backend(self):
        if not self.draft_model_runner.use_mla_backend:
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferMultiStepDraftBackend,
            )

            return (
                "flashinfer",
                FlashInferMultiStepDraftBackend(
                    self.draft_model_runner, self.topk, self.speculative_num_steps
                ),
            )
        else:
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAMultiStepDraftBackend,
            )

            return (
                "flashinfer",
                FlashInferMLAMultiStepDraftBackend(
                    self.draft_model_runner, self.topk, self.speculative_num_steps
                ),
            )

    def _create_triton_decode_backend(self):
        from sglang.srt.layers.attention.triton_backend import (
            TritonMultiStepDraftBackend,
        )

        return (
            "triton",
            TritonMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            ),
        )

    def _create_hybrid_linear_attn_decode_backend(self):
        if is_blackwell():
            return self._create_triton_decode_backend()
        return self._create_fa3_decode_backend()

    def _create_hybrid_linear_attn_prefill_backend(self):
        if is_blackwell():
            return self._create_triton_prefill_backend()
        return self._create_fa3_prefill_backend()

    def _create_fa_decode_backend(self, fa_impl_ver: int = 3):
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionMultiStepBackend,
        )

        return (
            f"fa{fa_impl_ver}",
            FlashAttentionMultiStepBackend(
                self.draft_model_runner,
                self.topk,
                self.speculative_num_steps,
                fa_impl_ver=fa_impl_ver,
            ),
        )

    def _create_fa3_decode_backend(self):
        return self._create_fa_decode_backend(fa_impl_ver=3)

    def _create_flashinfer_prefill_backend(self):
        if not self.draft_model_runner.use_mla_backend:
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferAttnBackend,
            )

            return (
                "flashinfer",
                FlashInferAttnBackend(self.draft_model_runner, skip_prefill=False),
            )
        else:
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAAttnBackend,
            )

            return (
                "flashinfer",
                FlashInferMLAAttnBackend(self.draft_model_runner, skip_prefill=False),
            )

    def _create_triton_prefill_backend(self):
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        return (
            "triton",
            TritonAttnBackend(self.draft_model_runner, skip_prefill=False),
        )

    def _create_fa_prefill_backend(self, fa_impl_ver: int = 3):
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )

        return (
            f"fa{fa_impl_ver}",
            FlashAttentionBackend(
                self.draft_model_runner, skip_prefill=False, fa_impl_ver=fa_impl_ver
            ),
        )

    def _create_fa3_prefill_backend(self):
        return self._create_fa_prefill_backend(fa_impl_ver=3)
