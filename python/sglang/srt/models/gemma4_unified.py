# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gemma 4 "unified" checkpoints (e.g. google/gemma-4-12B-it), text only.

The unified checkpoints keep the Gemma 4 text decoder unchanged
(model.language_model.*: sliding-window + global layers, attention_k_eq_v,
final logit softcap, tied embeddings) and replace the ViT vision tower with a
patch embedder (model.vision_embedder.*) plus an audio projection
(model.embed_audio.*). This class serves the text decoder only: the
multimodal encoders are not built and their weights are skipped, and
multimodal is disabled for this architecture in ModelConfig. Everything else
(forward, EAGLE3 / DFlash aux-hidden capture, weight loading of the decoder)
is inherited from Gemma4ForConditionalGeneration.
"""

import logging
from typing import Iterable, Optional, Tuple

import torch

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.gemma4_causal import Gemma4TextModel
from sglang.srt.models.gemma4_mm import (
    Gemma4ForConditionalGeneration,
    _is_cpu,
    _is_cpu_amx_available,
)
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

# checkpoint prefixes of the unified multimodal encoders (not served)
_MM_PREFIXES = (
    "model.vision_embedder.",
    "model.embed_vision.",
    "model.embed_audio.",
    "model.audio_tower.",
    "model.vision_tower.",
)


class Gemma4UnifiedForConditionalGeneration(Gemma4ForConditionalGeneration):
    """Text-only serving of a Gemma 4 unified checkpoint."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Skip Gemma4ForConditionalGeneration.__init__ (it builds the ViT
        # tower from a vision_config the unified checkpoints do not have).
        super(Gemma4ForConditionalGeneration, self).__init__(config=config)
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        text_config = config.text_config
        prefix = add_prefix("model", prefix)

        self.vision_tower = None
        self.embed_vision = None
        self.audio_tower = None
        self.embed_audio = None

        self.vocab_size = text_config.vocab_size
        self.vocab_size_per_layer_input = getattr(
            text_config, "vocab_size_per_layer_input", text_config.vocab_size
        )
        self.language_model = Gemma4TextModel(
            text_config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        text_tie = getattr(text_config, "tie_word_embeddings", True)
        self.lm_head_is_tied = (
            self.pp_group.world_size == 1
            and text_tie
            and not (_is_cpu and _is_cpu_amx_available)
        )
        if self.lm_head_is_tied:
            self.lm_head = self.language_model.embed_tokens
        elif self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                text_config.vocab_size,
                text_config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(text_config)
        self.capture_aux_hidden_states = False
        self.post_init()
        logger.info(
            "Gemma4Unified: serving the text decoder only "
            "(vision_embedder / embed_audio weights are skipped)"
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        return super().load_weights(
            (n, w) for n, w in weights if not n.startswith(_MM_PREFIXES)
        )


EntryClass = Gemma4UnifiedForConditionalGeneration
