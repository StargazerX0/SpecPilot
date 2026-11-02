# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py
from __future__ import annotations

from typing import Dict, Type

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp8Config,
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8Config
from sglang.srt.platforms import current_platform

# Kept for this checkout: the Qwen3.8-27B-FP8 checkpoint (quant_method "fp8"),
# plus the common FP8/FP4 export formats. AWQ/GPTQ/GGUF/bitsandbytes/INT8/MXFP4
# and platform-specific (NPU/CPU/XPU/MLX) methods were pruned.
BASE_QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "fp8": Fp8Config,
    "mxfp8": Fp8Config,
    "modelopt": ModelOptFp8Config,  # Auto-detect, defaults to FP8
    "modelopt_fp8": ModelOptFp8Config,
    "modelopt_fp4": ModelOptFp4Config,
    "modelopt_mixed": ModelOptMixedPrecisionConfig,
    "w8a8_fp8": W8A8Fp8Config,
    "compressed-tensors": CompressedTensorsConfig,
}

QUANTIZATION_METHODS = {**BASE_QUANTIZATION_METHODS}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )

    if current_platform.is_out_of_tree():
        config = current_platform.get_quantization_config(quantization)

        # If the platform has a quantization config, use it else use the default
        if config is not None:
            return config

    return QUANTIZATION_METHODS[quantization]
