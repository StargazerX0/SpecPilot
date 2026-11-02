"""Weight-only quantization of drafter weights: per-channel int8 / fp8 (e4m3) and group-wise int4 references.

Literature: drafter weight-only quantization (arXiv 2607.04244 "Quantize the Target, Quantize the Drafter"), QSpec
(arXiv 2410.11305), Marlin (arXiv 2408.11743), QServe (arXiv 2405.04532), Machete (Neural Magic / Red Hat blog,
2024-10-14). See `research/knowledge/methods/drafter_efficiency/drafter_weight_only_quant.md`.

Scope in this engine: the DFlash2 drafter's own linears (per layer qkv_proj [6144x5120], o_proj [5120x4096],
gate_up_proj [34816x5120], down_proj [5120x17408], two conv kernel_projection [1280x5120]; plus fc [5120x25600])
and, as a *derived copy*, the target lm_head rows used only for candidate extraction. The target's own weights and
its verify path are fixed. Quantizing the drafter changes drafts, never outputs: drafts enter only through the
accept check (EXECUTION_PATH.md section 4, invariant 2).

This module is a numerical reference (pure torch, device-agnostic, slow): use it to (1) bound/measure the weight
error, (2) fake-quantize drafter weights at load time (`fake_quantize_`) so an acceptance run measures the
acceptance cost before any fast kernel exists, and (3) check a fast kernel's output in a unit test. It does not
make anything faster. Fast paths that exist in this checkout (verified by reading; none timed here):

- FP8 W8A8 (per-channel weight scale, dynamic per-token activation quant): `Fp8LinearMethod`
  (python/sglang/srt/layers/quantization/fp8.py). With a bf16 checkpoint it quantizes online in
  `process_weights_after_loading` (`per_token_group_quant_fp8(weight, K)` -> per-output-channel scales) and runs
  the sgl-kernel cutlass `fp8_scaled_mm` (or `torch._scaled_mm` rowwise). Reached for the drafter only through
  `--speculative-draft-model-quantization fp8` (ModelConfig.from_server_args, is_draft_model=True). This is W8A8,
  not weight-only: activations are rounded too.
- FP8 weight-only (W8A16) via Marlin: `marlin_utils_fp8.apply_fp8_marlin_linear`; auto-enabled only on SM80-SM88
  (`can_auto_enable_marlin_fp8`), forced with SGLANG_FORCE_FP8_MARLIN=1. Marlin is Ampere-tuned; Hopper speed
  unverified.
- INT8 W8A8: compressed-tensors `CompressedTensorsW8A8Int8` (sgl_kernel `int8_scaled_mm`), needs a
  compressed-tensors checkpoint layout (not applicable to the fixed bf16 drafter files without a loader change).
- INT4/INT8 weight-only (wNa16): compressed-tensors `CompressedTensorsWNA16` -> gptq-marlin
  (kernels/ops/quantization/gptq_marlin.py). AWQ/GPTQ config classes were pruned from this checkout.
Layers built as plain `nn.Linear` in models/dflash.py (fc, conv kernel_projection, selector hidden_projection)
receive no quant_config and stay bf16 under every flag above.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import nn

__all__ = [
    "INT8_QMAX",
    "FP8_E4M3_MAX",
    "quantize_int8_per_channel",
    "dequantize_int8",
    "quantize_fp8_per_channel",
    "dequantize_fp8",
    "quantize_int4_groupwise",
    "dequantize_int4_groupwise",
    "quantize_dequantize",
    "fake_quantize_",
    "weight_only_matmul",
    "WeightOnlyLinear",
    "int8_error_bound",
    "fp8_error_bound",
    "weight_bytes",
]

INT8_QMAX = 127
INT4_QMIN, INT4_QMAX = -8, 7
# torch.finfo(torch.float8_e4m3fn).max; e4m3 has 3 mantissa bits (relative rounding error <= 2**-4).
FP8_E4M3_MAX = 448.0
_FP8_REL_EPS = 2.0**-4
# Smallest e4m3fn subnormal is 2**-9; round-to-nearest error below the normal range is half of it.
_FP8_SUBNORMAL_HALF_ULP = 2.0**-10


def _amax_scale(weight: torch.Tensor, qmax: float, dim: int) -> torch.Tensor:
    amax = weight.detach().abs().float().amax(dim=dim, keepdim=True)
    return torch.where(amax > 0, amax / qmax, torch.ones_like(amax))


def quantize_int8_per_channel(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8 for an [N, K] (out, in) weight. Returns (q int8 [N, K], scale fp32 [N, 1])."""
    scale = _amax_scale(weight, INT8_QMAX, dim=1)
    q = torch.round(weight.float() / scale).clamp_(-INT8_QMAX, INT8_QMAX).to(torch.int8)
    return q, scale


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return (q.float() * scale).to(dtype)


def quantize_fp8_per_channel(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel fp8 e4m3fn with amax scaling (the layout Fp8LinearMethod builds online).
    Returns (q float8_e4m3fn [N, K], scale fp32 [N, 1])."""
    scale = _amax_scale(weight, FP8_E4M3_MAX, dim=1)
    q = (weight.float() / scale).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale


def dequantize_fp8(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return (q.float() * scale).to(dtype)


def quantize_int4_groupwise(weight: torch.Tensor, group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric ([-8, 7]) group-wise int4 along K, round-to-nearest. Values are stored unpacked in int8
    ([N, K]); scales fp32 [N, K // group_size]. Packing/permutation for a real kernel (Marlin) is out of scope."""
    n, k = weight.shape
    if k % group_size:
        raise ValueError(f"K={k} not divisible by group_size={group_size}")
    w = weight.float().reshape(n, k // group_size, group_size)
    amax = w.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / INT4_QMAX, torch.ones_like(amax))
    q = torch.round(w / scale).clamp_(INT4_QMIN, INT4_QMAX).to(torch.int8)
    return q.reshape(n, k), scale.squeeze(-1)


def dequantize_int4_groupwise(
    q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    n, k = q.shape
    groups = scale.shape[-1]
    return (q.float().reshape(n, groups, k // groups) * scale.unsqueeze(-1)).reshape(n, k).to(dtype)


_QUANTIZERS: dict = {
    "int8": (quantize_int8_per_channel, dequantize_int8),
    "fp8": (quantize_fp8_per_channel, dequantize_fp8),
    "int4": (quantize_int4_groupwise, dequantize_int4_groupwise),
}


def quantize_dequantize(weight: torch.Tensor, mode: str) -> torch.Tensor:
    """Round-trip `weight` through `mode` in {'int8', 'fp8', 'int4'}; returns a tensor of weight's dtype."""
    quantize, dequantize = _QUANTIZERS[mode]
    q, scale = quantize(weight)
    return dequantize(q, scale, dtype=weight.dtype)


@torch.no_grad()
def fake_quantize_(
    module: nn.Module,
    mode: str,
    *,
    include: Optional[Callable[[str, nn.Parameter], bool]] = None,
) -> list:
    """In place: replace every 2-D floating `*.weight` parameter of `module` accepted by `include(name, param)`
    with its quantize-dequantize round trip. Returns the names touched. The model keeps its dtype and kernels, so
    this measures the *acceptance* cost of a quantized drafter, not its speed. Run once at load time, before graph
    capture (draft-only derived data). Default `include`: every 2-D weight except codebooks and norms."""
    touched = []
    for name, param in module.named_parameters():
        if param.ndim != 2 or not param.is_floating_point() or not name.endswith("weight"):
            continue
        if include is not None and not include(name, param):
            continue
        if include is None and ("codebook" in name or "norm" in name):
            continue
        param.data.copy_(quantize_dequantize(param.data, mode))
        touched.append(name)
    return touched


def weight_only_matmul(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, mode: str) -> torch.Tensor:
    """Reference W*A16 GEMM: y = x @ dequant(q, scale).T, computed in x.dtype. A fast kernel must match this to
    within accumulation-order drift."""
    _, dequantize = _QUANTIZERS[mode]
    return torch.matmul(x, dequantize(q, scale, dtype=x.dtype).T)


class WeightOnlyLinear(nn.Module):
    """Reference drop-in for a bias-free linear with quantized weights (dequantize on the fly)."""

    def __init__(self, q: torch.Tensor, scale: torch.Tensor, mode: str) -> None:
        super().__init__()
        if mode not in _QUANTIZERS:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)

    @classmethod
    def from_weight(cls, weight: torch.Tensor, mode: str) -> "WeightOnlyLinear":
        quantize, _ = _QUANTIZERS[mode]
        q, scale = quantize(weight)
        return cls(q, scale, mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return weight_only_matmul(x, self.q, self.scale, self.mode)


def int8_error_bound(scale: torch.Tensor) -> torch.Tensor:
    """Per-element |dequant - w| bound for per-channel symmetric int8 RTN: scale / 2 (amax maps exactly)."""
    return scale / 2


def fp8_error_bound(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Per-element |dequant - w| bound for per-channel e4m3fn: 2**-4 relative in the normal range plus half a
    subnormal ulp (scaled) below it (fp32 dequantization assumed)."""
    return _FP8_REL_EPS * weight.float().abs() + _FP8_SUBNORMAL_HALF_ULP * scale


def weight_bytes(shape: Tuple[int, int], mode: str, group_size: int = 128) -> int:
    """Bytes a kernel must stream for one [N, K] weight (packed values + fp32 scales): the decode-time cost proxy."""
    n, k = shape
    if mode == "bf16":
        return 2 * n * k
    if mode in ("int8", "fp8"):
        return n * k + 4 * n
    if mode == "int4":
        return n * k // 2 + 4 * n * (k // group_size)
    raise ValueError(f"unknown mode {mode!r}")
