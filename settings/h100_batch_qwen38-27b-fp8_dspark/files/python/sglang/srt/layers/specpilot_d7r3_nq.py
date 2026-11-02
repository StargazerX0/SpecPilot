"""SPECPILOT_D7R3_NQ=1: residual add + GemmaRMSNorm + FP8 1x128 group quant of the next block-FP8 linear's input.

The seed ran flashinfer's gemma_fused_add_rmsnorm (residual += x; x = norm(residual) in bf16) and then the FP8 linear
re-read x in ``sglang_per_token_group_quant_fp8(column_major_scales=True, scale_tma_aligned=True)``. Here one kernel
(specpilot_d7r3_nq.cu) updates the residual, quantises the normed row and hands the linear a pre-quantized (q, s)
tuple; the bf16 normed row is written only where another consumer needs it (the GDN layer's BF16 in_proj_ba).
Call sites: the input norm of every GDN / attention layer (-> in_proj_qkvz / qkv_proj) and every post-attention norm
(-> gate_up_proj) after the first layer. Knob read once at import; unset = the parent's path.
"""
from __future__ import annotations

import functools
import hashlib
import os

import torch

NQ = os.environ.get("SPECPILOT_D7R3_NQ", "0") not in ("", "0")
GROUP = 128
HIDDEN = 5120
_PDL = True  # programmatic dependent launch of the wide kernel (weight load overlaps the producer's tail)
_RPB = 20  # launch shape: 20 = one row per 320-thread block (fastest at M 128-384); 1/2/4 = 64-thread rows


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load_inline

    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "specpilot_d7r3_nq.cu")).read()
    cpp = ("void d7r3_add_norm_quant(torch::Tensor x, torch::Tensor r, torch::Tensor w, torch::Tensor q, "
           "torch::Tensor s_base, double eps, bool write_y, int64_t rpb, bool pdl);")
    # --use_fast_math: the reference per_token_group_quant JIT module is built with it on sm_90
    flags = ["-O3", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++17"]
    key = hashlib.sha1("\0".join([cpp, src, *flags]).encode()).hexdigest()[:16]
    base = os.environ.get("SGLANG_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "sglang")
    bdir = os.path.join(base, "specpilot_d7r3_nq", key)
    os.makedirs(bdir, exist_ok=True)
    return load_inline(name=f"specpilot_d7r3_nq_{key}", cpp_sources=cpp, cuda_sources=src,
                       functions=["d7r3_add_norm_quant"], extra_cuda_cflags=flags, build_directory=bdir, verbose=False)


def prepare() -> None:
    """Build the extension outside any CUDA-graph capture."""
    if NQ:
        _ext()


def add_norm_quant(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float, write_y: bool,
                   rpb: int | None = None, pdl: bool | None = None):
    """x, residual [M, 5120] bf16 (residual updated in place; x overwritten by the normed row iff write_y) ->
    (q [M, 5120] fp8 e4m3, s [M, 40] fp32 with stride (1, align4(M)))."""
    M = x.shape[0]
    m_al = (M + 3) // 4 * 4
    q = torch.empty((M, HIDDEN), dtype=torch.float8_e4m3fn, device=x.device)
    s_base = torch.empty((HIDDEN // GROUP, m_al), dtype=torch.float32, device=x.device)
    _ext().d7r3_add_norm_quant(x, residual, weight, q, s_base, float(eps), bool(write_y), int(rpb or _RPB),
                               _PDL if pdl is None else bool(pdl))
    return q, s_base.t()[:M]


def _linear_takes_q(linear) -> bool:
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    qm = getattr(linear, "quant_method", None)
    return (
        isinstance(qm, Fp8LinearMethod)
        and qm.block_quant
        and not qm.use_marlin
        and not getattr(qm, "use_mxfp8", False)
        and qm.w8a8_block_fp8_linear is fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback
        and not fp8_utils.deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        and linear.weight.shape[0] % 64 == 0
        and linear.weight.shape[1] == HIDDEN
        and getattr(linear, "tp_size", 1) == 1
    )


def eligible(x, residual, norm, linear) -> bool:
    """Plain TP=1 residual-add GemmaRMSNorm (no post-residual addition) feeding a block-FP8 DeepGEMM-path linear."""
    return (
        NQ
        and isinstance(x, torch.Tensor)
        and residual is not None
        and x.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and x.dim() == 2
        and x.shape == residual.shape
        and x.shape[1] == HIDDEN
        and x.shape[0] > 0
        and x.is_contiguous()
        and residual.is_contiguous()
        and norm.weight.dtype == torch.bfloat16
        and norm.weight.is_contiguous()
        and _linear_takes_q(linear)
    )


class NormQuant(tuple):
    """(head, q, s): head = the normed bf16 row when it was materialised (GDN input norm, for in_proj_ba), else q;
    (q, s) = the pre-quantized input of the next FP8 linear. A tuple so the decoder layers' ``hidden_states[0]``
    shape probes keep working."""

    __slots__ = ()

    def __new__(cls, head, q, s):
        return super().__new__(cls, (head, q, s))

    @property
    def qs(self):
        return (self[1], self[2])


def _plain_communicator(comm) -> bool:
    """The TP=1 communicator path: prepare_attn / prepare_mlp reduce to ``norm(x, residual)`` (no SP, no fused
    all-reduce, no latent qkv, trivial gather)."""
    from sglang.srt.layers.communicator import CommunicateSimpleFn, CommunicateWithAllReduceAndLayerNormFn

    f1 = comm._communicate_simple_fn
    f2 = comm._communicate_with_all_reduce_and_layer_norm_fn
    return (
        type(comm).__name__ == "LayerCommunicator"
        and getattr(comm, "_sp_variant", None) is None
        and getattr(comm, "qkv_latent_func", None) is None
        and comm._context.tp_size == 1
        and comm._context.attn_tp_size == 1
        and getattr(f1, "func", f1) is CommunicateSimpleFn._trivial
        and getattr(f2, "func", f2) is CommunicateWithAllReduceAndLayerNormFn._simple
    )


def prepare_attn(comm, x, residual, linear, write_y: bool, captured=None):
    """Replacement of ``comm.prepare_attn_and_capture_last_layer_outputs`` for a layer whose input norm feeds the FP8
    ``linear``; None when the fused path does not apply (the caller runs the parent's path)."""
    from sglang.srt.layers.communicator import get_attn_tp_context

    if not (
        eligible(x, residual, comm.input_layernorm, linear)
        and not getattr(x, "_sglang_needs_allreduce_fusion", False)
        and not get_attn_tp_context().input_scattered
        and _plain_communicator(comm)
    ):
        return None
    norm = comm.input_layernorm
    q, s = add_norm_quant(x, residual, norm.weight, norm.variance_epsilon, write_y)
    if captured is not None:  # parent: gathered residual (trivial gather) cloned unless the accumulator copies
        captured.append(residual if getattr(captured, "copies_on_append", False) else residual.clone())
    return NormQuant(x if write_y else q, q, s), residual


def prepare_mlp(comm, x, residual, linear):
    """Replacement of ``comm.prepare_mlp`` when the post-attention norm feeds the FP8 ``linear`` (gate_up)."""
    if not (eligible(x, residual, comm.post_attention_layernorm, linear) and _plain_communicator(comm)):
        return None
    norm = comm.post_attention_layernorm
    q, s = add_norm_quant(x, residual, norm.weight, norm.variance_epsilon, False)
    return NormQuant(q, q, s), residual
