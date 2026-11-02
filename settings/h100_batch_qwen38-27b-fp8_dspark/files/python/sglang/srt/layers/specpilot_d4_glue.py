"""SPECPILOT_D4: bitwise glue kernels of the target step (knobs read once at import).

SPECPILOT_D4_ACTQ=1  MLP SiLU*mul + FP8 per-1x128-group quant of the down_proj input in one kernel
                     (specpilot_d4_actq.cu); (q, s) equal sglang act_and_mul + sglang_per_token_group_quant_fp8(
                     column_major_scales, scale_tma_aligned) bit for bit, so down_proj's DeepGEMM call is unchanged.
SPECPILOT_D4_SPLIT=1 rv_pack's packed -> dense in-projection split, one 4-warp program per dense row (same copies).
"""
from __future__ import annotations

import functools
import hashlib
import os

import torch
import triton
import triton.language as tl

ACTQ = os.environ.get("SPECPILOT_D4_ACTQ", "0") not in ("", "0")
SPLIT = os.environ.get("SPECPILOT_D4_SPLIT", "0") not in ("", "0")
GROUP = 128


@functools.lru_cache(maxsize=None)
def _actq_ext():
    from torch.utils.cpp_extension import load_inline

    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "specpilot_d4_actq.cu")).read()
    cpp = "void d4_act_mul_quant(torch::Tensor in, torch::Tensor q, torch::Tensor s_base, int64_t M, int64_t I);"
    # --use_fast_math: the reference act_and_mul and per_token_group_quant JIT modules are built with it on sm_90
    flags = ["-O3", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++17"]
    key = hashlib.sha1("\0".join([cpp, src, *flags]).encode()).hexdigest()[:16]
    base = os.environ.get("SGLANG_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "sglang")
    bdir = os.path.join(base, "specpilot_d4_actq", key)
    os.makedirs(bdir, exist_ok=True)
    return load_inline(name=f"specpilot_d4_actq_{key}", cpp_sources=cpp, cuda_sources=src,
                       functions=["d4_act_mul_quant"], extra_cuda_cflags=flags, build_directory=bdir, verbose=False)


def prepare() -> None:
    """Build the extension outside any CUDA-graph capture."""
    if ACTQ:
        _actq_ext()


def act_mul_quant(gate_up: torch.Tensor):
    """gate_up [M, 2I] bf16 -> (q [M, I] fp8 e4m3, s [M, I/128] fp32, stride (1, align4(M)))."""
    M, I2 = gate_up.shape
    I = I2 // 2
    m_al = (M + 3) // 4 * 4
    q = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=gate_up.device)
    s_base = torch.empty((I // GROUP, m_al), dtype=torch.float32, device=gate_up.device)
    _actq_ext().d4_act_mul_quant(gate_up, q, s_base, M, I)
    return q, s_base.t()[:M]


def actq_eligible(gate_up: torch.Tensor, down_proj) -> bool:
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    qm = down_proj.quant_method
    return (
        isinstance(qm, Fp8LinearMethod)
        and qm.block_quant
        and not qm.use_marlin
        and qm.w8a8_block_fp8_linear is fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback
        and not fp8_utils.deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        and gate_up.dtype == torch.bfloat16
        and gate_up.dim() == 2
        and gate_up.is_contiguous()
        and gate_up.shape[0] > 0
        and (gate_up.shape[1] // 2) % GROUP == 0
        and down_proj.weight.shape[0] % 64 == 0
    )


@triton.jit
def _split_row_kernel(mixed_qkv, z, b, a, mixed_qkvz, mixed_ba, qkvz_row_stride, ba_row_stride, unpack_ptr,
                      rows_out_ptr, QKV: tl.constexpr, TOTAL_V: tl.constexpr, HV: tl.constexpr, HVP: tl.constexpr,
                      BLOCK: tl.constexpr):
    d = tl.program_id(0).to(tl.int64)
    src = tl.load(unpack_ptr + d)
    dst_z = tl.load(rows_out_ptr + d)
    in_row = mixed_qkvz + src * qkvz_row_stride
    out_row = mixed_qkv + d * QKV
    for c0 in tl.static_range(0, QKV, BLOCK):
        o = c0 + tl.arange(0, BLOCK)
        m = o < QKV
        tl.store(out_row + o, tl.load(in_row + o, mask=m), mask=m)
    if dst_z >= 0:
        for c0 in tl.static_range(0, TOTAL_V, BLOCK):
            o = c0 + tl.arange(0, BLOCK)
            m = o < TOTAL_V
            tl.store(z + dst_z * TOTAL_V + o, tl.load(in_row + QKV + o, mask=m), mask=m)
    oh = tl.arange(0, HVP)
    mh = oh < HV
    ba_row = mixed_ba + src * ba_row_stride
    tl.store(b + d * HV + oh, tl.load(ba_row + oh, mask=mh), mask=mh)
    tl.store(a + d * HV + oh, tl.load(ba_row + HV + oh, mask=mh), mask=mh)


def split_qkvzba_dense(p, mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v):
    """Same contract and outputs as specpilot_rv.split_qkvzba_dense."""
    rows = p.bs * p.n_nodes
    T = mixed_qkvz.shape[0]
    qkv = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    tv = num_heads_v * head_v
    mixed_qkv = mixed_qkvz.new_empty((rows, qkv))
    z = mixed_qkvz.new_empty((T, num_heads_v, head_v))
    b = mixed_ba.new_empty((rows, num_heads_v))
    a = torch.empty_like(b)
    _split_row_kernel[(rows,)](mixed_qkv, z, b, a, mixed_qkvz, mixed_ba, mixed_qkvz.stride(0), mixed_ba.stride(0),
                               p.unpack, p.rows_out, QKV=qkv, TOTAL_V=tv, HV=num_heads_v,
                               HVP=triton.next_power_of_2(num_heads_v), BLOCK=2048, num_warps=4)
    return mixed_qkv, z, b, a
