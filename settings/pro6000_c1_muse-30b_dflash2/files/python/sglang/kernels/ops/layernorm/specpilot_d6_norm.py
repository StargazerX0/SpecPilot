# SPECPILOT_D6: Muse residual join, norm1 -> add -> norm2 in one CuTe-DSL kernel.
"""Fused ``h = residual + rmsnorm(x, w1, eps1)``; ``y = rmsnorm(h, w2, eps2)``.

Bitwise equal to stock's composition ``sgl_kernel.rmsnorm`` (flashinfer CuTe-DSL
RMSNormKernel) -> torch bf16 add -> ``sgl_kernel.rmsnorm``: the kernel reuses
flashinfer's RMSNormKernel configuration (thread layout, vector size, async copy)
and its reduction helper, and evaluates the same expressions with a bf16 rounding
after every stock kernel boundary. Weights are loaded before the PDL wait.
``y`` is optional (``w2 is None``: norm + add only, the last layer before the
final norm, which the model applies itself).
"""


import functools
import math
from typing import Optional, Tuple

import torch

_H_SUPPORTED = (6656,)


def eligible(x: torch.Tensor, residual: torch.Tensor) -> bool:
    return (
        x.dim() == 2
        and x.shape[0] > 0
        and x.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and x.shape == residual.shape
        and x.shape[1] in _H_SUPPORTED
        and x.is_contiguous()
        and residual.is_contiguous()
    )


@functools.cache
def _kernel_class():
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int64
    from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
    from flashinfer.norm.utils import predicate_k, row_reduce_sum_multirow

    class NormAddNormKernel(RMSNormKernel):
        def __init__(self, dtype, H: int, has_norm2: bool):
            super().__init__(dtype, H, 0.0)
            assert self.cluster_n == 1 and self.rows_per_block == 1, (
                self.cluster_n,
                self.rows_per_block,
            )
            self.has_norm2 = has_norm2

        @cute.jit
        def __call__(
            self,
            mX: cute.Tensor,
            mR: cute.Tensor,
            mW1: cute.Tensor,
            mW2: cute.Tensor,
            mH: cute.Tensor,
            mY: cute.Tensor,
            M: Int64,
            eps1: Float32,
            eps2: Float32,
            enable_pdl: cutlass.Constexpr[bool],
            stream,
        ):
            tv_shape, tv_stride = self._make_tv_layout(
                self.threads_per_row,
                self.rows_per_block,
                self.vec_size,
                self.num_vec_blocks,
            )
            tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
            tiler_mn = (self.rows_per_block, self.cols_per_tile)
            self.kernel(
                mX, mR, mW1, mW2, mH, mY, M, eps1, eps2, enable_pdl, tv_layout, tiler_mn
            ).launch(
                grid=[cute.ceil_div(M, self.rows_per_block), 1, 1],
                block=[self.num_threads, 1, 1],
                smem=self._smem_size_in_bytes(),
                stream=stream,
                use_pdl=enable_pdl,
            )

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mR: cute.Tensor,
            mW1: cute.Tensor,
            mW2: cute.Tensor,
            mH: cute.Tensor,
            mY: cute.Tensor,
            M: Int64,
            eps1: Float32,
            eps2: Float32,
            enable_pdl: cutlass.Constexpr[bool],
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            H = self.H
            copy_bits = self.copy_bits
            threads_per_row = tv_layout.shape[0][0]
            rows_per_block = tiler_mn[0]
            warps_per_row = max(threads_per_row // 32, 1)

            smem = cutlass.utils.SmemAllocator()
            if cutlass.const_expr(self.use_async_copy):
                sX = smem.allocate_tensor(
                    mX.element_type,
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                    byte_alignment=16,
                )
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )

            idX = cute.make_identity_tensor(mX.shape)
            gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
            gR = cute.local_tile(mR, tiler_mn, (bidx, 0))
            gH = cute.local_tile(mH, tiler_mn, (bidx, 0))
            cX = cute.local_tile(idX, tiler_mn, (bidx, 0))
            w_layout = cute.prepend(
                mW1.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
            )
            gW1 = cute.local_tile(
                cute.make_tensor(mW1.iterator, w_layout), tiler_mn, (0, 0)
            )
            gW2 = cute.local_tile(
                cute.make_tensor(mW2.iterator, w_layout), tiler_mn, (0, 0)
            )

            copy_atom_sync = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                mX.element_type,
                num_bits_per_copy=copy_bits,
            )
            if cutlass.const_expr(self.use_async_copy):
                copy_atom_async = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(),
                    mX.element_type,
                    num_bits_per_copy=copy_bits,
                )
                tiled_copy_load = cute.make_tiled_copy(
                    copy_atom_async, tv_layout, tiler_mn
                )
            else:
                tiled_copy_load = cute.make_tiled_copy(
                    copy_atom_sync, tv_layout, tiler_mn
                )
            tiled_copy_sync = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)

            thr_copy_X = tiled_copy_load.get_slice(tidx)
            thr_copy_S = tiled_copy_sync.get_slice(tidx)

            tXgX = thr_copy_X.partition_S(gX)
            tXcX = thr_copy_X.partition_S(cX)
            tXrX = cute.make_fragment_like(tXgX)
            if cutlass.const_expr(self.use_async_copy):
                tXsX = thr_copy_X.partition_D(sX)

            tRgR = thr_copy_S.partition_S(gR)
            tRrR = cute.make_fragment_like(tRgR)
            tXrR = thr_copy_X.retile(tRrR)

            tWgW1 = thr_copy_S.partition_S(gW1)
            tWrW1 = cute.make_fragment_like(tWgW1)
            tXrW1 = thr_copy_X.retile(tWrW1)
            tWgW2 = thr_copy_S.partition_S(gW2)
            tWrW2 = cute.make_fragment_like(tWgW2)
            tXrW2 = thr_copy_X.retile(tWrW2)

            tHgH = thr_copy_S.partition_D(gH)
            tHrH = cute.make_fragment_like(tHgH)

            tXpX = predicate_k(tXcX, limit=H)
            tSpS = predicate_k(thr_copy_S.partition_S(cX), limit=H)

            # Weights do not depend on the previous kernel: load before the PDL wait.
            # Zero-filled so the padded columns (H < cols_per_tile) give h = 0 there.
            tWrW1.store(cute.zeros_like(tWrW1, dtype=mW1.element_type))
            cute.copy(copy_atom_sync, tWgW1, tWrW1, pred=tSpS)
            if cutlass.const_expr(self.has_norm2):
                tWrW2.store(cute.zeros_like(tWrW2, dtype=mW2.element_type))
                cute.copy(copy_atom_sync, tWgW2, tWrW2, pred=tSpS)

            if enable_pdl:
                cute.arch.griddepcontrol_wait()

            # ===== norm1 (as RMSNormKernel pass 1 / pass 2) =====
            if cutlass.const_expr(self.use_async_copy):
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
                cute.arch.cp_async_commit_group()
                tRrR.store(cute.zeros_like(tRrR, dtype=mR.element_type))
                cute.copy(copy_atom_sync, tRgR, tRrR, pred=tSpS)
                cute.arch.cp_async_wait_group(0)
                cute.autovec_copy(tXsX, tXrX)
            else:
                tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)
                tRrR.store(cute.zeros_like(tRrR, dtype=mR.element_type))
                cute.copy(copy_atom_sync, tRgR, tRrR, pred=tSpS)

            x = tXrX.load().to(Float32)
            x_sq = x * x
            sum_sq = row_reduce_sum_multirow(
                x_sq, threads_per_row, reduction_buffer, None, 1
            )
            mean_sq = sum_sq / Float32(H)
            rstd = cute.math.rsqrt(mean_sq + eps1, fastmath=True)
            cute.arch.barrier()

            if cutlass.const_expr(self.use_async_copy):
                cute.autovec_copy(tXsX, tXrX)
                x = tXrX.load().to(Float32)
            w1 = tXrW1.load().to(Float32)
            y1 = (x * rstd * (w1 + Float32(0.0))).to(mX.element_type)

            # ===== residual add (torch bf16 add: fp32 sum, one rounding) =====
            h = (tXrR.load().to(Float32) + y1.to(Float32)).to(mH.element_type)
            tHrH.store(h)
            cute.copy(copy_atom_sync, tHrH, tHgH, pred=tSpS)

            # ===== norm2 on the rounded residual =====
            if cutlass.const_expr(self.has_norm2):
                x2 = h.to(Float32)
                x2_sq = x2 * x2
                sum_sq2 = row_reduce_sum_multirow(
                    x2_sq, threads_per_row, reduction_buffer, None, 1
                )
                mean_sq2 = sum_sq2 / Float32(H)
                rstd2 = cute.math.rsqrt(mean_sq2 + eps2, fastmath=True)
                w2 = tXrW2.load().to(Float32)
                y2 = (x2 * rstd2 * (w2 + Float32(0.0))).to(mY.element_type)
                gY = cute.local_tile(mY, tiler_mn, (bidx, 0))
                tYgY = thr_copy_S.partition_D(gY)
                tYrY = cute.make_fragment_like(tYgY)
                tYrY.store(y2)
                cute.copy(copy_atom_sync, tYrY, tYgY, pred=tSpS)

            if enable_pdl:
                cute.arch.griddepcontrol_launch_dependents()

    return NormAddNormKernel


@functools.cache
def _compiled(H: int, has_norm2: bool, enable_pdl: bool):
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int64

    dtype = cutlass.BFloat16
    kernel_obj = _kernel_class()(dtype, H, has_norm2)
    sym_m = cute.sym_int(64)
    align = math.gcd(128, H * 2)

    def t2d():
        return cute.runtime.make_fake_compact_tensor(
            dtype, (sym_m, H), stride_order=(1, 0), assumed_align=align
        )

    def t1d():
        return cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)

    return cute.compile(
        kernel_obj,
        t2d(),
        t2d(),
        t1d(),
        t1d(),
        t2d(),
        t2d(),
        Int64(1),
        Float32(1e-6),
        Float32(1e-6),
        enable_pdl,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@functools.cache
def _pdl(device_index: int) -> bool:
    from flashinfer.utils import device_support_pdl

    return bool(device_support_pdl(torch.device("cuda", device_index)))


def norm_add_norm(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    w1: torch.Tensor,
    eps1: float,
    w2: Optional[torch.Tensor],
    eps2: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Returns (h, y): new residual ``residual + rmsnorm(x, w1)`` and ``rmsnorm(h, w2)``
    (None when ``w2`` is None). Fresh output tensors; inputs are not modified."""
    M, H = x.shape
    has_norm2 = w2 is not None
    h = torch.empty_like(residual)
    y = torch.empty_like(x) if has_norm2 else h
    kernel = _compiled(H, has_norm2, _pdl(x.device.index))
    kernel(x, residual, w1, w2 if has_norm2 else w1, h, y, M, eps1, eps2)
    return h, (y if has_norm2 else None)
