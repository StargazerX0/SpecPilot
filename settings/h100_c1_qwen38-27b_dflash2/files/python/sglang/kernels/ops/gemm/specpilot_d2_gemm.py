"""SPECPILOT_D2: verify-width (M <= 64) BF16 GEMM with PDL weight pre-staging (Gluon, SM90).

out[M, N] = x[M, K] @ w[N, K]^T, swap-AB (weight rows = wgmma M, verify rows = wgmma N), one CTA per BN weight rows,
no split-K (deterministic).  Each CTA owns a SMEM ring of STAGES k-tiles.  The weight tiles of the first STAGES k-steps
are issued with TMA *before* ``griddepcontrol.wait``: they do not depend on the predecessor kernel, so when this GEMM is
launched with programmatic dependent launch (PDL) and its predecessor triggers early, the weight stream of the GEMM
overlaps the latency-bound predecessor (norms, gated norm, WY GDN).  The activation tiles are loaded after the wait.
``gate_up_silu``: MergedColumn gate_up ([2I, K]) with the SiLU(gate) * up epilogue; gate and up accumulators are rounded
to bf16 before the activation, as the cuBLAS + act_and_mul path (and lib tma_gemm) does.

Knobs (read once at import; all unset = the parent's path, this module is not used):
  SPECPILOT_D2_PRE=1      route the x1 hook's tuned shapes through this kernel
  SPECPILOT_D2_PDL=1      launch with PDL (0: same kernel, plain launch: ablation)
  SPECPILOT_D2_TRIG       where the GEMM triggers its dependents: 0 start, 1 after its last weight tile is issued, 2 exit
  SPECPILOT_D2_SMEM       SMEM ring budget per CTA in KB (sets STAGES)
  SPECPILOT_D2_BK         k-tile           (TRIG / SMEM / BK unset: the per-shape table _CFG)
  SPECPILOT_D2_PREST      k-tiles issued before the wait (default the whole ring; PRESTS per shape)
  SPECPILOT_D2_EARLY=15   bitmask: 1 add-RMSNorm, 2 gated norm, 4 WY GDN, 8 attention merge+gate: the latency-bound predecessors (add-RMSNorm, gated norm, WY GDN, attention merge / gate)
                          trigger their dependents at start (edits marked SPECPILOT_D2 in those files)
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def read_d2_knobs() -> dict:
    return {
        "pre": _env_int("SPECPILOT_D2_PRE", 0),
        "pdl": _env_int("SPECPILOT_D2_PDL", 1),
        "trig": _env_int("SPECPILOT_D2_TRIG", -1),
        "smem": _env_int("SPECPILOT_D2_SMEM", 0),
        "bk": _env_int("SPECPILOT_D2_BK", 0),
        "prest": _env_int("SPECPILOT_D2_PREST", -1),
    }


D2 = read_d2_knobs()
# SPECPILOT_D2_SPLIT="N:K:S,..." split-K factor per shape (deterministic fixup), default none
D2_SPLIT = {
    (int(a), int(b)): int(c)
    for a, b, c in (t.split(":") for t in os.environ.get("SPECPILOT_D2_SPLIT", "").replace(" ", "").split(",") if t)
}
D2_PRE = bool(D2["pre"])
_MAX_M = 64
_BN = 64
# (N_out, K, silu) shapes of the target linears (same table as lib tma_gemm)
# (N_out, K, silu) -> (SMEM ring KB, BK, trigger) tuned on H100 SXM (scratch/d2/bench_shape.py: CUDA graph, cold
# weights; gu_tune.py and prescreen 1e15ac1a0ca9: triggering after the last weight tile is issued (1) beats triggering
# at start, which lets the successor's CTAs compete with this GEMM); SPECPILOT_D2_SMEM / _BK / _TRIG override all.
_CFG = {
    (17408, 5120, True): (200, 128, 1),  # gate_up + SiLU*mul
    (5120, 17408, False): (200, 256, 1),  # down_proj
    (16384, 5120, False): (100, 128, 1),  # in_proj_qkvz
    (16480, 5120, False): (100, 128, 1),  # qkvz + ba folded
    (5120, 6144, False): (200, 256, 1),  # out_proj / o_proj
    (14336, 5120, False): (100, 128, 1),  # attention qkv(+gate)
}
_SHAPES = set(_CFG)
# k-tiles issued before griddepcontrol.wait per shape (default: the whole ring); SPECPILOT_D2_PREST overrides all,
# SPECPILOT_D2_PRESTS="N:K:S:n,..." per shape (S = 1 for the fused gate_up)
D2_PREST = {
    (int(a), int(b), bool(int(c))): int(d)
    for a, b, c, d in (t.split(":") for t in os.environ.get("SPECPILOT_D2_PRESTS", "").replace(" ", "").split(",") if t)
}


@gluon.jit
def _gdc_wait():
    gl.inline_asm_elementwise("griddepcontrol.wait; // dummy $0", "=r", [], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _gdc_launch():
    gl.inline_asm_elementwise(
        "griddepcontrol.launch_dependents; // dummy $0", "=r", [], dtype=gl.int32, is_pure=False, pack=1
    )


@gluon.jit
def _d2_gemm_kernel(w_desc, x_desc, out_ptr, ws_ptr, cnt_ptr, M, NOUT, K, stride_om,
                    BN: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr, STAGES: gl.constexpr,
                    SILU: gl.constexpr, PDL: gl.constexpr, TRIG: gl.constexpr, SPLIT: gl.constexpr,
                    PRE: gl.constexpr, num_warps: gl.constexpr):
    pid = gl.program_id(0)
    n0 = pid * BN
    nk = (K // BK) // SPLIT
    kb = gl.program_id(1) * nk * BK  # this CTA's k range (split-K: deterministic last-CTA fixup below)
    NW: gl.constexpr = 2 if SILU else 1
    w_s = gl.allocate_shared_memory(gl.bfloat16, [STAGES * NW, BN, BK], w_desc.layout)
    x_s = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BM, BK], x_desc.layout)
    wbar = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    xbar = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    for s in gl.static_range(STAGES):
        mbarrier.init(wbar.index(s), count=1)
        mbarrier.init(xbar.index(s), count=1)
    fence_async_shared()
    # weight pre-staging: the first PRE k-tiles (activation-independent) are issued before the dependency wait
    # (PRE bounds the burst the latency-bound predecessor has to share HBM with); the rest of the ring after it
    for s in gl.static_range(PRE):
        p = s < nk
        mbarrier.expect(wbar.index(s), NW * BN * BK * 2, pred=p)
        tma.async_copy_global_to_shared(w_desc, [n0, kb + s * BK], wbar.index(s), w_s.index(s * NW), pred=p)
        if SILU:
            tma.async_copy_global_to_shared(w_desc, [NOUT + n0, kb + s * BK], wbar.index(s), w_s.index(s * NW + 1),
                                            pred=p)
    if TRIG == 0:
        _gdc_launch()
    if PDL:
        _gdc_wait()
    for s in gl.static_range(PRE, STAGES):
        p = s < nk
        mbarrier.expect(wbar.index(s), NW * BN * BK * 2, pred=p)
        tma.async_copy_global_to_shared(w_desc, [n0, kb + s * BK], wbar.index(s), w_s.index(s * NW), pred=p)
        if SILU:
            tma.async_copy_global_to_shared(w_desc, [NOUT + n0, kb + s * BK], wbar.index(s), w_s.index(s * NW + 1),
                                            pred=p)
    for s in gl.static_range(STAGES):
        p = s < nk
        mbarrier.expect(xbar.index(s), BM * BK * 2, pred=p)
        tma.async_copy_global_to_shared(x_desc, [0, kb + s * BK], xbar.index(s), x_s.index(s), pred=p)

    acc_l: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
                                                    instr_shape=[16, BM, 16])
    acc = gl.zeros([BN, BM], gl.float32, acc_l)
    acc_u = gl.zeros([BN, BM], gl.float32, acc_l)
    phase = 0
    for k in range(nk):
        st = k % STAGES
        mbarrier.wait(wbar.index(st), phase)
        mbarrier.wait(xbar.index(st), phase)
        xt = x_s.index(st).permute((1, 0))
        acc = warpgroup_mma(w_s.index(st * NW), xt, acc, is_async=True)
        if SILU:
            acc_u = warpgroup_mma(w_s.index(st * NW + 1), xt, acc_u, is_async=True)
        # keep this k-step's MMA in flight: wait for the previous one, then refill the stage it consumed
        if SILU:
            acc, acc_u = warpgroup_mma_wait(2, deps=[acc, acc_u])
        else:
            acc = warpgroup_mma_wait(1, deps=[acc])
        sp = (k + STAGES - 1) % STAGES
        nxt = k + STAGES - 1
        pn = (k > 0) & (nxt < nk)
        mbarrier.expect(wbar.index(sp), NW * BN * BK * 2, pred=pn)
        tma.async_copy_global_to_shared(w_desc, [n0, kb + nxt * BK], wbar.index(sp), w_s.index(sp * NW), pred=pn)
        if SILU:
            tma.async_copy_global_to_shared(w_desc, [NOUT + n0, kb + nxt * BK], wbar.index(sp),
                                            w_s.index(sp * NW + 1), pred=pn)
        mbarrier.expect(xbar.index(sp), BM * BK * 2, pred=pn)
        tma.async_copy_global_to_shared(x_desc, [0, kb + nxt * BK], xbar.index(sp), x_s.index(sp), pred=pn)
        if TRIG == 1:
            if k == nk - STAGES + 1:
                _gdc_launch()
        if st == STAGES - 1:
            phase ^= 1
    if SILU:
        acc, acc_u = warpgroup_mma_wait(0, deps=[acc, acc_u])
    else:
        acc = warpgroup_mma_wait(0, deps=[acc])
    for s in gl.static_range(STAGES):
        mbarrier.invalidate(wbar.index(s))
        mbarrier.invalidate(xbar.index(s))
    offs_n = n0 + gl.arange(0, BN, layout=gl.SliceLayout(1, acc_l))
    offs_m = gl.arange(0, BM, layout=gl.SliceLayout(0, acc_l))
    o_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
    msk = (offs_m[None, :] < M) & (offs_n[:, None] < NOUT)
    if SPLIT > 1:
        # each split stores its fp32 partial; the last CTA of the tile (atomic counter) sums them in split order
        # 0..SPLIT-1 (deterministic regardless of arrival order), stores bf16 and resets the counter.
        ws_off = offs_m[None, :] * NOUT + offs_n[:, None]
        wmask = offs_n[:, None] < NOUT
        gl.store(ws_ptr + gl.program_id(1) * (BM * NOUT) + ws_off, acc, mask=wmask)
        gl.barrier()
        old = gl.atomic_add(cnt_ptr + pid, 1, sem="acq_rel", scope="gpu")
        if old == SPLIT - 1:
            tot = gl.zeros([BN, BM], gl.float32, acc_l)
            for j in gl.static_range(SPLIT):
                tot += gl.load(ws_ptr + j * (BM * NOUT) + ws_off, mask=wmask, other=0.0, cache_modifier=".cg")
            gl.store(o_ptrs, tot.to(gl.bfloat16), mask=msk)
            gl.atomic_xchg(cnt_ptr + pid, 0, sem="relaxed", scope="gpu")
    elif SILU:
        g = acc.to(gl.bfloat16).to(gl.float32)
        u = acc_u.to(gl.bfloat16).to(gl.float32)
        res = g * (1.0 / (1.0 + gl.exp(-g))) * u
    else:
        res = acc
    if SPLIT == 1:
        gl.store(o_ptrs, res.to(gl.bfloat16), mask=msk)


_WDESC: dict = {}


def _bm_for(m: int) -> int:
    return 16 if m <= 16 else (32 if m <= 32 else 64)


def stages_for(bm: int, bk: int, silu: bool, k: int, smem_kb: int, bn: int = 64) -> int:
    per = ((2 if silu else 1) * bn * bk + bm * bk) * 2
    return max(2, min(k // bk, (smem_kb * 1024) // per))


def d2_gemm_out(x2: torch.Tensor, w: torch.Tensor, nout: int, silu: bool, *, pdl: Optional[bool] = None,
                trig: Optional[int] = None, smem_kb: Optional[int] = None, bk: Optional[int] = None,
                split: Optional[int] = None, bn: Optional[int] = None, pre: Optional[int] = None,
                out: Optional[torch.Tensor] = None) -> torch.Tensor:
    m, k = x2.shape
    bm = _bm_for(m)
    c_smem, c_bk, c_trig = _CFG.get((nout, k, silu), (200, 128, 1))
    bk = bk or D2["bk"] or c_bk
    smem_kb = smem_kb or D2["smem"] or c_smem
    pdl = bool(D2["pdl"]) if pdl is None else pdl
    trig = (D2["trig"] if D2["trig"] >= 0 else c_trig) if trig is None else trig
    split = D2_SPLIT.get((nout, k), 1) if split is None else split
    bn = bn or _BN
    stages = stages_for(bm, bk, silu, k // split, smem_kb, bn)
    if pre is None:
        pre = D2["prest"] if D2["prest"] >= 0 else D2_PREST.get((nout, k, silu), stages)
    pre = max(0, min(pre if pdl else stages, stages))
    key = (w.data_ptr(), tuple(w.shape), bn, bk)
    wd = _WDESC.get(key)
    if wd is None:
        wd = TensorDescriptor.from_tensor(w, [bn, bk], gl.NVMMASharedLayout.get_default_for([bn, bk], gl.bfloat16))
        _WDESC[key] = wd
    xd = TensorDescriptor(x2, [m, k], [x2.stride(0), 1], [bm, bk],
                          gl.NVMMASharedLayout.get_default_for([bm, bk], gl.bfloat16))
    if out is None:
        out = torch.empty((m, nout), dtype=x2.dtype, device=x2.device)
    ntile = triton.cdiv(nout, bn)
    if split > 1:
        ws, cnt = _split_buffers(x2.device, nout, ntile, split)
    else:
        ws = cnt = out
    _d2_gemm_kernel[(ntile, split)](
        wd, xd, out, ws, cnt, m, nout, k, out.stride(0),
        BN=bn, BK=bk, BM=bm, STAGES=stages, SILU=silu, PDL=pdl, TRIG=trig, SPLIT=split, PRE=pre, num_warps=4,
        launch_pdl=pdl,
    )
    return out


_SPLIT_BUF: dict = {}


def _split_buffers(device, nout: int, ntile: int, split: int):
    """Static per-(device, N, split) fp32 partials [split, 64, N] and zeroed tile counters (the last CTA resets its
    counter, so the buffers stay valid across CUDA-graph replays)."""
    key = (str(device), nout, split)
    b = _SPLIT_BUF.get(key)
    if b is None:
        b = (torch.empty((split, _MAX_M, nout), dtype=torch.float32, device=device),
             torch.zeros((ntile,), dtype=torch.int32, device=device))
        _SPLIT_BUF[key] = b
    return b


def d2_eligible(x2: torch.Tensor, w: torch.Tensor, nout: int, silu: bool) -> bool:
    m, k = x2.shape
    return (
        D2_PRE
        and 0 < m <= _MAX_M
        and (nout, k, silu) in _SHAPES
        and k % ((D2["bk"] or _CFG[(nout, k, silu)][1]) * D2_SPLIT.get((nout, k), 1)) == 0
        and x2.stride(-1) == 1
        and (x2.stride(0) * 2) % 16 == 0
        and x2.data_ptr() % 16 == 0
        and w.is_contiguous()
    )
