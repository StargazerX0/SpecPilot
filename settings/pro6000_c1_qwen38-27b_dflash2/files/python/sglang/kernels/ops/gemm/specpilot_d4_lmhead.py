"""SPECPILOT_D4: certified exact-argmax target lm_head (greedy verify).

argmax_v bf16(h_r . W_v) for M verify rows without streaming the bf16 lm_head:

1. screen   : (also ||h_r||) l~[r, v] = s_v * (h_r . q_v), q = INT8 copy of W (per-vocab-row absmax scale s_v), streamed through a
              host TMA descriptor (1 byte/weight), fp32 accumulate; per row T_r = max_v (l~ - b_rv) (atomic max),
              b_rv = ||h_r||_2 * eb_v with eb_v = ||W_v - s_v q_v||_2 + c_acc * ||W_v||_2 (Cauchy-Schwarz bound of the
              quantization error plus a worst-case fp32 summation term for both the screen and the exact recompute).
2. select   : candidates C_r = {v : l~ + b_rv >= T_r - tie_r}, tie_r = one bf16 ulp at the largest possible |l_max|
              (so every v whose bf16-rounded logit can equal the rounded maximum is kept); compacted, at most KMAX.
3. rescore  : exact bf16 h_r . W_v (fp32 accumulate, rounded to bf16 like the cuBLAS logits) for v in C_r; argmax with
              first-index ties through a 64-bit (sortable bf16, -v) atomic max.
4. fallback : rows with |C_r| > KMAX get the full exact row (a bf16 GEMM over all of W restricted to those rows); grid
              programs exit at once when no row overflowed (CUDA-graph safe, no host sync).
5. finalize : ids[r] from the packed key.

The result equals argmax over the exact bf16 logits up to accumulation-order rounding (lane numerical). Everything
runs on static buffers (one workspace per (M, V)); no host syncs.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

# worst-case relative fp32 summation error for K = 5120 terms (gamma_K with u = 2^-23 to allow truncating
# accumulators), applied to ||h|| * ||W_v|| (>= sum |h_i W_vi|); covers screen + exact recompute disagreement.
C_ACC = 5120 * 2.0**-23 * 2.0


def build_screen(weight: torch.Tensor, chunk: int = 16384) -> Dict[str, torch.Tensor]:
    """INT8 screening copy of a bf16 [V, K] lm_head + per-row scale and error bound (derived data, built once)."""
    V, K = weight.shape
    q = torch.empty((V, K), dtype=torch.int8, device=weight.device)
    s = torch.empty((V,), dtype=torch.float32, device=weight.device)
    eb = torch.empty((V,), dtype=torch.float32, device=weight.device)
    for i in range(0, V, chunk):
        w = weight[i : i + chunk].float()
        am = w.abs().amax(dim=1).clamp_min(1e-30)
        sc = am / 127.0
        qi = torch.round(w / sc[:, None]).clamp_(-127, 127)
        err = (w - qi * sc[:, None]).norm(dim=1)
        q[i : i + chunk] = qi.to(torch.int8)
        s[i : i + chunk] = sc
        # 1 + 2^-20: slack for the fp32 evaluation of the norms themselves.
        eb[i : i + chunk] = (err + C_ACC * w.norm(dim=1)) * (1.0 + 2.0**-20)
    return {"q": q, "s": s, "eb": eb, "ebmax": eb.max().reshape(1).clone()}


@triton.jit
def _tie_threshold(t, hn, ebmax):
    # |l_max| <= |T| + 2 max_v b_rv; one bf16 ulp there is <= 2^-7 of it. Monotone increasing in t.
    return t - ((tl.abs(t) + 2.0 * hn * ebmax) * 0.0078125 + 1e-30)


@triton.jit
def _d4_screen_kernel(h_ptr, q_desc, s_ptr, eb_ptr, ebmax_ptr, hn_ptr, l_ptr, u_ptr, t_ptr, M, V, K, stride_h,
                      stride_l, stride_u, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BM: tl.constexpr):
    pid = tl.program_id(0)
    n0 = pid * BLOCK_N
    m0 = tl.program_id(1) * BM
    offs_m = m0 + tl.arange(0, BM)
    offs_k = tl.arange(0, BLOCK_K)
    mmask = offs_m < M
    h_ptrs = h_ptr + offs_m[None, :] * stride_h + offs_k[:, None]
    acc = tl.zeros((BLOCK_N, BM), dtype=tl.float32)
    hsq = tl.zeros((BLOCK_K, BM), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        ht = tl.load(h_ptrs, mask=mmask[None, :], other=0.0)
        w = q_desc.load([n0, k])
        acc = tl.dot(w.to(tl.bfloat16), ht, acc)
        hf = ht.to(tl.float32)
        hsq += hf * hf
        h_ptrs += BLOCK_K
    # ||h_r|| in a fixed order (identical in every program), rounded up by 2^-20 relative for fp32 evaluation slack
    hn = tl.sqrt(tl.sum(hsq, axis=0)) * (1.0 + 9.5367431640625e-07)
    if pid == 0:
        tl.store(hn_ptr + offs_m, hn, mask=mmask)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    nmask = offs_n < V
    s = tl.load(s_ptr + offs_n, mask=nmask, other=0.0)
    eb = tl.load(eb_ptr + offs_n, mask=nmask, other=0.0)
    ebmax = tl.load(ebmax_ptr)
    lt = acc * s[:, None]
    bnd = eb[:, None] * hn[None, :]
    lo = tl.where(nmask[:, None], lt - bnd, -float("inf"))
    up = tl.max(tl.where(nmask[:, None], lt + bnd, -float("inf")), axis=0)
    tmax = tl.max(lo, axis=0)
    old = tl.atomic_max(t_ptr + offs_m, tmax, mask=mmask)
    tcur = tl.maximum(old, tmax)
    # T only grows: a tile whose upper bound is below the current threshold can never hold a candidate.
    keep = (up >= _tie_threshold(tcur, hn, ebmax)) & mmask
    tl.store(u_ptr + offs_m * stride_u + pid, tl.where(keep, up, -float("inf")), mask=mmask)
    tl.store(l_ptr + offs_m[None, :] * stride_l + offs_n[:, None], lt, mask=nmask[:, None] & keep[None, :])


@triton.jit
def _d4_select_kernel(l_ptr, u_ptr, eb_ptr, ebmax_ptr, hn_ptr, t_ptr, cnt_ptr, cand_ptr, V, NT, stride_l, stride_u,
                      KMAX, TB: tl.constexpr, BLOCK_N: tl.constexpr):
    r = tl.program_id(1)
    offs_t = tl.program_id(0) * TB + tl.arange(0, TB)
    tm = offs_t < NT
    u = tl.load(u_ptr + r * stride_u + offs_t, mask=tm, other=-float("inf"))
    hn = tl.load(hn_ptr + r)
    thr = _tie_threshold(tl.load(t_ptr + r), hn, tl.load(ebmax_ptr))
    tpass = (u >= thr) & tm
    if tl.max(tpass.to(tl.int32), axis=0) > 0:
        offs = offs_t[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        vm = tpass[:, None] & (offs < V)
        lt = tl.load(l_ptr + r * stride_l + offs, mask=vm, other=-float("inf"))
        eb = tl.load(eb_ptr + offs, mask=vm, other=0.0)
        sel = (lt + eb * hn >= thr) & vm
        si = tl.reshape(sel.to(tl.int32), (TB * BLOCK_N,))
        n = tl.sum(si, axis=0)
        base = tl.atomic_add(cnt_ptr + r, n)
        pos = base + tl.cumsum(si, axis=0) - 1
        flat = tl.reshape(offs, (TB * BLOCK_N,))
        tl.store(cand_ptr + r * KMAX + pos, flat, mask=(si > 0) & (pos < KMAX))


@triton.jit
def _sortable_key(val_bf16, v):
    u = val_bf16.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    s = tl.where((u & 0x8000) != 0, 0xFFFF - u, u | 0x8000)
    return (s.to(tl.int64) << 32) | (0x7FFFFFFF - v).to(tl.int64)


@triton.jit
def _d4_rescore_kernel(h_ptr, w_ptr, cnt_ptr, cand_ptr, best_ptr, K, stride_h, stride_w, KMAX, BKF: tl.constexpr):
    # one program per (row, candidate slot): the whole row is loaded at once (one DRAM round trip)
    r = tl.program_id(0)
    j = tl.program_id(1)
    if j < tl.minimum(tl.load(cnt_ptr + r), KMAX):
        v = tl.load(cand_ptr + r * KMAX + j)
        offs_k = tl.arange(0, BKF)
        km = offs_k < K
        w = tl.load(w_ptr + v.to(tl.int64) * stride_w + offs_k, mask=km, other=0.0).to(tl.float32)
        hk = tl.load(h_ptr + r * stride_h + offs_k, mask=km, other=0.0).to(tl.float32)
        acc = tl.sum(w * hk, axis=0)
        tl.atomic_max(best_ptr + r, _sortable_key(acc.to(tl.bfloat16), v))


@triton.jit
def _d4_fallback_kernel(h_ptr, w_desc, cnt_ptr, best_ptr, M, V, K, stride_h, KMAX,
                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BM: tl.constexpr):
    m0 = tl.program_id(1) * BM
    offs_m = m0 + tl.arange(0, BM)
    mmask = offs_m < M
    cnt = tl.load(cnt_ptr + offs_m, mask=mmask, other=0)
    ovf = (cnt > KMAX) & mmask
    if tl.max(ovf.to(tl.int32), axis=0) > 0:
        offs_k = tl.arange(0, BLOCK_K)
        for n0 in range(tl.program_id(0) * BLOCK_N, V, tl.num_programs(0) * BLOCK_N):
            h_ptrs = h_ptr + offs_m[None, :] * stride_h + offs_k[:, None]
            acc = tl.zeros((BLOCK_N, BM), dtype=tl.float32)
            for k in range(0, K, BLOCK_K):
                ht = tl.load(h_ptrs, mask=ovf[None, :], other=0.0)
                w = w_desc.load([n0, k])
                acc = tl.dot(w, ht, acc)
                h_ptrs += BLOCK_K
            offs_n = n0 + tl.arange(0, BLOCK_N)
            key = _sortable_key(acc.to(tl.bfloat16), offs_n[:, None])
            key = tl.where((offs_n < V)[:, None], key, 0)
            tl.atomic_max(best_ptr + offs_m, tl.max(key, axis=0), mask=ovf)


@triton.jit
def _d4_finalize_kernel(best_ptr, ids_ptr, t_ptr, cnt_ptr, cnt_last_ptr, M, BLOCK: tl.constexpr):
    # also re-arms the per-row state for the next call (screen and select start from T = -inf, cnt = 0, best = 0)
    offs = tl.arange(0, BLOCK)
    m = offs < M
    key = tl.load(best_ptr + offs, mask=m, other=0)
    tl.store(ids_ptr + offs, 0x7FFFFFFF - (key & 0xFFFFFFFF), mask=m)
    tl.store(cnt_last_ptr + offs, tl.load(cnt_ptr + offs, mask=m, other=0), mask=m)
    tl.store(best_ptr + offs, tl.zeros((BLOCK,), dtype=tl.int64), mask=m)
    tl.store(t_ptr + offs, tl.full((BLOCK,), -float("inf"), dtype=tl.float32), mask=m)
    tl.store(cnt_ptr + offs, tl.zeros((BLOCK,), dtype=tl.int32), mask=m)


# Measured on H100 SXM (scratch/d4/an/screen_var.py, CUDA graph after an L2 flush); re-tune when the kernel changes.
SCREEN_CFG = (128, 128, 4, 4)
FALLBACK_CFG = (128, 64, 4, 4)
FALLBACK_CTAS = 264
SELECT_TB = 16
_DESC: dict = {}


def _desc(w: torch.Tensor, bn: int, bk: int):
    from triton.tools.tensor_descriptor import TensorDescriptor

    key = (w.data_ptr(), tuple(w.shape), w.dtype, bn, bk)
    d = _DESC.get(key)
    if d is None:
        d = TensorDescriptor.from_tensor(w, [bn, bk])
        _DESC[key] = d
    return d


class CertifiedArgmaxHead:
    """Static-buffer driver of the five kernels for one lm_head (weights + screen) and any M <= max_rows."""

    def __init__(self, weight: torch.Tensor, kmax: int = 128, screen: Optional[Dict[str, torch.Tensor]] = None,
                 max_rows: int = 64, screen_cfg: Tuple[int, int, int, int] = SCREEN_CFG):
        assert weight.dtype == torch.bfloat16 and weight.dim() == 2 and weight.is_contiguous()
        self.w = weight
        self.V, self.K = weight.shape
        self.kmax = int(kmax)
        self.screen = screen if screen is not None else build_screen(weight)
        self.screen_cfg = screen_cfg
        dev = weight.device
        self.max_rows = max_rows
        self.hn = torch.zeros(max_rows, dtype=torch.float32, device=dev)
        self.t = torch.full((max_rows,), -float("inf"), dtype=torch.float32, device=dev)
        self.cnt = torch.zeros(max_rows, dtype=torch.int32, device=dev)
        self.cnt_last = torch.zeros(max_rows, dtype=torch.int32, device=dev)  # candidate counts of the last call
        self.best = torch.zeros(max_rows, dtype=torch.int64, device=dev)
        self.ids = torch.zeros(max_rows, dtype=torch.int64, device=dev)
        self.cand = torch.zeros(max_rows * self.kmax, dtype=torch.int32, device=dev)
        self.lbuf = torch.empty((max_rows, self.V), dtype=torch.float32, device=dev)
        self.ntiles = triton.cdiv(self.V, screen_cfg[0])
        self.ubuf = torch.empty((max_rows, self.ntiles), dtype=torch.float32, device=dev)
        self.hbuf = torch.zeros((max_rows, self.K), dtype=torch.bfloat16, device=dev)
        # SPECPILOT_D1: stages fitted per BM to the device's smem (99 KB/block on sm_120; every H100 tile unchanged)
        props = torch.cuda.get_device_properties(dev)
        optin = props.shared_memory_per_block_optin
        budget = optin - (8192 if optin > 200 * 1024 else 16384)
        self.fallback_ctas = 2 * props.multi_processor_count  # H100: 264 = FALLBACK_CTAS
        bn, bk, ns, _ = screen_cfg
        fbn, fbk, fns, _ = FALLBACK_CFG
        self.stages = {
            bm: (max(2, min(ns, budget // (bn * bk + bm * bk * 2))), max(2, min(fns, budget // ((fbn + bm) * fbk * 2))))
            for bm in (16, 32, 64)
        }

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        """ids[M] (int64, a view of a static buffer) = certified argmax of bf16(h @ W^T); h [M, K] bf16."""
        M, K = h.shape
        assert K == self.K and M <= self.max_rows and h.stride(1) == 1 and h.dtype == torch.bfloat16
        V = self.V
        bn, bk, _, nw = self.screen_cfg
        bm = 16 if M <= 16 else (32 if M <= 32 else 64)
        mt = triton.cdiv(M, bm)
        ns, fns = self.stages[bm]
        _d4_screen_kernel[(self.ntiles, mt)](
            h, _desc(self.screen["q"], bn, bk), self.screen["s"], self.screen["eb"], self.screen["ebmax"], self.hn,
            self.lbuf, self.ubuf, self.t, M, V, K, h.stride(0), self.lbuf.stride(0), self.ubuf.stride(0),
            BLOCK_N=bn, BLOCK_K=bk, BM=bm, num_warps=nw, num_stages=ns)
        _d4_select_kernel[(triton.cdiv(self.ntiles, SELECT_TB), M)](
            self.lbuf, self.ubuf, self.screen["eb"], self.screen["ebmax"], self.hn, self.t, self.cnt, self.cand, V,
            self.ntiles, self.lbuf.stride(0), self.ubuf.stride(0), self.kmax, TB=SELECT_TB, BLOCK_N=bn, num_warps=4)
        _d4_rescore_kernel[(M, self.kmax)](
            h, self.w, self.cnt, self.cand, self.best, K, h.stride(0), self.w.stride(0), self.kmax,
            BKF=triton.next_power_of_2(K), num_warps=4)
        fbn, fbk, _, fnw = FALLBACK_CFG
        _d4_fallback_kernel[(min(self.fallback_ctas, triton.cdiv(V, fbn)), mt)](
            h, _desc(self.w, fbn, fbk), self.cnt, self.best, M, V, K, h.stride(0), self.kmax,
            BLOCK_N=fbn, BLOCK_K=fbk, BM=bm, num_warps=fnw, num_stages=fns)
        _d4_finalize_kernel[(1,)](self.best, self.ids, self.t, self.cnt, self.cnt_last, M, BLOCK=triton.next_power_of_2(max(M, 16)),
                                  num_warps=1)
        return self.ids[:M]

    def screened_logits(self, M: int) -> torch.Tensor:
        return self.lbuf[:M]


def reference_argmax(h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The parent's semantics: argmax (first index) of the bf16 logits of torch.matmul."""
    return torch.argmax(torch.matmul(h, w.T), dim=-1)
