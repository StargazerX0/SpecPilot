"""SPECPILOT_D4_L2PF: keep the W12 weight stream running through a layer's latency-bound section.

After a layer's first W12 GEMM (GDN in_proj or attention qkv_proj) at M <= 64, a forked stream issues
cp.async.bulk.prefetch.L2 of the next W12 records in the W12 producer's consumption order: the first OF % of every
stream-K CTA range of out_proj / o_proj, then the first GU MB of the layer's gate_up. The GDN tree section and the
split-KV attention section leave DRAM idle for ~20 us per layer; the following GEMMs then hit L2. The fork is joined
before the next layer's first W12 GEMM (or the lm_head). Data movement only: GEMM outputs are bitwise unchanged.
Three plans per layer: exact pass (12 KB W12 records), TLV Q pass on the nested view (first 8 KB of each record) and
TLV Q pass on d3's separately stored Q copies (D3QLinear records, SPECPILOT_D3_QCOPY; used at M <= 32).

Knobs (read once at import; SPECPILOT_D4_L2PF unset = off, parent behaviour):
  SPECPILOT_D4_L2PF=1        enable
  SPECPILOT_D4_L2PF_OF       percent of each out_proj / o_proj CTA range (default 100)
  SPECPILOT_D4_L2PF_GU       MB of the layer's gate_up records, first records of every CTA range (default 0)
  SPECPILOT_D4_L2PF_CTA      prefetch CTAs (default 16; 8 is issue-rate bound, microbench 6d1b39b2112a)
  SPECPILOT_D4_L2PF_QGU      gate_up MB in the TLV Q pass (8 KB Q records; default 64)
  SPECPILOT_D4_L2PF_QCTA     prefetch CTAs in the Q pass (default 32)
  (defaults = best of microbench 13420859dade: exact -13.2 us/layer, Q -14.3 us/layer in a steady W12 stream)
  SPECPILOT_D4_L2PF_POL      L2 cache hint: 2 evict_first (default), 0 none, 1 evict_last
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


D4_L2PF = _env_int("SPECPILOT_D4_L2PF", 0)
D4_L2PF_OF = _env_int("SPECPILOT_D4_L2PF_OF", 100)
D4_L2PF_GU = _env_int("SPECPILOT_D4_L2PF_GU", 0)
D4_L2PF_CTA = _env_int("SPECPILOT_D4_L2PF_CTA", 16)
D4_L2PF_QGU = _env_int("SPECPILOT_D4_L2PF_QGU", 64)
D4_L2PF_QCTA = _env_int("SPECPILOT_D4_L2PF_QCTA", 32)
D4_L2PF_POL = _env_int("SPECPILOT_D4_L2PF_POL", 2)

# W12 record layout (specpilot_d2_w12): 12 KB per (128-row tile, 64-k chunk); the Q view reads its first 8 KB.
REC_BYTES = 12288
QREC_BYTES = 8192

_CU = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__global__ void d4_l2pf_kernel(const long long* __restrict__ addr, const int* __restrict__ sizes, int n, int bytes,
                               int pol) {
  const int nth = gridDim.x * blockDim.x;
  const unsigned long long policy = pol == 1 ? 0x14F0000000000000ull : 0x12F0000000000000ull;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += nth) {
    const int b = sizes ? sizes[i] : bytes;
    if (pol)
      asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;\n"
                   ::"l"(addr[i]), "r"(b), "l"(policy) : "memory");
    else
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(addr[i]), "r"(b) : "memory");
  }
}

void d4_l2pf(torch::Tensor addr, torch::Tensor sizes, int64_t bytes, int64_t ctas, int64_t pol) {
  auto st = c10::cuda::getCurrentCUDAStream();
  const int* sz = sizes.numel() ? sizes.data_ptr<int>() : nullptr;
  d4_l2pf_kernel<<<(int)ctas, 128, 0, st>>>((const long long*)addr.data_ptr<int64_t>(), sz, (int)addr.numel(),
                                           (int)bytes, (int)pol);
}
"""
_CPP = "void d4_l2pf(torch::Tensor addr, torch::Tensor sizes, int64_t bytes, int64_t ctas, int64_t pol);\n"
_mod = None
_SIDE: dict = {}
_PENDING = [False]


def _ext():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
        cflags = ["-O3", "-gencode=arch=compute_120a,code=sm_120a"]
        # one build directory per source version (as specpilot_d2_w12)
        key = hashlib.sha1("\0".join([_CPP, _CU, *cflags]).encode()).hexdigest()[:16]
        base = os.environ.get("SGLANG_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "sglang")
        bdir = os.path.join(base, "specpilot_d4_l2pf", key)
        os.makedirs(bdir, exist_ok=True)
        _mod = load_inline(name=f"specpilot_d4_l2pf_{key}", cpp_sources=_CPP, cuda_sources=_CU,
                           functions=["d4_l2pf"], extra_cuda_cflags=cflags, build_directory=bdir, verbose=False)
    return _mod


def _geom(enc, d3: bool):
    """(T, S, P, base, tile stride, record bytes, rotation) of the stream the GEMM reads: W12 records or the D3Q copy."""
    from sglang.kernels.ops.gemm import specpilot_d2_w12 as W

    if d3:
        q = enc.d3q
        return q.T, q.S, q.P, q.recs.data_ptr(), q.recs.stride(0), q.rec_bytes, 0
    return enc.T, enc.S, enc.P, enc.planes.data_ptr(), enc.planes.stride(0), REC_BYTES, W.ROT


def walk_units(T: int, S: int, P: int, nrec_per_cta: int, rot: int) -> List[Tuple[int, int]]:
    """(tile, k-step) of the first nrec units of every stream-K CTA in its producer's walk order (contiguous unit
    range u0..u1 per CTA; W12 rotates the start inside each tile by (tile * rot + p) % len, D3Q has rot = 0),
    interleaved across CTAs (earliest-consumed first). Mirrors w12m_kernel's `walk` and d3q's unit loop."""
    U = T * S
    per = []
    for p in range(P):
        u0, u1 = (p * U) // P, ((p + 1) * U) // P
        out = []
        tile = sbeg = srot = 0
        slen = 1
        for u in range(u0, min(u1, u0 + nrec_per_cta)):
            if u == u0 or u - tile * S == S:
                tile = u // S
                sbeg = u - tile * S
                slen = min(u1, (tile + 1) * S) - u
                srot = ((tile * rot + p) % slen) if rot else 0
            s = (u - tile * S) - sbeg + srot
            if s >= slen:
                s -= slen
            out.append((tile, s + sbeg))
        per.append(out)
    units = []
    for j in range(max((len(x) for x in per), default=0)):
        for x in per:
            if j < len(x):
                units.append(x[j])
    return units


def walk_addrs(enc, nrec_per_cta: int, rot: int) -> List[int]:
    """W12 record addresses of walk_units (exact-pass geometry)."""
    base, tstride = enc.planes.data_ptr(), enc.planes.stride(0)
    return [base + t * tstride + s * REC_BYTES for t, s in walk_units(enc.T, enc.S, enc.P, nrec_per_cta, rot)]


def recs_per_cta(enc) -> int:
    return (enc.T * enc.S + enc.P - 1) // enc.P


def _stream(enc, d3: bool, nrec: Optional[int], mbytes: int, read_bytes: int):
    """(addresses, sizes) of the first nrec units per CTA (or the first mbytes MB) of one GEMM's stream."""
    T, S, P, base, tstride, rb, rot = _geom(enc, d3)
    if nrec is None:
        nrec = max(1, (mbytes * 1000000) // (read_bytes * P))
    units = walk_units(T, S, P, nrec, rot)
    return [base + t * tstride + s * rb for t, s in units], [read_bytes] * len(units)


def plan_mode(nxt_out, nxt_gate_up, of_pct: int, gu_mb: int, mode: str, device):
    """Prefetch list of one pass mode ('exact' | 'qnest' | 'qd3'): out_proj/o_proj first of_pct % of every CTA range,
    then gate_up's first gu_mb MB -> (int64 addresses, int32 sizes) or None."""
    a: List[int] = []
    z: List[int] = []
    for enc, pct, mb in ((nxt_out, of_pct, 0), (nxt_gate_up, 0, gu_mb)):
        if enc is None or (pct <= 0 and mb <= 0):
            continue
        d3 = mode == "qd3" and enc.d3q is not None
        rb = (enc.d3q.rec_bytes if d3 else (QREC_BYTES if mode != "exact" else REC_BYTES))
        T, S, P = (enc.d3q.T, enc.d3q.S, enc.d3q.P) if d3 else (enc.T, enc.S, enc.P)
        nrec = max(1, (pct * ((T * S + P - 1) // P) + 99) // 100) if pct > 0 else None
        aa, zz = _stream(enc, d3, nrec, mb, rb)
        a += aa
        z += zz
    if not a:
        return None
    return (torch.tensor(a, dtype=torch.int64, device=device), torch.tensor(z, dtype=torch.int32, device=device))


def build_plans(model: torch.nn.Module) -> int:
    """Attach `d4_pf` = {mode: plan} to each decoder layer's first W12 weight; returns the number of layers planned.
    Runs after the D3 Q copies are attached (their record tensors are the Q-pass stream)."""
    from sglang.kernels.ops.gemm import specpilot_d2_w12 as W
    from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer, Qwen3_5LinearDecoderLayer

    n = 0
    for mod in model.modules():
        if isinstance(mod, Qwen3_5LinearDecoderLayer):
            src = getattr(mod.linear_attn, "_d2_in_proj", None)
            nxt = getattr(mod.linear_attn.out_proj, "_d2_w12", None)
        elif isinstance(mod, Qwen3_5AttentionDecoderLayer):
            src = getattr(mod.qkv_proj, "_d2_w12", None)
            nxt = getattr(mod.o_proj, "_d2_w12", None)
        else:
            continue
        gu = getattr(getattr(mod.mlp, "gate_up_proj", None), "_d2_w12", None)  # dense MLP only (MoE: none)
        if src is None:
            continue
        dev = src.planes.device
        plans = {"exact": plan_mode(nxt, gu, D4_L2PF_OF, D4_L2PF_GU, "exact", dev)}
        if W.D4_QVIEW:
            plans["qnest"] = plan_mode(nxt, gu, D4_L2PF_OF, D4_L2PF_QGU, "qnest", dev)
            # without Q copies the Q pass at M <= 32 reads the nested view
            plans["qd3"] = (plan_mode(nxt, gu, D4_L2PF_OF, D4_L2PF_QGU, "qd3", dev) if src.d3q is not None
                            else plans["qnest"])
        if all(v is None for v in plans.values()):
            continue
        src.d4_pf = plans
        n += 1
    if n:
        _ext()
        dev = next(model.parameters()).device
        _SIDE[dev] = torch.cuda.Stream(device=dev)
    logger.info("SPECPILOT_D4_L2PF: %d layers planned (of %d%%, gate_up %d MB / Q %d MB, %d / Q %d CTAs, hint %d)", n,
                D4_L2PF_OF, D4_L2PF_GU, D4_L2PF_QGU, D4_L2PF_CTA, D4_L2PF_QCTA, D4_L2PF_POL)
    return n


def join() -> None:
    """Join the pending prefetch fork into the current stream (before the next layer's first W12 GEMM)."""
    if _PENDING[0]:
        cur = torch.cuda.current_stream()
        cur.wait_stream(_SIDE[cur.device])
        _PENDING[0] = False


def fork(plans: dict, mode: str) -> None:
    """Issue the prefetch of plans[mode] on the side stream, after the work queued so far on the current stream."""
    join()
    plan = plans.get(mode)
    if plan is None:
        return
    cur = torch.cuda.current_stream()
    side = _SIDE[cur.device]
    side.wait_stream(cur)
    with torch.cuda.stream(side):
        _ext().d4_l2pf(plan[0], plan[1], 0, D4_L2PF_CTA if mode == "exact" else D4_L2PF_QCTA, D4_L2PF_POL)
    _PENDING[0] = True
