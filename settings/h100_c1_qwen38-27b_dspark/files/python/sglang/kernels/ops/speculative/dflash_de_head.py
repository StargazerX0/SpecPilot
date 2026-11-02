"""SPECPILOT_DE: draft-only derived candidate head for DFlash2 candidate extraction.

DFlash2 projects the drafted hiddens through the *target* lm_head over the whole vocabulary to take a top-k
(`DFlash2DraftModel.compute_candidates`). This module builds, once at lm_head attach time (before CUDA-graph
capture), a derived copy of that head used only by the drafter:

- `SPECPILOT_DE_VOCAB_SIZE=S` (0/unset = full vocabulary): keep S rows. Ranking: calib-split target-completion token
  counts (`dflash_de_calib_vocab.json`, counted on the calib_trace outputs), then ascending token id (BPE merge rank,
  a frequency prior), always including ids >= `_SPECIAL_FLOOR` (Qwen special / think / EOS tokens). Sorted ascending.
- `SPECPILOT_DE_HEAD_PREC=fp8`: per-row amax-scaled float8_e4m3fn weights, multiplied by bf16 activations in a
  weight-only (W8A16) Triton GEMM; `bf16` keeps a contiguous bf16 copy of the kept rows.

The target keeps its own bf16 head for verification, so outputs are unchanged (drafts enter only through the accept
check). Callers get logits over the kept rows plus `sub_to_full` to map top-k positions back to full-vocab ids, so a
top-k with any k <= S (trees: k > 1, scores included) works unchanged. With both knobs unset nothing here runs.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_VOCAB_FILE = Path(__file__).with_name("dflash_de_calib_vocab.json")
_SPECIAL_FLOOR = 248000
_FP8_MAX = 448.0


def read_de_head_knobs() -> Optional[dict]:
    """Read the SPECPILOT_DE head knobs once (model init). None = stock path."""
    prec = os.environ.get("SPECPILOT_DE_HEAD_PREC", "").strip().lower() or "bf16"
    size = int(os.environ.get("SPECPILOT_DE_VOCAB_SIZE", "0") or 0)
    if prec not in ("bf16", "fp8"):
        raise ValueError(f"SPECPILOT_DE_HEAD_PREC must be bf16 or fp8, got {prec!r}")
    if prec == "bf16" and size <= 0:
        return None
    return {"prec": prec, "vocab_size": size}


def build_vocab_subset(size: int, num_org: int, ranked_ids=None) -> torch.Tensor:
    """Ascending full-vocab ids: specials (>= _SPECIAL_FLOOR) + calib-ranked ids + ascending-id fill, `size` total."""
    if ranked_ids is None:
        ranked_ids = json.loads(_VOCAB_FILE.read_text())["ranked_ids"]
    size = min(int(size), int(num_org))
    chosen = torch.zeros(int(num_org), dtype=torch.bool)
    chosen[_SPECIAL_FLOOR:num_org] = True
    have = int(chosen.sum())
    for i in ranked_ids:
        if have >= size:
            break
        if 0 <= i < num_org and not bool(chosen[i]):
            chosen[i] = True
            have += 1
    need = size - have
    if need > 0:
        free = torch.nonzero(~chosen, as_tuple=False).reshape(-1)[:need]
        chosen[free] = True
    return torch.nonzero(chosen, as_tuple=False).reshape(-1)


@triton.jit
def _w8a16_head_kernel(x_ptr, w_ptr, s_ptr, out_ptr, M, N, K, stride_xm, stride_wn, stride_om,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :], mask=offs_m[:, None] < M, other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + (k0 + offs_k)[None, :], mask=n_mask[:, None], other=0.0)
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)))
    s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc * s[None, :]
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :], acc.to(out_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & n_mask[None, :])


_SMEM_BUDGET = 196608  # SPECPILOT_INT: conservative vs the H100 limit (232448 B); bf16 x tile + bf16-converted w tile


def _smem_estimate(block_m: int, block_n: int, block_k: int, num_stages: int) -> int:  # SPECPILOT_INT
    return num_stages * (block_m * block_k * 2 + block_n * block_k * 2)


def w8a16_head(x: torch.Tensor, w_fp8: torch.Tensor, scale: torch.Tensor, block_n: int = 64,
               block_k: int = 256, num_warps: int = 4, num_stages: int = 4) -> torch.Tensor:
    """x [M, K] bf16, w_fp8 [N, K] float8_e4m3fn row-major, scale [N] fp32 -> [M, N] bf16. K % block_k == 0."""
    x = x.to(torch.bfloat16).contiguous()
    M, K = x.shape
    N = w_fp8.shape[0]
    assert K % block_k == 0, (K, block_k)
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    block_m = max(16, triton.next_power_of_2(M))
    # SPECPILOT_INT: verify block 12/16 at bs 3-4 gives 33-60 rows (block_m 64); the M<=32-tuned tiles then exceed H100
    # shared memory (OutOfResources at graph capture). Shrink block_k, then num_stages, until the estimate fits.
    # Tiles with block_m <= 32 (every batch-1 shape) are unchanged.
    while _smem_estimate(block_m, block_n, block_k, num_stages) > _SMEM_BUDGET and block_k > 64 and K % (block_k // 2) == 0:
        block_k //= 2
    while _smem_estimate(block_m, block_n, block_k, num_stages) > _SMEM_BUDGET and num_stages > 2:
        num_stages -= 1
    _w8a16_head_kernel[(triton.cdiv(N, block_n),)](
        x, w_fp8, scale, out, M, N, K, x.stride(0), w_fp8.stride(0), out.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=num_warps, num_stages=num_stages)
    return out


def quantize_rows_fp8(weight: torch.Tensor, chunk: int = 8192):
    """Per-row amax e4m3 copy, computed in row chunks (bounded fp32 scratch). -> (q [N, K] fp8, scale [N] fp32)."""
    n, k = weight.shape
    q = torch.empty((n, k), device=weight.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((n,), device=weight.device, dtype=torch.float32)
    for r0 in range(0, n, chunk):
        w = weight[r0:r0 + chunk].float()
        s = (w.abs().amax(dim=1) / _FP8_MAX).clamp_min(1e-12)
        q[r0:r0 + chunk] = (w / s[:, None]).clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        scale[r0:r0 + chunk] = s
    return q, scale


class DEDraftHead:
    """Derived draft-only head: `project(hidden) -> logits [N, S]`, `sub_to_full` (None = identity)."""

    def __init__(self, lm_head, *, prec: str, vocab_size: int) -> None:
        weight = lm_head.weight
        num_org = int(lm_head.org_vocab_size)
        self.prec = prec
        with torch.no_grad():
            if vocab_size and vocab_size < num_org:
                ids = build_vocab_subset(vocab_size, num_org).to(weight.device)
                self.sub_to_full = ids.long().contiguous()
                rows = weight.index_select(0, self.sub_to_full)
            else:
                self.sub_to_full = None
                rows = weight[:num_org]
            if prec == "fp8":
                self.weight, self.scale = quantize_rows_fp8(rows)
            else:
                self.weight, self.scale = rows.contiguous(), None
            del rows
        self.size = int(self.weight.shape[0])
        logger.info(
            "SPECPILOT_DE draft head: prec=%s rows=%d/%d bytes=%.1f MB (draft-only derived copy of the target lm_head)",
            prec, self.size, num_org, self.weight.numel() * self.weight.element_size() / 2**20)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.scale is not None:
            return w8a16_head(hidden, self.weight, self.scale)
        return torch.matmul(hidden.to(self.weight.dtype), self.weight.T)

    def to_full(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.long()
        return ids if self.sub_to_full is None else self.sub_to_full[ids]


# ---------------------------------------------------------------------------------------------------------------
# SPECPILOT_DE_DRAFTER_PREC=fp8: weight-only fp8 derived copies of the drafter backbone linears
# ---------------------------------------------------------------------------------------------------------------


def read_de_drafter_prec() -> Optional[str]:
    """Read SPECPILOT_DE_DRAFTER_PREC once (model init): None (stock) or 'fp8'."""
    prec = os.environ.get("SPECPILOT_DE_DRAFTER_PREC", "").strip().lower()
    if prec in ("", "bf16"):
        return None
    if prec != "fp8":
        raise ValueError(f"SPECPILOT_DE_DRAFTER_PREC must be bf16 or fp8, got {prec!r}")
    return prec


def _linear_config(n: int, k: int) -> dict:
    """Tile choice for the M<=32 weight-only GEMM, from microbench 0a67c4ef6ab1 (H100, M=8, after-GEMM):
    gate_up [34816x5120] and down [5120x17408] are fastest at block_n=32/block_k=512, qkv at 32/256."""
    if k % 512 == 0 and (n >= 16384 or k >= 16384):
        return {"block_n": 32, "block_k": 512, "num_stages": 3}
    return {"block_n": 32, "block_k": 256 if k % 256 == 0 else 128}


# SPECPILOT_X4: (verify width N, drafter block B) when the tree verifies N > B nodes over a block-B lattice; the
# draft rows >= B of each request are never read (draft keys stop at B, lattice uses rows < B), so the W8A16
# linears compute rows < B only. None = parent behaviour. Set once by the worker before graph capture.
_X4_BLOCK_ROWS: Optional[tuple] = None


def set_x4_block_rows(n: int, b: int) -> None:
    global _X4_BLOCK_ROWS
    _X4_BLOCK_ROWS = (int(n), int(b)) if 0 < int(b) < int(n) else None


def _x4_rows_forward(x2: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, cfg: dict):
    n_rows, b_rows = _X4_BLOCK_ROWS
    bs = x2.shape[0] // n_rows
    xs = x2.view(bs, n_rows, -1)[:, :b_rows].reshape(bs * b_rows, -1)
    out = x2.new_zeros((bs, n_rows, q.shape[0]))
    out[:, :b_rows] = w8a16_head(xs, q, scale, **cfg).view(bs, b_rows, -1)
    return out.view(bs * n_rows, -1)


def _make_w8a16_forward(q: torch.Tensor, scale: torch.Tensor, cfg: dict, weight: torch.Tensor):
    def forward(x: torch.Tensor):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if _X4_BLOCK_ROWS is not None and x2.shape[0] % _X4_BLOCK_ROWS[0] == 0:  # SPECPILOT_X4
            if x2.shape[0] // _X4_BLOCK_ROWS[0] * _X4_BLOCK_ROWS[1] <= 64:
                return _x4_rows_forward(x2, q, scale, cfg).reshape(*shape[:-1], q.shape[0]), None
        if x2.shape[0] > 64:  # small-M kernel only; larger batches keep the bf16 GEMM
            return torch.nn.functional.linear(x, weight), None
        y = w8a16_head(x2, q, scale, **cfg)
        return y.reshape(*shape[:-1], q.shape[0]), None

    return forward


def quantize_drafter_linears_(layers) -> list:
    """Attach fp8 W8A16 forwards to each drafter layer's qkv/gate_up/down linears (o_proj: no gain at M=8) (bias-free, unquantized only).
    The bf16 parameters stay in place (the fused draft-KV append slices qkv_proj.weight). Returns patched names."""
    patched = []
    with torch.no_grad():
        for i, layer in enumerate(layers):
            attn, mlp = getattr(layer, "self_attn", None), getattr(layer, "mlp", None)
            for name, mod in (("qkv_proj", getattr(attn, "qkv_proj", None)),
                              ("gate_up_proj", getattr(mlp, "gate_up_proj", None)),
                              ("down_proj", getattr(mlp, "down_proj", None))):
                w = getattr(mod, "weight", None)
                if w is None or w.dim() != 2 or w.dtype not in (torch.bfloat16, torch.float16):
                    continue
                if getattr(mod, "bias", None) is not None:
                    continue
                if getattr(mod, "tp_size", 1) != 1:
                    continue
                q, s = quantize_rows_fp8(w.data)
                n, k = q.shape
                if k % 128:
                    continue
                object.__setattr__(mod, "forward", _make_w8a16_forward(q, s, _linear_config(n, k), w))
                object.__setattr__(mod, "_de_fp8", (q, s))
                patched.append(f"layers.{i}.{name}[{n}x{k}]")
    logger.info("SPECPILOT_DE drafter fp8 W8A16 linears: %d patched (%s)", len(patched), ", ".join(patched[:4]))
    return patched
