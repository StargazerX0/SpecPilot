"""SPECPILOT_D4: engine glue of the certified exact-argmax target lm_head (kernels: kernels/ops/gemm/specpilot_d4_lmhead).

Knobs (read once at import; all unset = the parent's lm_head path):
  SPECPILOT_D4_HEAD=int8   certified argmax for target-verify forwards (INT8 screening copy built before graph capture)
  SPECPILOT_D4_KMAX=<n>    exact-recompute candidates per verify row (rows above it take the full-row fallback)

The verify graph returns the screened fp32 logits buffer as next_token_logits; the DFlash worker recognises that
buffer (``certified_ids``) and takes the certified argmax ids, or recomputes the exact bf16 logits from the stashed
lm_head input (``exact_logits``) for every consumer beyond the greedy argmax.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch

from sglang.srt.speculative import specpilot_d2q as _d2q  # SPECPILOT_D2Q

logger = logging.getLogger(__name__)

_MODE = os.environ.get("SPECPILOT_D4_HEAD", "").strip().lower()
ENABLED = _MODE == "int8"
KMAX = int(os.environ.get("SPECPILOT_D4_KMAX", "128") or 128)
# bs <= 4 verify graphs of the c1 protocol (cuda_graph max_bs 4) at 32 nodes each.
MAX_ROWS = int(os.environ.get("SPECPILOT_D4_MAX_ROWS", "128") or 128)

_heads: Dict[int, object] = {}
_by_lbuf: Dict[int, object] = {}


def _bf16_weight(mod: torch.nn.Module) -> Optional[torch.Tensor]:
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

    if not isinstance(mod, ParallelLMHead):
        return None
    w = mod.weight
    ok = w.dim() == 2 and w.dtype == torch.bfloat16 and w.is_cuda and w.is_contiguous()
    return w if ok else None


def _lm_heads(model: torch.nn.Module):
    for _, mod in model.named_modules():
        w = _bf16_weight(mod)
        if w is not None:
            yield w


def prepare(model: torch.nn.Module) -> None:
    """Builds the screening copy of every bf16 lm_head of ``model`` (call outside graph capture)."""
    if not ENABLED:
        return
    from sglang.kernels.ops.gemm.specpilot_d4_lmhead import CertifiedArgmaxHead

    for w in _lm_heads(model):
        if w.data_ptr() in _heads:
            continue
        head = CertifiedArgmaxHead(w.data, kmax=KMAX, max_rows=MAX_ROWS)
        _heads[w.data_ptr()] = head
        _d2q.prepare_head(head)  # SPECPILOT_D2Q (no-op unless SPECPILOT_D2Q_HEAD)
        _by_lbuf[head.lbuf.data_ptr()] = head
        logger.info("SPECPILOT_D4: certified lm_head screen built for %s (kmax %d, rows %d)", tuple(w.shape), KMAX,
                    MAX_ROWS)


def maybe_screen(hidden: torch.Tensor, lm_head: torch.nn.Module, is_target_verify: bool) -> Optional[torch.Tensor]:
    """Screened logits buffer [M, V] (fp32, NOT the logits) after computing the certified ids, or None (parent path)."""
    if not (ENABLED and is_target_verify) or hidden.dim() != 2 or hidden.dtype != torch.bfloat16:
        return None
    w = _bf16_weight(lm_head)
    head = _heads.get(w.data_ptr()) if w is not None else None
    if head is None or hidden.shape[0] > head.max_rows or hidden.shape[1] != head.K:
        return None
    m = hidden.shape[0]
    head.hbuf[:m].copy_(hidden)
    if _d2q.maybe_q_head(head, head.hbuf[:m]):  # SPECPILOT_D2Q: Q pass top-1 into head.ids (lbuf not written)
        return head.lbuf[:m]
    head(head.hbuf[:m])
    return head.lbuf[:m]


def _head_of(logits: Optional[torch.Tensor]):
    if not ENABLED or logits is None or logits.dtype != torch.float32:
        return None
    return _by_lbuf.get(logits.data_ptr())


def certified_ids(logits: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """The certified argmax ids [M] when ``logits`` is a screened buffer from ``maybe_screen``, else None."""
    head = _head_of(logits)
    return None if head is None else head.ids[: logits.shape[0]]


def exact_logits(logits: torch.Tensor) -> torch.Tensor:
    """The parent's exact bf16 logits for the rows behind a screened buffer (same GEMM as the parent lm_head)."""
    head = _head_of(logits)
    m = logits.shape[0]
    return torch.matmul(head.hbuf[:m], head.w.T)
