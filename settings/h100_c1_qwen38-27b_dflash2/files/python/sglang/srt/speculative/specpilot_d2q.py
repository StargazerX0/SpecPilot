"""SPECPILOT_D2Q: Q-pass lm_head for the two-level verify (SPECPILOT_D8R_TLV).

The Q verify pass (the d8q variant of the captured target-verify graph) only proposes a chain: every token it accepts
is re-verified by the exact pass, so a Q-only change keeps the outputs lossless. Here the Q pass's lm_head is restricted
to the drafter's vocabulary subset (SPECPILOT_DE_VOCAB_SIZE ids, default 98304): the tree's tokens are all subset
tokens, so the restriction can only change Q's bonus token, and only when the full-vocabulary top-1 lies outside the
subset (0 of 108k dev positions had a top-20 without a subset token). Within the subset the argmax is d4's certified
exact argmax (CertifiedArgmaxHead over a contiguous subset copy: bf16 1.0 GB + INT8 screen 0.5 GB, built before graph
capture), so the screen streams 0.5 GB instead of 1.27 GB (sweep 177cc5379978: -0.26 ms per two-level step, tokens
per exact pass unchanged, identical outputs 98/100).

Everything runs only while ``specpilot_d8_tlv.q_active()`` is True (capture of the d8q graph and its replays); the
exact pass (d8x graph) and every other forward are untouched.

Knob (read once at import; unset = the parent's path, nothing here runs):
  SPECPILOT_D2Q_HEAD=cert   (or 1) Q lm_head = certified exact argmax over the drafter's vocabulary subset
"""
from __future__ import annotations

import logging
import os

import torch

from sglang.srt.speculative import specpilot_d8_tlv as _tlv

logger = logging.getLogger(__name__)

_HEAD_MODE = (os.environ.get("SPECPILOT_D2Q_HEAD", "") or "0").strip().lower()
HEAD = _HEAD_MODE in ("1", "cert") and _tlv.enabled()
HEAD_VOCAB = int((os.environ.get("SPECPILOT_DE_VOCAB_SIZE", "") or "98304").strip() or 98304)

_SUB: dict = {}


def prepare_head(head) -> None:
    """Subset head for one d4 CertifiedArgmaxHead (call outside graph capture; idempotent)."""
    if not HEAD or id(head) in _SUB:
        return
    from sglang.kernels.ops.gemm.specpilot_d4_lmhead import CertifiedArgmaxHead
    from sglang.kernels.ops.speculative.dflash_de_head import build_vocab_subset

    sub = build_vocab_subset(HEAD_VOCAB, head.V).to(device=head.w.device, dtype=torch.int64)
    w_sub = head.w.index_select(0, sub).contiguous()
    _SUB[id(head)] = (sub, CertifiedArgmaxHead(w_sub, kmax=head.kmax, max_rows=head.max_rows))
    torch.cuda.empty_cache()
    logger.info("SPECPILOT_D2Q: Q head = certified argmax over %d of %d vocabulary rows", sub.numel(), head.V)


def maybe_q_head(head, h: torch.Tensor) -> bool:
    """While the Q pass runs: head.ids[:M] = certified top-1 over the subset for h [M, K]; returns True if done."""
    if not HEAD or not _tlv.q_active():
        return False
    st = _SUB.get(id(head))
    if st is None:
        return False
    sub, sub_head = st
    m = h.shape[0]
    ids_sub = sub_head(h)
    # a non-finite row (e.g. CUDA-graph capture inputs) yields the sentinel 2^31 - 1: clamp before the gather
    torch.index_select(sub, 0, ids_sub.clamp(0, sub.numel() - 1), out=head.ids[:m])
    return True
