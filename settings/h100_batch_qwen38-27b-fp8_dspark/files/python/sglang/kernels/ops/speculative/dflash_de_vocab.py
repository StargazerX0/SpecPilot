"""SPECPILOT_DE: static candidate vocabulary for the DFlash2 candidate head (drafter-only).

`DFlash2DraftModel.compute_candidates` projects every drafted hidden through the target
lm_head over the whole vocabulary and takes a radix top-k. With `SPECPILOT_DE_VOCAB_SIZE=S`
the projection and the top-k run over a contiguous draft-only copy of S rows of the head
(gathered once, outside graph capture); subset positions are mapped back to full ids.
Verification is unchanged, so outputs are unchanged (drafts enter only through accept).

Ranking (deterministic, training-free): tokens observed on the `calib` split
(`dflash_de_vocab_calib.json`: stock target completions + prompts of the calib_trace run,
count desc, id asc), then the tokenizer's added tokens (ids >= ADDED_START), then every
remaining id in ascending order (BPE merge order, a frequency prior). The subset is the
first S ids of that ranking, sorted ascending (row order of the head is kept).

Knobs (read once, at model init): SPECPILOT_DE_VOCAB_SIZE (unset/0 = stock, strict no-op).
Top-k returns k ids + scores per row for any k <= S (trees may ask for k > 16).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_CALIB_FILE = os.path.join(os.path.dirname(__file__), "dflash_de_vocab_calib.json")
# First added (special) token of the Qwen3.5 tokenizer; ids in [ADDED_START, tokenizer
# length) are chat/think/special tokens and always ranked right after the calib tokens.
ADDED_START = 248044
ADDED_END = 248077


def vocab_size_knob() -> int:
    raw = os.environ.get("SPECPILOT_DE_VOCAB_SIZE", "").strip()
    return int(raw) if raw else 0


def ranked_ids(num_org: int, calib_ids=None) -> torch.Tensor:
    """Full ranking of [0, num_org): calib ids, added tokens, then ascending id."""
    if calib_ids is None:
        with open(_CALIB_FILE) as f:
            calib_ids = json.load(f)["ids"]
    seen = torch.zeros(num_org, dtype=torch.bool)
    head = []
    for i in list(calib_ids) + list(range(ADDED_START, min(ADDED_END, num_org))):
        i = int(i)
        if 0 <= i < num_org and not seen[i]:
            seen[i] = True
            head.append(i)
    rest = torch.nonzero(~seen, as_tuple=False).reshape(-1)
    return torch.cat([torch.tensor(head, dtype=torch.int64), rest])


def subset_ids(size: int, num_org: int, calib_ids=None) -> torch.Tensor:
    """First `size` ranked ids, ascending."""
    size = min(int(size), int(num_org))
    return torch.sort(ranked_ids(num_org, calib_ids)[:size]).values


class StaticVocab:
    """Draft-only gathered head rows + sub->full map (allocated once, before capture)."""

    def __init__(self, weight: torch.Tensor, num_org: int, size: int, calib_ids=None):
        ids = subset_ids(size, num_org, calib_ids).to(weight.device)
        self.sub_to_full = ids
        self.weight = weight.index_select(0, ids).contiguous()
        self.size = int(ids.numel())
        self.fp8 = None  # SPECPILOT_X2: block-FP8 copy of the subset rows (candidate head GEMM)
        from sglang.kernels.ops.speculative import specpilot_draft_fp8 as _dfp8

        if _dfp8.head_fp8_enabled() and self.size % 128 == 0 and self.weight.shape[1] % 128 == 0:
            self.fp8 = _dfp8.prepare_head(self.weight)

    def topk(self, hidden: torch.Tensor, k: int, topk_fn) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fp8 is not None:  # SPECPILOT_X2
            from sglang.kernels.ops.speculative.specpilot_draft_fp8 import block_fp8_linear

            logits = block_fp8_linear(hidden.to(self.weight.dtype), *self.fp8)
        else:
            logits = torch.matmul(hidden.to(self.weight.dtype), self.weight.T)
        vals, sub = topk_fn(logits, k)
        return vals, self.sub_to_full[sub.long()]


def maybe_build(model, lm_head) -> Optional[StaticVocab]:
    """Build the subset for `model` (DFlash2DraftModel) once, never while capturing."""
    size = getattr(model, "_specpilot_de_vocab_size", 0)
    if not size:
        return None
    sv = getattr(model, "_specpilot_de_vocab", None)
    if sv is not None and getattr(model, "_specpilot_de_vocab_head", None) is lm_head:
        return sv
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SPECPILOT_DE vocab subset must be built before graph capture")
    num_org = int(lm_head.org_vocab_size)
    sv = StaticVocab(lm_head.weight, num_org, size)
    # Plain attributes (not submodules): keep the draft's module tree unchanged.
    model.__dict__["_specpilot_de_vocab"] = sv
    model.__dict__["_specpilot_de_vocab_head"] = lm_head
    logger.info(
        "SPECPILOT_DE vocab subset: %d of %d head rows (%.1f MiB draft-only copy)",
        sv.size,
        num_org,
        sv.weight.numel() * sv.weight.element_size() / 2**20,
    )
    return sv
