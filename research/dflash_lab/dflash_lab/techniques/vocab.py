"""Frequency-ranked draft vocabulary subsets for the DFlash2 candidate head (reference implementation).

Literature: FR-Spec (Zhao et al., arXiv 2502.14856), VocabTrim (arXiv 2506.22694), EAGLE-3's `d2t`/`t2d` draft
vocabulary (arXiv 2503.01840, SafeAILab/EAGLE `eagle/traineagle3/cnets.py`). See
`research/knowledge/methods/drafter_efficiency/fr_spec_vocab_subset.md`.

What it is for in this engine: `DFlash2DraftModel.compute_candidates` (python/sglang/srt/models/dflash.py) projects
the 7 drafted hiddens through the *target* lm_head over the whole 248,320-token vocabulary and takes a top-16
(`_project_candidate_logits` + `_radix_topk`). Restricting both to a frequency-ranked subset V_sub reads |V_sub|
instead of V rows of the head. Verification still uses the full target distribution, so outputs are unchanged
(drafts enter only through the accept check); the cost is top-16 recall, i.e. acceptance.

Pure torch, device-agnostic, no host syncs inside `subset_topk` (safe to call inside a CUDA graph once the remap
tables and the gathered head rows are allocated before capture). Building the subset (`count_token_frequencies`,
`build_vocab_subset`) is offline/one-off work: run it on the `calib` split (or on target generations of it, which
VocabTrim reports loses less acceptance than raw text) -- never on dev/confirm/final data.

Conventions: "full" ids are target-vocabulary ids in [0, vocab_size); "sub" ids are positions in the subset in
[0, S). `subset_ids` (== `sub_to_full`) is sorted ascending so the gathered head rows keep the original row order.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Tuple

import msgspec
import torch

__all__ = [
    "count_token_frequencies",
    "build_vocab_subset",
    "frequency_subset_from_streams",
    "default_always_include",
    "VocabRemap",
    "gather_head_rows",
    "subset_logits",
    "subset_topk",
    "reference_restricted_topk",
    "subset_coverage",
    "topk_recall",
    "save_subset",
    "load_subset",
]


# ---------------------------------------------------------------------------------------------------------------
# Building the subset (offline)
# ---------------------------------------------------------------------------------------------------------------


def count_token_frequencies(streams: Iterable[Sequence[int]], vocab_size: int) -> torch.Tensor:
    """Count token occurrences over an iterable of token-id lists (e.g. the tokenized `calib` split, prompts plus
    target completions). Returns int64 counts of shape [vocab_size]. Ids outside [0, vocab_size) raise."""
    counts = torch.zeros(int(vocab_size), dtype=torch.int64)
    for ids in streams:
        t = torch.as_tensor(list(ids) if not isinstance(ids, torch.Tensor) else ids, dtype=torch.int64).reshape(-1)
        if t.numel() == 0:
            continue
        if int(t.min()) < 0 or int(t.max()) >= vocab_size:
            raise ValueError(f"token id out of range [0, {vocab_size}): min={int(t.min())} max={int(t.max())}")
        counts += torch.bincount(t, minlength=int(vocab_size))
    return counts


def build_vocab_subset(
    counts: torch.Tensor,
    size: int,
    *,
    always_include: Iterable[int] = (),
    num_org: Optional[int] = None,
) -> torch.Tensor:
    """Frequency-ranked subset of `size` full-vocab ids, sorted ascending.

    `always_include` ids (special / EOS / think-tag / mask tokens, or any id the caller insists on) are taken first;
    the rest is filled by descending count, ties broken by smaller id (deterministic). Ids >= `num_org` (padding rows
    of a padded lm_head, see `lm_head.org_vocab_size`) are never selected. Zero-count tokens are used only if the
    observed vocabulary is smaller than `size`, in id order.
    """
    counts = torch.as_tensor(counts, dtype=torch.int64).reshape(-1)
    vocab = int(counts.numel())
    limit = vocab if num_org is None else min(int(num_org), vocab)
    must = sorted({int(i) for i in always_include})
    for i in must:
        if not 0 <= i < limit:
            raise ValueError(f"always_include id {i} outside [0, {limit})")
    size = int(size)
    if size < len(must):
        raise ValueError(f"size={size} is smaller than the {len(must)} always-included ids")
    if size > limit:
        raise ValueError(f"size={size} exceeds the selectable vocabulary ({limit})")
    # Deterministic ranking: stable sort on -count keeps ascending id order inside equal counts.
    order = torch.sort(-counts[:limit], stable=True).indices
    chosen = torch.zeros(limit, dtype=torch.bool)
    if must:
        chosen[torch.tensor(must, dtype=torch.int64)] = True
    need = size - len(must)
    if need > 0:
        rest = order[~chosen[order]][:need]
        chosen[rest] = True
    return torch.nonzero(chosen, as_tuple=False).reshape(-1)


def frequency_subset_from_streams(
    streams: Iterable[Sequence[int]],
    vocab_size: int,
    size: int,
    *,
    always_include: Iterable[int] = (),
    num_org: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience: (subset_ids, counts) from token-id streams."""
    counts = count_token_frequencies(streams, vocab_size)
    return build_vocab_subset(counts, size, always_include=always_include, num_org=num_org), counts


def default_always_include(tokenizer=None, extra: Iterable[int] = ()) -> list:
    """Ids that a draft subset should keep regardless of frequency: the tokenizer's special ids, its added tokens
    (Qwen chat/think tags live there) and `extra` (e.g. EOS ids from generation_config, the DFlash mask id is NOT
    needed: it is an input, never a candidate). Duck-typed on HF tokenizers; with tokenizer=None returns `extra`."""
    ids = {int(i) for i in extra}
    if tokenizer is not None:
        ids.update(int(i) for i in (getattr(tokenizer, "all_special_ids", None) or []))
        added = getattr(tokenizer, "added_tokens_decoder", None) or {}
        ids.update(int(i) for i in added)
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            ids.update(int(i) for i in (eos if isinstance(eos, (list, tuple)) else [eos]))
    return sorted(ids)


# ---------------------------------------------------------------------------------------------------------------
# Remap tables and the restricted head (runtime)
# ---------------------------------------------------------------------------------------------------------------


class VocabRemap(msgspec.Struct, frozen=True):
    """sub_to_full [S] int64 (ascending full ids); full_to_sub [V] int64 with -1 for ids not in the subset."""

    sub_to_full: torch.Tensor
    full_to_sub: torch.Tensor

    @classmethod
    def from_subset(cls, subset_ids: torch.Tensor, vocab_size: int) -> "VocabRemap":
        sub_to_full = torch.as_tensor(subset_ids, dtype=torch.int64).reshape(-1)
        if sub_to_full.numel() and bool((sub_to_full[1:] <= sub_to_full[:-1]).any()):
            raise ValueError("subset_ids must be strictly ascending (use build_vocab_subset)")
        if sub_to_full.numel() and (int(sub_to_full[0]) < 0 or int(sub_to_full[-1]) >= vocab_size):
            raise ValueError("subset ids out of range")
        full_to_sub = torch.full((int(vocab_size),), -1, dtype=torch.int64, device=sub_to_full.device)
        full_to_sub[sub_to_full] = torch.arange(sub_to_full.numel(), dtype=torch.int64, device=sub_to_full.device)
        return cls(sub_to_full=sub_to_full, full_to_sub=full_to_sub)

    @property
    def size(self) -> int:
        return int(self.sub_to_full.numel())

    @property
    def vocab_size(self) -> int:
        return int(self.full_to_sub.numel())

    def to(self, device) -> "VocabRemap":
        return VocabRemap(self.sub_to_full.to(device), self.full_to_sub.to(device))

    def to_full(self, sub_ids: torch.Tensor) -> torch.Tensor:
        """Map subset positions to full-vocab ids (any shape)."""
        return self.sub_to_full[sub_ids.long()]

    def to_sub(self, full_ids: torch.Tensor) -> torch.Tensor:
        """Map full-vocab ids to subset positions; -1 where the id is not in the subset."""
        return self.full_to_sub[full_ids.long()]

    def contains(self, full_ids: torch.Tensor) -> torch.Tensor:
        return self.to_sub(full_ids) >= 0


def gather_head_rows(weight: torch.Tensor, subset_ids: torch.Tensor) -> torch.Tensor:
    """Contiguous copy of the head rows of the subset: [S, H]. This is declared draft-only derived data (computed at
    load time from the fixed target lm_head; the target keeps using its own full head). Allocate before graph
    capture. Memory: S * H * itemsize (32K x 5120 bf16 = 335 MB)."""
    return weight.index_select(0, subset_ids.to(weight.device).long()).contiguous()


def subset_logits(hidden: torch.Tensor, sub_weight: torch.Tensor) -> torch.Tensor:
    """[N, H] x [S, H]^T -> [N, S] in the weight dtype (mirrors `_project_candidate_logits`)."""
    return torch.matmul(hidden.to(sub_weight.dtype), sub_weight.T)


def _torch_topk(scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.topk(scores, k, dim=-1, largest=True, sorted=True)


def subset_topk(
    hidden: torch.Tensor,
    sub_weight: torch.Tensor,
    sub_to_full: torch.Tensor,
    k: int,
    *,
    topk_fn: Optional[Callable[[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k over the subset, returned as (values [N, k], full-vocab ids [N, k] int64), sorted by value.

    `topk_fn(scores, k)` defaults to torch.topk; in the engine pass `sglang.srt.models.dflash._radix_topk`
    (flashinfer radix top-k, sorted, deterministic). No host sync; static shapes."""
    fn = topk_fn or _torch_topk
    vals, sub_ids = fn(subset_logits(hidden, sub_weight), int(k))
    return vals, sub_to_full[sub_ids.long()]


def reference_restricted_topk(
    full_logits: torch.Tensor, subset_ids: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Oracle: the full-vocab top-k after masking every id outside the subset to -inf."""
    masked = torch.full_like(full_logits, float("-inf"))
    idx = subset_ids.to(full_logits.device).long()
    masked[..., idx] = full_logits[..., idx]
    return torch.topk(masked, int(k), dim=-1, largest=True, sorted=True)


# ---------------------------------------------------------------------------------------------------------------
# Offline diagnostics (go/no-go before any GPU run)
# ---------------------------------------------------------------------------------------------------------------


def subset_coverage(counts: torch.Tensor, subset_ids: torch.Tensor) -> float:
    """Fraction of the counted token mass that falls inside the subset (FR-Spec's coverage curve)."""
    counts = torch.as_tensor(counts, dtype=torch.float64)
    total = float(counts.sum())
    return float(counts[subset_ids.long()].sum()) / total if total > 0 else 0.0


def topk_recall(full_logits: torch.Tensor, subset_ids: torch.Tensor, k: int) -> dict:
    """Per-row agreement between the full top-k and the subset-restricted top-k, from logged drafter logits
    (or from hiddens projected offline). Returns mean fractions: `top1` (full argmax inside the subset),
    `set` (|full top-k inter subset top-k| / k) and `exact` (identical ordered top-k)."""
    full_v, full_i = torch.topk(full_logits, int(k), dim=-1)
    sub_v, sub_i = reference_restricted_topk(full_logits, subset_ids, k)
    member = torch.zeros(full_logits.shape[-1], dtype=torch.bool, device=full_logits.device)
    member[subset_ids.to(full_logits.device).long()] = True
    top1 = member[full_i[..., 0]].float().mean()
    inter = (full_i.unsqueeze(-1) == sub_i.unsqueeze(-2)).any(-1).float().sum(-1) / int(k)
    exact = (full_i == sub_i).all(-1).float().mean()
    return {"top1": float(top1), "set": float(inter.mean()), "exact": float(exact)}


def save_subset(path: str, subset_ids: torch.Tensor, *, vocab_size: int, source: str, **meta) -> None:
    """Persist a subset with its provenance (`source` must name the split it was counted on, e.g. 'calib')."""
    torch.save({"subset_ids": subset_ids.cpu().long(), "vocab_size": int(vocab_size), "source": source, **meta}, path)


def load_subset(path: str) -> Tuple[torch.Tensor, dict]:
    blob = torch.load(path, weights_only=True)
    ids = blob.pop("subset_ids")
    return ids.long(), blob
