"""Sparse verification (approximate, lossy): the verify pass computes the target sparsely.

Reference implementation for the ``approximate`` family ``sparse``
(``research/knowledge/search_space/07_sparse_verification.md``; protocol ``approximate.kinds.sparse_verification``).
Pure PyTorch, batched, vectorised: no host syncs, no ``.item()``/``.tolist()``, no data-dependent shapes. Budgets
(pages, neurons, experts) are static Python ints or tensors used only in comparisons, so every output has a shape
fixed by the inputs and the config; Python control flow depends only on the (static) config. This module is a
numerical reference (slow, device-agnostic) to validate an engine kernel against in a unit test; it does not make
anything faster. In every snapshot under ``lab/dflash_lab/techniques/``, importable as ``dflash_lab.techniques.sparse``
(the engine never imports it).

Literature (method pages in ``research/knowledge/methods/sparse_verification/``):

* Quest (Tang et al., ICML 2024, arXiv 2406.10774): per page and per KV head, elementwise min ``m`` and max ``M`` of
  the keys; criticality upper bound ``U(q, page) = sum_d max(q_d * M_d, q_d * m_d) >= q . k`` for every key of the
  page; top-K pages; exact attention on the selected pages (the last, partial page always attended; first two layers
  dense). Decode-only, batch 1, MHA in the authors' kernels.
* Dustin (Lee et al., arXiv 2606.24957, "Accepted to ICML 2026" per the arXiv comment): importance = attention
  probabilities recomputed on a small offline-chosen subset of heads ("semantic retrieval heads"), summed over the
  draft positions of the verify block (draft lookahead) and, separately, over the last target token (history);
  tiered selection protected (sinks + recent window) | top-m by draft score | rest of the budget by target score;
  one index set shared by every layer. Token-granular in the paper (``page_size = 1`` here).
* Sparse Computation in Verification (Wang et al., arXiv 2512.21911): block top-N attention scored with the first
  verified token's query and reused by the other tokens of the step (inter-draft reuse); anchor layers chosen offline
  by low adjacent-layer Jaccard similarity, other layers reuse the nearest preceding anchor's selection (inter-layer
  reuse); FFN: per-token threshold on the gate activation, up/down only on the surviving channels; MoE: per-token
  skipping of the lowest-weight routed experts with a calibrated mass ratio ``beta_m``.

Layout (one verify step, engine shapes in brackets)::

    q        [bs, n, Hq, D]     queries of the n verify rows (row 0 = anchor/bonus, rows 1..n-1 = drafts)
    keys     [bs, Hkv, L, D]    committed context K of one full-attention layer (RoPE applied, as stored)
    values   [bs, Hkv, L, D]    committed context V
    kv_lens  [bs]               valid context length per request (tokens >= kv_lens are padding)
    k_blk    [bs, n, Hkv, D]    K/V of the n verify rows themselves (always attended, causal inside the block)
    pages    page p holds context tokens [p*P, (p+1)*P); NP = ceil(L / P)

The engine runs ``page_size = 1`` on CUDA (``--page-size`` unset), so a "page" here is a *logical* group of
``P`` consecutive token slots of ``req_to_token``; a selection becomes a compacted token table (``compact_token_table``)
that FA3 consumes as ``page_table`` with ``cache_seqlens`` = selected tokens + n verify rows. FA3's ``page_table`` is
per request, shared by all heads: per-KV-head selections need a head-split call or another kernel.

Selection semantics (all return boolean masks over pages, ``True`` = attend):

* ``topk_mask``: rank-based top-k with first-index tie breaking (stable sort). The selected set is monotone in k
  (``mask(k) <= mask(k + 1)``) and capped by the valid pages; forced pages (sinks, recent/last page) count inside k.
* ``draft_augmented_mask``: how the n rows of a verify block share pages. ``"per_position"`` (each row its own
  top-k), ``"union"`` (union of the per-row top-k: a superset of every row's selection, at most n*k pages),
  ``"sum"`` (one top-k over row-summed scores: Dustin-style aggregation), ``"first"`` (row 0's selection reused by all
  rows: inter-draft reuse of 2512.21911), ``"max"`` (one top-k over the row-wise max).
* ``reuse_map`` / ``choose_anchor_layers``: inter-layer reuse over the *full-attention* layers only (on a hybrid
  target the previous full-attention layer is 4 layers earlier).

Lossless limit: every page / neuron / expert selected reproduces the dense computation exactly in this reference
(same arithmetic, masked entries contribute exact zeros). In an engine the sparse kernel may still differ from the
dense kernel in reduction order; the limit must be checked output-identical to stock with the candidate's own code
path (``07_sparse_verification.md``, "Approximate contract").

State: sparse verification changes the hidden states of every layer after the first sparse one, hence the KV written
for the verify rows and the GDN/conv states of later layers. The commit path is unchanged (``commit_lens - 1`` GDN
scatter, KV of the committed rows, prefix-valid draft-KV append), so the committed state belongs to the emitted tokens
as computed by the sparse pass. Page metadata (``update_page_metadata``) must be folded in for *committed* tokens only:
min/max are not invertible, so a rejected draft row folded in would only loosen the bound (safe but stale), never
tighten it back.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import msgspec
import torch
import torch.nn.functional as F

Number = Union[int, float, torch.Tensor]

DRAFT_MODES = ("per_position", "union", "sum", "max", "first")
HEAD_SHARING = ("per_q_head", "per_kv_head", "per_request")
HEAD_REDUCE = ("max", "sum")
FFN_MODES = ("dense", "threshold", "token_topk", "block_topk")
MOE_MODES = ("dense", "fixed", "adaptive", "block_shared")

__all__ = [
    "SparseVerifyConfig",
    "DENSE",
    "PageMeta",
    "page_metadata",
    "update_page_metadata",
    "quest_page_scores",
    "reduce_heads",
    "forced_page_mask",
    "topk_mask",
    "draft_augmented_mask",
    "mask_to_indices",
    "compact_token_table",
    "block_sparse_attention",
    "gathered_sparse_attention",
    "attention_importance",
    "pool_to_pages",
    "dustin_select",
    "jaccard",
    "choose_anchor_layers",
    "reuse_map",
    "swiglu_ffn",
    "ffn_neuron_mask",
    "sparse_swiglu_ffn",
    "block_shared_ffn_gathered",
    "route_topk",
    "moe_forward",
    "truncate_experts_fixed",
    "calibrate_beta",
    "truncate_experts_adaptive",
    "block_shared_experts",
    "expected_union_experts",
]


class SparseVerifyConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Static configuration of one sparse-verification policy (all knobs frozen at launch).

    Defaults are the lossless limit (dense everything). ``page_budget = None`` means all pages.
    """

    # attention (full-attention layers only)
    page_size: int = 16                    # logical page of P token slots (Quest default 16; Dustin = 1)
    page_budget: Optional[int] = None      # selected pages per selection unit, sinks/recent included
    n_sink_pages: int = 0                  # always-attended first pages (attention sinks)
    n_recent_pages: int = 1                # always-attended last pages (Quest: the current, partial page)
    draft_mode: str = "union"              # how the n verify rows share pages (DRAFT_MODES)
    head_sharing: str = "per_request"      # selection unit (HEAD_SHARING); FA3 page_table is per request
    head_reduce: str = "max"               # how q heads of a unit are combined (HEAD_REDUCE)
    dense_attn_layers: Tuple[int, ...] = ()   # attention-layer ordinals kept dense (Quest: first two layers)
    anchor_attn_layers: Optional[Tuple[int, ...]] = None  # inter-layer reuse anchors (None: every layer selects)
    min_context_pages: int = 0             # below this many valid pages a request stays dense (2512.21911 L0)
    # FFN (dense SwiGLU layers)
    ffn_mode: str = "dense"
    ffn_tau: float = 0.0                   # threshold mode: drop channel i if |silu(gate_i)| < tau
    ffn_keep: Optional[int] = None         # topk modes: kept channels per token / per request block
    # MoE (routed experts; shared expert always dense)
    moe_mode: str = "dense"
    moe_keep: Optional[int] = None         # fixed: routed experts kept per token; block_shared: experts per request
    moe_max_skip: int = 0                  # adaptive: at most m experts skipped per token
    moe_beta: Optional[float] = None       # adaptive: calibrated kept-mass ratio

    def __post_init__(self) -> None:
        if self.page_size < 1:
            raise ValueError("page_size must be >= 1")
        if self.page_budget is not None and self.page_budget < 1:
            raise ValueError("page_budget must be >= 1 or None")
        if self.draft_mode not in DRAFT_MODES:
            raise ValueError(f"draft_mode must be one of {DRAFT_MODES}")
        if self.head_sharing not in HEAD_SHARING:
            raise ValueError(f"head_sharing must be one of {HEAD_SHARING}")
        if self.head_reduce not in HEAD_REDUCE:
            raise ValueError(f"head_reduce must be one of {HEAD_REDUCE}")
        if self.ffn_mode not in FFN_MODES:
            raise ValueError(f"ffn_mode must be one of {FFN_MODES}")
        if self.moe_mode not in MOE_MODES:
            raise ValueError(f"moe_mode must be one of {MOE_MODES}")
        if self.ffn_tau < 0:
            raise ValueError("ffn_tau must be >= 0")
        if self.moe_max_skip < 0:
            raise ValueError("moe_max_skip must be >= 0")

    def is_lossless_limit(self, num_pages: int, d_ff: int, top_k: int, num_experts: int) -> bool:
        """True when every page / channel / expert is computed for these model dimensions."""
        attn = self.page_budget is None or self.page_budget >= num_pages
        ffn = (
            self.ffn_mode == "dense"
            or (self.ffn_mode == "threshold" and self.ffn_tau == 0.0)
            or (self.ffn_mode in ("token_topk", "block_topk") and (self.ffn_keep is None or self.ffn_keep >= d_ff))
        )
        moe = (
            self.moe_mode == "dense"
            or (self.moe_mode == "fixed" and (self.moe_keep is None or self.moe_keep >= top_k))
            or (self.moe_mode == "adaptive" and self.moe_max_skip == 0)
            or (self.moe_mode == "block_shared" and (self.moe_keep is None or self.moe_keep >= num_experts))
        )
        return attn and ffn and moe


DENSE = SparseVerifyConfig()


# ----------------------------------------------------------------------------------------------------------------
# Quest page metadata and upper-bound scores
# ----------------------------------------------------------------------------------------------------------------


class PageMeta(NamedTuple):
    """Per-page key bounds. ``kmin``/``kmax`` are ``+inf``/``-inf`` on empty pages."""

    kmin: torch.Tensor        # [bs, Hkv, NP, D]
    kmax: torch.Tensor        # [bs, Hkv, NP, D]
    page_valid: torch.Tensor  # [bs, NP] bool: page holds at least one valid token
    page_size: int


def _num_pages(length: int, page_size: int) -> int:
    return -(-length // page_size)


def _token_valid(kv_lens: torch.Tensor, length: int) -> torch.Tensor:
    pos = torch.arange(length, device=kv_lens.device)
    return pos[None, :] < kv_lens[:, None]


def page_metadata(keys: torch.Tensor, kv_lens: torch.Tensor, page_size: int) -> PageMeta:
    """Elementwise min / max of the valid keys of every page (Quest Algorithm 1, metadata)."""
    bs, hkv, length, dim = keys.shape
    num_pages = _num_pages(length, page_size)
    pad = num_pages * page_size - length
    k = F.pad(keys, (0, 0, 0, pad))
    valid = _token_valid(kv_lens, num_pages * page_size)  # [bs, NP*P]
    v = valid[:, None, :, None]
    inf = torch.tensor(float("inf"), dtype=keys.dtype, device=keys.device)
    kmax = torch.where(v, k, -inf).view(bs, hkv, num_pages, page_size, dim).amax(dim=3)
    kmin = torch.where(v, k, inf).view(bs, hkv, num_pages, page_size, dim).amin(dim=3)
    page_valid = valid.view(bs, num_pages, page_size).any(dim=-1)
    return PageMeta(kmin, kmax, page_valid, page_size)


def empty_page_metadata(bs: int, hkv: int, num_pages: int, dim: int, page_size: int, *, dtype=torch.float32,
                        device=None) -> PageMeta:
    """Metadata of an empty cache (static buffers an engine would allocate once)."""
    kmin = torch.full((bs, hkv, num_pages, dim), float("inf"), dtype=dtype, device=device)
    kmax = torch.full((bs, hkv, num_pages, dim), float("-inf"), dtype=dtype, device=device)
    page_valid = torch.zeros(bs, num_pages, dtype=torch.bool, device=device)
    return PageMeta(kmin, kmax, page_valid, page_size)


def update_page_metadata(meta: PageMeta, new_keys: torch.Tensor, positions: torch.Tensor,
                         commit_mask: torch.Tensor) -> PageMeta:
    """Fold newly *committed* keys into the page bounds (out of place; static shapes).

    new_keys    [bs, n, Hkv, D]  K of the verify rows (as written to the KV cache)
    positions   [bs, n]          context position of each row (seq_len + j)
    commit_mask [bs, n]          True for rows that were committed (j < commit_len); others are ignored
    """
    bs, hkv, num_pages, dim = meta.kmax.shape
    page = (positions // meta.page_size).clamp(0, num_pages - 1)  # [bs, n]
    idx = page[:, None, :, None].expand(bs, hkv, page.shape[1], dim)
    k = new_keys.permute(0, 2, 1, 3).to(meta.kmax.dtype)  # [bs, Hkv, n, D]
    cm = commit_mask[:, None, :, None]
    inf = torch.tensor(float("inf"), dtype=k.dtype, device=k.device)
    kmax = meta.kmax.scatter_reduce(2, idx, torch.where(cm, k, -inf), reduce="amax", include_self=True)
    kmin = meta.kmin.scatter_reduce(2, idx, torch.where(cm, k, inf), reduce="amin", include_self=True)
    hit = torch.zeros_like(meta.page_valid).scatter_reduce(
        1, page, commit_mask, reduce="amax", include_self=True)
    return PageMeta(kmin, kmax, meta.page_valid | hit, meta.page_size)


def quest_page_scores(q: torch.Tensor, meta: PageMeta) -> torch.Tensor:
    """Upper bound of ``q . k`` over each page, per query head and verify row: ``[bs, Hq, n, NP]``.

    ``max(q_d M_d, q_d m_d) = q_d^+ M_d + q_d^- m_d`` (since ``M >= m``), so the bound is two matmuls. Invalid pages
    score ``-inf``. Unscaled dot products (the softmax scale does not change the ranking).
    """
    bs, n, hq, dim = q.shape
    hkv = meta.kmax.shape[1]
    group = hq // hkv
    valid = meta.page_valid[:, None, None, :]
    kmax = torch.where(meta.page_valid[:, None, :, None], meta.kmax, torch.zeros_like(meta.kmax))
    kmin = torch.where(meta.page_valid[:, None, :, None], meta.kmin, torch.zeros_like(meta.kmin))
    kmax = kmax.repeat_interleave(group, dim=1)  # [bs, Hq, NP, D]
    kmin = kmin.repeat_interleave(group, dim=1)
    qh = q.permute(0, 2, 1, 3).to(kmax.dtype)    # [bs, Hq, n, D]
    bound = qh.clamp(min=0) @ kmax.transpose(-1, -2) + qh.clamp(max=0) @ kmin.transpose(-1, -2)
    return torch.where(valid, bound, torch.full_like(bound, float("-inf")))


def reduce_heads(scores: torch.Tensor, num_kv_heads: int, sharing: str = "per_kv_head",
                 how: str = "max") -> torch.Tensor:
    """Combine per-q-head scores ``[bs, Hq, ...]`` into the selection unit.

    ``per_q_head`` -> ``[bs, Hq, ...]``; ``per_kv_head`` (a GQA group shares pages) -> ``[bs, Hkv, ...]``;
    ``per_request`` (all heads share pages, the FA3 ``page_table`` granularity) -> ``[bs, 1, ...]``.
    ``max`` keeps a page that is critical for any head; ``sum`` ranks by total bound.
    """
    if sharing not in HEAD_SHARING or how not in HEAD_REDUCE:
        raise ValueError("bad sharing / how")
    if sharing == "per_q_head":
        return scores
    bs, hq = scores.shape[:2]
    groups = num_kv_heads if sharing == "per_kv_head" else 1
    s = scores.reshape(bs, groups, hq // groups, *scores.shape[2:])
    return s.amax(dim=2) if how == "max" else s.sum(dim=2)


# ----------------------------------------------------------------------------------------------------------------
# Top-k page selection (static k, monotone, forced pages inside the budget)
# ----------------------------------------------------------------------------------------------------------------


def forced_page_mask(kv_lens: torch.Tensor, num_pages: int, page_size: int, n_sink: int = 0,
                     n_recent: int = 1) -> torch.Tensor:
    """Pages always attended: the first ``n_sink`` and the last ``n_recent`` valid pages. ``[bs, NP]`` bool."""
    p = torch.arange(num_pages, device=kv_lens.device)[None, :]
    n_valid = (kv_lens[:, None] + page_size - 1) // page_size
    sink = p < n_sink
    recent = (p >= n_valid - n_recent) & (p < n_valid)
    return (sink | recent) & (p < n_valid)


def _ranks(scores: torch.Tensor) -> torch.Tensor:
    """Descending rank of every entry along the last dim; ties broken by lower index first."""
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    ar = torch.arange(scores.shape[-1], device=scores.device).expand_as(order)
    return torch.empty_like(order).scatter_(-1, order, ar)


def topk_mask(scores: torch.Tensor, k: Number, valid: Optional[torch.Tensor] = None,
              forced: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Boolean mask of the top-``k`` entries along the last dim (static shape, no host sync).

    ``valid`` / ``forced`` broadcast to ``scores``. Forced entries rank first and count inside ``k``; invalid entries
    are never selected (fewer than ``k`` valid entries -> all valid ones). ``k`` may be an int or a tensor
    broadcastable to ``scores.shape[:-1] + (1,)`` (a per-row budget). Monotone in ``k``.
    """
    s = scores
    if valid is not None:
        s = torch.where(valid, s, torch.full_like(s, float("-inf")))
    if forced is not None:
        f = forced if valid is None else (forced & valid)
        s = torch.where(f, torch.full_like(s, float("inf")), s)
    mask = _ranks(s) < k
    if valid is not None:
        mask = mask & valid
    return mask


def draft_augmented_mask(scores: torch.Tensor, k: int, mode: str = "union",
                         valid: Optional[torch.Tensor] = None,
                         forced: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Share a page selection across the n rows of a verify block.

    scores ``[bs, U, n, NP]`` (U = selection units). Returns ``[bs, U, n, NP]`` for ``per_position`` and
    ``[bs, U, NP]`` otherwise. ``valid``/``forced``: ``[bs, NP]``.
    """
    if mode not in DRAFT_MODES:
        raise ValueError(f"mode must be one of {DRAFT_MODES}")
    v = None if valid is None else valid[:, None, :]
    f = None if forced is None else forced[:, None, :]
    if mode in ("per_position", "union"):
        vv = None if v is None else v[:, :, None, :]
        ff = None if f is None else f[:, :, None, :]
        per_pos = topk_mask(scores, k, vv, ff)
        return per_pos if mode == "per_position" else per_pos.any(dim=2)
    if mode == "first":
        agg = scores[:, :, 0, :]
    elif mode == "sum":
        agg = scores.sum(dim=2)
    else:  # max
        agg = scores.amax(dim=2)
    return topk_mask(agg, k, v, f)


def mask_to_indices(mask: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Selected page ids in ascending order, padded to a static width ``k``: ``(idx, idx_valid)``.

    Padding entries point at page 0 and are marked invalid. Selections larger than ``k`` are truncated (callers
    size ``k`` to the budget, ``n * k`` for a union).
    """
    num_pages = mask.shape[-1]
    pid = torch.arange(num_pages, device=mask.device).expand_as(mask)
    key = torch.where(mask, pid, pid + num_pages)
    srt = torch.sort(key, dim=-1, stable=True).values[..., :k]
    if k > num_pages:  # static padding: the budget may exceed the page count
        srt = F.pad(srt, (0, k - num_pages), value=2 * num_pages)
    idx_valid = srt < num_pages
    idx = torch.where(idx_valid, srt, torch.zeros_like(srt))
    return idx, idx_valid


def compact_token_table(page_mask: torch.Tensor, kv_lens: torch.Tensor, page_size: int,
                        max_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-request compacted context token positions of the selected pages (FA3 ``page_table`` at page size 1).

    page_mask ``[bs, NP]`` (one selection per request). Returns ``positions [bs, max_tokens]`` (ascending, padding
    = 0) and ``count [bs]`` (selected valid tokens; the engine appends the n verify rows after them and sets
    ``cache_seqlens = count + n``). ``max_tokens`` is static (``page_budget * page_size`` or the union bound).
    The positions index ``req_to_token[req, pos]`` to get KV slots.
    """
    bs, num_pages = page_mask.shape
    length = num_pages * page_size
    tok_sel = page_mask.repeat_interleave(page_size, dim=1) & _token_valid(kv_lens, length)
    count = tok_sel.sum(dim=1)
    dest = torch.cumsum(tok_sel.to(torch.int64), dim=1) - 1
    dest = torch.where(tok_sel & (dest < max_tokens), dest, torch.full_like(dest, max_tokens))
    pos = torch.arange(length, device=page_mask.device).expand(bs, length)
    out = torch.zeros(bs, max_tokens + 1, dtype=torch.int64, device=page_mask.device)
    out.scatter_(1, dest, pos)
    return out[:, :max_tokens], count.clamp(max=max_tokens)


# ----------------------------------------------------------------------------------------------------------------
# Sparse attention over the selected pages + dense causal attention inside the verify block
# ----------------------------------------------------------------------------------------------------------------


def _expand_mask_to_q(page_mask: torch.Tensor, bs: int, hq: int, n: int) -> torch.Tensor:
    """Page mask [bs, U, NP] or [bs, U, n, NP] (U in {1, Hkv, Hq}) -> [bs, Hq, n, NP]."""
    if page_mask.dim() == 3:
        page_mask = page_mask[:, :, None, :].expand(bs, page_mask.shape[1], n, page_mask.shape[-1])
    units = page_mask.shape[1]
    return page_mask.repeat_interleave(hq // units, dim=1)


def block_sparse_attention(q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, kv_lens: torch.Tensor,
                           k_blk: torch.Tensor, v_blk: torch.Tensor, page_mask: Optional[torch.Tensor] = None,
                           page_size: int = 16, scale: Optional[float] = None) -> torch.Tensor:
    """Verify-block attention: selected context pages (masked) + causal attention over the n block rows.

    ``page_mask = None`` is dense attention (the lossless limit). Returns ``[bs, n, Hq, D]`` in ``q``'s dtype; the
    softmax runs in float32 (float64 inputs stay float64).
    """
    bs, n, hq, dim = q.shape
    hkv, length = keys.shape[1], keys.shape[2]
    group = hq // hkv
    scale = dim ** -0.5 if scale is None else scale
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    qh = q.permute(0, 2, 1, 3).to(acc)                                    # [bs, Hq, n, D]
    kc = keys.repeat_interleave(group, dim=1).to(acc)                     # [bs, Hq, L, D]
    vc = values.repeat_interleave(group, dim=1).to(acc)
    kb = k_blk.permute(0, 2, 1, 3).repeat_interleave(group, dim=1).to(acc)  # [bs, Hq, n, D]
    vb = v_blk.permute(0, 2, 1, 3).repeat_interleave(group, dim=1).to(acc)
    ctx_logits = (qh @ kc.transpose(-1, -2)) * scale                      # [bs, Hq, n, L]
    tok_ok = _token_valid(kv_lens, length)[:, None, None, :]
    if page_mask is not None:
        pm = _expand_mask_to_q(page_mask, bs, hq, n)
        tok_ok = tok_ok & pm.repeat_interleave(page_size, dim=-1)[..., :length]
    ctx_logits = ctx_logits.masked_fill(~tok_ok, float("-inf"))
    blk_logits = (qh @ kb.transpose(-1, -2)) * scale                      # [bs, Hq, n, n]
    causal = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
    blk_logits = blk_logits.masked_fill(~causal, float("-inf"))
    logits = torch.cat([ctx_logits, blk_logits], dim=-1)
    probs = torch.softmax(logits, dim=-1)
    out = probs[..., :length] @ vc + probs[..., length:] @ vb             # [bs, Hq, n, D]
    return out.permute(0, 2, 1, 3).to(q.dtype)


def gathered_sparse_attention(q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, kv_lens: torch.Tensor,
                              k_blk: torch.Tensor, v_blk: torch.Tensor, page_mask: torch.Tensor, page_size: int,
                              max_pages: int, scale: Optional[float] = None) -> torch.Tensor:
    """Same result as ``block_sparse_attention`` for a shared ``[bs, U, NP]`` mask, computed the way a kernel would:
    gather a static ``max_pages * page_size`` compact KV per unit, attend only to it (bytes read ~ budget)."""
    bs, n, hq, dim = q.shape
    hkv, length = keys.shape[1], keys.shape[2]
    units = page_mask.shape[1]
    scale = dim ** -0.5 if scale is None else scale
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    num_pages = page_mask.shape[-1]
    pad = num_pages * page_size - length
    kp = F.pad(keys, (0, 0, 0, pad)).view(bs, hkv, num_pages, page_size, dim)
    vp = F.pad(values, (0, 0, 0, pad)).view(bs, hkv, num_pages, page_size, dim)
    tok_valid = _token_valid(kv_lens, num_pages * page_size).view(bs, num_pages, page_size)
    idx, idx_ok = mask_to_indices(page_mask, max_pages)                    # [bs, U, K]
    # broadcast units to kv heads (a unit covers hkv // units kv heads, or several q heads of one kv head)
    if units <= hkv:
        idx_kv = idx.repeat_interleave(hkv // units, dim=1)
        ok_kv = idx_ok.repeat_interleave(hkv // units, dim=1)
        gi = idx_kv[..., None, None].expand(bs, hkv, max_pages, page_size, dim)
        kg = torch.gather(kp, 2, gi).reshape(bs, hkv, max_pages * page_size, dim)
        vg = torch.gather(vp, 2, gi).reshape(bs, hkv, max_pages * page_size, dim)
        tv = torch.gather(tok_valid[:, None].expand(bs, hkv, num_pages, page_size), 2,
                          idx_kv[..., None].expand(bs, hkv, max_pages, page_size))
        ok = (tv & ok_kv[..., None]).reshape(bs, hkv, max_pages * page_size)
        group = hq // hkv
        kg, vg, ok = (t.repeat_interleave(group, dim=1) for t in (kg, vg, ok))
    else:  # per q head
        group = hq // hkv
        kq = kp.repeat_interleave(group, dim=1)
        vq = vp.repeat_interleave(group, dim=1)
        gi = idx[..., None, None].expand(bs, hq, max_pages, page_size, dim)
        kg = torch.gather(kq, 2, gi).reshape(bs, hq, max_pages * page_size, dim)
        vg = torch.gather(vq, 2, gi).reshape(bs, hq, max_pages * page_size, dim)
        tv = torch.gather(tok_valid[:, None].expand(bs, hq, num_pages, page_size), 2,
                          idx[..., None].expand(bs, hq, max_pages, page_size))
        ok = (tv & idx_ok[..., None]).reshape(bs, hq, max_pages * page_size)
    qh = q.permute(0, 2, 1, 3).to(acc)
    kb = k_blk.permute(0, 2, 1, 3).repeat_interleave(hq // hkv, dim=1).to(acc)
    vb = v_blk.permute(0, 2, 1, 3).repeat_interleave(hq // hkv, dim=1).to(acc)
    ctx = (qh @ kg.to(acc).transpose(-1, -2)) * scale
    ctx = ctx.masked_fill(~ok[:, :, None, :], float("-inf"))
    blk = (qh @ kb.transpose(-1, -2)) * scale
    causal = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
    blk = blk.masked_fill(~causal, float("-inf"))
    probs = torch.softmax(torch.cat([ctx, blk], dim=-1), dim=-1)
    width = kg.shape[2]
    out = probs[..., :width] @ vg.to(acc) + probs[..., width:] @ vb
    return out.permute(0, 2, 1, 3).to(q.dtype)


# ----------------------------------------------------------------------------------------------------------------
# Dustin: head-subset importance, draft lookahead + target history, tiered selection
# ----------------------------------------------------------------------------------------------------------------


def attention_importance(q: torch.Tensor, keys: torch.Tensor, kv_lens: torch.Tensor,
                         heads: Optional[Sequence[int]] = None, scale: Optional[float] = None) -> torch.Tensor:
    """Attention probability mass on each *context* token, summed over the chosen q heads and the n rows.

    q ``[bs, n, Hq, D]`` (drafter block queries for the draft lookahead score, or the last committed target token's
    query for the history score, ``n = 1``); keys ``[bs, Hkv, L, D]`` of the same model and layer. ``heads`` = the
    offline-chosen head subset (Dustin's semantic retrieval heads); ``None`` = all heads. Softmax over the context
    only (Dustin Eq. 5 sums attention weights; whether the verify rows are in the normaliser is not stated in the
    paper). Returns ``[bs, L]`` (0 on padding).
    """
    bs, n, hq, dim = q.shape
    hkv, length = keys.shape[1], keys.shape[2]
    scale = dim ** -0.5 if scale is None else scale
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    head_ids = torch.arange(hq, device=q.device) if heads is None else torch.as_tensor(heads, device=q.device)
    kv_ids = head_ids // (hq // hkv)
    qh = q.permute(0, 2, 1, 3)[:, head_ids].to(acc)        # [bs, S, n, D]
    kh = keys[:, kv_ids].to(acc)                           # [bs, S, L, D]
    logits = (qh @ kh.transpose(-1, -2)) * scale
    ok = _token_valid(kv_lens, length)[:, None, None, :]
    probs = torch.softmax(logits.masked_fill(~ok, float("-inf")), dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)  # empty context
    return probs.sum(dim=(1, 2))


def pool_to_pages(token_scores: torch.Tensor, page_size: int) -> torch.Tensor:
    """Sum token scores ``[bs, L]`` into pages ``[bs, NP]`` (page_size 1 = Dustin's token granularity)."""
    bs, length = token_scores.shape
    num_pages = _num_pages(length, page_size)
    pad = num_pages * page_size - length
    return F.pad(token_scores, (0, pad)).view(bs, num_pages, page_size).sum(dim=-1)


def dustin_select(draft_scores: torch.Tensor, target_scores: torch.Tensor, budget: int, m_draft: int,
                  kv_lens: torch.Tensor, page_size: int = 1, n_sink: int = 0,
                  n_recent: int = 0) -> torch.Tensor:
    """Dustin Eq. 6, tiered under one budget ``k`` (pages; tokens at ``page_size = 1``)::

        I_prot   = sinks U recent window                          (inside the budget)
        I_draft  = TopK(S_draft  \\ I_prot, m)
        I_target = TopK(S_target \\ (I_prot U I_draft), k - m - |I_prot|)
        I_verify = I_prot U I_draft U I_target

    ``draft_scores``/``target_scores`` ``[bs, NP]`` (e.g. ``pool_to_pages(attention_importance(...))``). Returns one
    ``[bs, NP]`` mask shared by every row of the verify block and, in the paper, by every layer. Static shapes:
    ``|I_prot|`` is a per-row tensor used only in a comparison.
    """
    bs, num_pages = draft_scores.shape
    pid = torch.arange(num_pages, device=draft_scores.device)[None, :]
    valid = pid < ((kv_lens[:, None] + page_size - 1) // page_size)
    prot = forced_page_mask(kv_lens, num_pages, page_size, n_sink, n_recent) if (n_sink or n_recent) else \
        torch.zeros_like(valid)
    prot = topk_mask(torch.zeros_like(draft_scores), budget, valid, prot) & prot  # protected set capped at k
    n_prot = prot.sum(dim=-1, keepdim=True)
    m_eff = torch.clamp(torch.minimum(torch.full_like(n_prot, m_draft), budget - n_prot), min=0)
    drafted = topk_mask(draft_scores, m_eff, valid & ~prot)
    rest = torch.clamp(budget - n_prot - drafted.sum(dim=-1, keepdim=True), min=0)
    hist = topk_mask(target_scores, rest, valid & ~prot & ~drafted)
    return prot | drafted | hist


# ----------------------------------------------------------------------------------------------------------------
# Inter-layer reuse (anchor layers)
# ----------------------------------------------------------------------------------------------------------------


def jaccard(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|a & b| / |a | b| over the last dim (1 when both are empty)."""
    inter = (a & b).sum(dim=-1).to(torch.float64)
    union = (a | b).sum(dim=-1).to(torch.float64)
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))


def choose_anchor_layers(masks: Sequence[torch.Tensor], num_anchors: int) -> Tuple[int, ...]:
    """Offline (calibration-time) anchor choice of 2512.21911: the first attention layer always selects; the others
    are ranked by mean ``1 - J(M_l, M_{l-1})`` over the calibration masks and the top ``num_anchors - 1`` select.
    ``masks[l]`` are the oracle (own-score) selections of attention-layer ordinal ``l``. Host-side, not per step."""
    if num_anchors < 1:
        raise ValueError("num_anchors must be >= 1")
    num = len(masks)
    dissim = [float((1.0 - jaccard(masks[i], masks[i - 1])).mean()) for i in range(1, num)]
    order = sorted(range(1, num), key=lambda i: (-dissim[i - 1], i))
    return tuple(sorted([0] + order[: max(0, num_anchors - 1)]))


def reuse_map(num_layers: int, anchors: Optional[Sequence[int]]) -> List[int]:
    """Source layer of each attention-layer ordinal: anchors select (map to themselves), every other layer reuses the
    nearest preceding anchor; layers before the first anchor select for themselves. ``None`` = every layer selects.
    Ordinals count full-attention layers only (Qwen3.5-4B: ordinal i = model layer 4i + 3)."""
    if anchors is None:
        return list(range(num_layers))
    anchor_set = set(int(a) for a in anchors)
    out, last = [], None
    for i in range(num_layers):
        if i in anchor_set:
            last = i
        out.append(i if last is None else last)
    return out


# ----------------------------------------------------------------------------------------------------------------
# FFN neuron sparsity (dense SwiGLU: Qwen3.5-4B, Qwen3.8-27B, the MoE shared expert)
# ----------------------------------------------------------------------------------------------------------------


def swiglu_ffn(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
               channel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``down(silu(gate x) * up x)``; x ``[..., d]``, w_gate/w_up ``[F, d]``, w_down ``[d, F]``.

    ``channel_mask`` (broadcastable to ``[..., F]``) zeroes dropped channels before ``down``; all-True = dense.
    """
    h = F.silu(x @ w_gate.T) * (x @ w_up.T)
    if channel_mask is not None:
        h = h * channel_mask.to(h.dtype)
    return h @ w_down.T


def ffn_neuron_mask(gate_act: torch.Tensor, mode: str, tau: float = 0.0, keep: Optional[int] = None,
                    block_rows: Optional[int] = None) -> torch.Tensor:
    """Channels kept, from the gate activation ``a = silu(gate x)`` ``[bs*n, F]`` (or ``[..., F]``).

    * ``threshold``  (2512.21911 Eq. 10-11): drop channel i of a token if ``|a_i| < tau``; ``tau = 0`` keeps all.
    * ``token_topk`` keep the ``keep`` largest ``|a_i|`` per token.
    * ``block_topk`` one channel set per request shared by its ``block_rows`` verify rows (inter-draft reuse applied
      to the FFN; not in the paper): top ``keep`` of ``sum_rows |a_i|``. Returns ``[bs, 1, F]`` broadcastable to the
      rows when the input is ``[bs, n, F]``.
    """
    if mode == "dense":
        return torch.ones_like(gate_act, dtype=torch.bool)
    mag = gate_act.abs()
    if mode == "threshold":
        return mag >= tau
    if mode == "token_topk":
        return topk_mask(mag, gate_act.shape[-1] if keep is None else keep)
    if mode == "block_topk":
        if gate_act.dim() == 2:
            if block_rows is None:
                raise ValueError("block_topk on [bs*n, F] needs block_rows")
            mag = mag.view(-1, block_rows, mag.shape[-1])
        return topk_mask(mag.sum(dim=-2, keepdim=True), gate_act.shape[-1] if keep is None else keep)
    raise ValueError(f"mode must be one of {FFN_MODES}")


def sparse_swiglu_ffn(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
                      mode: str = "dense", tau: float = 0.0, keep: Optional[int] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference sparse FFN on ``x [bs, n, d]``: gate computed densely (as in the paper), up/down only on kept
    channels (emulated by masking). Returns ``(out [bs, n, d], channel_mask)``."""
    a = F.silu(x @ w_gate.T)
    mask = ffn_neuron_mask(a, mode, tau, keep)
    h = a * (x @ w_up.T) * mask.to(a.dtype)
    return h @ w_down.T, mask


def block_shared_ffn_gathered(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
                              keep: int) -> torch.Tensor:
    """Block-shared channel set computed with gathered weights (static ``keep``): reads ``w_up``/``w_down`` rows /
    columns of the kept channels only, the form that saves weight bytes at batch 1. Equals
    ``sparse_swiglu_ffn(x, ..., mode="block_topk", keep=keep)[0]``. x ``[bs, n, d]``."""
    a = F.silu(x @ w_gate.T)                                         # [bs, n, F]
    _, idx = torch.sort(_ranks(a.abs().sum(dim=1)), dim=-1, stable=True)  # channel ids by rank
    idx = idx[:, :keep]                                              # [bs, K]
    a_k = torch.gather(a, 2, idx[:, None, :].expand(-1, x.shape[1], -1))
    up_k = w_up[idx]                                                 # [bs, K, d]
    down_k = w_down.T[idx]                                           # [bs, K, d]
    h = a_k * torch.einsum("bnd,bkd->bnk", x, up_k)
    return torch.einsum("bnk,bkd->bnd", h, down_k)


# ----------------------------------------------------------------------------------------------------------------
# MoE expert truncation (Qwen3.6-35B-A3B: 256 routed experts, top-8, + one gated shared expert kept dense)
# ----------------------------------------------------------------------------------------------------------------


def route_topk(router_logits: torch.Tensor, top_k: int, renormalize: bool = True
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softmax routing, top-k experts (descending weight, first-index ties), optional renormalisation.
    Returns ``(weights [T, k], ids [T, k])``."""
    probs = torch.softmax(router_logits.to(torch.float64 if router_logits.dtype == torch.float64
                                           else torch.float32), dim=-1)
    order = torch.argsort(probs, dim=-1, descending=True, stable=True)[..., :top_k]
    w = torch.gather(probs, -1, order)
    if renormalize:
        w = w / w.sum(dim=-1, keepdim=True)
    return w, order


def moe_forward(x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, w13: torch.Tensor,
                w2: torch.Tensor) -> torch.Tensor:
    """Routed-expert output ``sum_j w_j * expert_{id_j}(x)``; ``ids < 0`` or ``w == 0`` contribute nothing.
    x ``[T, d]``, w13 ``[E, 2F, d]`` (gate rows then up rows), w2 ``[E, d, F]``."""
    num_experts, two_f, _ = w13.shape
    f = two_f // 2
    out = torch.zeros_like(x)
    for j in range(ids.shape[-1]):  # static loop over k slots
        e = ids[:, j].clamp(min=0)
        live = (ids[:, j] >= 0).to(x.dtype) * weights[:, j].to(x.dtype)
        gu = torch.einsum("td,tfd->tf", x, w13[e])
        h = F.silu(gu[:, :f]) * gu[:, f:]
        out = out + live[:, None] * torch.einsum("tf,tdf->td", h, w2[e])
    return out


def truncate_experts_fixed(weights: torch.Tensor, ids: torch.Tensor, keep: int,
                           renormalize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep the ``keep`` highest-weight routed experts per token (ids sorted by weight, as ``route_topk``); dropped
    slots get ``id = -1`` and weight 0. ``renormalize`` rescales the kept weights to the original routed total.
    ``keep >= k`` is the lossless limit."""
    keep_t = torch.full(ids.shape[:-1], keep, dtype=torch.int64, device=ids.device)
    return _keep_first(weights, ids, keep_t, renormalize)


def calibrate_beta(weights: torch.Tensor, m: int) -> torch.Tensor:
    """Per-layer ``beta_m`` of 2512.21911: median over calibration tokens of ``sum_{j<=k-m} w_j / sum_j w_j``
    (weights sorted descending). Calibration-time only (``calib`` split)."""
    k = weights.shape[-1]
    kept = weights[..., : k - m].sum(dim=-1) / weights.sum(dim=-1)
    return kept.median()


def truncate_experts_adaptive(weights: torch.Tensor, ids: torch.Tensor, max_skip: int, beta: Number,
                              renormalize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per token, skip the largest ``i <= max_skip`` lowest-weight experts such that the kept top-``(k - i)`` still
    carry at least ``beta`` of the routed mass (``sum_{j<=k-i} w_j / sum_j w_j >= beta``).

    This is the mass-retention reading of 2512.21911's skipping rule; the inequality as extracted from the paper's
    HTML reads ``< beta_m`` (see the method page: not verified, possibly an extraction artefact). ``max_skip = 0`` is
    the lossless limit; ``beta > 1`` never skips.
    """
    k = ids.shape[-1]
    total = weights.sum(dim=-1, keepdim=True)
    ratio = torch.cumsum(weights, dim=-1) / total                       # ratio[:, k-1-i] = kept mass after skipping i
    i = torch.arange(k, device=ids.device)
    kept_after_skip = torch.flip(ratio, dims=[-1])                      # [:, i] = mass of top-(k-i)
    ok = (kept_after_skip >= beta) & (i[None, :] <= max_skip) & (i[None, :] < k)
    n_skip = torch.where(ok, i[None, :].expand_as(ok), torch.zeros_like(ok, dtype=i.dtype)).amax(dim=-1)
    return _keep_first(weights, ids, k - n_skip, renormalize)


def _keep_first(weights: torch.Tensor, ids: torch.Tensor, keep: torch.Tensor, renormalize: bool
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    slot = torch.arange(ids.shape[-1], device=ids.device)
    live = slot[None, :] < keep[:, None]
    w = torch.where(live, weights, torch.zeros_like(weights))
    if renormalize:
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(w.dtype).tiny) * weights.sum(-1, keepdim=True)
    return w, torch.where(live, ids, torch.full_like(ids, -1))


def block_shared_experts(weights: torch.Tensor, ids: torch.Tensor, block_rows: int, num_experts: int,
                         budget: int, renormalize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """One expert set per request for its ``block_rows`` verify rows (inter-draft reuse applied to MoE; **novel, not
    in the papers**): rank experts by routed mass summed over the block, keep the top ``budget``, and drop each
    token's routed experts outside that set. Bounds the experts streamed per request per layer by ``budget`` instead
    of the union of up to ``n * k``. ``budget >= num_experts`` (or >= the union) is the lossless limit.
    weights/ids ``[bs * n, k]``."""
    t, k = ids.shape
    bs = t // block_rows
    mass = torch.zeros(bs, num_experts, dtype=weights.dtype, device=weights.device)
    mass.scatter_add_(1, ids.reshape(bs, block_rows * k), weights.reshape(bs, block_rows * k))
    keep_set = topk_mask(mass, budget, mass > 0)                        # [bs, E]
    in_set = torch.gather(keep_set, 1, ids.reshape(bs, block_rows * k)).view(t, k)
    w = torch.where(in_set, weights, torch.zeros_like(weights))
    if renormalize:
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(w.dtype).tiny) * weights.sum(-1, keepdim=True)
    return w, torch.where(in_set, ids, torch.full_like(ids, -1))


def expected_union_experts(num_experts: int, top_k: int, tokens: int) -> float:
    """Distinct experts touched by ``tokens`` independent uniformly routed tokens: ``E (1 - (1 - k/E)^t)``
    (MoESD's uniform-routing model; real routing is correlated, so measure the union on a trace)."""
    return num_experts * (1.0 - (1.0 - top_k / num_experts) ** tokens)
