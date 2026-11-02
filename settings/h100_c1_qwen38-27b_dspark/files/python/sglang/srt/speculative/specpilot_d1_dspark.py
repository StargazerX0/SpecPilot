"""SPECPILOT_D1_DS: the DSpark drafter inside the DFlash2 lineage worker (DFlashWorkerV2 + tree / TLV stack).

DSpark (``DSparkDraftModel``) and DFlash2 share the drafter backbone (``DFlashDraftModel``: 5 layers, fc over the target
taps, the same context-KV path). They differ in the proposal head:

- DFlash2 samples block rows 1..B-1 (row 0 = anchor) and walks the selector lattice
  ``score[b,s,p,c] = unary[s,c] + <A[pred_p] * proj(h_s), B[c]>``.
- DSpark samples ALL query rows ``[bonus, MASK x (gamma-1)]`` (row i predicts the token at depth i + 1) and walks a
  VanillaMarkov bigram over the full vocabulary: ``logits_t = base_t + markov_w2(markov_w1[prev_t])``.

The Markov bias is the selector's edge term with hidden factor 1, ``A = markov_w1`` (predecessor table) and
``B = markov_w2.weight`` (successor table, both [V, 256]). Restricted to the per-row top-K of the base logits it
gives the K x K lattice the lineage's chain walk / best-first tree / TLV consume unchanged. The greedy lattice walk
equals DSpark's greedy chain whenever DSpark's token is inside the base top-K of its row.

Knobs (read once; every knob unset = parent behaviour, nothing here runs):
  SPECPILOT_D1_DS_LINEAGE=1   route ``--speculative-algorithm DSPARK`` to the DFLASH worker and adapt the DSpark
                              drafter (class swap after load: selector, candidate head, DE / D7 attach hooks)
  SPECPILOT_D1_DS_TOPK=K      candidates per row (power of two, default 16)
Drafter rows: ``SPECPILOT_X4_DRAFT_BLOCK=B`` (tree mode) else ``block_size - 1`` (chain: DSpark's native gamma = N - 1);
the draft attention keys only those B block rows (flashattention_backend ``_x4_draft_kv_block``), so a block-N draft
graph computes DSpark's B-row block exactly on rows 0..B-1.
"""
from __future__ import annotations

import functools
import logging
import os
from typing import Optional, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def enabled() -> bool:
    raw = os.environ.get("SPECPILOT_D1_DS_LINEAGE", "").strip().lower()
    if raw in ("", "0", "off", "false"):
        return False
    if raw not in ("1", "on", "true"):
        raise ValueError(f"SPECPILOT_D1_DS_LINEAGE={raw!r}: expected 1 or unset")
    return True


@functools.lru_cache(maxsize=None)
def top_k() -> int:
    k = int(os.environ.get("SPECPILOT_D1_DS_TOPK", "") or (32 if edges_mode() == "wide" else 16))
    if k <= 0 or k & (k - 1):
        raise ValueError(f"SPECPILOT_D1_DS_TOPK={k} must be a power of two")
    return k


@functools.lru_cache(maxsize=None)
def edges_mode() -> str:
    """'wide' (default): per-parent Markov rows over a base top-M shortlist, beam of top_k() tokens per slot, edges
    pre-normalised log-probs; 'topk': the parent selector form over the base top-K of each row (K x K lattice)."""
    mode = os.environ.get("SPECPILOT_D1_DS_EDGES", "").strip().lower() or "wide"
    if mode not in ("wide", "topk"):
        raise ValueError(f"SPECPILOT_D1_DS_EDGES={mode!r}: expected wide or topk")
    return mode


@functools.lru_cache(maxsize=None)
def shortlist_size() -> int:
    m = int(os.environ.get("SPECPILOT_D1_DS_SHORTLIST", "2048") or 2048)
    if m < 1:
        raise ValueError(f"SPECPILOT_D1_DS_SHORTLIST={m} must be >= 1")
    return m


def prenormalized() -> bool:
    """Lattice scores are ready edge log-probs (the tree builder must not renormalise over the K candidates)."""
    return enabled() and edges_mode() == "wide"


def _x4_raw() -> int:
    raw = os.environ.get("SPECPILOT_X4_DRAFT_BLOCK", "").strip()
    return int(raw) if raw else 0


def draft_rows(block_size: int) -> int:
    """DSpark query rows B (each one sampled): X4's B when set, else block_size - 1 (native gamma)."""
    n = int(block_size)
    b = _x4_raw()
    b = min(b, n) if b > 0 else n - 1
    if b < 1:
        raise ValueError(f"SPECPILOT_D1_DS: draft rows {b} for block {n}")
    return b


def is_lineage_drafter(draft_model) -> bool:
    return enabled() and isinstance(draft_model, lineage_class())


def row_range(draft_model, block_size: int, parent_lattice_rows: int) -> Tuple[int, int]:
    """Hidden rows [lo, hi) feeding the lattice: DSpark (0, B); parent DFlash2 (1, lattice_rows)."""
    if is_lineage_drafter(draft_model):
        return 0, draft_rows(block_size)
    return 1, int(parent_lattice_rows)


def wide_lattice(draft_model) -> bool:
    """The adapted DSpark drafter in wide mode: its selector returns (cand, ready edge log-probs) itself."""
    return prenormalized() and is_lineage_drafter(draft_model)


def draft_kv_block(num_draft_tokens) -> Optional[int]:
    """Draft-side block key length under DSpark (None = all N rows)."""
    if not enabled() or num_draft_tokens is None:
        return None
    b = draft_rows(int(num_draft_tokens))
    return b if b < int(num_draft_tokens) else None


def resolve_algorithm(name: Optional[str]) -> Optional[str]:
    """DSPARK -> DFLASH under the knob (the lineage worker serves the DSpark checkpoint)."""
    if name == "DSPARK" and enabled():
        _log_alias_once()
        return "DFLASH"
    return name


@functools.lru_cache(maxsize=None)
def _log_alias_once() -> None:
    logger.info("SPECPILOT_D1_DS_LINEAGE: --speculative-algorithm DSPARK served by the DFLASH lineage worker.")


# ------------------------------------------------------------------------------------------------
# lattice: Markov bigram edges over the per-row base top-K
# ------------------------------------------------------------------------------------------------


def _markov_edges_ref(
    *,
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """score[b,s,p,c] = unary[b,s,c] + <W1[pred[b,s,p]], W2[cand[b,s,c]]>; pred = cand[b,s-1], anchor at s = 0."""
    keys = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        [anchor_token_ids[:, None, None].expand(-1, 1, top_k), candidate_ids[:, :-1]],
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum("blpr,blcr->blpc", predecessors, keys)


class _UnitGate(nn.Module):
    """hidden [.., H] -> ones [.., rank]: the selector's hidden factor for an ungated (vanilla) Markov head."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = int(rank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.new_ones((*hidden_states.shape[:-1], self.rank))


def _selector_base():
    from sglang.srt.models.dflash import CandidateSelector

    return CandidateSelector


def make_selector(markov_head: nn.Module, k: int) -> nn.Module:
    """CandidateSelector over DSpark's VanillaMarkov tables: predecessor = markov_w1, successor = markov_w2.weight,
    hidden factor 1. The parent's build_lattice / _score_edges / sample_path run unchanged (so the lab's lattice tools
    see it as a CandidateSelector); the tables stay owned (and loaded) by the draft model."""
    return _selector_class()(markov_head, k)


_SELECTOR_CLS = None


def _selector_class():
    global _SELECTOR_CLS
    if _SELECTOR_CLS is not None:
        return _SELECTOR_CLS
    base = _selector_base()

    class DSparkMarkovSelector(base):
        def __init__(self, markov_head: nn.Module, k: int) -> None:
            nn.Module.__init__(self)  # no codebook parameters of its own
            self.top_k = int(k)
            object.__setattr__(self, "_markov", markov_head)
            self.hidden_projection = _UnitGate(int(markov_head.markov_rank))

        @property
        def predecessor_codebook(self) -> torch.Tensor:
            return self._markov.markov_w1.weight

        @property
        def successor_codebook(self) -> torch.Tensor:
            return self._markov.markov_w2.weight

        def lattice_from_hidden(self, draft_model, pred_hidden: torch.Tensor, anchor_token_ids: torch.Tensor):
            """Wide mode: (cand [bs, S, K], edge log-probs [bs, S, K, K]) straight from the draft hiddens."""
            bs, S, H = pred_hidden.shape
            logits, ids_map = base_logits(draft_model, pred_hidden.reshape(-1, H))
            return wide_markov_lattice(
                logits=logits.view(bs, S, -1),
                ids_map=ids_map,
                anchor_token_ids=anchor_token_ids,
                w1=self.predecessor_codebook,
                w2=self.successor_codebook,
                shortlist=shortlist_size(),
                beam=self.top_k,
            )

    _SELECTOR_CLS = DSparkMarkovSelector
    return _SELECTOR_CLS


def base_logits(draft_model, hidden: torch.Tensor):
    """Draft base logits over the candidate vocabulary: (logits [N, Vc] fp32, ids_map or None) via the DE head when
    attached (its sub_to_full maps positions to full ids), else the target lm_head over the org vocab."""
    from sglang.srt.models.dflash import _project_candidate_logits
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

    de_head = getattr(draft_model.candidate_selector, "de_head", None)
    if de_head is not None:
        return draft_model._transform_unary_logits(de_head.project(hidden)), de_head.sub_to_full
    lm_head = draft_model.lm_head
    quant = should_apply_lm_head_quant_method(lm_head, getattr(lm_head, "quant_method", None))
    org = int(lm_head.org_vocab_size)
    return draft_model._transform_unary_logits(
        _project_candidate_logits(hidden, lm_head, num_org=org, use_quant_head=quant)
    ), None


def wide_markov_lattice(
    *,
    logits: torch.Tensor,
    ids_map: Optional[torch.Tensor],
    anchor_token_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    shortlist: int,
    beam: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """logits [bs, S, Vc] fp32 -> (cand [bs, S, beam] full ids, edge log-probs [bs, S, beam, beam]).

    Slot s row of parent p: s_pc = base_s[c] + <W1[p], W2[c]> over the base top-M shortlist c, normalised by
    logaddexp(lse_c s_pc, tail_s) (tail_s = base mass outside the shortlist, Markov bias taken as 0 there). The slot-s
    tokens are the top-`beam` by best path log-prob over the slot-(s-1) beam (Viterbi), so with beam >= N - 1 every node
    of the best-first N-node tree over these rows is in the lattice. Edge [s, p, c] = log-prob of beam token c under
    beam parent p (slot 0: the anchor, broadcast over p)."""
    from sglang.srt.models.dflash import _radix_topk

    if logits.is_cuda:  # SPECPILOT_D3_DS_FAST: the same lattice from one cooperative Triton kernel
        from sglang.srt.speculative import specpilot_d3_dsfast as _d3_fast

        if _d3_fast.enabled() and _d3_fast.supported(shortlist, beam, logits.shape[-1]):
            return _d3_fast.wide_markov_lattice(
                logits=logits, ids_map=ids_map, anchor_token_ids=anchor_token_ids, w1=w1, w2=w2,
                shortlist=shortlist, beam=beam,
            )

    bs, S, _ = logits.shape
    flat = logits.reshape(bs * S, -1)
    m = min(int(shortlist), flat.shape[-1])
    vals, pos = _radix_topk(flat, m) if flat.is_cuda else torch.topk(flat, m, dim=-1)  # sorted, descending
    vals = vals.float().view(bs, S, m)
    pos = pos.long()
    ids = (pos if ids_map is None else ids_map[pos]).view(bs, S, m)
    lse = torch.logsumexp(flat, dim=-1).view(bs, S)
    head = torch.logsumexp(vals, dim=-1)
    tail = lse + torch.log1p(-torch.exp(head - lse).clamp(max=1.0 - 1e-7))
    keys = w2[ids]  # [bs, S, M, r]
    cand = torch.empty((bs, S, beam), dtype=torch.int64, device=logits.device)
    edges = torch.empty((bs, S, beam, beam), dtype=torch.float32, device=logits.device)
    parents = anchor_token_ids.long().view(bs, 1)
    path = torch.zeros((bs, 1), dtype=torch.float32, device=logits.device)
    for s in range(S):
        bias = torch.einsum("bpr,bmr->bpm", w1[parents], keys[:, s]).float()
        row = vals[:, s, None, :] + bias  # [bs, P, M]
        z = torch.logaddexp(torch.logsumexp(row, dim=-1), tail[:, s, None])
        logp = row - z[..., None]
        best, _ = (path[..., None] + logp).max(dim=1)  # [bs, M]
        bv, bi = torch.topk(best, beam, dim=-1)
        cand[:, s] = ids[:, s].gather(1, bi)
        e = logp.gather(2, bi[:, None, :].expand(-1, logp.shape[1], -1))
        edges[:, s] = e.expand(-1, beam, -1) if e.shape[1] == 1 else e
        parents, path = cand[:, s], bv
    return cand, edges


# ------------------------------------------------------------------------------------------------
# drafter adaptation (class swap after load)
# ------------------------------------------------------------------------------------------------


def _lineage_class():
    from sglang.srt.models.dflash import DFlash2DraftModel
    from sglang.srt.models.dspark import DSparkDraftModel

    class D1DSparkLineageDraftModel(DSparkDraftModel):
        """DSpark drafter with the DFlash2 lineage interface (compute_candidates, candidate_selector, DE/D7 hooks)."""

        # The DFLASH worker stages input_embeds itself; hide DSpark's in-graph embedding hook from the graph runner.
        @property
        def forward_embed(self):
            raise AttributeError("forward_embed")

        compute_candidates = DFlash2DraftModel.compute_candidates
        _transform_unary_logits = DFlash2DraftModel._transform_unary_logits

        def __setattr__(self, name, value):
            nn.Module.__setattr__(self, name, value)
            if name != "lm_head" or value is None:
                return
            _attach_hooks(self, value)

    D1DSparkLineageDraftModel.__name__ = "D1DSparkLineageDraftModel"
    return D1DSparkLineageDraftModel


_LINEAGE_CLS = None


def lineage_class():
    global _LINEAGE_CLS
    if _LINEAGE_CLS is None:
        _LINEAGE_CLS = _lineage_class()
    return _LINEAGE_CLS


def _attach_hooks(model, lm_head) -> None:
    """DFlash2DraftModel.__setattr__'s lm_head hooks: SPECPILOT_DE derived head, fp8 drafter, SPECPILOT_D7 INT4."""
    from sglang.kernels.ops.speculative.dflash_d7_w4 import apply_d7_w4_
    from sglang.kernels.ops.speculative.dflash_de_head import DEDraftHead, quantize_drafter_linears_
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.speculative.dflash_utils import is_dense_head_weight

    tp1 = get_parallel().tp_size == 1
    sel = getattr(model, "candidate_selector", None)
    if (
        getattr(model, "_de_head_knobs", None) is not None
        and sel is not None
        and getattr(sel, "de_head", None) is None
        and tp1
        and is_dense_head_weight(getattr(lm_head, "weight", None))
    ):
        object.__setattr__(sel, "de_head", DEDraftHead(lm_head, **model._de_head_knobs))
    if getattr(model, "_de_drafter_prec", None) == "fp8" and not getattr(model, "_de_drafter_patched", False) and tp1:
        object.__setattr__(model, "_de_drafter_patched", True)
        quantize_drafter_linears_(model.layers)
    if getattr(model, "_d7_knobs", None) is not None and not getattr(model, "_d7_patched", False) and tp1:
        object.__setattr__(model, "_d7_patched", True)
        apply_d7_w4_(model, lm_head, model._d7_knobs)


def adapt_draft_model(model: nn.Module) -> nn.Module:
    """Turn a loaded DSparkDraftModel into the lineage drafter in place (before lm_head attach / graph capture)."""
    from sglang.kernels.ops.speculative.dflash_d7_w4 import read_d7_knobs
    from sglang.kernels.ops.speculative.dflash_de_head import read_de_drafter_prec, read_de_head_knobs
    from sglang.srt.models.dspark import DSparkDraftModel

    if not isinstance(model, DSparkDraftModel):
        raise RuntimeError(
            f"SPECPILOT_D1_DS_LINEAGE needs the DSpark drafter, got {type(model).__name__}"
        )
    head = getattr(model, "markov_head", None)
    if head is None or getattr(head, "markov_head_type", None) != "vanilla":
        raise RuntimeError("SPECPILOT_D1_DS_LINEAGE supports the vanilla Markov head only")
    if getattr(model, "lm_head", None) is not None:
        raise RuntimeError("SPECPILOT_D1_DS_LINEAGE: adapt before the lm_head is attached")
    if prenormalized():
        from sglang.srt.speculative import specpilot_tree as _dtree

        if _dtree.d3m_calib_enabled():
            raise RuntimeError("SPECPILOT_D1_DS_EDGES=wide gives normalised edges: unset SPECPILOT_D3M_CALIB")
    model.__class__ = lineage_class()
    object.__setattr__(model, "_de_head_knobs", read_de_head_knobs())
    object.__setattr__(model, "_de_drafter_prec", read_de_drafter_prec())
    object.__setattr__(model, "_d7_knobs", read_d7_knobs())
    # The confidence head only feeds DSpark's ragged-verify planner (unused here).
    model.confidence_head = None
    model.candidate_selector = make_selector(head, top_k())
    logger.info(
        "SPECPILOT_D1_DS: DSpark drafter on the lineage worker (Markov lattice, top_k=%d, de_head=%s, drafter=%s, d7=%s).",
        top_k(),
        model._de_head_knobs,
        model._de_drafter_prec,
        model._d7_knobs is not None,
    )
    return model
