"""SPECPILOT_D1_LD: the DSpark drafter inside the FP8 load lineage worker (DFlashWorkerV2 + packed tree verify).

Port of lib ``dspark_lineage`` (batch-1 int-015 lineage) to the batched load lineage (int-210: rv_pack / rv_topt
packed ragged verify, drafter rblock, FP8 drafter + 98k FP8 candidate head).

DSpark (``DSparkDraftModel``) and DFlash2 share the drafter backbone (``DFlashDraftModel``: 5 layers, fc over the
target taps, the same context-KV path). They differ in the proposal head:

- DFlash2 samples block rows 1..B-1 (row 0 = anchor) and walks the selector lattice
  ``score[b,s,p,c] = unary[s,c] + <A[pred_p] * proj(h_s), B[c]>``.
- DSpark samples query rows ``[bonus, MASK x (B-1)]`` from row 0 (``sample_from_anchor``: row i predicts the token at
  depth i + 1) and walks a VanillaMarkov bigram over the full vocabulary: ``logits_t = base_t + W2 @ W1[prev_t]``.

The Markov bias is the selector's edge term with hidden factor 1, ``A = markov_w1`` (predecessor table) and
``B = markov_w2.weight`` (successor table, both [V, 256]). Restricted to the per-row top-K of the base logits it gives
the K x K lattice the lineage's best-first tree, rv_topt policy and rblock consume unchanged. Rows 0..B-2 of the
B = 16 block feed lattice slots 1..15 (the same count as DFlash2's rows 1..15), so every int-210 shape (M = bs*16
drafter rows, rblock tables, the [bs, 15, K, K] builder input, the verify tiers) is unchanged.

Knobs (read once; every knob unset = parent behaviour, nothing here runs):
  SPECPILOT_D1_LD=1          route ``--speculative-algorithm DSPARK`` to the DFLASH lineage worker and adapt the DSpark
                             drafter in place (class swap after load: Markov selector, DFlash2 candidate head with the
                             SPECPILOT_DE FP8 vocabulary subset)
  SPECPILOT_D1_LD_EDGES=topk lattice form: 'topk' (default) = base top-K per row, K x K Markov edges renormalised over
                             the K by the tree builder; 'wide' = per-parent Markov rows over a base top-M shortlist,
                             normalised by logaddexp(lse_shortlist, base tail), Viterbi beam of K tokens per slot
                             (ready edge log-probs: build_tree norm=False); with K >= N - 1 the best-first N-node tree
                             over the beam lattice is the best-first tree over the shortlist rows (torch ops here)
  SPECPILOT_D1_LD_TOPK=K     candidates (topk) / beam (wide) per slot (power of two, default 16; rblock and the tree
                             builder use the same K)
  SPECPILOT_D1_LD_SHORTLIST=M  wide mode: base shortlist per row (default 512); union mode: base top-B (default 128)
  SPECPILOT_D1_LD_EDGES=union  d4's batched union-candidate builder (specpilot_d4_ldwide, slot d4-001-ldwide): per parent
                             base top-B U static Markov top-MS of the parent, exact lse + tail normaliser, beam K, one
                             CTA per request (graph-safe); same output contract as 'wide'
  SPECPILOT_D1_LD_MS=MS      union mode: static per-parent Markov candidates (default 32; 0 = plain wide over top-B)
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
    raw = os.environ.get("SPECPILOT_D1_LD", "").strip().lower()
    if raw in ("", "0", "off", "false"):
        return False
    if raw not in ("1", "on", "true"):
        raise ValueError(f"SPECPILOT_D1_LD={raw!r}: expected 1 or unset")
    return True


@functools.lru_cache(maxsize=None)
def edges_mode() -> str:
    mode = os.environ.get("SPECPILOT_D1_LD_EDGES", "").strip().lower() or "topk"
    if mode not in ("topk", "wide", "union"):
        raise ValueError(f"SPECPILOT_D1_LD_EDGES={mode!r}: expected topk, wide or union")
    return mode


@functools.lru_cache(maxsize=None)
def shortlist_size() -> int:
    m = int(os.environ.get("SPECPILOT_D1_LD_SHORTLIST", "") or (128 if edges_mode() == "union" else 512))
    if m < 1:
        raise ValueError(f"SPECPILOT_D1_LD_SHORTLIST={m} must be >= 1")
    return m


@functools.lru_cache(maxsize=None)
def union_ms() -> int:
    return int(os.environ.get("SPECPILOT_D1_LD_MS", "") or 32)


@functools.lru_cache(maxsize=None)
def top_k() -> int:
    k = int(os.environ.get("SPECPILOT_D1_LD_TOPK", "") or 16)
    if k <= 0 or k & (k - 1):
        raise ValueError(f"SPECPILOT_D1_LD_TOPK={k} must be a power of two")
    return k


@functools.lru_cache(maxsize=None)
def debug_every() -> int:
    """SPECPILOT_D1_LD_DEBUG=k: every k-th draft step log sum(n_buf), bs and the tree score range (host sync;
    untimed diagnostic only). Unset = nothing."""
    return int(os.environ.get("SPECPILOT_D1_LD_DEBUG", "") or 0)


_DBG = {"step": 0}


def debug_step(worker, bs: int, tree) -> None:
    from sglang.srt.speculative import specpilot_rv as _drv

    _DBG["step"] += 1
    if _DBG["step"] % debug_every():
        return
    n = _drv.state(worker.device).n_buf[:bs]
    sc = tree["score"][:bs] if tree is not None and "score" in tree else None
    rv = _drv.rv_config()
    logger.info(
        "SPECPILOT_D1_LD_DEBUG step=%d bs=%d sum_n=%d n_min=%d n_max=%d tier=%s score_min=%s nan=%s",
        _DBG["step"], bs, int(n.sum()), int(n.min()), int(n.max()),
        rv.tokens_for_bs(bs, int(worker.block_size)) if rv is not None else None,
        None if sc is None else float(sc.min()), None if sc is None else bool(torch.isnan(sc).any()),
    )


def clear_caches() -> None:
    for f in (enabled, edges_mode, top_k, shortlist_size, union_ms):
        f.cache_clear()


def resolve_algorithm(name: Optional[str]) -> Optional[str]:
    """DSPARK -> DFLASH under the knob (the lineage worker serves the DSpark checkpoint; server args keep DSPARK)."""
    if name == "DSPARK" and enabled():
        _log_alias_once()
        return "DFLASH"
    return name


@functools.lru_cache(maxsize=None)
def _log_alias_once() -> None:
    logger.info("SPECPILOT_D1_LD: --speculative-algorithm DSPARK served by the DFLASH load lineage worker.")


def is_lineage_drafter(draft_model) -> bool:
    return _LINEAGE_CLS is not None and isinstance(draft_model, _LINEAGE_CLS)


def row_offset(draft_model) -> int:
    """First drafter row feeding lattice slot 1: 0 for the adapted DSpark drafter, 1 (the parent) otherwise."""
    return 0 if is_lineage_drafter(draft_model) else 1


def wide(draft_model) -> bool:
    """The adapted DSpark drafter in wide mode: its selector returns (cand, ready edge log-probs) itself."""
    return is_lineage_drafter(draft_model) and edges_mode() in ("wide", "union")


def union(draft_model) -> bool:
    return is_lineage_drafter(draft_model) and edges_mode() == "union"


def tree_norm(draft_model) -> bool:
    """build_tree's norm flag for this drafter's lattice (False: ready edge log-probs)."""
    return not wide(draft_model)


def row_range(draft_model, block_size: int) -> Tuple[int, int]:
    """Hidden rows [lo, hi) feeding the block_size - 1 lattice slots: DSpark (0, B-1); parent DFlash2 (1, B)."""
    lo = row_offset(draft_model)
    return lo, lo + int(block_size) - 1


# ------------------------------------------------------------------------------------------------
# lattice: Markov bigram edges over the per-row base top-K (reference used by the tests)
# ------------------------------------------------------------------------------------------------


def markov_edges_ref(
    *,
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    anchor_token_ids: torch.Tensor,
) -> torch.Tensor:
    """score[b,s,p,c] = unary[b,s,c] + <W1[pred[b,s,p]], W2[cand[b,s,c]]>; pred = cand[b,s-1], anchor at s = 0."""
    k = candidate_ids.shape[-1]
    keys = successor_table[candidate_ids].float()
    predecessor_ids = torch.cat(
        [anchor_token_ids[:, None, None].expand(-1, 1, k), candidate_ids[:, :-1]], dim=1
    )
    predecessors = predecessor_table[predecessor_ids].float()
    return unary_logits.float()[:, :, None] + torch.einsum("blpr,blcr->blpc", predecessors, keys)


def subset_logits(draft_model, hidden: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Draft base logits [N, Vc] fp32 over the candidate vocabulary and the sub->full id map (None = full vocab):
    the SPECPILOT_DE subset head (FP8 when SPECPILOT_X2_DFP8_HEAD) when built, else the target lm_head."""
    from sglang.kernels.ops.speculative import dflash_de_vocab as _de_vocab
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
    from sglang.srt.models.dflash import _project_candidate_logits

    lm_head = draft_model.lm_head
    quant = should_apply_lm_head_quant_method(lm_head, getattr(lm_head, "quant_method", None))
    sv = _de_vocab.maybe_build(draft_model, lm_head) if not quant else None
    if sv is not None:
        if sv.fp8 is not None:
            from sglang.kernels.ops.speculative.specpilot_draft_fp8 import block_fp8_linear

            logits = block_fp8_linear(hidden.to(sv.weight.dtype), *sv.fp8)
        else:
            logits = torch.matmul(hidden.to(sv.weight.dtype), sv.weight.T)
        return draft_model._transform_unary_logits(logits), sv.sub_to_full
    org = int(lm_head.org_vocab_size)
    logits = _project_candidate_logits(hidden, lm_head, num_org=org, use_quant_head=quant)
    return draft_model._transform_unary_logits(logits), None


def subset_logits_raw(draft_model, hidden: torch.Tensor) -> torch.Tensor:
    """Raw draft logits [N, Vs] over the SPECPILOT_DE subset (head dtype; union mode: the transform is the identity)."""
    from sglang.kernels.ops.speculative import dflash_de_vocab as _de_vocab

    sv = _de_vocab.maybe_build(draft_model, draft_model.lm_head)
    if sv is None:
        raise RuntimeError("SPECPILOT_D1_LD_EDGES=union needs the SPECPILOT_DE vocabulary subset")
    if sv.fp8 is not None:
        from sglang.kernels.ops.speculative.specpilot_draft_fp8 import block_fp8_linear

        return block_fp8_linear(hidden.to(sv.weight.dtype), *sv.fp8)
    return torch.matmul(hidden.to(sv.weight.dtype), sv.weight.T)


def prepare(draft_model, max_bs: int) -> None:
    """Union mode: build d4's UnionLatticeBuilder (static Markov table, scratch for max_bs) before graph capture."""
    if not union(draft_model):
        return
    draft_model.candidate_selector.union_builder(draft_model, max_bs)


def shortlist(logits: torch.Tensor, ids_map: Optional[torch.Tensor], m: int):
    """logits [N, Vc] fp32 -> (vals [N, m] fp32, full ids [N, m] int64, tail [N] fp32 = log base mass outside)."""
    from sglang.srt.models.dflash import _radix_topk

    m = min(int(m), logits.shape[-1])
    vals, pos = _radix_topk(logits, m) if logits.is_cuda else torch.topk(logits, m, dim=-1)
    vals = vals.float()
    pos = pos.long()
    ids = pos if ids_map is None else ids_map[pos]
    lse = torch.logsumexp(logits, dim=-1)
    head = torch.logsumexp(vals, dim=-1)
    tail = lse + torch.log1p(-torch.exp(head - lse).clamp(max=1.0 - 1e-6))
    return vals, ids, tail


def markov_beam(
    *,
    vals: torch.Tensor,
    ids: torch.Tensor,
    tail: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    beam: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Wide lattice. vals/ids [bs, S, M], tail [bs, S] -> (cand [bs, S, K] full ids, edges [bs, S, K, K] fp32).

    Slot s row of parent p: s_pc = base_s[c] + <W1[p], W2[c]> over the shortlist c, normalised by
    logaddexp(lse_c s_pc, tail_s) (Markov bias taken as 0 outside the shortlist). Slot-s tokens = the top-K by best
    path log-prob over the slot-(s-1) beam (Viterbi), so with K >= N - 1 every node of the best-first N-node tree over
    these rows is in the lattice. Edge [s, p, c] = log-prob of beam token c under beam parent p (slot 0: the anchor in
    every row). Static shapes, no host sync (graph-capturable)."""
    bs, S, M = vals.shape
    K = int(beam)
    dev = vals.device
    cand = torch.empty((bs, S, K), dtype=torch.int64, device=dev)
    edges = torch.empty((bs, S, K, K), dtype=torch.float32, device=dev)
    parents = anchor_token_ids.long().view(bs, 1).expand(bs, K)
    path = torch.full((bs, K), float("-inf"), dtype=torch.float32, device=dev)
    path[:, 0] = 0.0
    for s in range(S):
        keys = w2[ids[:, s]]  # [bs, M, r]
        bias = torch.bmm(w1[parents], keys.transpose(1, 2)).float()  # [bs, K, M]
        row = vals[:, s, None, :] + bias
        z = torch.logaddexp(torch.logsumexp(row, dim=-1), tail[:, s, None])
        logp = row - z[..., None]
        best = (path[..., None] + logp).max(dim=1).values  # [bs, M]
        bv, bi = torch.topk(best, K, dim=-1)
        cand[:, s] = ids[:, s].gather(1, bi)
        edges[:, s] = logp.gather(2, bi[:, None, :].expand(-1, K, -1))
        parents, path = cand[:, s], bv
    return cand, edges


class _UnitGate(nn.Module):
    """hidden [.., H] -> ones [.., rank]: the selector's hidden factor for an ungated (vanilla) Markov head."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = int(rank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.new_ones((*hidden_states.shape[:-1], self.rank))


_SELECTOR_CLS = None


def _selector_class():
    global _SELECTOR_CLS
    if _SELECTOR_CLS is not None:
        return _SELECTOR_CLS
    from sglang.srt.models.dflash import CandidateSelector

    class DSparkMarkovSelector(CandidateSelector):
        """CandidateSelector over DSpark's VanillaMarkov tables (predecessor = markov_w1, successor =
        markov_w2.weight, hidden factor 1). build_lattice / _score_edges / sample_path run unchanged; the tables
        stay owned (and loaded) by the draft model."""

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

        def union_builder(self, draft_model, max_bs: int = 24):
            bld = self.__dict__.get("_union_bld")
            if bld is not None:
                return bld
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("SPECPILOT_D1_LD union builder must be built before graph capture")
            from sglang.kernels.ops.speculative import dflash_de_vocab as _de_vocab
            from sglang.srt.speculative import specpilot_d4_ldwide as _d4

            sv = _de_vocab.maybe_build(draft_model, draft_model.lm_head)
            if sv is None:
                raise RuntimeError("SPECPILOT_D1_LD_EDGES=union needs SPECPILOT_DE_VOCAB_SIZE")
            cfg = draft_model.draft_config
            if cfg.output_multiplier != 1.0 or cfg.final_logit_softcapping is not None:
                raise RuntimeError("SPECPILOT_D1_LD_EDGES=union needs identity unary logits")
            bld = _d4.UnionLatticeBuilder(
                self.predecessor_codebook, self.successor_codebook, sv.sub_to_full, max_bs=int(max_bs), n_slots=15,
                K=self.top_k, B=shortlist_size(), MS=union_ms())
            self.__dict__["_union_bld"] = bld
            logger.info("SPECPILOT_D1_LD: union lattice builder (d4) max_bs=%d K=%d B=%d MS=%d Vs=%d",
                        int(max_bs), self.top_k, shortlist_size(), union_ms(), bld.Vs)
            return bld

        def lattice_from_hidden(self, draft_model, pred_hidden: torch.Tensor, anchor_token_ids: torch.Tensor):
            """Wide / union mode: (cand [bs, S, K], edge log-probs [bs, S, K, K]) straight from the drafter rows."""
            bs, S, H = pred_hidden.shape
            if edges_mode() == "union":
                logits = subset_logits_raw(draft_model, pred_hidden.reshape(-1, H)).view(bs, S, -1)
                # anchors arrive as a strided column view (block_ids[:, 0]); the Triton builder reads them densely
                return self.union_builder(draft_model).lattice(logits, anchor_token_ids.contiguous())
            logits, ids_map = subset_logits(draft_model, pred_hidden.reshape(-1, H))
            vals, ids, tail = shortlist(logits, ids_map, shortlist_size())
            return self.lattice_from_shortlist(vals.view(bs, S, -1), ids.view(bs, S, -1), tail.view(bs, S),
                                               anchor_token_ids)

        def lattice_from_shortlist(self, vals, ids, tail, anchor_token_ids):
            return markov_beam(vals=vals, ids=ids, tail=tail, anchor_token_ids=anchor_token_ids,
                               w1=self.predecessor_codebook, w2=self.successor_codebook, beam=self.top_k)

    _SELECTOR_CLS = DSparkMarkovSelector
    return _SELECTOR_CLS


def make_selector(markov_head: nn.Module, k: int) -> nn.Module:
    return _selector_class()(markov_head, k)


# ------------------------------------------------------------------------------------------------
# drafter adaptation (class swap after load, before the worker attaches the target lm_head)
# ------------------------------------------------------------------------------------------------

_LINEAGE_CLS = None


def lineage_class():
    global _LINEAGE_CLS
    if _LINEAGE_CLS is not None:
        return _LINEAGE_CLS
    from sglang.srt.models.dflash import DFlash2DraftModel
    from sglang.srt.models.dspark import DSparkDraftModel

    class D1LDDSparkDraftModel(DSparkDraftModel):
        """DSpark drafter with the DFlash2 lineage interface: compute_candidates (SPECPILOT_DE FP8 subset head),
        candidate_selector (Markov edges), DE subset built when the worker attaches the target lm_head."""

        # The DFLASH worker (and rblock) stage input_embeds themselves; hide DSpark's in-graph embedding hook.
        @property
        def forward_embed(self):
            raise AttributeError("forward_embed")

        compute_candidates = DFlash2DraftModel.compute_candidates
        _transform_unary_logits = DFlash2DraftModel._transform_unary_logits

        def __setattr__(self, name, value):
            nn.Module.__setattr__(self, name, value)
            # SPECPILOT_DE: gather the subset rows when the worker attaches the target head (eager, before capture),
            # as DFlash2DraftModel.__setattr__ does.
            if name == "lm_head" and value is not None and self.__dict__.get("_specpilot_de_vocab_size", 0):
                from sglang.kernels.ops.speculative import dflash_de_vocab as _de_vocab
                from sglang.srt.runtime_context import get_parallel

                if get_parallel().tp_size == 1:
                    _de_vocab.maybe_build(self, value)

    D1LDDSparkDraftModel.__name__ = "D1LDDSparkDraftModel"
    _LINEAGE_CLS = D1LDDSparkDraftModel
    return _LINEAGE_CLS


def adapt_draft_model(model: nn.Module) -> nn.Module:
    """Turn a loaded DSparkDraftModel into the lineage drafter in place."""
    from sglang.kernels.ops.speculative import dflash_de_vocab as _de_vocab
    from sglang.srt.models.dspark import DSparkDraftModel

    if not isinstance(model, DSparkDraftModel):
        raise RuntimeError(f"SPECPILOT_D1_LD needs the DSpark drafter, got {type(model).__name__}")
    head = getattr(model, "markov_head", None)
    if head is None or getattr(head, "markov_head_type", None) != "vanilla":
        raise RuntimeError("SPECPILOT_D1_LD supports the vanilla Markov head only")
    if not getattr(model, "sample_from_anchor", True):
        raise RuntimeError("SPECPILOT_D1_LD needs a sample_from_anchor DSpark checkpoint (rows 0..B-2 -> slots)")
    if getattr(model, "lm_head", None) is not None:
        raise RuntimeError("SPECPILOT_D1_LD: adapt before the lm_head is attached")
    edges_mode()
    if edges_mode() in ("wide", "union"):
        from sglang.srt.speculative import specpilot_tree as _dtree

        cfg = _dtree.tree_config()
        if cfg is None or cfg.temp != 1.0 or cfg.temps:
            raise RuntimeError("SPECPILOT_D1_LD_EDGES=wide needs SPECPILOT_TREE_MODE with temperature 1 (ready log-probs)")
    model.__class__ = lineage_class()
    model.__dict__["_specpilot_de_vocab_size"] = _de_vocab.vocab_size_knob()
    model.__dict__["_specpilot_de_vocab"] = None
    # The confidence head only feeds DSpark's ragged-verify planner (unused here): free it.
    model.confidence_head = None
    model.candidate_selector = make_selector(head, top_k())
    logger.info(
        "SPECPILOT_D1_LD: DSpark drafter on the load lineage worker (Markov lattice %s, top_k=%d, shortlist=%s, "
        "de_vocab=%d).",
        edges_mode(),
        top_k(),
        (shortlist_size(), union_ms()) if edges_mode() != "topk" else "-",
        model._specpilot_de_vocab_size,
    )
    return model
