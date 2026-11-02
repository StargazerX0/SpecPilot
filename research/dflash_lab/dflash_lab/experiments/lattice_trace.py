"""Dump the selector lattice per verify step for offline path screening (untimed runs only).

Wraps ``CandidateSelector.build_lattice`` (the original still computes the scores the
walk uses; the wrapper only copies them into a buffer allocated before capture), and
after every decode step appends, per request row, the fp16 scores ``[S, K, K]`` and the
int32 candidate ids ``[S, K]`` to ``$DFLASH_LAB_OUT/lattice_<pid>.bin``, with a JSON
index line ``{rid, P, i}`` in ``lattice_<pid>.idx.jsonl`` (``i`` = record number).
With ``tpred=true`` it also observes ``_accept_block`` (read-only) and appends the target's
greedy argmax at all block positions (int32 ``[block]``, -1 for non-greedy batches) to each
record. With ``unary=true`` it appends the transformed unary top-K logits (fp32 ``[S, K]``,
same order as the candidates) after that, and with ``hidden=true`` the draft hiddens fed to
``compute_candidates`` (bf16 ``[S, H]``, read-only observe) last. The dump syncs the host every step. Outputs are unchanged.
"""
from __future__ import annotations

import json
import os

import torch

from dflash_lab import core

DIAGNOSTIC = True

_S = {"orig": None, "scores": None, "cand": None, "tpred": None, "hidden": None, "f": None, "idx": None, "n": 0}
_P = {}


def _init(worker):
    S, k = worker.block_size - 1, worker.draft_model.candidate_selector.top_k
    cap = core.STATE.cap_rows // S
    _S["scores"] = torch.zeros((cap, S, k, k), dtype=torch.float32, device=worker.device)
    _S["cand"] = torch.zeros((cap, S, k), dtype=torch.int64, device=worker.device)
    if _P["hidden"]:
        H = worker.draft_model.config.hidden_size
        _S["hidden"] = torch.zeros((cap * S, H), dtype=torch.bfloat16, device=worker.device)
    if _P["tpred"]:
        _S["tpred"] = torch.full((cap, S + 1), -1, dtype=torch.int64, device=worker.device)


def _after_accept(result, *args, **kwargs):
    target_predict = result[5]
    if target_predict is None:
        _S["tpred"].fill_(-1)
        return
    bs = target_predict.shape[0]
    _S["tpred"][:bs].copy_(target_predict)


def _capture_hidden(result, self, hidden):
    n = hidden.shape[0]
    if _S["hidden"] is not None and n <= _S["hidden"].shape[0]:
        _S["hidden"][:n].copy_(hidden)


def _lattice(self, *, candidate_ids, unary_logits, hidden_states, anchor_token_ids):
    scores = _S["orig"](self, candidate_ids=candidate_ids, unary_logits=unary_logits,
                        hidden_states=hidden_states, anchor_token_ids=anchor_token_ids)
    bs = scores.shape[0]
    if _S["scores"] is not None and bs <= _S["scores"].shape[0]:
        _S["scores"][:bs].copy_(scores)
        _S["cand"][:bs].copy_(candidate_ids)
    return scores


def _files():
    if _S["f"] is None:
        out = os.environ.get("DFLASH_LAB_OUT")
        if not out:
            raise RuntimeError("lattice_trace needs DFLASH_LAB_OUT")
        os.makedirs(out, exist_ok=True)
        _S["f"] = open(os.path.join(out, f"lattice_{os.getpid()}.bin"), "ab")
        _S["idx"] = open(os.path.join(out, f"lattice_{os.getpid()}.idx.jsonl"), "a", buffering=1)
    return _S["f"], _S["idx"]


def _observe(*, worker, batch, result, ctx, is_extend):
    if is_extend:
        return
    f, idx = _files()
    bs = ctx.bs
    scores = _S["scores"][:bs].to(torch.float16).cpu().numpy()
    cand = _S["cand"][:bs].to(torch.int32).cpu().numpy()
    tpred = _S["tpred"][:bs].to(torch.int32).cpu().numpy() if _P["tpred"] else None
    unary = core.STATE.unary[: bs * scores.shape[1]].view(bs, -1).cpu().numpy() if _P["unary"] else None
    hidden = _S["hidden"][: bs * scores.shape[1]].view(bs, -1).view(torch.int16).cpu().numpy() if _P["hidden"] else None
    anchors = ctx.anchor_pos.tolist()
    for i, req in enumerate(batch.reqs):
        f.write(scores[i].tobytes())
        f.write(cand[i].tobytes())
        if tpred is not None:
            f.write(tpred[i].tobytes())
        if unary is not None:
            f.write(unary[i].tobytes())
        if hidden is not None:
            f.write(hidden[i].tobytes())
        idx.write(json.dumps({"rid": req.rid, "P": anchors[i], "i": _S["n"]}) + "\n")
        _S["n"] += 1
    f.flush()


def apply(params):
    _P["tpred"] = bool(params.pop("tpred", False))
    _P["unary"] = bool(params.pop("unary", False))
    _P["hidden"] = bool(params.pop("hidden", False))
    if params:
        raise ValueError(f"unknown lattice_trace params {params}")
    from sglang.srt.models.dflash import CandidateSelector

    _S["orig"] = CandidateSelector.build_lattice
    if _P["unary"]:
        core.need_unary()
    if _P["hidden"]:
        core.hook(core.CANDIDATES, _capture_hidden, "observe")
    core.on_init(_init)
    core.hook(core.LATTICE, _lattice, "replace")
    core.add_step_observer(_observe)
    if _P["tpred"]:
        core.hook(core.ACCEPT, _after_accept, "observe")
