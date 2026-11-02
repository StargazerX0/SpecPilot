"""Per-verify-step trace for diagnostics and offline replay (untimed runs only).

Every decode step writes one JSON line per request row to
``$DFLASH_LAB_OUT/trace_<pid>.jsonl`` (flushed each step; this syncs the host, so
trace runs are never timed):

  rid, P (anchor position), raw (drafter chain before any refiner, slots 1..7),
  final (proposal that was verified), conf (top-K-normalized unary top-1 mass),
  commit (accepted drafts + 1), out (committed tokens, first ``commit`` valid),
  cand (top-K candidate ids per slot; with ``cand=true``),
  hist_tail (last ``tail`` history tokens up to P; with token history enabled).

Load with ``research/autoresearch/replay.py``; completeness is checked there
(steps per rid == the request's ``spec_verify_ct``). Register ``trace`` first in
``DFLASH_LAB`` so ``raw`` is captured before other refiners run.
"""
from __future__ import annotations

import json
import os

import torch

from dflash_lab import core

DIAGNOSTIC = True

_P = {}
_S = {"f": None, "raw": None, "conf": None, "cand": None}


def _file():
    if _S["f"] is None:
        out = os.environ.get("DFLASH_LAB_OUT")
        if not out:
            raise RuntimeError("trace needs DFLASH_LAB_OUT")
        os.makedirs(out, exist_ok=True)
        _S["f"] = open(os.path.join(out, f"trace_{os.getpid()}.jsonl"), "a", buffering=1)
    return _S["f"]


def _capture_raw(ctx):
    _S["raw"] = ctx.draft_tokens[:, 1:].clone()
    _S["conf"] = core.unary_top1_conf(ctx.bs).clone()
    if _P["cand"]:
        k = core.STATE.cand.shape[1]
        _S["cand"] = core.STATE.cand[: ctx.bs * (ctx.draft_tokens.shape[1] - 1)].view(ctx.bs, -1, k).clone()


def _observe(*, worker, batch, result, ctx, is_extend):
    f = _file()
    if is_extend:
        f.write(json.dumps({"ev": "extend", "rids": [r.rid for r in batch.reqs],
                            "extend_lens": [int(x) for x in batch.extend_lens],
                            "prefix_lens": [int(x) for x in batch.prefix_lens]}) + "\n")
        return
    bs, block = ctx.bs, ctx.draft_tokens.shape[1]
    rows = {"P": ctx.anchor_pos, "final": ctx.draft_tokens[:, 1:], "raw": _S["raw"], "conf": _S["conf"],
            "commit": result.accept_lens.view(-1)[:bs], "out": result.next_token_ids.view(bs, block)}
    if _P["cand"]:
        rows["cand"] = _S["cand"]
    hist = core.STATE.hist
    if hist is not None and _P["tail"]:
        pos = (ctx.anchor_pos.long()[:, None] - torch.arange(_P["tail"] - 1, -1, -1, device=hist.device)).clamp_min(0)
        rows["hist_tail"] = hist[ctx.rows.long()[:, None], pos]
    host = {k: v.tolist() for k, v in rows.items()}
    for i, req in enumerate(batch.reqs):
        rec = {"rid": req.rid, **{k: v[i] for k, v in host.items()}}
        rec["conf"] = [round(x, 4) for x in rec["conf"]]
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def apply(params):
    _P.update(cand=bool(params.pop("cand", True)), tail=int(params.pop("tail", 16)))
    if params:
        raise ValueError(f"unknown trace params {params}")
    core.need_unary()
    if _P["tail"]:
        core.need_hist()
    core.STATE.refiners.insert(0, _capture_raw)
    core.add_step_observer(_observe)
