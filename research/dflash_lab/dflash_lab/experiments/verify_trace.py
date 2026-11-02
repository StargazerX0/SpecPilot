"""Record the target's verify-shape scores along the committed path (diagnostic; untimed runs only).

Observes ``DFlashWorkerV2._accept_block`` read-only. For every decode step and request row it
writes one JSON line to ``$DFLASH_LAB_OUT/verify_<pid>.jsonl``:

  rid, P (anchor position), commit (accepted drafts + 1),
  top (per committed position j < commit: top-``k`` [logprob, token] of the verify logits row j,
       i.e. the target distribution for position P+1+j given the committed prefix),
  emitted (logprob of the committed token at each of those positions).

Rows past ``commit`` are conditioned on rejected drafts and are not recorded. This is the layout of
the stock chain verify (logits row j <-> block position j); an engine that verifies another layout
(for example a tree) must expose the accepted path in the same form for `ar validate --layer
numerics` to compare it. A ragged verify (per-request widths packed back to back) passes ``verify_lens`` (one width per row) to
``_accept_block``; the trace then splits the logits rows by those widths. The dump syncs the host every step.
Outputs are unchanged.
"""
from __future__ import annotations

import json
import os

import torch

from dflash_lab import core

DIAGNOSTIC = True
_S = {"f": None, "rows": None}
_P = {}


def _file():
    if _S["f"] is None:
        out = os.environ.get("DFLASH_LAB_OUT")
        if not out:
            raise RuntimeError("verify_trace needs DFLASH_LAB_OUT")
        os.makedirs(out, exist_ok=True)
        _S["f"] = open(os.path.join(out, f"verify_{os.getpid()}.jsonl"), "a", buffering=1)
    return _S["f"]


def _row_spans(n_rows: int, bs: int, verify_lens) -> list:
    """(start, width) of each request's logits rows: uniform blocks for the stock chain verify; a ragged verify
    (per-request widths packed back to back) passes ``verify_lens`` to ``_accept_block``."""
    if verify_lens is None:
        block = n_rows // bs
        return [(i * block, block) for i in range(bs)]
    widths = [int(w) for w in (verify_lens.tolist() if torch.is_tensor(verify_lens) else verify_lens)][:bs]
    starts = [sum(widths[:i]) for i in range(len(widths))]
    if sum(widths) != n_rows:
        raise RuntimeError(f"verify_trace: verify_lens sum {sum(widths)} != {n_rows} logits rows")
    return list(zip(starts, widths))


def _after_accept(result, *args, **kwargs):
    logits = kwargs["next_token_logits"]
    bs = kwargs["bs"]
    commit_lens, out_tokens = result[1], result[3]
    spans = _row_spans(logits.shape[0], bs, kwargs.get("verify_lens"))
    lp = torch.log_softmax(logits.float(), dim=-1)
    top_v, top_i = lp.topk(_P["k"], dim=-1)
    flat_out = out_tokens.reshape(-1).long()
    emitted = lp.gather(1, flat_out.view(-1, 1)).view(-1) if flat_out.numel() == logits.shape[0] else None
    tv, ti = top_v.tolist(), top_i.tolist()
    em = emitted.tolist() if emitted is not None else None
    _S["rows"] = (commit_lens.view(-1)[:bs].tolist(), [tv[a:a + w] for a, w in spans], [ti[a:a + w] for a, w in spans],
                  [em[a:a + w] if em is not None else [] for a, w in spans])


def _observe(*, worker, batch, result, ctx, is_extend):
    if is_extend or _S["rows"] is None:
        return
    commit, tv, ti, em = _S["rows"]
    _S["rows"] = None
    anchors = ctx.anchor_pos.tolist()
    f = _file()
    for i, req in enumerate(batch.reqs):
        c = min(int(commit[i]), len(tv[i]))
        f.write(json.dumps({"rid": req.rid, "P": int(anchors[i]), "commit": c,
                            "top": [[[round(v, 5), int(t)] for v, t in zip(tv[i][j], ti[i][j])] for j in range(c)],
                            "emitted": [round(x, 5) for x in em[i][:c]]}, separators=(",", ":")) + "\n")


def apply(params):
    _P["k"] = int(params.pop("k", 5))
    if params:
        raise ValueError(f"unknown verify_trace params {params}")
    core.hook(core.ACCEPT, _after_accept, "observe")
    core.add_step_observer(_observe)
