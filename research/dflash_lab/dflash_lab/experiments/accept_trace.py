"""Record the per-verify-step acceptance histogram (instrumentation only).

Observes ``DFlashWorkerV2._accept_block`` (eager, after the target verify) and
counts ``commit_lens`` (= accepted drafts + 1 bonus token) on the GPU; every
``every`` steps (param, default 200) the counts are written to
``$DFLASH_LAB_OUT/accept_hist_<pid>.json``. The flush syncs: diagnostic runs only.
"""
from __future__ import annotations

import json
import os

import torch

from dflash_lab import core

DIAGNOSTIC = True

_state = {"hist": None, "steps": 0, "every": 200}


def _after(result, *args, **kwargs):
    commit_lens = result[1]
    hist = _state["hist"]
    if hist is None or hist.device != commit_lens.device:
        hist = _state["hist"] = torch.zeros(64, dtype=torch.int64, device=commit_lens.device)
    # index_add_ instead of bincount: bincount on CUDA syncs the host every step.
    idx = commit_lens.to(torch.int64).clamp_max(63)
    hist.index_add_(0, idx, torch.ones_like(idx))
    _state["steps"] += 1
    if _state["steps"] % _state["every"] == 0:
        _flush()


def _flush():
    out = os.environ.get("DFLASH_LAB_OUT", ".")
    os.makedirs(out, exist_ok=True)
    counts = _state["hist"].tolist()
    last = max((i for i, c in enumerate(counts) if c), default=0)
    with open(os.path.join(out, f"accept_hist_{os.getpid()}.json"), "w") as f:
        json.dump({"verify_steps": _state["steps"], "commit_len_counts": counts[: last + 1]}, f)


def apply(params):
    _state["every"] = int(params.pop("every", 200))
    if params:
        raise ValueError(f"unknown accept_trace params {params}")
    core.hook(core.ACCEPT, _after, "observe")
