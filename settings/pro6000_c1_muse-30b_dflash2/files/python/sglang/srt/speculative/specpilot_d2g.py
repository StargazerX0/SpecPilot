"""SPECPILOT_D2G: predictive gate for the two-level cycle's trailing Q-step (SPECPILOT_D8R_TLV + SPECPILOT_D4_TAILSKIP).

Under the overlap scheduler, scheduler step k+1 is launched before step k's result is processed. In the two-level verify
a step is Q-step (draft + Q verify, no commit) -> second draft -> exact verify. The D4 certain-finish gate (it waits on
exact step k's pinned copy) sits before the exact verify of step k+1, so the Q-step of the trailing step k+1 always runs
and is discarded: every request ends with ~13 ms of GPU work that the next request's prefill queues behind.

Gating the Q-step on the D4 gate unconditionally would drain the GPU every cycle (the host would wait for exact step k
before launching anything). Here the host predicts the end from data that is complete BEFORE exact step k starts: the
composed tree of step k (Q's chain + Q's bonus + the second draft's subtree, copied to pinned memory right after the
compose). Exact step k commits a path of that tree plus the target's bonus, so:
  - a stop id among the tree's tokens, or
  - output_count(k-1) + max tree depth + 2 >= max_new_tokens (exact step k may reach the length cap; 1 of margin)
make the finish of step k possible. Only then does the host take the D4 gate before the Q-step of step k+1; when the
gate says finished, the Q-step is skipped (the trailing step keeps only its plain draft; its outputs are discarded by
the scheduler as before). Otherwise the cycle runs unchanged, so outputs are identical in every case; a false
positive costs one host round trip with an idle GPU (~1 ms). A miss (e.g. the target's own bonus is the stop id)
costs what it costs today.

Waiting for the pinned copy of step k at the start of step k+1 does not drain the GPU: exact step k (~15 ms) is
already queued behind it.

Knob (read once at import): SPECPILOT_D2G_QGATE=1 enables it (needs SPECPILOT_D8R_TLV and SPECPILOT_D4_TAILSKIP);
unset = the parent's path, untouched.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

_ENABLED = os.environ.get("SPECPILOT_D2G_QGATE", "0") == "1"


def enabled() -> bool:
    return _ENABLED


def possible_finish(
    *, tokens, max_depth: int, output_count: int, max_new_tokens: Optional[int], stops: frozenset,
    ignore_eos: bool, vocab_size: Optional[int],
) -> bool:
    """True when exact step k, committing a path of a tree with these tokens plus a bonus, could finish the request."""
    # A committed path below the root has at most max_depth tokens, plus the bonus; one token of margin.
    if max_new_tokens is not None and output_count + max_depth + 2 >= max_new_tokens:
        return True
    for t in tokens:
        if t < 0 or (vocab_size is not None and t >= vocab_size):
            return True
        if not ignore_eos and t in stops:
            return True
    return False


class QGate:
    """Per-worker state: the pinned copy of the last exact step's composed tree (batch 1)."""

    def __init__(self, *, n: int):
        self._tokens_host = torch.zeros(n, dtype=torch.int64, pin_memory=True)
        self._depth_host = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        self._event = torch.cuda.Event()
        self._req = None
        self._pending = False
        self.checks = 0
        self.predicted = 0
        self.skipped = 0

    def record_tree(self, *, req, tokens: torch.Tensor, depth: torch.Tensor) -> None:
        """Right after the compose of an exact step (stream order): stage its tree tokens and max depth."""
        tok = tokens.reshape(-1)
        if self._tokens_host.dtype != tok.dtype or self._tokens_host.numel() != tok.numel():
            self._tokens_host = torch.zeros(tok.numel(), dtype=tok.dtype, pin_memory=True)
        self._tokens_host.copy_(tok, non_blocking=True)
        self._depth_host.copy_(depth.reshape(-1).amax().reshape(1).to(torch.int64), non_blocking=True)
        self._event.record()
        self._req = req
        self._pending = True

    def take(self, req) -> Optional[tuple]:
        """At the start of the next step: (tokens, max_depth) of the previous exact step's tree, or None."""
        if not self._pending or req is not self._req:
            self._pending = False
            return None
        self._pending = False
        self._event.synchronize()
        return self._tokens_host.tolist(), int(self._depth_host[0])

    def gate(self, *, req, tail) -> Optional[bool]:
        """Before the Q-step of step k+1. Returns None (no prediction: the D4 gate runs at its usual place), or the D4
        gate's verdict taken here (True = step k finished the request: skip the Q-step and the verify replay)."""
        prev = self.take(req)
        # The D4 tracker must hold exact step k's record for this request, else its gate can only say False.
        if prev is None or tail._req is not req or not tail._pending:
            return None
        self.checks += 1
        tokens, max_depth = prev
        if not possible_finish(
            tokens=tokens, max_depth=max_depth, output_count=tail._count,
            max_new_tokens=req.sampling_params.max_new_tokens, stops=tail._stops,
            ignore_eos=req.sampling_params.ignore_eos, vocab_size=req.vocab_size,
        ):
            return None
        self.predicted += 1
        verdict = tail.should_skip(req)
        if verdict:
            self.skipped += 1
        return verdict


_STATE: Optional[QGate] = None


def state(*, n: int) -> QGate:
    global _STATE
    if _STATE is None:
        _STATE = QGate(n=n)
    return _STATE
