"""SPECPILOT_D4 trailing-step skip for batch-1 DFlash decode under the overlap scheduler.

The overlap scheduler launches step N+1 before it processes step N's result, so every request ends with one verify
step whose output is discarded (``req.finished()`` in the result processor) and the next request's prefill queues
behind it. At batch 1 the worker copies step N's commit length and committed tokens to pinned host memory; before
replaying step N+1's verify graph it waits for that copy (step N is then complete; step N+1's draft work is already
queued, so the GPU does not idle) and checks whether ``Req.update_finish_state`` is certain to finish the request on
those tokens. Only conditions that finish unconditionally are used (out-of-vocab id, stop/eos id without
ignore_eos, length cap), a strict subset of the host check. When certain, the worker reuses step N's static
verify-graph outputs instead of replaying the graph; everything step N+1 produces is discarded by the scheduler.

Knob (read once at import): SPECPILOT_D4_TAILSKIP=1 enables it; unset = the parent's path, untouched.
"""
from __future__ import annotations

import copy
import os
from typing import List, Optional

import torch

_ENABLED = os.environ.get("SPECPILOT_D4_TAILSKIP", "0") == "1"


def enabled() -> bool:
    return _ENABLED


def stop_ids(req) -> frozenset:
    """Token ids that make ``Req._check_token_based_finish`` fire (same sources, same order of precedence)."""
    ids = set()
    if req.sampling_params.stop_token_ids:
        ids.update(req.sampling_params.stop_token_ids)
    if req.eos_token_ids:
        ids.update(req.eos_token_ids)
    tok = req.tokenizer
    if tok is not None:
        if tok.eos_token_id is not None:
            ids.add(tok.eos_token_id)
        if tok.additional_stop_token_ids:
            ids.update(tok.additional_stop_token_ids)
    return frozenset(ids)


def certain_finish(
    *, tokens: List[int], output_len: int, max_new_tokens: Optional[int], vocab_size: Optional[int],
    ignore_eos: bool, stops: frozenset,
) -> bool:
    """True only if the host finish check fires on ``tokens`` (output length ``output_len`` after appending them)."""
    for t in tokens:
        if t < 0 or (vocab_size is not None and t >= vocab_size):
            return True
    if not ignore_eos and any(t in stops for t in tokens):
        return True
    return max_new_tokens is not None and output_len >= max_new_tokens


class TailSkip:
    """Per-worker state: one tracked batch-1 request, its host-side output count and step N's pinned copy."""

    def __init__(self, *, block_size: int, device: torch.device):
        self._lens_host = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        self._tokens_host = torch.zeros(block_size, dtype=torch.int64, pin_memory=True)
        self._event = torch.cuda.Event()
        self._req = None
        self._dropped = None
        self._stops = frozenset()
        self._count = 0  # output_ids length once the pending step is appended (excluding it until consumed)
        self._pending = False
        self._stale_out = None
        self.skip_ct = 0

    def reset(self) -> None:
        # A request whose count could not be kept is never tracked again (its next step is not its first).
        self._dropped = self._req
        self._req = None
        self._pending = False

    def record_step(self, *, req, commit_lens: torch.Tensor, out_tokens: torch.Tensor) -> None:
        """End of a batch-1 decode step (stream order): stage its commit length and tokens for the next gate."""
        if req is self._dropped:
            return
        if req is not self._req:
            if len(req.output_ids) > 1:
                # Not the request's first decode step: its output count is unknown here.
                self._dropped = req
                return
            # First decode step: the prefill appends exactly one token.
            self._req = req
            self._stops = stop_ids(req)
            self._count = 1
            self._pending = False
        elif self._pending:
            # The previous record was never consumed by a gate: the count would be wrong.
            self.reset()
            return
        lens, tokens = commit_lens.reshape(-1)[:1], out_tokens.reshape(-1)
        if self._lens_host.dtype != lens.dtype:
            self._lens_host = torch.zeros(1, dtype=lens.dtype, pin_memory=True)
        if self._tokens_host.dtype != tokens.dtype or self._tokens_host.numel() != tokens.numel():
            self._tokens_host = torch.zeros(tokens.numel(), dtype=tokens.dtype, pin_memory=True)
        self._lens_host.copy_(lens, non_blocking=True)
        self._tokens_host.copy_(tokens, non_blocking=True)
        self._event.record()
        self._pending = True

    def stash_outputs(self, target_out) -> None:
        """Right after a graph-replayed verify forward: keep the static-buffer views before the step mutates them."""
        if not target_out.can_run_cuda_graph:
            self._stale_out = None
            return
        out = copy.copy(target_out)
        out.logits_output = copy.copy(target_out.logits_output)
        self._stale_out = out

    def should_skip(self, req) -> bool:
        """Gate before step N+1's verify: True when step N certainly finished ``req``."""
        if req is not self._req:
            return False
        if not self._pending or self._stale_out is None:
            self.reset()
            return False
        self._event.synchronize()
        n = int(self._lens_host[0])
        tokens = self._tokens_host[:n].tolist()
        host_len = len(req.output_ids)
        if host_len not in (self._count, self._count + n):
            # The host has not processed exactly step N-1 (or N): do not trust the count.
            self.reset()
            return False
        self._count += n
        self._pending = False
        finish = certain_finish(
            tokens=tokens, output_len=self._count, max_new_tokens=req.sampling_params.max_new_tokens,
            vocab_size=req.vocab_size, ignore_eos=req.sampling_params.ignore_eos, stops=self._stops,
        )
        if finish:
            self.skip_ct += 1
            self.reset()
        return finish

    def stale_outputs(self):
        out = copy.copy(self._stale_out)
        out.logits_output = copy.copy(self._stale_out.logits_output)
        return out


_STATE: Optional[TailSkip] = None


def state(*, block_size: int, device: torch.device) -> TailSkip:
    """The process's tracker (one DFlash worker per scheduler process)."""
    global _STATE
    if _STATE is None:
        _STATE = TailSkip(block_size=block_size, device=device)
    return _STATE
