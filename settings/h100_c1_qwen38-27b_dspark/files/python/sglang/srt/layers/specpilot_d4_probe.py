"""SPECPILOT_D4 probe: dump the verify lm_head inputs (final hidden states) and the target argmax.

Diagnostic only (untimed workspace runs): SPECPILOT_D4_DUMP=<dir> enables it; unset = no-op. The logits processor
copies the lm_head input of verify-width batches into a persistent per-shape buffer (capturable: a plain copy_);
the DFlash worker host-copies that buffer plus the step's argmax after the verify forward (host sync: probe only).
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch

_DUMP_DIR = os.environ.get("SPECPILOT_D4_DUMP") or None
_ROWS = int(os.environ.get("SPECPILOT_D4_DUMP_ROWS", "32"))
_EVERY = int(os.environ.get("SPECPILOT_D4_DUMP_EVERY", "2"))
_FLUSH = 128

_bufs: Dict[tuple, torch.Tensor] = {}
_last: Optional[torch.Tensor] = None
_pending: list = []
_step = 0
_part = 0


def enabled() -> bool:
    return _DUMP_DIR is not None


def stash(hidden: torch.Tensor) -> None:
    """Inside the (captured) logits path: persist the lm_head input rows of a verify-width batch."""
    global _last
    if _DUMP_DIR is None or hidden.dim() != 2 or hidden.shape[0] != _ROWS:
        return
    key = (tuple(hidden.shape), hidden.dtype, hidden.device)
    buf = _bufs.get(key)
    if buf is None:
        buf = torch.empty_like(hidden)
        _bufs[key] = buf
    buf.copy_(hidden)
    _last = buf


def record(target_predict: torch.Tensor) -> None:
    """After the verify forward (eager host code): keep every _EVERY-th step's hidden rows + argmax."""
    global _step
    if _DUMP_DIR is None or _last is None:
        return
    _step += 1
    if _step % _EVERY:
        return
    pred = target_predict.reshape(-1)
    if pred.numel() != _ROWS:
        return
    _pending.append((_last.detach().to("cpu", copy=True), pred.detach().to("cpu", copy=True).to(torch.int32)))
    if len(_pending) >= _FLUSH:
        flush()


def flush() -> None:
    global _part
    if not _pending:
        return
    os.makedirs(_DUMP_DIR, exist_ok=True)
    h = torch.stack([p[0] for p in _pending])
    t = torch.stack([p[1] for p in _pending])
    torch.save({"h": h, "pred": t}, os.path.join(_DUMP_DIR, f"d4dump_{os.getpid()}_{_part:04d}.pt"))
    _part += 1
    _pending.clear()


if _DUMP_DIR is not None:
    import atexit

    atexit.register(flush)
