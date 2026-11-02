"""SPECPILOT_DB_RP: per-request node budget on the best-first DDTree (dynamic budget family, db-2).

The best-first tree kernel (``specpilot_tree._tree_build_kernel``) picks nodes in non-increasing order of
their cumulative path log-prob, and every node's parent precedes it. So the first n nodes of the N-node
build are exactly the n-node build (a valid tree), and a per-request budget n_i in 1..N is a node *prefix*.
Greedy accept restricted to the first n_i nodes equals greedy accept on the n_i-node tree.

Knobs (read once; every knob unset or ``SPECPILOT_DB_RP_MODE=off`` = L's behaviour, L's graphs):
  SPECPILOT_DB_RP_MODE    off | trace (store node scores, record full-width traces; untimed diagnostic)
                        | mask (policy n[bs] in-graph + accept restricted to the first n_i nodes)
                        | rv (policy writes specpilot_rv n_buf, filling the packed tier
                        specpilot_rv.tokens_for_bs; needs SPECPILOT_DB_RV_MODE=pack, SPECPILOT_DB_RV_POLICY=external)
  SPECPILOT_DB_RP_TRACE   trace directory (mode trace/mask: record per-step rows there; a host sync per step)
  SPECPILOT_DB_RP_POLICY  topt (global top-T split filling a per-bs total, default) | thresh (price threshold) | fixed
  SPECPILOT_DB_RP_LAMBDA  per-bs price table "bs:lam,bs:lam,..." (lambda in acceptance-probability units;
                        linear interpolation in bs); a single float = constant price
  SPECPILOT_DB_RP_GAMMA   calibration exponent: p_accept(node) ~ exp(gamma * score) (default 1.0)
  SPECPILOT_DB_RP_NODES   topt: mean verified nodes per request (root included) by bs, "bs:nodes,..." or a float;
                        total T(bs) = round(nodes(bs) * bs) (linear interpolation in bs)
  SPECPILOT_DB_RP_FIXED   fixed: every request gets this many nodes
  SPECPILOT_DB_RP_MIN     minimum nodes per request (default 1 = root only)
"""
from __future__ import annotations

import functools
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_MODES = ("off", "trace", "mask", "rv")
_POLICIES = ("thresh", "topt", "fixed")


def _parse_table(raw: str, default: float) -> Tuple[Tuple[float, float], ...]:
    raw = (raw or "").strip()
    if not raw:
        return ((1.0, default),)
    if ":" not in raw:
        return ((1.0, float(raw)),)
    pts = []
    for item in raw.split(","):
        if not item.strip():
            continue
        k, v = item.split(":")
        pts.append((float(k), float(v)))
    pts.sort()
    return tuple(pts)


def _interp(table: Tuple[Tuple[float, float], ...], x: float) -> float:
    if len(table) == 1 or x <= table[0][0]:
        return table[0][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return table[-1][1]


@dataclass(frozen=True)
class RPConfig:
    mode: str
    trace_dir: str
    policy: str
    lam: Tuple[Tuple[float, float], ...]
    gamma: float
    nodes: Tuple[Tuple[float, float], ...]
    fixed: int
    min_nodes: int

    @property
    def store_scores(self) -> bool:
        return self.mode in ("trace", "mask", "rv")

    @property
    def writes_rv(self) -> bool:
        return self.mode == "rv"

    @property
    def masked(self) -> bool:
        return self.mode == "mask"

    def lam_at(self, bs: int) -> float:
        return _interp(self.lam, float(bs))

    def total_for_bs(self, bs: int, n_nodes: int) -> int:
        """Verified tokens per step (roots included) for a captured bs; a host constant per graph."""
        t = int(round(_interp(self.nodes, float(bs)) * bs))
        return max(bs, min(bs * int(n_nodes), t))


def apply_policy(tree: dict, cfg: RPConfig, valid: Optional[torch.Tensor] = None) -> None:
    """Draft-tail hook: tree["score"] [bs, N] -> per-request node budget (in-graph)."""
    if cfg.writes_rv:
        from sglang.srt.speculative import specpilot_rv as _drv

        rv = _drv.rv_config()
        bs, N = tree["score"].shape
        rp_policy(
            tree["score"],
            _drv.state(tree["score"].device).n_buf,
            cfg,
            valid=valid,
            total=rv.tokens_for_bs(bs, N),
        )
    elif cfg.masked:
        rp_policy(tree["score"], tree["n_budget"], cfg, valid=valid)


def check_rv(cfg: Optional[RPConfig]) -> None:
    if cfg is None or not cfg.writes_rv:
        return
    from sglang.srt.speculative import specpilot_rv as _drv

    rv = _drv.rv_config()
    if rv is None or rv.policy != "external":
        raise ValueError("SPECPILOT_DB_RP_MODE=rv needs SPECPILOT_DB_RV_MODE=pack and SPECPILOT_DB_RV_POLICY=external")


@functools.lru_cache(maxsize=None)
def rp_config() -> Optional[RPConfig]:
    mode = os.environ.get("SPECPILOT_DB_RP_MODE", "").strip().lower()
    if mode in ("", "0", "off", "none"):
        return None
    if mode not in _MODES:
        raise ValueError(f"SPECPILOT_DB_RP_MODE={mode!r}: expected one of {_MODES}")
    policy = os.environ.get("SPECPILOT_DB_RP_POLICY", "topt").strip().lower()
    if policy not in _POLICIES:
        raise ValueError(f"SPECPILOT_DB_RP_POLICY={policy!r}: expected one of {_POLICIES}")
    return RPConfig(
        mode=mode,
        trace_dir=os.environ.get("SPECPILOT_DB_RP_TRACE", "").strip(),
        policy=policy,
        lam=_parse_table(os.environ.get("SPECPILOT_DB_RP_LAMBDA", ""), 0.0),
        gamma=float(os.environ.get("SPECPILOT_DB_RP_GAMMA", "1.0")),
        nodes=_parse_table(os.environ.get("SPECPILOT_DB_RP_NODES", ""), 16.0),
        fixed=int(os.environ.get("SPECPILOT_DB_RP_FIXED", "16")),
        min_nodes=max(1, int(os.environ.get("SPECPILOT_DB_RP_MIN", "1"))),
    )


# ------------------------------------------------------------------------------------------------
# policy: scores [bs, N] (node 0 = root = 0, non-increasing) -> n [bs] int32 in [min_nodes, N]
# in-graph, static shapes, no host sync. bs is static per captured graph, so the per-bs price /
# fraction is a Python constant folded at capture (graph keyed by bs).
# ------------------------------------------------------------------------------------------------


@triton.jit
def _rp_thresh_kernel(score_ptr, n_out, log_lam, gamma, N: tl.constexpr, NP: tl.constexpr, MIN_N: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, NP)
    s = tl.load(score_ptr + b * N + j, mask=j < N, other=float("-inf"))
    fail = (j >= N) | ((gamma * s < log_lam) & (j > 0))
    n = tl.min(tl.where(fail, j, NP), 0).to(tl.int32)
    n = tl.maximum(n, MIN_N)
    tl.store(n_out + b, n)


def rp_policy(
    scores: torch.Tensor,
    n_out: torch.Tensor,
    cfg: RPConfig,
    valid: Optional[torch.Tensor] = None,
    total: Optional[int] = None,
) -> None:
    """Write n_out[:bs] from scores[:bs] (static shapes; bs taken from scores).

    ``total``: the verified-token tier of this (captured) bs; default from SPECPILOT_DB_RP_NODES."""
    bs, N = scores.shape
    if cfg.policy == "fixed":
        n_out[:bs].fill_(max(cfg.min_nodes, min(N, cfg.fixed)))
        return
    if cfg.policy == "thresh":
        lam = cfg.lam_at(bs)
        log_lam = float("-inf") if lam <= 0 else math.log(lam)
        _rp_thresh_kernel[(bs,)](
            scores, n_out, log_lam, float(cfg.gamma),
            N=N, NP=1 << max(0, (N - 1).bit_length()), MIN_N=min(cfg.min_nodes, N),
        )
        return
    if total is None:
        total = cfg.total_for_bs(bs, N)
    rp_topt(scores, n_out, total=total, min_nodes=min(cfg.min_nodes, N), valid=valid)


def rp_topt(
    scores: torch.Tensor,
    n_out: torch.Tensor,
    *,
    total: int,
    min_nodes: int = 1,
    valid: Optional[torch.Tensor] = None,
) -> None:
    """Exact global top-(total - bs) split over (request, node >= 1); sum(n) == total.

    Keys are the running min of the node scores minus a per-index epsilon, strictly decreasing along
    each row, so any top-k set is a node prefix of every row (a valid best-first tree). bs, N and total
    are static per captured graph: one topk, one scatter, one row sum; no host sync.
    """
    bs, N = scores.shape
    extra = int(total) - bs
    if extra <= 0:
        n_out[:bs].fill_(1)
        return
    if extra >= bs * (N - 1):
        n_out[:bs].fill_(N)
        return
    keys = torch.cummin(scores[:, 1:].float(), dim=1).values
    # 1e-4 per node index: arbitrary tie-break, far below the score resolution that matters.
    keys = keys - torch.arange(N - 1, dtype=torch.float32, device=scores.device) * 1e-4
    if valid is not None:  # graph padding rows verify the root only
        keys = torch.where(valid[:bs, None], keys, float("-inf"))
    idx = torch.topk(keys.reshape(-1), extra, sorted=False).indices
    sel = torch.zeros(bs * (N - 1), dtype=torch.int32, device=scores.device)
    sel.index_fill_(0, idx, 1)
    n = sel.view(bs, N - 1).sum(dim=1, dtype=torch.int32) + 1
    n_out[:bs].copy_(n if valid is None else torch.where(valid[:bs], n, 1))
    if min_nodes > 1:  # breaks sum(n) == total; only for policy studies
        n_out[:bs].clamp_(min=min_nodes)


# ------------------------------------------------------------------------------------------------
# trace recorder (diagnostic only: one device->host copy per step)
# ------------------------------------------------------------------------------------------------


class RPTrace:
    CHUNK = 256

    def __init__(self, trace_dir: str, n_nodes: int):
        self.dir = trace_dir
        os.makedirs(trace_dir, exist_ok=True)
        self.N = int(n_nodes)
        self.pid = os.getpid()
        self.chunk = 0
        self._reset()
        self.lock = threading.Lock()

    def _reset(self):
        self.steps = []

    def record(self, *, req_pool_indices, prefix_lens, scores, depth, parents, path, accept_len, n_budget=None):
        bs = int(scores.shape[0])
        cols = [
            req_pool_indices.to(torch.float64).view(bs, 1),
            prefix_lens.to(torch.float64).view(bs, 1),
            accept_len.to(torch.float64).view(bs, 1),
            scores.to(torch.float64),
            depth.to(torch.float64),
            parents.to(torch.float64),
            path.to(torch.float64),
        ]
        if n_budget is not None:
            cols.append(n_budget.to(torch.float64).view(bs, 1))
        else:
            cols.append(torch.full((bs, 1), float(self.N), dtype=torch.float64, device=scores.device))
        host = torch.cat(cols, dim=1).cpu().numpy()
        self.steps.append((time.time(), host))
        if len(self.steps) >= self.CHUNK:
            self.flush()

    def flush(self):
        import numpy as np

        if not self.steps:
            return
        steps, self.steps = self.steps, []
        N = self.N
        bs = np.array([h.shape[0] for _, h in steps], dtype=np.int32)
        rows = np.concatenate([h for _, h in steps], axis=0)
        row0 = np.concatenate([[0], np.cumsum(bs)[:-1]]).astype(np.int64)
        c = 3
        out = dict(
            step_bs=bs,
            step_t=np.array([t for t, _ in steps], dtype=np.float64),
            step_row0=row0,
            row_step=np.repeat(np.arange(len(steps), dtype=np.int32), bs),
            row_req=rows[:, 0].astype(np.int32),
            row_prefix=rows[:, 1].astype(np.int32),
            accept_len=rows[:, 2].astype(np.int8),
            score=rows[:, c : c + N].astype(np.float32),
            depth=rows[:, c + N : c + 2 * N].astype(np.int8),
            parent=rows[:, c + 2 * N : c + 3 * N].astype(np.int8),
            path=rows[:, c + 3 * N : c + 4 * N].astype(np.int8),
            n_budget=rows[:, c + 4 * N].astype(np.int8),
        )
        fn = os.path.join(self.dir, f"tr_{self.pid}_{self.chunk:05d}.npz")
        self.chunk += 1
        np.savez(fn, **out)


_TRACE: Optional[RPTrace] = None


def get_trace(n_nodes: int) -> Optional[RPTrace]:
    global _TRACE
    cfg = rp_config()
    if cfg is None or not cfg.trace_dir:
        return None
    if _TRACE is None:
        _TRACE = RPTrace(cfg.trace_dir, n_nodes)
        import atexit

        atexit.register(_TRACE.flush)
    return _TRACE
