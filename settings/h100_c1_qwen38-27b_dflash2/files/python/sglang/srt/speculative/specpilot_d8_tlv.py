"""SPECPILOT_D8R_TLV: two-level verify (lossless, training-free).

One scheduler step = a Q-step + an exact step:
1. Q-step: DFlash draft + DDTree build + tree verify with the target's quantized view Q (the d8q variant of the
   captured target-verify graph; the linear kernels read ``q_active()``), greedy tree accept WITHOUT commit (no
   recurrent-state scatter, no seq-len publish). Q's accepted path (root .. last accepted node) is the chain C and
   Q's bonus the anchor of the next draft; the drafter KV of the chain rows is written from Q's taps.
2. Exact step: a second draft anchored at Q's bonus (position seq_len + |C|), its best-first tree truncated to
   N - |C| nodes and hung below the chain (``compose_chain_tree``); this tree is verified by the exact target in
   ONE pass, accepted and committed as usual. Every emitted token is the exact target's greedy choice: the chain
   only proposes tokens, like any draft.

Q view: q_mode() also enters specpilot_d5_w12.q_pass() (nested W12 planes: the Q pass streams the 1-byte Q regions).

Knobs (read once at import; unset = parent behaviour):
  SPECPILOT_D8R_TLV         1 = two-level verify (greedy tree mode only)
  SPECPILOT_D8R_TLV_QNOISE  test only: probability that a Q prediction is replaced by a different token (exercises
                            chain mismatches while Q runs the exact kernels); 0 default
"""
from __future__ import annotations

import contextlib
import os

import torch


def _flag(name: str) -> bool:
    return (os.environ.get(name, "") or "0").strip() not in ("", "0")


_ENABLED = _flag("SPECPILOT_D8R_TLV")
_QNOISE = float((os.environ.get("SPECPILOT_D8R_TLV_QNOISE", "") or "0").strip() or 0)
_Q = [False]

VARIANT_EXACT = "d8x"
VARIANT_Q = "d8q"


def enabled() -> bool:
    return _ENABLED


def q_noise() -> float:
    return _QNOISE


def q_active() -> bool:
    """True while the Q view of the target runs (graph capture of the d8q variant and its replays)."""
    return _Q[0]


def _d5_q_pass(on: bool):
    """SPECPILOT_D5 nested W12 planes: the Q view of the W12 linears (read at call time / graph capture)."""
    try:
        from sglang.kernels.ops.gemm import specpilot_d5_w12 as _d5
    except ImportError:
        return contextlib.nullcontext()
    q_pass = getattr(_d5, "q_pass", None)
    return q_pass(bool(on)) if q_pass is not None else contextlib.nullcontext()


@contextlib.contextmanager
def q_mode(on: bool = True):
    prev = _Q[0]
    _Q[0] = bool(on)
    try:
        with _d5_q_pass(on):
            yield
    finally:
        _Q[0] = prev


def perturb_pred(pred: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    """Test hook: replace each prediction by pred + 1 with probability q_noise() (no host sync)."""
    if _QNOISE <= 0:
        return pred
    flip = torch.rand(pred.shape, device=pred.device, generator=gen) < _QNOISE
    return torch.where(flip, pred + 1, pred)


def chain_from_path(tokens: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
    """tokens [bs, N] (tree nodes), path [bs, N] node index per depth (-1 past the leaf) -> chain tokens per depth."""
    return torch.gather(tokens, 1, path.clamp(min=0))


def alloc_compose_buffers(max_bs: int, n_nodes: int, device) -> dict:
    N = int(n_nodes)
    return {
        "tokens": torch.zeros((max_bs, N), dtype=torch.int64, device=device),
        "parents": torch.full((max_bs, N), -1, dtype=torch.int64, device=device),
        "depth": torch.zeros((max_bs, N), dtype=torch.int64, device=device),
        "next_token": torch.full((max_bs, N), -1, dtype=torch.int32, device=device),
        "next_sibling": torch.full((max_bs, N), -1, dtype=torch.int32, device=device),
        "mask": torch.zeros((max_bs, N, N), dtype=torch.bool, device=device),
    }


def compose_chain_tree(chain_tokens: torch.Tensor, chain_len: torch.Tensor, sub: dict, out: dict) -> dict:
    """Tree = chain (nodes 0..C-1, node d at depth d; node 0 = the previous bonus) + the first N - C nodes of the
    best-first subtree ``sub`` (its node 0 = Q's bonus, child of chain node C - 1). Same layout as
    specpilot_tree.build_tree (parents precede children, mask[i, j] = j is an ancestor of i or j == i).
    chain_tokens [bs, N], chain_len [bs] (1 <= C <= N - 1), sub/out: tree dicts with the first bs rows used."""
    bs, N = chain_tokens.shape
    dev = chain_tokens.device
    i = torch.arange(N, device=dev).unsqueeze(0)  # [1, N]
    C = chain_len.to(torch.int64).view(bs, 1)
    is_chain = i < C
    s = (i - C).clamp(min=0)
    sub_tok = torch.gather(sub["tokens"][:bs], 1, s)
    sub_par = torch.gather(sub["parents"][:bs], 1, s)
    sub_dep = torch.gather(sub["depth"][:bs], 1, s)
    tok = torch.where(is_chain, chain_tokens, sub_tok)
    par = torch.where(is_chain, i - 1, torch.where(sub_par < 0, C - 1, sub_par + C))
    dep = torch.where(is_chain, i, sub_dep + C)
    # ancestor mask
    sm = torch.gather(sub["mask"][:bs], 1, s.unsqueeze(-1).expand(bs, N, N))  # row s of the sub mask
    sm = torch.gather(sm, 2, s.unsqueeze(1).expand(bs, N, N))  # column s' = j - C
    jj = i.unsqueeze(1)  # [1, 1, N] column index
    ii = i.unsqueeze(2)  # [1, N, 1] row index
    Cc = C.view(bs, 1, 1)
    row_chain = ii < Cc
    col_chain = jj < Cc
    mask = torch.where(row_chain, col_chain & (jj <= ii), col_chain | (sm & ~col_chain))
    # first child / next sibling (ascending node order), as the tree build kernel
    is_child = (par.unsqueeze(1) == ii) & (jj > 0)  # [bs, i, j]: j is a child of i
    big = torch.full_like(is_child, N, dtype=torch.int64)
    first = torch.where(is_child, jj.expand_as(big), big).amin(2)
    first = torch.where(first >= N, -1, first)
    same = (par.unsqueeze(1) == par.unsqueeze(2)) & (par.unsqueeze(2) >= 0) & (jj > ii)
    nsib = torch.where(same, jj.expand_as(big), big).amin(2)
    nsib = torch.where(nsib >= N, -1, nsib)
    out["tokens"][:bs].copy_(tok)
    out["parents"][:bs].copy_(par)
    out["depth"][:bs].copy_(dep)
    out["mask"][:bs].copy_(mask)
    out["next_token"][:bs].copy_(first.to(torch.int32))
    out["next_sibling"][:bs].copy_(nsib.to(torch.int32))
    return {k: v[:bs] for k, v in out.items()}
