"""SPECPILOT_TREE: draft trees on the DFlash2 selector lattice (TREE family).

A tree replaces the single selector walk with N - 1 = block_size - 1 draft nodes chosen best-first
from the drafter's per-slot top-K candidates, scored by the selector's first-order conditional
``log_softmax(score[b, slot, parent_cand, :] / T)`` (path-conditional DDTree). The verify width
stays N = block_size, so every block-sized buffer of the engine (KV reservation, verify CUDA graph,
GDN intermediate snapshots) is reused unchanged.

Layout per request (branch_oracle convention): node 0 = root = the previous bonus at ``seq_len``;
``parents[0] = -1``; parents precede children; node position = seq_len + depth. Siblings carry
distinct tokens (distinct candidates of one slot).

Knobs (read once, at import of the first user; every knob unset = stock chain behaviour):
  SPECPILOT_TREE_MODE   ''/'off' (stock) | 'ddtree' (best-first conditional tree)
  SPECPILOT_TREE_TEMP   edge softmax temperature (default 1.0)
  SPECPILOT_D3M_CALIB   1 = calibrated edge scores (1 - B) log p + B log q[slot, rank, bin] from the constant
                        table in specpilot_d3m_calib_table.py, without per-row renormalisation (d3-003-calib)

Accept/commit interface (kept independent of the acceptance rule so a relaxed rule can plug in):
``tree_accept_commit`` takes per-node target predictions ``pred[b, i]`` (the token the target emits
after node i) and a per-node "child token matches" rule (greedy: ``tok[c] == pred[parent(c)]``);
it returns the accepted path (node index per depth), commit length, bonus, out tokens and permutes
the reserved ``req_to_token`` block so the accepted path occupies positions ``seq_len + d``.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

PAD = -2


@dataclass(frozen=True)
class TreeConfig:
    mode: str
    temp: float
    # optional per-depth temperatures (depth 1, 2, ...; the last value repeats)
    temps: Optional[tuple] = None

    def inv_temps(self, n_slots: int, device) -> torch.Tensor:
        t = list(self.temps) if self.temps else [self.temp]
        t = (t + [t[-1]] * n_slots)[:n_slots]
        return torch.tensor([1.0 / x for x in t], dtype=torch.float32, device=device)


@functools.lru_cache(maxsize=None)
def tree_config() -> Optional[TreeConfig]:
    mode = os.environ.get("SPECPILOT_TREE_MODE", "").strip().lower()
    if mode in ("", "0", "off", "none", "chain"):
        return None
    if mode not in ("ddtree",):
        raise ValueError(f"SPECPILOT_TREE_MODE={mode!r}: expected 'ddtree' or unset")
    temp = float(os.environ.get("SPECPILOT_TREE_TEMP", "1.0"))
    if not temp > 0:
        raise ValueError(f"SPECPILOT_TREE_TEMP must be > 0, got {temp}")
    raw = os.environ.get("SPECPILOT_TREE_TEMPS", "").strip()
    temps = tuple(float(x) for x in raw.replace(":", ",").split(",") if x.strip()) if raw else None
    if temps is not None and not all(t > 0 for t in temps):
        raise ValueError(f"SPECPILOT_TREE_TEMPS must be > 0, got {raw!r}")
    return TreeConfig(mode=mode, temp=temp, temps=temps)


def tree_enabled() -> bool:
    return tree_config() is not None


@functools.lru_cache(maxsize=None)
def gdn_paths_enabled() -> bool:
    """SPECPILOT_TREE_GDN=paths: snapshot-free path-parallel GDN tree verify + replay commit."""
    mode = os.environ.get("SPECPILOT_TREE_GDN", "").strip().lower()
    if mode in ("", "0", "off", "snapshots"):
        return False
    if mode != "paths":
        raise ValueError(f"SPECPILOT_TREE_GDN={mode!r}: expected 'paths' or unset")
    if not tree_enabled():
        raise ValueError("SPECPILOT_TREE_GDN=paths needs SPECPILOT_TREE_MODE")
    return True


@functools.lru_cache(maxsize=None)
def x4_draft_block() -> int:
    """SPECPILOT_X4_DRAFT_BLOCK=B: the tree is built over the lattice of the first B drafter rows (B - 1 slots)
    while the verify width stays the engine block N >= B; 0/unset = the whole block (parent behaviour)."""
    raw = os.environ.get("SPECPILOT_X4_DRAFT_BLOCK", "").strip()
    if not raw:
        return 0
    if not tree_enabled():
        raise ValueError("SPECPILOT_X4_DRAFT_BLOCK needs SPECPILOT_TREE_MODE")
    return int(raw)


def x4_lattice_rows(block_size: int) -> int:
    """Number of drafter rows (anchor included) feeding the lattice for an engine block of ``block_size``."""
    b = x4_draft_block()
    return min(b, int(block_size)) if b > 0 else int(block_size)


@functools.lru_cache(maxsize=None)
def d3m_calib_enabled() -> bool:
    """SPECPILOT_D3M_CALIB=1: calibrated edge scores for the best-first builder."""
    raw = os.environ.get("SPECPILOT_D3M_CALIB", "").strip().lower()
    if raw in ("", "0", "off", "false"):
        return False
    if raw not in ("1", "on", "true"):
        raise ValueError(f"SPECPILOT_D3M_CALIB={raw!r}: expected 1 or unset")
    if not tree_enabled():
        raise ValueError("SPECPILOT_D3M_CALIB needs SPECPILOT_TREE_MODE")
    return True


_D3M_TABLES: dict = {}


@functools.lru_cache(maxsize=None)
def _d3m_tab():
    """The constant table module (loaded from this file's directory, so the file-path test import works too)."""
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specpilot_d3m_calib_table.py")
    spec = importlib.util.spec_from_file_location("specpilot_d3m_calib_table", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def d3m_calib_table(device) -> torch.Tensor:
    """Device copy of LOG_Q (flattened) followed by the bin edges; created once per device, before graph capture
    (the worker pre-warms it in __init__)."""
    key = str(torch.device(device))
    t = _D3M_TABLES.get(key)
    if t is None:
        tab = _d3m_tab()
        vals = list(tab.LOG_Q) + list(tab.BIN_EDGES)
        assert len(tab.LOG_Q) == tab.N_SLOT * tab.N_RANK * tab.N_BIN
        t = torch.tensor(vals, dtype=torch.float32, device=device)
        _D3M_TABLES[key] = t
    return t


@triton.jit
def _d3m_calib_kernel(
    scores_ptr,  # [rows, K] float (contiguous; rows = bs * S * K, slot = (row // K) % S)
    out_ptr,  # [rows, K] float32 calibrated edge log-scores
    inv_temp_ptr,  # [S] float32
    table_ptr,  # [N_SLOT * N_RANK * N_BIN + N_EDGE] float32: log q, then the bin edges
    S: tl.constexpr,
    K: tl.constexpr,
    N_SLOT: tl.constexpr,
    N_RANK: tl.constexpr,
    N_BIN: tl.constexpr,
    N_EDGE: tl.constexpr,
    BLEND: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    s = (r // K) % S
    kk = tl.arange(0, K)
    row = tl.load(scores_ptr + r * K + kk).to(tl.float32) * tl.load(inv_temp_ptr + s)
    m = tl.max(row, 0)
    e = tl.exp(row - m)
    p = e / tl.sum(e, 0)
    pj = p[None, :]
    pc = p[:, None]
    ahead = (pj > pc) | ((pj == pc) & (kk[None, :] < kk[:, None]))
    rank = tl.sum(ahead.to(tl.int32), 1)
    rb = tl.where(rank < 5, rank, tl.where(rank < 8, 5, 6))
    pb = tl.zeros([K], dtype=tl.int32)
    for i in tl.static_range(N_EDGE):
        edge = tl.load(table_ptr + N_SLOT * N_RANK * N_BIN + i)
        pb += (p >= edge).to(tl.int32)
    sl = tl.minimum(s, N_SLOT - 1)
    lq = tl.load(table_ptr + sl * (N_RANK * N_BIN) + rb * N_BIN + pb)
    out = (1.0 - BLEND) * tl.log(tl.maximum(p, 1e-12)) + BLEND * lq
    tl.store(out_ptr + r * K + kk, out)


def d3m_calibrate(scores: torch.Tensor, inv_temps: torch.Tensor) -> torch.Tensor:
    """[bs, S, K, K] lattice scores -> calibrated edge log-scores (same shape, float32)."""
    tab = _d3m_tab()
    bs, S, K, _ = scores.shape
    scores = scores.contiguous()
    out = torch.empty((bs, S, K, K), dtype=torch.float32, device=scores.device)
    _d3m_calib_kernel[(bs * S * K,)](
        scores,
        out,
        inv_temps,
        d3m_calib_table(scores.device),
        S=S,
        K=K,
        N_SLOT=tab.N_SLOT,
        N_RANK=tab.N_RANK,
        N_BIN=tab.N_BIN,
        N_EDGE=len(tab.BIN_EDGES),
        BLEND=tab.BLEND,
    )
    return out


def _np2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


# ------------------------------------------------------------------------------------------------
# tree construction (in-graph: static shapes, no host sync)
# ------------------------------------------------------------------------------------------------


@triton.jit
def _tree_build_kernel(
    scores_ptr,  # [bs, S, K, K] float (contiguous)
    cand_ptr,  # [bs, S, K] int64 (contiguous)
    anchor_ptr,  # [bs] int64
    tokens_out,  # [bs, N] int64
    parents_out,  # [bs, N] int64
    depth_out,  # [bs, N] int64
    next_tok_out,  # [bs, N] int32
    next_sib_out,  # [bs, N] int32
    mask_out,  # [bs, N, N] bool (as int8)
    inv_temp_ptr,  # [S] float32: 1 / temperature of the children at slot s
    S: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    NP: tl.constexpr,
    NORM: tl.constexpr,  # False: scores are ready edge log-scores (SPECPILOT_D3M_CALIB), no temperature / lse
):
    b = tl.program_id(0).to(tl.int64)
    P: tl.constexpr = NP * K
    pidx = tl.arange(0, P)
    kidx = tl.arange(0, K)
    nidx = tl.arange(0, NP)
    s_base = scores_ptr + b * (S * K * K)
    c_base = cand_ptr + b * (S * K)

    # node arrays (registers)
    n_tok = tl.zeros([NP], dtype=tl.int64)
    n_par = tl.full([NP], -1, dtype=tl.int64)
    n_dep = tl.zeros([NP], dtype=tl.int64)
    n_cid = tl.zeros([NP], dtype=tl.int64)
    n_sc = tl.zeros([NP], dtype=tl.float32)
    n_anc = tl.where(nidx == 0, 1, 0).to(tl.int64)
    anchor = tl.load(anchor_ptr + b).to(tl.int64)
    n_tok = tl.where(nidx == 0, anchor, n_tok)

    # pool entry e = parent_node * K + child candidate index; root children: slot 0, row 0
    if NORM:
        inv_temp = tl.load(inv_temp_ptr)
        row = tl.load(s_base + kidx).to(tl.float32) * inv_temp
        m = tl.max(row, 0)
        lse = m + tl.log(tl.sum(tl.exp(row - m), 0))
    else:
        inv_temp = 1.0
        lse = 0.0
    vals = tl.load(s_base + (pidx % K), mask=pidx < K, other=0.0).to(tl.float32)
    pool = tl.where(pidx < K, vals * inv_temp - lse, float("-inf"))

    for n in tl.static_range(1, N):
        best = tl.max(pool, 0)
        e = tl.min(tl.where(pool == best, pidx, P), 0)
        e = tl.minimum(e, P - 1)
        p = (e // K).to(tl.int64)
        c = (e % K).to(tl.int64)
        p_dep = tl.sum(tl.where(nidx == p, n_dep, 0), 0)
        p_anc = tl.sum(tl.where(nidx == p, n_anc, 0), 0)
        d = p_dep + 1
        tok = tl.load(c_base + (d - 1) * K + c)
        n_tok = tl.where(nidx == n, tok, n_tok)
        n_par = tl.where(nidx == n, p, n_par)
        n_dep = tl.where(nidx == n, d, n_dep)
        n_cid = tl.where(nidx == n, c, n_cid)
        n_sc = tl.where(nidx == n, best, n_sc)
        n_anc = tl.where(nidx == n, p_anc | (1 << n), n_anc)
        pool = tl.where(pidx == e, float("-inf"), pool)
        # children of the new node live at slot d (depth d + 1), row = its candidate index
        has_kids = d < S
        r_base = s_base + tl.minimum(d, S - 1) * (K * K) + c * K
        if NORM:
            inv_temp = tl.load(inv_temp_ptr + tl.minimum(d, S - 1))
            row = tl.load(r_base + kidx).to(tl.float32) * inv_temp
            m = tl.max(row, 0)
            lse = m + tl.log(tl.sum(tl.exp(row - m), 0))
        seg = (pidx // K) == n
        vals = tl.load(r_base + (pidx % K), mask=seg, other=0.0).to(tl.float32)
        pool = tl.where(seg & has_kids, best + vals * inv_temp - lse, pool)

    out_mask = nidx < N
    o_base = b * N
    tl.store(tokens_out + o_base + nidx, n_tok, mask=out_mask)
    tl.store(parents_out + o_base + nidx, n_par, mask=out_mask)
    tl.store(depth_out + o_base + nidx, n_dep, mask=out_mask)
    # first child / next sibling (ascending node order)
    jj = nidx[None, :]
    ii = nidx[:, None]
    valid_j = jj < N
    is_child = (n_par[None, :] == ii) & valid_j & (jj > 0)
    first = tl.min(tl.where(is_child, jj, NP), 1)
    first = tl.where(first >= NP, -1, first)
    same = (n_par[None, :] == n_par[:, None]) & (n_par[:, None] >= 0) & (jj > ii) & valid_j
    nsib = tl.min(tl.where(same, jj, NP), 1)
    nsib = tl.where(nsib >= NP, -1, nsib)
    tl.store(next_tok_out + o_base + nidx, first.to(tl.int32), mask=out_mask)
    tl.store(next_sib_out + o_base + nidx, nsib.to(tl.int32), mask=out_mask)
    anc_bits = (n_anc[:, None] >> jj.to(tl.int64)) & 1
    m_ptr = mask_out + b * (N * N) + ii * N + jj
    tl.store(m_ptr, anc_bits.to(tl.int8), mask=(ii < N) & valid_j)


def build_tree(
    scores: torch.Tensor,
    cand_ids: torch.Tensor,
    anchor: torch.Tensor,
    *,
    n_nodes: int,
    temp,
    out: dict,
) -> None:
    """Best-first conditional tree, written into the static buffers of ``out`` (first bs rows):
    tokens/parents/depth [bs, N] int64, next_token/next_sibling [bs, N] int32, mask [bs, N, N] bool."""
    bs, S, K = cand_ids.shape
    assert scores.shape == (bs, S, K, K), (scores.shape, cand_ids.shape)
    assert K & (K - 1) == 0, "K must be a power of two"
    N = int(n_nodes)
    assert N <= 63, "ancestor bitmask holds at most 63 nodes"
    scores = scores.contiguous()
    cand_ids = cand_ids.contiguous()
    if cand_ids.dtype != torch.int64:
        cand_ids = cand_ids.long()
    inv_temps = _inv_temps(temp, S, scores.device)
    norm = not d3m_calib_enabled()
    if not norm:  # SPECPILOT_D3M_CALIB
        scores = d3m_calibrate(scores, inv_temps)
    _tree_build_kernel[(bs,)](
        scores,
        cand_ids,
        anchor.reshape(-1).long(),
        out["tokens"],
        out["parents"],
        out["depth"],
        out["next_token"],
        out["next_sibling"],
        out["mask"].view(torch.int8),
        inv_temps,
        S=S,
        K=K,
        N=N,
        NP=_np2(N),
        NORM=norm,
    )


def _inv_temps(temp, S: int, device) -> torch.Tensor:
    """``temp``: a float (one temperature) or a ready [>= S] float32 tensor of inverse temperatures."""
    if torch.is_tensor(temp):
        assert temp.numel() >= S and temp.dtype == torch.float32
        return temp
    return torch.full((S,), 1.0 / float(temp), dtype=torch.float32, device=device)


def alloc_tree_buffers(max_bs: int, n_nodes: int, device) -> dict:
    N = int(n_nodes)
    return {
        "tokens": torch.zeros((max_bs, N), dtype=torch.int64, device=device),
        "parents": torch.full((max_bs, N), -1, dtype=torch.int64, device=device),
        "depth": torch.zeros((max_bs, N), dtype=torch.int64, device=device),
        "next_token": torch.full((max_bs, N), -1, dtype=torch.int32, device=device),
        "next_sibling": torch.full((max_bs, N), -1, dtype=torch.int32, device=device),
        "mask": torch.zeros((max_bs, N, N), dtype=torch.bool, device=device),
    }


# ------------------------------------------------------------------------------------------------
# greedy tree accept + commit (eager, after verify; no host sync)
# ------------------------------------------------------------------------------------------------


@triton.jit
def _tree_accept_kernel(
    tokens_ptr,  # [bs, N] int64, node 0 = root
    parents_ptr,  # [bs, N] int64
    pred_ptr,  # [bs, N] int64 target argmax per node
    prefix_ptr,  # [bs] committed length (root position)
    req_pool_ptr,  # [bs]
    req_to_token_ptr,  # [pool, ctx] int32
    stride_rtt,
    accept_len_out,  # [bs] int32
    commit_out,  # [bs] int32
    bonus_out,  # [bs] int64
    out_tokens_out,  # [bs, N] int64
    new_seq_out,  # [bs] int64
    leaf_out,  # [bs] int64
    path_out,  # [bs, N] int64, node index per depth, -1 past the leaf
    slots_out,  # [bs, N] int64, req_to_token[seq_len + d] after the permutation
    N: tl.constexpr,
    NP: tl.constexpr,
    PERMUTE: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    nidx = tl.arange(0, NP)
    live = nidx < N
    base = b * N
    tok = tl.load(tokens_ptr + base + nidx, mask=live, other=-1)
    par = tl.load(parents_ptr + base + nidx, mask=live, other=-2)
    pred = tl.load(pred_ptr + base + nidx, mask=live, other=-1)
    ii = nidx[:, None]
    jj = nidx[None, :]
    is_par = par[:, None] == jj  # [i, j]: j is the parent of i
    ppred = tl.sum(tl.where(is_par, pred[None, :], 0), 1)
    ok = live & (par >= 0) & (tok == ppred)
    # accepted = every edge on the root path matches its parent's prediction (longest path wins)
    acc = nidx == 0
    dep = tl.zeros([NP], dtype=tl.int64)
    for _ in tl.static_range(N - 1):
        acc_par = tl.sum(tl.where(is_par, acc.to(tl.int32)[None, :], 0), 1) > 0
        acc = (nidx == 0) | (ok & acc_par)
        dep_par = tl.sum(tl.where(is_par, dep[None, :], 0), 1)
        dep = tl.where(par >= 0, dep_par + 1, 0)
    key = tl.where(acc, dep * (NP + 1) + (NP - nidx), -1)
    best = tl.max(key, 0)
    leaf = tl.min(tl.where(key == best, nidx, NP), 0).to(tl.int64)
    a = tl.sum(tl.where(nidx == leaf, dep, 0), 0)
    # path[d] = ancestor of the leaf at depth d
    path = tl.full([NP], -1, dtype=tl.int64)
    cur = leaf
    for _ in tl.static_range(N):
        dcur = tl.sum(tl.where(nidx == cur, dep, 0), 0)
        path = tl.where((nidx == dcur) & (cur >= 0), cur, path)
        nxt = tl.sum(tl.where(nidx == cur, par, 0), 0)
        cur = tl.where(cur > 0, nxt, -1)
    path_next = tl.sum(tl.where(jj == ii + 1, path[None, :], 0), 1)
    tok_next = tl.sum(tl.where(jj == path_next[:, None], tok[None, :], 0), 1)
    bonus = tl.sum(tl.where(nidx == leaf, pred, 0), 0)
    out = tl.where(nidx < a, tok_next, tl.where(nidx == a, bonus, 0))
    cur = leaf

    prefix = tl.load(prefix_ptr + b).to(tl.int64)
    tl.store(accept_len_out + b, a.to(tl.int32))
    tl.store(commit_out + b, (a + 1).to(tl.int32))
    tl.store(bonus_out + b, bonus)
    tl.store(new_seq_out + b, prefix + a + 1)
    tl.store(leaf_out + b, cur)
    tl.store(out_tokens_out + base + nidx, out, mask=live)
    tl.store(path_out + base + nidx, path, mask=live)

    r = tl.load(req_pool_ptr + b).to(tl.int64)
    row = req_to_token_ptr + r * stride_rtt + prefix
    slots = tl.load(row + nidx, mask=live, other=0).to(tl.int64)
    if PERMUTE:
        for d in tl.static_range(1, N):
            nd = tl.sum(tl.where(nidx == d, path, 0), 0)
            do = (d <= a) & (nd != d) & (nd > 0)
            vd = tl.sum(tl.where(nidx == d, slots, 0), 0)
            vn = tl.sum(tl.where(nidx == nd, slots, 0), 0)
            slots = tl.where(do & (nidx == d), vn, tl.where(do & (nidx == nd), vd, slots))
        tl.store(row + nidx, slots.to(tl.int32), mask=live)
    tl.store(slots_out + base + nidx, slots, mask=live)



def tree_accept_commit(
    *,
    tokens: torch.Tensor,
    parents: torch.Tensor,
    pred: torch.Tensor,
    prefix_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    out: dict,
    permute: bool = True,
) -> None:
    """Greedy longest-path acceptance on the verified tree and zero-copy KV commit.

    Writes into ``out`` (first bs rows): accept_len int32, commit_lens int32, bonus int64,
    out_tokens [bs, N] int64 (accepted drafts then the bonus), new_seq_lens int64, leaf int64
    (accepted node index: the GDN/conv state to commit), path [bs, N] int64 and slots [bs, N]
    int64 (the KV slot of each committed position after the permutation, for the draft KV
    append). With ``permute`` the reserved ``req_to_token[req, seq_len : seq_len + N]`` entries
    are permuted in place so the accepted path's slots sit at ``seq_len + d``.
    """
    bs, N = tokens.shape
    assert req_to_token.dtype == torch.int32
    _tree_accept_kernel[(bs,)](
        tokens,
        parents,
        pred,
        prefix_lens,
        req_pool_indices,
        req_to_token,
        req_to_token.stride(0),
        out["accept_len"],
        out["commit_lens"],
        out["bonus"],
        out["out_tokens"],
        out["new_seq_lens"],
        out["leaf"],
        out["path"],
        out["slots"],
        N=N,
        NP=_np2(N),
        PERMUTE=permute,
    )


def alloc_accept_buffers(max_bs: int, n_nodes: int, device) -> dict:
    N = int(n_nodes)
    return {
        "accept_len": torch.zeros((max_bs,), dtype=torch.int32, device=device),
        "commit_lens": torch.zeros((max_bs,), dtype=torch.int32, device=device),
        "bonus": torch.zeros((max_bs,), dtype=torch.int64, device=device),
        "out_tokens": torch.zeros((max_bs, N), dtype=torch.int64, device=device),
        "new_seq_lens": torch.zeros((max_bs,), dtype=torch.int64, device=device),
        "leaf": torch.zeros((max_bs,), dtype=torch.int64, device=device),
        "path": torch.full((max_bs, N), -1, dtype=torch.int64, device=device),
        "slots": torch.zeros((max_bs, N), dtype=torch.int64, device=device),
    }


def slice_buffers(bufs: dict, bs: int) -> dict:
    return {k: v[:bs] for k, v in bufs.items()}


def tree_mask_rows(tree_mask: torch.Tensor, bs: int) -> torch.Tensor:
    """[b, N, N] ancestor mask -> [bs * N, N] rows for the FA3 cascade expand page table; graph
    padding rows (bs > b) see only themselves."""
    b, N, _ = tree_mask.shape
    m = tree_mask.reshape(b * N, N)
    if bs > b:
        eye = torch.eye(N, dtype=torch.bool, device=m.device).repeat(bs - b, 1)
        m = torch.cat([m, eye], 0)
    return m


# ------------------------------------------------------------------------------------------------
# SPECPILOT_D1: tree mask for the triton target-verify custom-mask path (sm_120, no FA3)
# ------------------------------------------------------------------------------------------------


@triton.jit
def _triton_verify_mask_kernel(
    mask_ptr,  # uint8 [*], layout per request: N rows of (seq_len + N) columns at mask_indptr[b]
    tree_ptr,  # uint8 [b_tree, N, N] ancestor-or-self mask (row = query node, col = key node)
    mask_indptr_ptr,
    seq_lens_ptr,
    b_tree,
    N: tl.constexpr,
    NP: tl.constexpr,
    HAS_TREE: tl.constexpr,
    X4_BLOCK: tl.constexpr,  # > 0: drafter key block (every row keys the first X4_BLOCK block rows)
):
    pid = tl.program_id(0)
    start = tl.load(mask_indptr_ptr + pid).to(tl.int64)
    seq = tl.load(seq_lens_ptr + pid).to(tl.int64)
    i = tl.arange(0, NP)[:, None]
    j = tl.arange(0, NP)[None, :]
    valid = (i < N) & (j < N)
    causal = (j <= i).to(tl.uint8)
    if X4_BLOCK > 0:
        m = (j < X4_BLOCK).to(tl.uint8) + (i * 0).to(tl.uint8)
    elif HAS_TREE:
        m = tl.load(
            tree_ptr + pid * N * N + i * N + j, mask=valid & (pid < b_tree), other=0
        ).to(tl.uint8)
        m = tl.where(pid < b_tree, m, causal)  # graph padding rows: causal
    else:
        m = causal
    # Only the suffix columns [seq, seq + N) are written: the extend kernel runs with
    # skip_prefix_custom_mask=True and never reads the prefix columns.
    tl.store(mask_ptr + start + i * (seq + N) + seq + j, m, mask=valid)


def fill_triton_verify_mask(
    mask: torch.Tensor,
    tree_mask: Optional[torch.Tensor],
    mask_indptr: torch.Tensor,
    seq_lens: torch.Tensor,
    bs: int,
    n_nodes: int,
    x4_block: int = 0,
) -> None:
    """Scatter DFlash's [b, N, N] tree mask into the triton backend's custom-mask buffer (suffix
    block of each request's N x (seq_len + N) rows). GPU only, no host sync; ``tree_mask=None`` (graph
    capture) writes the causal block. Rows of padded requests (b <= r < bs) get the causal block.
    ``x4_block`` = B > 0 (SPECPILOT_X4 draft side): every row keys block columns < B (exact block-B
    non-causal drafter inside a block-N forward); the tree mask is ignored."""
    if bs == 0:
        return
    N = int(n_nodes)
    if x4_block:
        tree_mask = None
    if tree_mask is not None:
        assert tree_mask.shape[1] == N and tree_mask.shape[2] == N, tree_mask.shape
        tm = tree_mask.contiguous()
        if tm.dtype == torch.bool:
            tm = tm.view(torch.uint8)
        b_tree = int(tm.shape[0])
    else:
        tm = mask  # dummy pointer
        b_tree = 0
    _triton_verify_mask_kernel[(bs,)](
        mask,
        tm,
        mask_indptr,
        seq_lens,
        b_tree,
        N=N,
        NP=_np2(N),
        HAS_TREE=tree_mask is not None,
        X4_BLOCK=int(x4_block),
        num_warps=1 if N <= 16 else 4,
    )
