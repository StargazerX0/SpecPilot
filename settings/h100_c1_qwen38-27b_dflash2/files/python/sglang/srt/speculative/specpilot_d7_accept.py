"""SPECPILOT_D7: faster greedy tree accept (same outputs as specpilot_tree._tree_accept_kernel, bit for bit).

The parent kernel resolves depth / path acceptance with N-1 sequential parent gathers, walks the accepted path with
N scalar steps and permutes the KV slots with N-1 static iterations (~73 us per step at N=32, one program per
request, 4 warps: every step is a block-wide reduction with barriers). Here: pointer jumping (ceil(log2 N) rounds of
simultaneous gathers give depth, all-edges-match and the ancestor bitmask), the path is read from the leaf's ancestor
bitmask in one reduction, the permutation loop stops at the accept length (its iterations past it are no-ops in the
parent), 8 warps per request (measured 9 us vs 35 us for the parent kernel, dev loop N=32). Knob SPECPILOT_D7_ACCEPT=1 (read
once at worker init); unset = the parent kernel.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# SPECPILOT_INT_D3W32=1 (int-012): n <= 32 trees also take specpilot_d3w_gdn's register-gather accept (same outputs)
_INT_D3W32 = os.environ.get("SPECPILOT_INT_D3W32", "0") not in ("", "0")


def read_d7_accept_knob() -> bool:
    return (os.environ.get("SPECPILOT_D7_ACCEPT", "") or "0").strip() not in ("", "0")


@triton.jit
def _gather(v, idx, jj):
    # v[idx[i]] for 0 <= idx[i] < NP, else 0
    return tl.sum(tl.where(idx[:, None] == jj, v[None, :], 0), 1)


@triton.jit
def _d7_tree_accept_kernel(
    tokens_ptr, parents_ptr, pred_ptr, prefix_ptr, req_pool_ptr, req_to_token_ptr, stride_rtt,
    accept_len_out, commit_out, bonus_out, out_tokens_out, new_seq_out, leaf_out, path_out, slots_out,
    N: tl.constexpr, NP: tl.constexpr, LOGN: tl.constexpr, PERMUTE: tl.constexpr,
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
    ppred = _gather(pred, par, jj)
    ok = live & (par >= 0) & (tok == ppred)
    # pointer jumping: dep = edges to the root, okp = every edge on the root path matches, anc = ancestor bitmask
    jmp = par
    dep = tl.where(par >= 0, 1, 0).to(tl.int64)
    okp = ((nidx == 0) | ok).to(tl.int32)
    anc = tl.where(live, tl.full([NP], 1, tl.int64) << nidx.to(tl.int64), 0)
    for _ in tl.static_range(LOGN):
        has = jmp >= 0
        dep = tl.where(has, dep + _gather(dep, jmp, jj), dep)
        okp = tl.where(has, okp & _gather(okp, jmp, jj), okp)
        anc = tl.where(has, anc | _gather(anc, jmp, jj), anc)
        jmp = tl.where(has, _gather(jmp, jmp, jj), jmp)
    acc = okp > 0
    key = tl.where(acc, dep * (NP + 1) + (NP - nidx), -1)
    best = tl.max(key, 0)
    leaf = tl.min(tl.where(key == best, nidx, NP), 0).to(tl.int64)
    a = tl.sum(tl.where(nidx == leaf, dep, 0), 0)
    lmask = tl.sum(tl.where(nidx == leaf, anc, 0), 0)
    onpath = ((lmask >> nidx.to(tl.int64)) & 1) > 0
    sel = (dep[None, :] == ii) & onpath[None, :]
    path = tl.where(tl.sum(sel.to(tl.int32), 1) > 0, tl.sum(tl.where(sel, jj, 0), 1).to(tl.int64), -1)
    path_next = tl.sum(tl.where(jj == ii + 1, path[None, :], 0), 1)
    tok_next = tl.sum(tl.where(jj == path_next[:, None], tok[None, :], 0), 1)
    bonus = tl.sum(tl.where(nidx == leaf, pred, 0), 0)
    out = tl.where(nidx < a, tok_next, tl.where(nidx == a, bonus, 0))

    prefix = tl.load(prefix_ptr + b).to(tl.int64)
    tl.store(accept_len_out + b, a.to(tl.int32))
    tl.store(commit_out + b, (a + 1).to(tl.int32))
    tl.store(bonus_out + b, bonus)
    tl.store(new_seq_out + b, prefix + a + 1)
    tl.store(leaf_out + b, leaf)
    tl.store(out_tokens_out + base + nidx, out, mask=live)
    tl.store(path_out + base + nidx, path, mask=live)

    r = tl.load(req_pool_ptr + b).to(tl.int64)
    row = req_to_token_ptr + r * stride_rtt + prefix
    slots = tl.load(row + nidx, mask=live, other=0).to(tl.int64)
    if PERMUTE:
        for d in range(1, a + 1):
            nd = tl.sum(tl.where(nidx == d, path, 0), 0)
            do = (nd != d) & (nd > 0)
            vd = tl.sum(tl.where(nidx == d, slots, 0), 0)
            vn = tl.sum(tl.where(nidx == nd, slots, 0), 0)
            slots = tl.where(do & (nidx == d), vn, tl.where(do & (nidx == nd), vd, slots))
        tl.store(row + nidx, slots.to(tl.int32), mask=live)
    tl.store(slots_out + base + nidx, slots, mask=live)


def d7_tree_accept_commit(*, tokens, parents, pred, prefix_lens, req_pool_indices, req_to_token, out, permute=True):
    """Drop-in for specpilot_tree.tree_accept_commit (same buffers, same outputs)."""
    bs, n = tokens.shape
    assert req_to_token.dtype == torch.int32
    if n > 32 or _INT_D3W32:
        from sglang.srt.speculative import specpilot_d3w_gdn as _d3w

        # SPECPILOT_D3W_GDN (n > 32) / SPECPILOT_INT_D3W32 (n <= 32): register-gather variant (same outputs, ~7 us vs
        # ~50 us at n = 48, 7.9 vs 11.5 us at n = 32)
        if _d3w.enabled() if n > 32 else _d3w.small_enabled():
            return _d3w.d3w_tree_accept_commit(tokens=tokens, parents=parents, pred=pred, prefix_lens=prefix_lens,
                                               req_pool_indices=req_pool_indices, req_to_token=req_to_token,
                                               out=out, permute=permute)
    np2 = triton.next_power_of_2(n)
    assert n <= 63, n  # int64 ancestor bitmask
    _d7_tree_accept_kernel[(bs,)](
        tokens, parents, pred, prefix_lens, req_pool_indices, req_to_token, req_to_token.stride(0),
        out["accept_len"], out["commit_lens"], out["bonus"], out["out_tokens"], out["new_seq_lens"], out["leaf"],
        out["path"], out["slots"], N=n, NP=np2, LOGN=max(1, (n - 1).bit_length()), PERMUTE=permute, num_warps=8)
