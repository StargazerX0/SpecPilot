"""SPECPILOT_D3_DS_FAST: one cooperative Triton kernel for the wide DSpark Markov lattice (specpilot_d1_dspark
``wide_markov_lattice``, SPECPILOT_D1_DS_EDGES=wide).

Same lattice: shortlist = base top-M of every slot row; slot-s row of beam parent p over the shortlist columns c is
``vals[s, c] + bf16(<W1[p], W2[ids[s, c]]>)``, normalised by logaddexp(lse_c row, tail_s) with tail_s the base mass outside
the shortlist; the slot-s beam is the top-K of max_p (path_p + logp[p, c]) (descending, lower shortlist index first on
ties); edges[s, p, c] = logp[p, beam_c] (slot 0: the anchor row broadcast over p).

Kernel: grid (M / CM, bs); CTA j owns shortlist columns [j*CM, (j+1)*CM).
- phase 0: partial logsumexp of the full logits rows (vocabulary slice j) and of the shortlist values; grid barrier;
  every CTA combines the partials in the same order -> tails.
- per slot: K x CM Markov rows (in-kernel dot), row max / sumexp partials; grid barrier; lse (fixed combine order),
  logp chunk -> scratch, local top-K packed keys; grid barrier; global top-K (identical in every CTA) -> beam; CTA 0
  writes cand / edges. The next slot's shortlist ids / values / W2 rows are loaded before the barriers.
- exact early exit: tau = the K-th best beam path value found so far (a lower bound of the threshold of any best-first
  tree of N <= K + 1 nodes, since beam entries are distinct prefixes); once every entry of a slot's beam is below it no
  deeper node can enter the tree, so the remaining slots get edges -inf and cand = the anchor (never read by the build).
Grid barriers spin on a per-request counter (zeroed in-graph before each launch); all M / CM * bs CTAs must be
co-resident (64 x 4 = 128 CTAs at most here, one per SM).

Results equal the torch builder up to fp32 rounding of the reductions (trees identical except swaps of nodes whose path
scores tie within a few ulp). Knob (read once): SPECPILOT_D3_DS_FAST=1; unset = parent behaviour.
"""
from __future__ import annotations

import functools
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_CM = 64
_NUM_WARPS = 8
_VB = 512


@functools.lru_cache(maxsize=None)
def enabled() -> bool:
    raw = os.environ.get("SPECPILOT_D3_DS_FAST", "").strip().lower()
    if raw in ("", "0", "off", "false"):
        return False
    if raw not in ("1", "on", "true"):
        raise ValueError(f"SPECPILOT_D3_DS_FAST={raw!r}: expected 1 or unset")
    return True


def supported(shortlist: int, beam: int, vocab: int) -> bool:
    m = min(int(shortlist), int(vocab))
    return m >= _CM and m % _CM == 0 and m & (m - 1) == 0 and beam & (beam - 1) == 0 and 16 <= beam <= _CM


@triton.jit
def _pack_desc(work, idx, M: tl.constexpr):
    """int64 key whose descending order is descending value, lower index first on ties (-0.0 taken as +0.0)."""
    w = work + 0.0
    bits = w.to(tl.int32, bitcast=True)
    k32 = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    return (k32.to(tl.int64) << 32) | (M - 1 - idx).to(tl.int64)


@triton.jit
def _unpack(key, M: tl.constexpr):
    k32 = (key >> 32).to(tl.int32)
    bits = k32 ^ ((k32 >> 31) & 0x7FFFFFFF)
    return (M - 1 - (key & 0xFFFFFFFF)).to(tl.int64), bits.to(tl.float32, bitcast=True)


@triton.jit
def _grid_barrier(cnt_ptr, target):
    tl.debug_barrier()
    tl.atomic_add(cnt_ptr, 1, sem="release", scope="gpu")
    while tl.atomic_add(cnt_ptr, 0, sem="acquire", scope="gpu") < target:
        pass
    tl.debug_barrier()


@triton.jit
def _coop_lattice_kernel(
    logits_ptr,  # [bs, S, VC] f32
    vals_ptr,  # [bs, S, M] f32 shortlist logits (descending)
    pos_ptr,  # [bs, S, M] int shortlist positions in the logits row
    map_ptr,  # [VC] int64 ids map (HAS_MAP)
    anchor_ptr,  # [bs] int64
    w1_ptr,  # [V, R] bf16 Markov predecessor table
    w2_ptr,  # [V, R] bf16 Markov successor table
    cand_ptr,  # [bs, S, K] int64 out
    edges_ptr,  # [bs, S, K, K] f32 out
    lsep_ptr,  # [bs, NC, 2, SP] f32 scratch
    headp_ptr,  # [bs, NC, 2, SP] f32 scratch
    part_ptr,  # [bs, NC, 2, K] f32 scratch
    key_ptr,  # [bs, NC, K] int64 scratch
    logp_ptr,  # [bs, K, M] f32 scratch
    cnt_ptr,  # [bs] int32 barrier counters (zero at launch)
    VC,
    S: tl.constexpr, SP: tl.constexpr, M: tl.constexpr, K: tl.constexpr, CM: tl.constexpr, R: tl.constexpr,
    NC: tl.constexpr, VB: tl.constexpr, HAS_MAP: tl.constexpr, EARLY: tl.constexpr,
):
    j = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    kidx = tl.arange(0, K)
    cidx = tl.arange(0, CM)
    ridx = tl.arange(0, R)
    sidx = tl.arange(0, SP)
    qidx = tl.arange(0, NC * K)
    nidx = tl.arange(0, NC)
    cols = j * CM + cidx
    cnt = cnt_ptr + b
    smask = sidx < S
    # ---- phase 0: tails
    span = (VC + NC - 1) // NC
    lo = j * span
    vidx = tl.arange(0, VB)
    m_run = tl.full([SP], float("-inf"), dtype=tl.float32)
    s_run = tl.zeros([SP], dtype=tl.float32)
    for v0 in range(0, span, VB):
        vcol = lo + v0 + vidx
        ok = smask[:, None] & (vcol[None, :] < VC) & ((v0 + vidx)[None, :] < span)
        x = tl.load(logits_ptr + (b * S + sidx[:, None]) * VC + vcol[None, :], mask=ok, other=float("-inf"))
        nm = tl.maximum(m_run, tl.max(x, 1))
        nm_safe = tl.where(nm == float("-inf"), 0.0, nm)
        s_run = s_run * tl.exp(m_run - nm_safe) + tl.sum(tl.exp(x - nm_safe[:, None]), 1)
        m_run = nm
    tl.store(lsep_ptr + ((b * NC + j) * 2) * SP + sidx, m_run)
    tl.store(lsep_ptr + ((b * NC + j) * 2 + 1) * SP + sidx, s_run)
    hv = tl.load(vals_ptr + (b * S + sidx[:, None]) * M + cols[None, :], mask=smask[:, None], other=float("-inf"))
    hm = tl.max(hv, 1)
    hm_safe = tl.where(hm == float("-inf"), 0.0, hm)
    tl.store(headp_ptr + ((b * NC + j) * 2) * SP + sidx, hm)
    tl.store(headp_ptr + ((b * NC + j) * 2 + 1) * SP + sidx, tl.sum(tl.exp(hv - hm_safe[:, None]), 1))
    nb = 1
    _grid_barrier(cnt, nb * NC)
    am = tl.load(lsep_ptr + (b * NC + nidx[:, None]) * 2 * SP + sidx[None, :], cache_modifier=".cg")
    asum = tl.load(lsep_ptr + ((b * NC + nidx[:, None]) * 2 + 1) * SP + sidx[None, :], cache_modifier=".cg")
    gm = tl.max(am, 0)
    gm_safe = tl.where(gm == float("-inf"), 0.0, gm)
    lse_full = gm_safe + tl.log(tl.sum(asum * tl.exp(am - gm_safe[None, :]), 0))
    hm2 = tl.load(headp_ptr + (b * NC + nidx[:, None]) * 2 * SP + sidx[None, :], cache_modifier=".cg")
    hs2 = tl.load(headp_ptr + ((b * NC + nidx[:, None]) * 2 + 1) * SP + sidx[None, :], cache_modifier=".cg")
    hg = tl.max(hm2, 0)
    hg_safe = tl.where(hg == float("-inf"), 0.0, hg)
    head = hg_safe + tl.log(tl.sum(hs2 * tl.exp(hm2 - hg_safe[None, :]), 0))
    tails = lse_full + tl.log(tl.maximum(1.0 - tl.exp(head - lse_full), 1e-7))  # [SP]

    # ---- slots
    anchor = tl.load(anchor_ptr + b)
    ptok = tl.zeros([K], dtype=tl.int64) + anchor
    path = tl.where(kidx == 0, 0.0, float("-inf"))
    found = tl.full([K], float("-inf"), dtype=tl.float32)
    p0 = tl.load(pos_ptr + (b * S) * M + cols).to(tl.int64)
    if HAS_MAP:
        cid = tl.load(map_ptr + p0).to(tl.int64)
    else:
        cid = p0
    vv = tl.load(vals_ptr + (b * S) * M + cols)
    keys = tl.load(w2_ptr + cid[:, None] * R + ridx[None, :])
    done = 0
    for s in range(0, S):
        if done == 0:
            pw = tl.load(w1_ptr + ptok[:, None] * R + ridx[None, :])
            bias = tl.dot(pw, tl.trans(keys)).to(tl.bfloat16).to(tl.float32)  # [K, CM]
            rows = vv[None, :] + bias
            pm = tl.max(rows, 1)
            ps = tl.sum(tl.exp(rows - pm[:, None]), 1)
            pp = part_ptr + (b * NC + j) * 2 * K
            tl.store(pp + kidx, pm)
            tl.store(pp + K + kidx, ps)
            sn = tl.minimum(s + 1, S - 1)  # prefetch the next slot
            pn = tl.load(pos_ptr + (b * S + sn) * M + cols).to(tl.int64)
            if HAS_MAP:
                cid = tl.load(map_ptr + pn).to(tl.int64)
            else:
                cid = pn
            vv = tl.load(vals_ptr + (b * S + sn) * M + cols)
            keys = tl.load(w2_ptr + cid[:, None] * R + ridx[None, :])
            nb += 1
            _grid_barrier(cnt, nb * NC)
            allm = tl.load(part_ptr + (b * NC + nidx[:, None]) * 2 * K + kidx[None, :], cache_modifier=".cg")
            alls = tl.load(part_ptr + (b * NC + nidx[:, None]) * 2 * K + K + kidx[None, :], cache_modifier=".cg")
            mx = tl.max(allm, 0)
            lse = mx + tl.log(tl.sum(alls * tl.exp(allm - mx[None, :]), 0))
            ts = tl.sum(tl.where(sidx == s, tails, 0.0), 0)
            hi = tl.maximum(lse, ts)
            z = hi + tl.log(tl.exp(lse - hi) + tl.exp(ts - hi))
            logp = rows - z[:, None]
            tl.store(logp_ptr + b * K * M + kidx[:, None] * M + cols[None, :], logp)
            best = tl.max(path[:, None] + logp, 0)
            tl.store(key_ptr + (b * NC + j) * K + kidx, tl.topk(_pack_desc(best, cols, M), K))
            nb += 1
            _grid_barrier(cnt, nb * NC)
            top = tl.topk(tl.load(key_ptr + b * NC * K + qidx, cache_modifier=".cg"), K)
            nbeam, npath = _unpack(top, M)
            npos = tl.load(pos_ptr + (b * S + s) * M + nbeam).to(tl.int64)
            if HAS_MAP:
                ntok = tl.load(map_ptr + npos).to(tl.int64)
            else:
                ntok = npos
            if j == 0:
                tl.store(cand_ptr + (b * S + s) * K + kidx, ntok)
                if s == 0:
                    e = tl.load(logp_ptr + b * K * M + nbeam[None, :] + kidx[:, None] * 0, cache_modifier=".cg")
                else:
                    e = tl.load(logp_ptr + b * K * M + kidx[:, None] * M + nbeam[None, :], cache_modifier=".cg")
                tl.store(edges_ptr + ((b * S + s) * K) * K + kidx[:, None] * K + kidx[None, :], e)
            ptok = ntok
            path = npath
            if EARLY:
                found = tl.topk(tl.reshape(tl.join(found, npath), [2 * K]), K)
                tau = tl.min(found, 0)
                if s + 1 < S:
                    if tl.max(npath, 0) < tau:
                        done = 1
        else:
            if j == 0:
                tl.store(cand_ptr + (b * S + s) * K + kidx, tl.zeros([K], dtype=tl.int64) + anchor)
                tl.store(edges_ptr + ((b * S + s) * K) * K + kidx[:, None] * K + kidx[None, :],
                         tl.full([K, K], float("-inf"), dtype=tl.float32))


class _Scratch:
    def __init__(self, bs: int, S: int, M: int, K: int, device) -> None:
        nc, sp = M // _CM, triton.next_power_of_2(S)
        f32 = dict(dtype=torch.float32, device=device)
        self.cap = bs
        self.lsep = torch.empty((bs, nc, 2, sp), **f32)
        self.headp = torch.empty((bs, nc, 2, sp), **f32)
        self.part = torch.empty((bs, nc, 2, K), **f32)
        self.keys = torch.empty((bs, nc, K), dtype=torch.int64, device=device)
        self.logp = torch.empty((bs, K, M), **f32)
        self.cnt = torch.zeros((bs,), dtype=torch.int32, device=device)


# every scratch ever handed to a captured graph stays alive (keyed by device / shape, grown by capacity)
_SCRATCH: dict = {}
_ALL_SCRATCH: list = []


def _scratch(bs: int, S: int, M: int, K: int, device) -> _Scratch:
    key = (str(device), S, M, K)
    sc = _SCRATCH.get(key)
    if sc is None or sc.cap < bs:
        sc = _Scratch(max(bs, 4), S, M, K, device)
        _SCRATCH[key] = sc
        _ALL_SCRATCH.append(sc)
    return sc


def wide_markov_lattice(
    *,
    logits: torch.Tensor,
    ids_map: Optional[torch.Tensor],
    anchor_token_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    shortlist: int,
    beam: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for specpilot_d1_dspark.wide_markov_lattice on CUDA (no host sync, graph-capturable)."""
    from sglang.srt.models.dflash import _radix_topk

    bs, S, VC = logits.shape
    M, K = min(int(shortlist), VC), int(beam)
    logits = logits.float().contiguous()
    vals, pos = _radix_topk(logits.view(bs * S, VC), M)
    vals = vals.float().contiguous()
    pos = pos.contiguous()
    sc = _scratch(bs, S, M, K, logits.device)
    cand = torch.empty((bs, S, K), dtype=torch.int64, device=logits.device)
    edges = torch.empty((bs, S, K, K), dtype=torch.float32, device=logits.device)
    sc.cnt[:bs].zero_()
    nc = M // _CM
    _coop_lattice_kernel[(nc, bs)](
        logits, vals, pos, ids_map if ids_map is not None else pos, anchor_token_ids.reshape(-1).long().contiguous(),
        w1.contiguous(), w2.contiguous(), cand, edges, sc.lsep, sc.headp, sc.part, sc.keys, sc.logp, sc.cnt, VC,
        S=S, SP=triton.next_power_of_2(S), M=M, K=K, CM=_CM, R=w1.shape[1], NC=nc, VB=_VB,
        HAS_MAP=ids_map is not None, EARLY=True, num_warps=_NUM_WARPS,
    )
    return cand, edges
