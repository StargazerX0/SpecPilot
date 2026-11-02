"""SPECPILOT_D4: batched union-candidate DSpark Markov lattice builder (load regime, bs <= 24, in-graph).

DSpark's VanillaMarkov head makes the child distribution of a draft node depend on its parent token:
child score s_d(c | p) = base_d[c] + <W1[p], W2[c]> (rank 256). The load lineage's best-first ``build_tree``
consumes a per-slot beam lattice: ``cand[b, s, j]`` (K candidate tokens of depth s+1) and ``edges[b, s, i, j]`` =
log p(cand[s, j] | cand[s-1, i]) (slot 0: row 0 = children of the anchor), already normalised (NORM off).

This module builds that lattice for every request of the batch with ONE CTA PER REQUEST (grid (bs,), no grid
barrier, so it is CUDA-graph safe at any bs). Candidate set of a parent p at depth d (the "union" policy):
  base top-B of row d (shared by all parents)  U  static Markov top-MS of p (precomputed once from W1/W2),
with per-parent normaliser logaddexp(lse over the union scores, tail_d) where tail_d is the base mass outside the
base top-B (Markov bias 0 on the tail). Per slot: tensor-core rank-R dots for the base candidates, precomputed bias
for the static ones, Viterbi beam: value(c) = max over parents (beam[p] + logp(c | p)) over each parent's own
union, top-K distinct tokens (ties: lower pool index), then exact K x K edges for the new beam.
With MS = 0 the same kernel is the plain "wide" builder over a base shortlist of B.

Inputs are in the draft-vocab SUBSET index space (dflash_de_vocab.StaticVocab: logits over Vs rows, sub_to_full map);
parents (and the anchor) are FULL token ids (W1 is indexed by full id). Outputs: cand = full ids (int64), edges fp32.

Lossless: only the draft tree changes; verification is untouched.
Knobs: none read here (pure library; the caller decides). Scratch: ``UnionLatticeBuilder`` allocates everything for
``max_bs`` before graph capture; the kernel restores its scratch (mark = -1) before it returns.
Base top-B / lse / tail come from ``RowStats`` (one CTA per row, one read: block top-TB + max/sum-exp), the beam from
``_union_beam_kernel`` (one CTA per request, serial over slots; one CTA barrier per slot for the stamps; the static
window is only sorted when a static entry beats the K-th base value), the K x K edges from ``_union_edges_kernel``
(grid (bs, S), parallel).
Cross-parent duplicate static tokens are removed inside the best-2K static window (exact unless more than K of the
window's entries are duplicates; then fewer static tokens compete: a draft-quality approximation only).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------------------------------------
# static per-parent Markov top-MS table (built once, outside graph capture)
# ------------------------------------------------------------------------------------------------


@torch.no_grad()
def build_static_table(
    w1: torch.Tensor,  # [V, R] bf16 (markov_w1, full vocab)
    w2s: torch.Tensor,  # [Vs, R] bf16 (markov_w2 rows of the draft subset)
    ms: int,
    chunk: int = 2048,
    parents: Optional[torch.Tensor] = None,  # optional subset of parent ids (default: all V)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """st_ids [V, ms] int32 (subset index), st_bias [V, ms] fp32 = <W1[p], W2s[c]> (fp32 accumulate),
    ordered by bias descending. Selection runs on a bf16 GEMM; the stored bias is recomputed in fp32."""
    V, R = w1.shape
    dev = w1.device
    st_ids = torch.zeros((V, ms), dtype=torch.int32, device=dev)
    st_bias = torch.zeros((V, ms), dtype=torch.float32, device=dev)
    rows = torch.arange(V, device=dev) if parents is None else parents.to(dev).long()
    w2t = w2s.t().contiguous()
    for i in range(0, rows.numel(), chunk):
        r = rows[i : i + chunk]
        a = w1.index_select(0, r)
        sc = a @ w2t  # [c, Vs] bf16 output (fp32 accumulate)
        ids = torch.topk(sc, ms, dim=1, sorted=True).indices
        bias = torch.einsum("cr,cmr->cm", a.float(), w2s[ids].float())
        bias, o = torch.sort(bias, dim=1, descending=True)
        st_ids[r] = ids.gather(1, o).to(torch.int32)
        st_bias[r] = bias
        del sc
    return st_ids, st_bias


# ------------------------------------------------------------------------------------------------
# row statistics (base top-B, full-row lse, tail): one batched pass over [rows, Vs]
# ------------------------------------------------------------------------------------------------


@torch.no_grad()
def row_stats_ref(logits: torch.Tensor, B: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """logits [..., Vs] -> top ids [..., B] int32, top vals [..., B] fp32, tail [...] fp32 (log of the base mass
    outside the top-B; -inf if none). Graph-safe (torch ops, static shapes)."""
    vals, ids = torch.topk(logits, B, dim=-1, sorted=False)
    vals = vals.float()
    lse = torch.logsumexp(logits.float(), dim=-1)
    lse_top = torch.logsumexp(vals, dim=-1)
    frac = -torch.expm1(lse_top - lse)  # 1 - mass(top) / mass(all)
    tail = torch.where(frac > 1e-12, lse + torch.log(frac.clamp_min(1e-30)), torch.full_like(lse, float("-inf")))
    return ids.to(torch.int32), vals, tail


# ------------------------------------------------------------------------------------------------
# packed sort keys: fp32 value (desc) + pool index (asc on ties), non-negative int64
# ------------------------------------------------------------------------------------------------

_IDX_BITS = 20
_IDX_MASK = (1 << _IDX_BITS) - 1


@triton.jit
def _pack(v, idx, IDX_BITS: tl.constexpr):
    u = v.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    key = tl.where(u >= 0x80000000, (~u) & 0xFFFFFFFF, u | 0x80000000)
    return (key << IDX_BITS) | (((1 << IDX_BITS) - 1) - idx.to(tl.int64))


@triton.jit
def _unpack(p, IDX_BITS: tl.constexpr):
    idx = ((1 << IDX_BITS) - 1) - (p & ((1 << IDX_BITS) - 1))
    key = p >> IDX_BITS
    u = tl.where(key >= 0x80000000, key ^ 0x80000000, (~key) & 0xFFFFFFFF)
    v = u.to(tl.int32).to(tl.float32, bitcast=True)
    return idx.to(tl.int32), v


@triton.jit
def _row_stats_kernel(
    x_ptr,  # [rows, Vs] float (row stride s_row, unit inner stride)
    s_row,
    ids_out,  # [rows, B] int32
    vals_out,  # [rows, B] fp32
    tail_out,  # [rows] fp32
    scratch_ptr,  # [rows, NCP] int64 (block winners)
    Vs: tl.constexpr,
    B: tl.constexpr,
    NBLK: tl.constexpr,
    BW: tl.constexpr,
    TB: tl.constexpr,
    NC: tl.constexpr,
    NCP: tl.constexpr,
    IDX_BITS: tl.constexpr,
):
    """One CTA per row, one read of the row: online max / sum-exp and the top-TB packed keys of every BW-column block
    (TB rounds of max + remove), then the top-B of the NC block winners, lse and tail."""
    r = tl.program_id(0).to(tl.int64)
    xr = x_ptr + r * s_row
    bl = tl.arange(0, NBLK)
    cc = tl.arange(0, BW)
    tb = tl.arange(0, TB)
    m = tl.max(tl.load(xr + cc).to(tl.float32), 0)
    se = m * 0.0
    srow = scratch_ptr + r * NCP
    ITER: tl.constexpr = NBLK * BW
    for it in range(0, Vs // ITER):
        cols = it * ITER + bl[:, None] * BW + cc[None, :]
        v = tl.load(xr + cols).to(tl.float32)
        mn = tl.maximum(m, tl.max(tl.max(v, 1), 0))
        se = se * tl.exp(m - mn) + tl.sum(tl.sum(tl.exp(v - mn), 1), 0)
        m = mn
        keys = _pack(v, cols, IDX_BITS)
        out = tl.zeros([NBLK, TB], dtype=tl.int64)
        for t in tl.static_range(TB):
            km = tl.max(keys, 1)
            out = tl.where(tb[None, :] == t, km[:, None], out)
            keys = tl.where(keys == km[:, None], 0, keys)
        tl.store(srow + it * NBLK * TB + bl[:, None] * TB + tb[None, :], out)
    tl.debug_barrier()
    kk = tl.arange(0, NCP)
    top = tl.topk(tl.load(srow + kk, mask=kk < NC, other=0), B)
    idx, val = _unpack(top, IDX_BITS)
    jb = tl.arange(0, B)
    tl.store(ids_out + r * B + jb, idx)
    tl.store(vals_out + r * B + jb, val)
    lse = m + tl.log(se)
    mt = tl.max(val, 0)
    lse_top = mt + tl.log(tl.sum(tl.exp(val - mt), 0))
    frac = 1.0 - tl.exp(lse_top - lse)
    tl.store(tail_out + r, tl.where(frac > 1e-6, lse + tl.log(tl.maximum(frac, 1e-30)), float("-inf")))


def _np2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


class RowStats:
    """Base top-B, full-row lse and tail for ``max_rows`` rows with static outputs (graph-safe): one Triton CTA per
    row, one read (``_row_stats_kernel``). Exact top-B unless more than TB of a row's top-B fall into one BW-column
    block (random rows at Vs 98304, B 128, BW 256, TB 4: 99.4% of rows exact); otherwise a near-top-B set (draft
    quality only)."""

    def __init__(self, max_rows: int, B: int, device, BW: int = 256, TB: int = 4, NBLK: int = 32):
        assert B & (B - 1) == 0 and BW & (BW - 1) == 0 and TB & (TB - 1) == 0
        self.B, self.BW, self.TB, self.NBLK, self.max_rows, self.device = B, BW, TB, NBLK, max_rows, device
        self.ids = torch.zeros((max_rows, B), dtype=torch.int32, device=device)
        self.vals = torch.zeros((max_rows, B), dtype=torch.float32, device=device)
        self.tail = torch.zeros((max_rows,), dtype=torch.float32, device=device)
        self._scr = {}

    def __call__(self, logits: torch.Tensor):
        lead = logits.shape[:-1]
        Vs = logits.shape[-1]
        x = logits.reshape(-1, Vs)
        rows = x.shape[0]
        assert x.stride(1) == 1 and rows <= self.max_rows
        BW, TB = self.BW, self.TB
        while (Vs // BW) * TB < 2 * self.B and BW > 16:  # small vocabularies: smaller blocks
            BW //= 2
        nblk = min(self.NBLK, Vs // BW)
        assert Vs % (nblk * BW) == 0, "Vs must be a multiple of NBLK * BW"
        nc = Vs // BW * TB
        assert nc >= self.B
        ncp = _np2(nc)
        if ncp not in self._scr:
            self._scr[ncp] = torch.zeros((self.max_rows, ncp), dtype=torch.int64, device=self.device)
        _row_stats_kernel[(rows,)](
            x, x.stride(0), self.ids, self.vals, self.tail, self._scr[ncp],
            Vs=Vs, B=self.B, NBLK=nblk, BW=BW, TB=TB, NC=nc, NCP=ncp, IDX_BITS=_IDX_BITS, num_warps=8,
        )
        return (self.ids[:rows].view(*lead, self.B), self.vals[:rows].view(*lead, self.B), self.tail[:rows].view(*lead))


@triton.jit
def _union_beam_kernel(
    logits_ptr,  # [bs, S, Vs] float (row stride s_lb / s_ls, unit inner stride)
    s_lb,
    s_ls,
    top_ids_ptr,  # [bs, S, B] int32 (subset idx)
    top_vals_ptr,  # [bs, S, B] fp32
    tail_ptr,  # [bs, S] fp32
    anchor_ptr,  # [bs] int64 full id
    w1_ptr,  # [V, R] bf16
    w2s_ptr,  # [Vs, R] bf16
    sub2full_ptr,  # [Vs] int64
    st_ids_ptr,  # [V, MS] int32
    st_bias_ptr,  # [V, MS] fp32
    mark_ptr,  # [max_bs, 2, Vs] int8 scratch (-1 at rest): slot-parity stamps of the base top-B
    cand_ptr,  # [bs, S, K] int64 out (full ids)
    csub_ptr,  # [bs, S, K] int32 out (subset idx)
    norm_ptr,  # [bs, S, K] fp32 out (per-parent normaliser of slot s)
    beam_ptr,  # [bs, S, K] fp32 out (Viterbi beam values)
    slots_ptr,  # [bs] int32 valid slots per request (only if HAS_SLOTS; rblock L_b - 1)
    Vs,
    S: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
    MS: tl.constexpr,
    R: tl.constexpr,
    RB: tl.constexpr,
    HAS_STATIC: tl.constexpr,
    IDX_BITS: tl.constexpr,
    HAS_SLOTS: tl.constexpr = False,
):
    """Serial part (one CTA per request): per slot the Viterbi beam over each parent's union candidates. The K x K
    edges of every slot are computed afterwards, in parallel, by ``_union_edges_kernel``."""
    b = tl.program_id(0).to(tl.int64)
    kk = tl.arange(0, K)
    jb = tl.arange(0, B)
    rr = tl.arange(0, RB)
    NEG: tl.constexpr = float("-inf")

    anchor = tl.load(anchor_ptr + b).to(tl.int64)
    ptok = tl.zeros([K], dtype=tl.int64) + anchor
    pv = tl.where(kk == 0, 0.0, NEG)
    mark_row = mark_ptr + b * 2 * Vs
    if HAS_SLOTS:  # slots >= s_lim are never read by build_tree(slots=): skip them (their rows may be stale)
        s_lim = tl.minimum(tl.load(slots_ptr + b), S)
    else:
        s_lim = S

    for s in range(s_lim):
        row = b * S + s
        lrow = logits_ptr + b * s_lb + s * s_ls
        tid = tl.load(top_ids_ptr + row * B + jb).to(tl.int64)
        tv = tl.load(top_vals_ptr + row * B + jb)
        tail = tl.load(tail_ptr + row)
        if HAS_STATIC:
            mk = mark_row + (s % 2) * Vs
            tl.store(mk + tid, tl.zeros([B], dtype=tl.int8) + s.to(tl.int8))
            mm = tl.arange(0, MS)
            sid = tl.load(st_ids_ptr + ptok[:, None] * MS + mm[None, :]).to(tl.int64)
            sbias = tl.load(st_bias_ptr + ptok[:, None] * MS + mm[None, :])
            lv = tl.load(lrow + sid).to(tl.float32)
        # base candidates: bias[p, j] = <W1[ptok[p]], W2s[tid[j]]>
        acc = tl.zeros([K, B], dtype=tl.float32)
        for r0 in tl.static_range(0, R, RB):
            a = tl.load(w1_ptr + ptok[:, None] * R + r0 + rr[None, :])
            w = tl.load(w2s_ptr + tid[None, :] * R + r0 + rr[:, None])
            acc = tl.dot(a, w, acc)
        sb = tv[None, :] + acc
        mx = tl.max(sb, 1)
        if HAS_STATIC:
            tl.debug_barrier()  # this slot's stamps visible
            isb = tl.load(mk + sid) == s.to(tl.int8)
            ss = tl.where(isb, NEG, lv + sbias)
            mx = tl.maximum(mx, tl.max(ss, 1))
            se = tl.sum(tl.exp(sb - mx[:, None]), 1) + tl.sum(tl.exp(ss - mx[:, None]), 1)
        else:
            se = tl.sum(tl.exp(sb - mx[:, None]), 1)
        lse_u = mx + tl.log(se)
        hi = tl.maximum(lse_u, tail)
        norm = hi + tl.log(tl.exp(lse_u - hi) + tl.exp(tail - hi))
        vb = tl.max(pv[:, None] + sb - norm[:, None], 0)  # [B] Viterbi over parents
        top = tl.topk(_pack(vb, jb, IDX_BITS), K)
        if HAS_STATIC:
            sid_f = tl.reshape(sid, [K * MS])
            keys = _pack(pv[:, None] + ss - norm[:, None], B + kk[:, None] * MS + mm[None, :], IDX_BITS)
            keys = tl.where(keys > tl.min(top, 0), keys, 0)  # only entries above the K-th base value can enter
            n_up = tl.sum(tl.sum((keys > 0).to(tl.int32), 1), 0)
            if n_up > 0:
                win = tl.topk(tl.reshape(keys, [K * MS]), 2 * K)
                widx, _ = _unpack(win, IDX_BITS)
                wtok = tl.gather(sid_f, tl.minimum(tl.maximum(widx - B, 0), K * MS - 1), 0)
                # cross-parent duplicates inside the window: keep the best-keyed entry of each token
                dup = tl.max(((wtok[:, None] == wtok[None, :]) & (win[:, None] > win[None, :])).to(tl.int32), 0)
                win = tl.where(dup > 0, 0, win)
                top = tl.topk(tl.reshape(tl.join(top, tl.topk(win, K)), [2 * K]), K)
        idx, val = _unpack(top, IDX_BITS)
        ct = tl.gather(tid, tl.minimum(idx, B - 1), 0)
        if HAS_STATIC:
            ct = tl.where(idx < B, ct, tl.gather(sid_f, tl.minimum(tl.maximum(idx - B, 0), K * MS - 1), 0))
        cfull = tl.load(sub2full_ptr + ct).to(tl.int64)
        tl.store(cand_ptr + row * K + kk, cfull)
        tl.store(csub_ptr + row * K + kk, ct.to(tl.int32))
        tl.store(norm_ptr + row * K + kk, norm)
        tl.store(beam_ptr + row * K + kk, val)
        ptok = cfull
        pv = val
    if HAS_STATIC:  # restore the scratch for the next launch
        tl.debug_barrier()
        for s in range(s_lim):
            tid = tl.load(top_ids_ptr + (b * S + s) * B + jb).to(tl.int64)
            tl.store(mark_row + (s % 2) * Vs + tid, tl.full([B], -1, dtype=tl.int8))


@triton.jit
def _union_edges_kernel(
    logits_ptr,
    s_lb,
    s_ls,
    anchor_ptr,  # [bs] int64
    w1_ptr,  # [V, R] bf16
    w2s_ptr,  # [Vs, R] bf16
    cand_ptr,  # [bs, S, K] int64 (full ids; parents of slot s = cand[s - 1])
    csub_ptr,  # [bs, S, K] int32
    norm_ptr,  # [bs, S, K] fp32
    edges_ptr,  # [bs, S, K, K] fp32 out
    slots_ptr,  # [bs] int32 (only if HAS_SLOTS)
    S: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    HAS_SLOTS: tl.constexpr = False,
):
    """edges[b, s, i, j] = base_s[cand_s[j]] + <W1[parent_i], W2[cand_s[j]]> - norm_s[i]; grid (bs, S)."""
    b = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    kk = tl.arange(0, K)
    rr = tl.arange(0, R)
    row = b * S + s
    if HAS_SLOTS:
        if s >= tl.load(slots_ptr + b):
            return
    if s == 0:
        ptok = tl.zeros([K], dtype=tl.int64) + tl.load(anchor_ptr + b).to(tl.int64)
    else:
        ptok = tl.load(cand_ptr + (row - 1) * K + kk)
    ct = tl.load(csub_ptr + row * K + kk).to(tl.int64)
    a = tl.load(w1_ptr + ptok[:, None] * R + rr[None, :])
    w = tl.load(w2s_ptr + ct[None, :] * R + rr[:, None])
    dot = tl.dot(a, w)
    lvc = tl.load(logits_ptr + b * s_lb + s * s_ls + ct).to(tl.float32)
    norm = tl.load(norm_ptr + row * K + kk)
    tl.store(edges_ptr + (row * K + kk[:, None]) * K + kk[None, :], lvc[None, :] + dot - norm[:, None])


class UnionLatticeBuilder:
    """Holds the static table, subset W2, scratch and output buffers for ``max_bs`` (allocate before capture)."""

    def __init__(
        self,
        w1: torch.Tensor,  # [V, R] bf16
        w2: torch.Tensor,  # [V, R] bf16 (markov_w2.weight, full vocab)
        sub_to_full: Optional[torch.Tensor],  # [Vs] int64 (None = identity, Vs = V)
        *,
        max_bs: int,
        n_slots: int,
        K: int = 16,
        B: int = 128,
        MS: int = 32,
        st: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_warps: int = 8,
        RB: int = 64,
    ):
        dev = w1.device
        V, R = w1.shape
        assert w2.shape == (V, R)
        assert K >= 16 and K & (K - 1) == 0 and B & (B - 1) == 0 and B >= K
        assert MS == 0 or (MS & (MS - 1) == 0 and MS * K + B < (1 << _IDX_BITS) and MS * K >= 2 * K)
        assert R % RB == 0 and R & (R - 1) == 0 and n_slots < 127
        if sub_to_full is None:
            sub_to_full = torch.arange(V, device=dev)
        self.sub_to_full = sub_to_full.to(dev).long().contiguous()
        self.Vs = int(self.sub_to_full.numel())
        self.w1 = w1.to(torch.bfloat16).contiguous()
        self.w2s = w2.to(torch.bfloat16).index_select(0, self.sub_to_full).contiguous()
        self.K, self.B, self.MS, self.R, self.RB, self.S = K, B, MS, R, RB, n_slots
        self.num_warps = num_warps
        if MS:
            self.st_ids, self.st_bias = st if st is not None else build_static_table(self.w1, self.w2s, MS)
            self.mark = torch.full((max_bs, 2, self.Vs), -1, dtype=torch.int8, device=dev)
        else:
            self.st_ids = torch.zeros((1, 1), dtype=torch.int32, device=dev)
            self.st_bias = torch.zeros((1, 1), dtype=torch.float32, device=dev)
            self.mark = torch.zeros((1, 1), dtype=torch.int8, device=dev)
        self.stats = RowStats(max_bs * n_slots, B, dev)
        self.cand = torch.zeros((max_bs, n_slots, K), dtype=torch.int64, device=dev)
        self.csub = torch.zeros((max_bs, n_slots, K), dtype=torch.int32, device=dev)
        self.norm = torch.zeros((max_bs, n_slots, K), dtype=torch.float32, device=dev)
        self.edges = torch.zeros((max_bs, n_slots, K, K), dtype=torch.float32, device=dev)
        self.beam = torch.zeros((max_bs, n_slots, K), dtype=torch.float32, device=dev)

    def lattice(
        self, logits: torch.Tensor, anchor: torch.Tensor, stats=None, slots: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """logits [bs, S, Vs] (subset rows, any float dtype, unit inner stride); anchor [bs] full ids; ``slots``
        (optional, [bs] int32): valid slots per request (rblock: L_b - 1, the same tensor build_tree(slots=) gets);
        slots >= slots[b] are left stale. Returns views cand [bs, S, K] int64 and edges [bs, S, K, K] fp32 (static
        buffers, valid for the first bs rows). No host sync; CUDA-graph safe."""
        bs, S, Vs = logits.shape
        assert S == self.S and Vs == self.Vs and logits.stride(2) == 1
        if stats is None:
            stats = self.stats(logits)
        tid, tv, tail = stats
        anchor = anchor.reshape(-1).long()
        _union_beam_kernel[(bs,)](
            logits, logits.stride(0), logits.stride(1),
            tid.contiguous(), tv.contiguous(), tail.contiguous(), anchor,
            self.w1, self.w2s, self.sub_to_full, self.st_ids, self.st_bias, self.mark,
            self.cand, self.csub, self.norm, self.beam, slots if slots is not None else self.csub, self.Vs,
            S=S, K=self.K, B=self.B, MS=self.MS, R=self.R, RB=self.RB, HAS_STATIC=self.MS > 0, IDX_BITS=_IDX_BITS,
            HAS_SLOTS=slots is not None,
            num_warps=self.num_warps,
            num_stages=1,  # the slot loop carries the beam: no software pipelining
        )
        _union_edges_kernel[(bs, S)](
            logits, logits.stride(0), logits.stride(1), anchor, self.w1, self.w2s,
            self.cand, self.csub, self.norm, self.edges, slots if slots is not None else self.csub,
            S=S, K=self.K, R=self.R, HAS_SLOTS=slots is not None, num_warps=4,
        )
        return self.cand[:bs], self.edges[:bs]


# ------------------------------------------------------------------------------------------------
# PyTorch reference (same semantics; tests and replays)
# ------------------------------------------------------------------------------------------------


@torch.no_grad()
def union_lattice_ref(
    logits: torch.Tensor,  # [bs, S, Vs]
    anchor: torch.Tensor,  # [bs] full ids
    w1: torch.Tensor,  # [V, R]
    w2s: torch.Tensor,  # [Vs, R]
    sub_to_full: torch.Tensor,  # [Vs]
    st_ids: Optional[torch.Tensor],  # [V, MS] or None
    st_bias: Optional[torch.Tensor],
    K: int,
    B: int,
    stats=None,
):
    """Returns cand [bs,S,K] int64 (full ids), edges [bs,S,K,K] fp32, beam values [bs,S,K]."""
    bs, S, Vs = logits.shape
    dev = logits.device
    if stats is None:
        stats = row_stats_ref(logits, B)
    tid_all, tv_all, tail_all = stats
    w1f, w2f = w1.float(), w2s.float()
    s2f = sub_to_full.long()
    cand = torch.zeros((bs, S, K), dtype=torch.int64, device=dev)
    edges = torch.zeros((bs, S, K, K), dtype=torch.float32, device=dev)
    beam = torch.zeros((bs, S, K), dtype=torch.float32, device=dev)
    MS = 0 if st_ids is None else st_ids.shape[1]
    for b in range(bs):
        ptok = torch.full((K,), int(anchor[b]), dtype=torch.int64, device=dev)
        pv = torch.full((K,), float("-inf"), device=dev)
        pv[0] = 0.0
        for s in range(S):
            L = logits[b, s].float()
            tid = tid_all[b, s].long()
            tv = tv_all[b, s]
            tail = tail_all[b, s]
            W1p = w1f[ptok]  # [K, R]
            sb = tv[None, :] + W1p @ w2f[tid].T  # [K, B]
            parts = [sb]
            if MS:
                sid = st_ids[ptok].long()  # [K, MS]
                isb = torch.isin(sid, tid)
                ss = torch.where(isb, torch.full_like(st_bias[ptok], float("-inf")), L[sid] + st_bias[ptok])
                parts.append(ss)
            lse_u = torch.logsumexp(torch.cat(parts, 1), 1)
            norm = torch.logaddexp(lse_u, tail.expand_as(lse_u))
            vb = (pv[:, None] + sb - norm[:, None]).max(0).values  # [B]
            pool_v = [vb]
            pool_t = [tid]
            if MS:
                # static pool: the best 2K entries (value desc, flat index asc); within that window keep the first
                # occurrence of each token (cross-parent duplicates)
                vs = (pv[:, None] + ss - norm[:, None]).reshape(-1)
                st = sid.reshape(-1)
                order = torch.argsort(-vs, stable=True)[: 2 * K]
                keep = torch.zeros_like(vs, dtype=torch.bool)
                seen = set()
                for e in order.tolist():
                    t = int(st[e])
                    if t not in seen:
                        seen.add(t)
                        keep[e] = True
                pool_v.append(torch.where(keep, vs, torch.full_like(vs, float("-inf"))))
                pool_t.append(st)
            pv_all = torch.cat(pool_v)
            pt_all = torch.cat(pool_t)
            sel = torch.argsort(-pv_all, stable=True)[:K]
            ct = pt_all[sel]
            val = pv_all[sel]
            e = L[ct][None, :] + W1p @ w2f[ct].T - norm[:, None]
            edges[b, s] = e
            cand[b, s] = s2f[ct]
            beam[b, s] = val
            ptok = s2f[ct]
            pv = val
    return cand, edges, beam
