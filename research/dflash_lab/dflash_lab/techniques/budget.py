"""Dynamic verification budget: how many drafted positions each request verifies per step.

Reference implementation for the ``budget`` family
(``research/knowledge/search_space/06_dynamic_budget.md``). Pure PyTorch, no sglang import.
In every snapshot under ``lab/dflash_lab/techniques/``, importable as ``dflash_lab.techniques.budget``
(the engine never imports it). CPU tests: ``research/validation/tests/test_techniques_budget.py``.

Layout (the engine's verify block, ``EXECUTION_PATH.md`` section 1 steps 7-10)::

    candidates  [bs, n]   int    column 0 = anchor (previous bonus), columns 1..n-1 = drafts
    conf        [bs, D]   float  D = n - 1; conf[:, t-1] = drafter confidence of draft position t
                                 (any number in [0, 1]; e.g. the selector walk probability of the
                                 chosen candidate, or a calibrated per-slot acceptance estimate)
    s           [bs, n]   float  prefix survival, s[:, 0] = 1 (anchor/bonus is always committed),
                                 s[:, k] ~ Pr(the first k drafts are all accepted); -inf = not
                                 selectable (inactive / padded row, or k >= max_len)
    verify_lens [bs]      int32  positions verified per request, anchor included (1..n; 0 for an
                                 inactive row). Request i verifies candidates[i, :verify_lens[i]].
    cost        [T+1]     float  measured step time (seconds or ms, any unit) when the batch verifies
                                 T positions in total; index = total verified positions

Semantics. Verification stays exact greedy: request i accepts the longest prefix of its
*verified* drafts that equals the target argmax, so ``accept = min(L_i, verify_lens - 1)`` where
L_i is what the full block would have accepted, and the bonus is the target argmax at row
``accept`` (that row is verified because ``accept <= verify_lens - 1``). Rows are causal, so the
target rows of a truncated block equal the first rows of the full block: a budget changes only
*how many tokens a step commits*, never *which tokens*. Lossless limit: every active request
verifies the full block (``verify_lens == n``), which is stock (``truncated_greedy_accept`` with
``verify_lens = n`` is the stock rule, and every selector below returns ``n`` for all active rows
when its candidate totals include ``B * n`` and the cost is flat, or when the only candidate is
``B * n``).

Expected commit of a prefix of length v (D-Cut Eq. 2-3, Leviathan et al.):
``E[commit] = 1 + sum_{k=1}^{v-1} Pr(L >= k) = sum_{k<v} s_k``. Because s is non-increasing in k,
the global top-K positions (ties to smaller k, then smaller request index) are a contiguous prefix
per request, and the top-K sum is the best expected commit any allocation of K positions can get
(the exchange argument of D-Cut / TETRIS / DSpark's ``schedule_verify_lens_topk``).

Methods

1. D-Cut (Liu et al. 2026, arXiv 2607.14647, Alg. 1): ``prefix_survival`` (Eq. 3, inclusive
   product; ``shifted=True`` is the App. B shifted score for stochastic drafts), ``dcut_totals``
   (Eq. 4, ``K_rho = max(B, ceil(rho * B * n))``), ``dcut`` / ``select_budget`` (Eq. 5: pick the
   candidate total maximising ``sum(top-K scores) / step_time(K)``), ``lens_from_total``.
2. TurboSpec (Liu et al. 2024/2025, arXiv 2406.14066): ``expected_accept_len`` (Eq. 4,
   ``(1 - a^(k+1)) / (1 - a)``), ``turbospec_uniform`` (Listing 2: argmax over k of expected
   generated tokens / estimated batch time, one k for the whole batch), ``turbospec_per_request``
   (per-request lengths: the same top-K machinery with geometric survival ``a_i^k``),
   ``linear_step_time`` / ``fit_linear_step_time`` (Eq. 3 cost model), ``update_acceptance``
   (moving-average acceptance estimate; the paper's accepted/proposed ratio and an unbiased
   censored-geometric variant).
3. BanditSpec (Hou et al. ICML 2025, arXiv 2505.15141): ``UCBSpec`` (Alg. 4, Eq. 6 confidence
   radius), ``EXP3Spec`` (Alg. 5, anytime EXP3 with importance-weighted losses), the
   stopping-time formulation (``run_stopping_time``, ``TruncatedGeometricArms``) and
   ``BucketedPolicy`` (one bandit per batch-size bucket; not in the paper).

Graph safety. Everything in 1-2 is tensor-only: no ``.item()``/``.tolist()``, no data-dependent
shapes (outputs are ``[bs]`` / ``[bs, n]`` / ``[]`` / ``[m]`` with m = number of candidate totals),
Python control flow only on static arguments; the sort is over the static ``bs * n`` positions.
They can run in a captured region or eagerly after the draft graph. Choosing *which captured
verify graph* to replay (a total-token tier or a uniform width) is a host decision: feed it
host-known values (batch size, lagged acceptance, lagged confidence) or pay a sync; the
per-request split under a host-chosen total can then run on device from the current step's
confidence (``lens_from_total`` with a tensor total). Section 3 keeps host-side Python state by
design (one arm choice per step, rewards observed one or two steps late under the overlap
scheduler); it must never read device tensors synchronously on the timed path.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch

Number = Union[float, int, torch.Tensor]
NEG_INF = float("-inf")
DCUT_RATIOS = (0.25, 0.5, 0.75, 1.0)  # D-Cut's ratio buckets R (Sec. 3, Eq. 4)


# ----------------------------------------------------------------------------------------------
# Scores and allocation (shared by D-Cut and per-request TurboSpec)
# ----------------------------------------------------------------------------------------------

def _active_mask(bs: int, active: Optional[torch.Tensor], device) -> torch.Tensor:
    if active is None:
        return torch.ones(bs, dtype=torch.bool, device=device)
    if active.shape != (bs,):
        raise ValueError(f"active must be [bs] = [{bs}], got {tuple(active.shape)}")
    return active.to(device=device, dtype=torch.bool)


def _finalize_scores(s: torch.Tensor, active: Optional[torch.Tensor],
                     max_len: Optional[int]) -> torch.Tensor:
    bs, n = s.shape
    s = torch.cummin(s, dim=1).values  # guard: survival must be non-increasing in k
    act = _active_mask(bs, active, s.device)
    keep = act[:, None].expand(bs, n)
    if max_len is not None:
        if max_len < 1:
            raise ValueError("max_len must be >= 1 (the anchor is always verified)")
        keep = keep & (torch.arange(n, device=s.device) < max_len)[None, :]
    return torch.where(keep, s, torch.full_like(s, NEG_INF))


def prefix_survival(conf: torch.Tensor, *, shifted: bool = False, max_len: Optional[int] = None,
                    active: Optional[torch.Tensor] = None) -> torch.Tensor:
    """D-Cut Eq. 3: ``s[:, 0] = 1``, ``s[:, k] = prod_{t=1}^{k} c_t`` (float32 ``[bs, D+1]``).

    ``shifted=True`` is D-Cut App. B's score ``prod_{t=1}^{k-1} c_t`` (``s[:, 1] = 1``), which the
    authors give for stochastic drafts so that the cut does not depend on the sampled token itself
    (analysis only in the paper; for greedy drafts and greedy verify both are exact).
    Confidences are clamped to [0, 1]. ``max_len`` caps the verify length (positions k >= max_len
    get -inf); inactive rows are all -inf (verify_lens 0).
    """
    if conf.ndim != 2:
        raise ValueError(f"conf must be [bs, D], got {tuple(conf.shape)}")
    c = conf.float().clamp(0.0, 1.0)
    bs = c.shape[0]
    ones = c.new_ones(bs, 1)
    cp = torch.cumprod(c, dim=1)
    if shifted:
        s = torch.cat([ones, ones, cp[:, :-1]], dim=1) if c.shape[1] >= 1 else ones
    else:
        s = torch.cat([ones, cp], dim=1)
    return _finalize_scores(s, active, max_len)


def geometric_survival(alpha: Number, n: int, *, bs: Optional[int] = None,
                       active: Optional[torch.Tensor] = None,
                       max_len: Optional[int] = None) -> torch.Tensor:
    """``s[:, k] = alpha_i ** k`` (``[bs, n]``): the i.i.d. acceptance model of TurboSpec Eq. 4.

    ``alpha`` is a per-request ``[bs]`` tensor or a scalar (then ``bs`` is required).
    """
    if not isinstance(alpha, torch.Tensor):
        if bs is None:
            raise ValueError("bs is required with a scalar alpha")
        alpha = torch.full((bs,), float(alpha))
    a = alpha.float().clamp(0.0, 1.0).reshape(-1)
    k = torch.arange(n, device=a.device, dtype=torch.float32)
    s = torch.where(k[None, :] == 0, torch.ones_like(a)[:, None], a[:, None] ** k[None, :])
    return _finalize_scores(s, active, max_len)


def global_rank(s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rank of every position in the batch-wide descending order of ``s``.

    Ties go to the smaller position k, then the smaller request index (D-Cut Alg. 1 "ties toward
    smaller k"; DSpark ``_value_independent_descending_order``). Returns ``rank [bs, n]`` (int64,
    0 = best) and the sorted scores ``[bs * n]`` (-inf last). One static-shape stable sort.
    """
    bs, n = s.shape
    flat = s.t().reshape(-1)  # index = k * bs + i  ->  original order is (k, i) ascending
    sorted_s, order = torch.sort(flat, descending=True, stable=True)
    rank_flat = torch.empty_like(order)
    rank_flat[order] = torch.arange(order.numel(), device=s.device)
    return rank_flat.view(n, bs).t().contiguous(), sorted_s


def lens_from_total(s: torch.Tensor, total: Number,
                    rank: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-request verify lengths (int32 ``[bs]``) when the batch verifies the global top-``total``
    positions. ``total`` may be a Python int or a 0-d tensor (device-side, no sync). Anchors of
    active rows come first, so ``total >= B`` verifies every anchor; selections are prefixes."""
    if rank is None:
        rank, _ = global_rank(s)
    valid = torch.isfinite(s)
    tot = total if isinstance(total, torch.Tensor) else torch.tensor(int(total), device=s.device)
    return ((rank < tot.to(rank.device)) & valid).sum(dim=1).to(torch.int32)


def verify_mask(verify_lens: torch.Tensor, n: int) -> torch.Tensor:
    """bool ``[bs, n]``: position k of request i is verified (fixed shape; a kernel can skip work,
    e.g. GDN snapshot writes, at masked positions)."""
    return torch.arange(n, device=verify_lens.device)[None, :] < verify_lens.to(torch.int64)[:, None]


def expected_commit(s: torch.Tensor, verify_lens: torch.Tensor) -> torch.Tensor:
    """Predicted committed tokens per request, ``sum_{k < verify_lens} s_k`` (float32 ``[bs]``)."""
    m = verify_mask(verify_lens, s.shape[1]) & torch.isfinite(s)
    return torch.where(m, s, torch.zeros_like(s)).sum(dim=1)


class BudgetDecision(NamedTuple):
    verify_lens: torch.Tensor  # int32 [bs]   positions verified per request (anchor included)
    total: torch.Tensor        # int64 []     chosen total verified positions
    choice: torch.Tensor       # int64 []     index into the candidate totals
    utility: torch.Tensor      # f32 [m]      predicted committed tokens per candidate total
    step_time: torch.Tensor    # f32 [m]      draft_cost + cost[total] per candidate
    rate: torch.Tensor         # f32 [m]      utility / step_time (the objective)
    totals: torch.Tensor       # int64 [m]    candidate totals after clamping to [B, #valid]


def _lookup_cost(cost: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    cost = cost.to(device=total.device, dtype=torch.float32)
    last = cost.numel() - 1
    c = cost[total.clamp(0, last)]
    return torch.where(total <= last, c, torch.full_like(c, float("inf")))


def select_budget(s: torch.Tensor, cost: torch.Tensor, *, totals: Optional[torch.Tensor] = None,
                  draft_cost: Number = 0.0, count_anchor: bool = True) -> BudgetDecision:
    """Pick the total verified positions K maximising ``utility(K) / (draft_cost + cost[K])``,
    then split K across requests by the global top-K of ``s`` (D-Cut Eq. 5 + Alg. 1).

    ``totals``: candidate totals (int ``[m]``), e.g. D-Cut's ``dcut_totals`` or the captured
    verify-graph token tiers; ``None`` = every integer 0..bs*n (finest grid, the per-request
    TurboSpec / TETRIS optimum). Totals are clamped to ``[B, #selectable]``. ``utility(K)`` is the
    sum of the top-K scores (anchors included, = predicted committed tokens) or, with
    ``count_anchor=False``, only the draft part (what the D-Cut vLLM PR #47131 code sums; with a
    ratio objective this can move the argmax). ``draft_cost`` is a scalar or ``[m]`` tensor added
    to every candidate's step time. Ties pick the first candidate (smallest total if ascending).
    Unknown cost (K beyond the table) is +inf.
    """
    bs, n = s.shape
    rank, sorted_s = global_rank(s)
    valid = torch.isfinite(s)
    csum = torch.cat([sorted_s.new_zeros(1),
                      torch.cumsum(torch.where(torch.isfinite(sorted_s), sorted_s,
                                               torch.zeros_like(sorted_s)), 0)])
    num_active = valid[:, 0].sum()
    num_valid = valid.sum()
    if totals is None:
        cand = torch.arange(bs * n + 1, device=s.device)
    else:
        cand = totals.to(device=s.device, dtype=torch.int64).reshape(-1)
    k = torch.minimum(torch.maximum(cand, num_active), num_valid)
    util = csum[k]
    if not count_anchor:
        util = util - num_active.to(util.dtype)
    dc = draft_cost.to(device=s.device, dtype=torch.float32) if isinstance(draft_cost, torch.Tensor) \
        else torch.tensor(float(draft_cost), device=s.device)
    step = dc + _lookup_cost(cost, k)
    rate = torch.where(step > 0, util / step, torch.zeros_like(util))
    rate = torch.where(torch.isfinite(step), rate, torch.full_like(rate, NEG_INF))
    choice = torch.argmax(rate)
    total = k[choice]
    lens = ((rank < total) & valid).sum(dim=1).to(torch.int32)
    return BudgetDecision(lens, total, choice, util, step, rate, k)


def dcut_totals(num_active: Number, n: int, ratios: Sequence[float] = DCUT_RATIOS,
                device=None) -> torch.Tensor:
    """D-Cut Eq. 4: ``K_rho(B) = max(B, ceil(rho * B * n))`` for every rho (int64 ``[R]``).
    ``num_active`` may be a 0-d tensor (B counted on device)."""
    b = num_active if isinstance(num_active, torch.Tensor) else torch.tensor(int(num_active))
    b = b.to(device=device if device is not None else b.device, dtype=torch.float64)
    r = torch.tensor([float(x) for x in ratios], dtype=torch.float64, device=b.device)
    # The 1e-9 guard keeps exact products (0.75 * 16 * 3 = 36) from rounding up.
    k = torch.ceil(r * b * n - 1e-9)
    return torch.maximum(k, b).to(torch.int64)


def dcut(conf: torch.Tensor, cost: torch.Tensor, *, ratios: Sequence[float] = DCUT_RATIOS,
         shifted: bool = False, active: Optional[torch.Tensor] = None,
         max_len: Optional[int] = None, draft_cost: Number = 0.0,
         count_anchor: bool = True) -> BudgetDecision:
    """D-Cut Alg. 1 end to end: prefix scores, the R candidate budgets, the cost-aware choice, and
    per-request prefixes. ``ratios=(rho,)`` is the fixed-ratio mode; ``(1.0,)`` is stock."""
    s = prefix_survival(conf, shifted=shifted, max_len=max_len, active=active)
    n_eff = s.shape[1] if max_len is None else min(s.shape[1], max_len)
    b = torch.isfinite(s[:, 0]).sum()
    return select_budget(s, cost, totals=dcut_totals(b, n_eff, ratios, device=s.device),
                         draft_cost=draft_cost, count_anchor=count_anchor)


# ----------------------------------------------------------------------------------------------
# Truncated exact-greedy accept (what the engine must do with a verify budget)
# ----------------------------------------------------------------------------------------------

class TruncatedAccept(NamedTuple):
    accept_len: torch.Tensor   # int32 [bs]  accepted drafts, <= verify_lens - 1
    commit_lens: torch.Tensor  # int32 [bs]  accept_len + 1 (GDN scatter index = commit_lens - 1)
    bonus: torch.Tensor        # int64 [bs]  target argmax at row accept_len
    capped: torch.Tensor       # int32 [bs]  drafts beyond the cut that would also match (all rows real)


def truncated_greedy_accept(candidates: torch.Tensor, target_top1: torch.Tensor,
                            verify_lens: Optional[torch.Tensor] = None) -> TruncatedAccept:
    """Exact greedy accept on per-request verify prefixes (``_dflash_accept_bonus_contig_kernel``
    semantics plus DSpark's ``cap_correct_len``). ``verify_lens=None`` (or all ``n``) is stock.

    ``target_top1 [bs, n]`` is the target argmax of each verify row. Rows at or beyond
    ``verify_lens`` are never read, so they may hold anything (a fixed-shape verify can leave
    them uncomputed or unsnapshotted). Inactive rows (``verify_lens == 0``) return zeros.
    """
    bs, n = candidates.shape
    if verify_lens is None:
        verify_lens = torch.full((bs,), n, dtype=torch.int32, device=candidates.device)
    v = verify_lens.to(torch.int64)
    match = candidates[:, 1:].to(torch.int64) == target_top1[:, :-1].to(torch.int64)
    full = match.to(torch.int64).cumprod(dim=1).sum(dim=1)
    allowed = torch.arange(n - 1, device=candidates.device)[None, :] < (v - 1)[:, None]
    acc = (match & allowed).to(torch.int64).cumprod(dim=1).sum(dim=1)
    live = v > 0
    acc = torch.where(live, acc, torch.zeros_like(acc))
    bonus = target_top1.to(torch.int64).gather(1, acc[:, None]).squeeze(1)
    bonus = torch.where(live, bonus, torch.zeros_like(bonus))
    commit = torch.where(live, acc + 1, torch.zeros_like(acc))
    capped = torch.where(live, full - acc, torch.zeros_like(full))
    return TruncatedAccept(acc.to(torch.int32), commit.to(torch.int32), bonus, capped.to(torch.int32))


# ----------------------------------------------------------------------------------------------
# Cost models
# ----------------------------------------------------------------------------------------------

def cost_table_from_samples(tokens: Sequence[int], seconds: Sequence[float], max_tokens: int, *,
                            monotone: bool = True) -> torch.Tensor:
    """Piecewise-linear cost table ``[max_tokens + 1]`` from profiled (total tokens, step time)
    samples, constant beyond the end points; ``monotone`` applies a running max (step time never
    falls with more tokens). D-Cut instead looks up the nearest profiled entry; DSpark's
    ``SpsCostTable.lookup`` takes the floor probe."""
    xs = torch.tensor([float(t) for t in tokens], dtype=torch.float64)
    ys = torch.tensor([float(y) for y in seconds], dtype=torch.float64)
    if xs.numel() == 0 or xs.numel() != ys.numel():
        raise ValueError("need matching, non-empty tokens / seconds")
    order = torch.argsort(xs)
    xs, ys = xs[order], ys[order]
    t = torch.arange(max_tokens + 1, dtype=torch.float64)
    hi = torch.searchsorted(xs, t, right=True).clamp(1, max(xs.numel() - 1, 1))
    lo = hi - 1
    if xs.numel() == 1:
        out = ys.expand_as(t).clone()
    else:
        span = (xs[hi] - xs[lo]).clamp(min=1e-12)
        frac = ((t - xs[lo]) / span).clamp(0.0, 1.0)
        out = ys[lo] + frac * (ys[hi] - ys[lo])
    if monotone:
        out = torch.cummax(out, 0).values
    return out.to(torch.float32)


def round_up_to_tier(total: torch.Tensor, tiers: Sequence[int]) -> torch.Tensor:
    """Smallest captured token tier >= total (-1 above the largest tier)."""
    tt = torch.tensor(sorted(set(int(x) for x in tiers)), dtype=torch.int64, device=total.device)
    idx = torch.searchsorted(tt, total.to(torch.int64))
    return torch.where(idx < tt.numel(), tt[idx.clamp(max=tt.numel() - 1)], torch.full_like(idx, -1))


def tiered_cost(cost: torch.Tensor, tiers: Sequence[int]) -> torch.Tensor:
    """``cost_eff[T] = cost[tier(T)]``: what a padded graph replay costs (inf above the top tier).
    With this table the best total is always a tier itself (fill the bucket: DSpark
    ``align_verify_tokens_to_graph_tier``)."""
    t = torch.arange(cost.numel(), dtype=torch.int64)
    tier = round_up_to_tier(t, tiers)
    out = cost.float()[tier.clamp(min=0)]
    return torch.where(tier >= 0, out, torch.full_like(out, float("inf")))


def roofline_step_cost(max_tokens: int, *, weight_bytes: float, flops_per_token: float,
                       state_bytes_per_token: float = 0.0, fixed: float = 0.0,
                       bandwidth: float = 3.35e12, peak_flops: float = 989e12,
                       efficiency: float = 1.0) -> torch.Tensor:
    """Roofline *estimate* of a verify step vs total tokens T (seconds, ``[max_tokens + 1]``):
    ``fixed + max(weight_bytes / bw, flops_per_token * T / (peak * eff)) + state_bytes * T / bw``.
    The last term is the per-verified-token GDN snapshot traffic of a hybrid target. Synthetic
    tables and planning only: a campaign uses measured tables (``ar microbench`` / profile)."""
    t = torch.arange(max_tokens + 1, dtype=torch.float64)
    dense = torch.maximum(torch.full_like(t, weight_bytes / bandwidth),
                          flops_per_token * t / (peak_flops * efficiency))
    out = fixed + dense + state_bytes_per_token * t / bandwidth
    out[0] = fixed
    return out.to(torch.float32)


def linear_step_time(n_context: Number, n_batched: Number, coef: Sequence[float]) -> Number:
    """TurboSpec Eq. 3: ``T_fwd = a * N_context + g * N_batched + d`` with ``coef = (a, g, d)``
    (a: KV-cache loading per context token, g: compute per batched token, d: weights + overhead)."""
    a, g, d = (float(x) for x in coef)
    return a * n_context + g * n_batched + d


def fit_linear_step_time(n_context: Sequence[float], n_batched: Sequence[float],
                         seconds: Sequence[float]) -> Tuple[float, float, float]:
    """Least-squares fit of TurboSpec's Eq. 3 coefficients from offline profiling (host, float64)."""
    x = torch.tensor([[float(c), float(b), 1.0] for c, b in zip(n_context, n_batched)],
                     dtype=torch.float64)
    y = torch.tensor([float(v) for v in seconds], dtype=torch.float64)[:, None]
    sol = torch.linalg.lstsq(x, y).solution.reshape(-1)
    return float(sol[0]), float(sol[1]), float(sol[2])


# ----------------------------------------------------------------------------------------------
# TurboSpec: goodput-driven speculation length
# ----------------------------------------------------------------------------------------------

def expected_accept_len(alpha: Number, k: Number) -> torch.Tensor:
    """TurboSpec Eq. 4 / Leviathan et al.: expected generated tokens (accepted + bonus) when k
    drafts are verified with i.i.d. per-draft acceptance alpha: ``(1 - a^(k+1)) / (1 - a)``
    (``k + 1`` at ``a = 1``, ``1`` at ``k = 0``). Broadcasts tensors."""
    a = alpha.float() if isinstance(alpha, torch.Tensor) else torch.tensor(float(alpha))
    kk = k.to(a.device).float() if isinstance(k, torch.Tensor) else torch.tensor(float(k), device=a.device)
    a = a.clamp(0.0, 1.0)
    near_one = (1.0 - a).abs() < 1e-6
    safe = torch.where(near_one, torch.zeros_like(a), a)
    val = (1.0 - safe ** (kk + 1.0)) / (1.0 - safe)
    return torch.where(near_one, kk + 1.0 + torch.zeros_like(val), val)


class UniformDecision(NamedTuple):
    k: torch.Tensor          # int64 []   drafts verified per request (verify width k + 1)
    choice: torch.Tensor     # int64 []   index into ks
    generated: torch.Tensor  # f32 [m]    expected generated tokens of the batch per k
    step_time: torch.Tensor  # f32 [m]
    goodput: torch.Tensor    # f32 [m]


def turbospec_uniform(alpha: Number, cost: torch.Tensor, ks: Sequence[int], *,
                      bs: Optional[int] = None, active: Optional[torch.Tensor] = None,
                      draft_cost: Number = 0.0) -> UniformDecision:
    """TurboSpec Listing 2 (``ArgMaxGoodput``): one speculation length for the whole batch,
    ``k* = argmax_k sum_i l(alpha_i, k) / (draft_cost(k) + cost[B * (k + 1)])``.

    ``alpha``: per-request ``[bs]`` estimates or one global scalar (the paper's default
    granularity). ``ks``: allowed k (e.g. the captured verify widths minus one; include 0 only if
    the engine has a non-speculative path and pass its ``draft_cost`` as 0 there).
    ``draft_cost``: scalar or ``[m]`` per-k drafting time (a DFlash block draft costs the same for
    every k > 0). Ties pick the first k in ``ks``.
    """
    if isinstance(alpha, torch.Tensor) and alpha.ndim == 1:
        a = alpha.float()
    else:
        if bs is None:
            raise ValueError("bs is required with a scalar alpha")
        a = torch.full((bs,), float(alpha))
    act = _active_mask(a.numel(), active, a.device)
    kt = torch.tensor([int(x) for x in ks], dtype=torch.int64, device=a.device)
    lens = expected_accept_len(a[:, None], kt[None, :])  # [bs, m]
    gen = torch.where(act[:, None], lens, torch.zeros_like(lens)).sum(0)
    b = act.sum()
    total = b * (kt + 1)
    dc = draft_cost.to(device=a.device, dtype=torch.float32) if isinstance(draft_cost, torch.Tensor) \
        else torch.tensor(float(draft_cost), device=a.device)
    step = dc + _lookup_cost(cost, total)
    good = torch.where(torch.isfinite(step) & (step > 0), gen / step, torch.full_like(gen, NEG_INF))
    choice = torch.argmax(good)
    return UniformDecision(kt[choice], choice, gen, step, good)


def turbospec_per_request(alpha: torch.Tensor, cost: torch.Tensor, n: int, *,
                          active: Optional[torch.Tensor] = None, totals: Optional[torch.Tensor] = None,
                          draft_cost: Number = 0.0, max_len: Optional[int] = None) -> BudgetDecision:
    """Per-request lengths under TurboSpec's goodput objective (the request-level granularity the
    paper describes as an alternative): the marginal gain of request i's k-th draft is
    ``alpha_i^k``, so the optimum for each total is the global top-K of geometric survival and the
    total is chosen like D-Cut (``totals=None`` searches every integer)."""
    s = geometric_survival(alpha, n, active=active, max_len=max_len)
    return select_budget(s, cost, totals=totals, draft_cost=draft_cost)


class AcceptanceState(NamedTuple):
    num: torch.Tensor  # f32 [R]  discounted accepted drafts
    den: torch.Tensor  # f32 [R]  discounted trials

    def alpha(self, prior: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
        return torch.where(self.den > eps, self.num / self.den.clamp(min=eps),
                           torch.full_like(self.num, prior))


def init_acceptance(size: int, device=None) -> AcceptanceState:
    z = torch.zeros(size, dtype=torch.float32, device=device)
    return AcceptanceState(z, z.clone())


def update_acceptance(state: AcceptanceState, accepted: torch.Tensor, verified_drafts: torch.Tensor,
                      *, index: Optional[torch.Tensor] = None, decay: float = 0.9,
                      estimator: str = "censored") -> AcceptanceState:
    """Moving-average per-draft acceptance estimate (device tensors, no sync).

    ``estimator="ratio"`` is TurboSpec's accepted / proposed. It is biased low whenever a step
    accepts every verified draft (the run is censored): with i.i.d. acceptance alpha it converges
    to ``(l(alpha, m) - 1) / m``, not alpha. ``"censored"`` is the truncated-geometric MLE: a
    step with ``a < m`` observed a successes and one failure, a step with ``a == m`` observed m
    successes and no failure, ``alpha_hat = sum a / sum (a + 1[a < m])``. Verify budgets make
    censoring common (short verify lengths), so the choice matters for this family.

    ``index`` selects rows of a per-request-slot state (e.g. ``req_pool_indices``); without it the
    state has one row (global estimate) or one row per batch row.
    """
    a = accepted.float().reshape(-1)
    m = verified_drafts.float().reshape(-1)
    if estimator == "ratio":
        succ, trials = a, m
    elif estimator == "censored":
        succ, trials = a, a + (a < m).float()
    else:
        raise ValueError(f"unknown estimator {estimator!r}")
    if index is None:
        if state.num.numel() == 1:
            return AcceptanceState(decay * state.num + succ.sum(), decay * state.den + trials.sum())
        return AcceptanceState(decay * state.num + succ, decay * state.den + trials)
    num = state.num.clone()
    den = state.den.clone()
    idx = index.to(torch.int64).reshape(-1)
    num.index_copy_(0, idx, decay * num.index_select(0, idx) + succ)
    den.index_copy_(0, idx, decay * den.index_select(0, idx) + trials)
    return AcceptanceState(num, den)


def calibrate_power(conf: torch.Tensor, exponent: Number) -> torch.Tensor:
    """Per-slot power calibration ``c_t ** gamma_t`` (``exponent`` scalar or ``[D]``), a monotone,
    closed-form stand-in for DSpark's per-slot temperature scaling; fit gamma on calib traces."""
    e = exponent.to(conf.device).float() if isinstance(exponent, torch.Tensor) else float(exponent)
    return conf.float().clamp(0.0, 1.0) ** e


# ----------------------------------------------------------------------------------------------
# BanditSpec: online selection of a speculation configuration (host-side state)
# ----------------------------------------------------------------------------------------------

class UCBSpec:
    """UCBSpec (Hou et al., ICML 2025, Alg. 4 / Alg. 8, Eq. 6). One arm per speculation
    configuration (e.g. a verify-width cap, a D-Cut ratio set, a budget rule); the caller keeps the
    arm list. Round-robin over the K arms first, then ``argmax_i mu_hat_i + cr_i`` with

        cr_i = h * sqrt((1 + n_i) / n_i^2 * (1 + 2 log(K t^2 sqrt(1 + n_i) / delta)))

    where t is the number of completed rounds and ``h`` the sub-Gaussian half-width of the reward.
    The paper's reward is the round's generated tokens ``Y in [1, L + 1]`` (h = L / 2,
    ``max_len=L``); for any other bounded reward (e.g. tokens per second of a batched step, the
    paper's Sec. 6.2 experiment) pass ``reward_range=(lo, hi)`` (h = (hi - lo) / 2). Ties go to
    the smallest arm index (not specified in the paper). Host-side Python state; ``update`` may be
    called one or two steps late (overlap scheduler), which the analysis does not cover.
    """

    def __init__(self, num_arms: int, *, max_len: Optional[int] = None,
                 reward_range: Optional[Tuple[float, float]] = None, delta: float = 0.5):
        if num_arms < 1:
            raise ValueError("num_arms must be >= 1")
        if (max_len is None) == (reward_range is None):
            raise ValueError("give exactly one of max_len (paper reward) or reward_range")
        self.k = int(num_arms)
        self.half = (max_len / 2.0) if max_len is not None else (reward_range[1] - reward_range[0]) / 2.0
        self.delta = float(delta)
        self.t = 0
        self.n = [0] * self.k
        self.total = [0.0] * self.k

    def means(self) -> List[float]:
        return [s / c if c else float("nan") for s, c in zip(self.total, self.n)]

    def radius(self, arm: int) -> float:
        n = self.n[arm]
        if n == 0:
            return float("inf")
        t = max(self.t, 1)
        log_term = math.log(self.k * t * t * math.sqrt(1.0 + n) / self.delta)
        return self.half * math.sqrt((1.0 + n) / (n * n) * (1.0 + 2.0 * max(log_term, 0.0)))

    def index(self) -> List[float]:
        return [(self.total[i] / self.n[i] + self.radius(i)) if self.n[i] else float("inf")
                for i in range(self.k)]

    def select(self) -> int:
        for i in range(self.k):
            if self.n[i] == 0:
                return i
        idx = self.index()
        return max(range(self.k), key=lambda i: (idx[i], -i))

    def update(self, arm: int, reward: float) -> None:
        self.n[arm] += 1
        self.total[arm] += float(reward)
        self.t += 1


class EXP3Spec:
    """EXP3Spec (Hou et al., ICML 2025, Alg. 5 / Alg. 9; anytime EXP3). Losses are importance
    weighted and scaled to [0, 1]: ``Z_hat_i = 1{i = I_t} (hi - Y) / ((hi - lo) p_{t,i})`` (the
    paper: ``(L + 1 - Y) / (L p)`` with Y in [1, L + 1]); ``eta_t = sqrt(log K / (t K))``;
    ``p_{t,i} proportional to exp(-eta_t * sum_{s<t} Z_hat_{i,s})``; ``I_t ~ p_t``. No explicit
    uniform exploration (the decaying eta handles the unknown stopping time). For an oblivious
    adversary (non-stationary rewards). Host-side state with its own seeded RNG.
    """

    def __init__(self, num_arms: int, *, max_len: Optional[int] = None,
                 reward_range: Optional[Tuple[float, float]] = None, seed: int = 0):
        if num_arms < 1:
            raise ValueError("num_arms must be >= 1")
        if (max_len is None) == (reward_range is None):
            raise ValueError("give exactly one of max_len (paper reward) or reward_range")
        self.k = int(num_arms)
        self.lo, self.hi = ((1.0, float(max_len) + 1.0) if max_len is not None
                            else (float(reward_range[0]), float(reward_range[1])))
        if self.hi <= self.lo:
            raise ValueError("reward range must have hi > lo")
        self.rng = random.Random(seed)
        self.t = 0
        self.cum_loss = [0.0] * self.k
        self._p: Optional[List[float]] = None

    def probs(self) -> List[float]:
        t = self.t + 1
        eta = math.sqrt(math.log(self.k) / (t * self.k)) if self.k > 1 else 0.0
        m = min(self.cum_loss)
        w = [math.exp(-eta * (c - m)) for c in self.cum_loss]
        z = sum(w)
        return [x / z for x in w]

    def select(self) -> int:
        p = self.probs()
        self._p = p
        u = self.rng.random()
        acc = 0.0
        for i, pi in enumerate(p):
            acc += pi
            if u < acc:
                return i
        return self.k - 1

    def update(self, arm: int, reward: float) -> None:
        p = self._p if self._p is not None else self.probs()
        y = min(max(float(reward), self.lo), self.hi)
        loss = (self.hi - y) / (self.hi - self.lo)
        self.cum_loss[arm] += loss / max(p[arm], 1e-12)
        self.t += 1
        self._p = None


class FixedArm:
    """Always plays one arm (ALG_i in the stopping-time regret)."""

    def __init__(self, arm: int):
        self.arm = int(arm)

    def select(self) -> int:
        return self.arm

    def update(self, arm: int, reward: float) -> None:
        pass


class UniformRandomArm:
    """Uniform random arm (a sanity baseline, not in the paper)."""

    def __init__(self, num_arms: int, seed: int = 0):
        self.k = int(num_arms)
        self.rng = random.Random(seed)

    def select(self) -> int:
        return self.rng.randrange(self.k)

    def update(self, arm: int, reward: float) -> None:
        pass


class BucketedPolicy:
    """One independent bandit per context bucket (e.g. the running batch size under load). Not in
    BanditSpec (a plain K-armed bandit; contextual / non-stationary variants are its future work);
    the batch size is host-known, so routing costs no sync."""

    def __init__(self, make_policy: Callable[[], object], edges: Sequence[int]):
        self.edges = sorted(int(e) for e in edges)
        self.make = make_policy
        self.policies: Dict[int, object] = {}
        self._last: Optional[int] = None

    def bucket(self, context: int) -> int:
        for i, e in enumerate(self.edges):
            if context <= e:
                return i
        return len(self.edges)

    def policy(self, context: int):
        b = self.bucket(context)
        if b not in self.policies:
            self.policies[b] = self.make()
        return self.policies[b]

    def select(self, context: int) -> int:
        return self.policy(context).select()

    def update(self, context: int, arm: int, reward: float) -> None:
        self.policy(context).update(arm, reward)


class TruncatedGeometricArms:
    """Synthetic stationary environment (BanditSpec Prop. 4.5): arm i draws
    ``Y = 1 + min(G, L_i)`` with ``Pr(G >= k) = p_i^k`` (accepted drafts + bonus)."""

    def __init__(self, p: Sequence[float], max_len: Union[int, Sequence[int]]):
        self.p = [float(x) for x in p]
        self.L = [int(max_len)] * len(self.p) if isinstance(max_len, int) else [int(x) for x in max_len]

    def mean(self, arm: int) -> float:
        return float(expected_accept_len(self.p[arm], self.L[arm]))

    def best_arm(self) -> int:
        return max(range(len(self.p)), key=lambda i: (self.mean(i), -i))

    def pull(self, arm: int, rng: random.Random) -> int:
        g = 0
        while g < self.L[arm] and rng.random() < self.p[arm]:
            g += 1
        return 1 + g


def run_stopping_time(policy, env, length: int, seed: int = 0) -> Tuple[int, List[int]]:
    """BanditSpec Alg. 3 with a length stop instead of EOS: rounds until ``length`` tokens are
    generated (the stopping time ST) and the per-arm pull counts."""
    rng = random.Random(seed)
    k = len(env.p)
    pulls = [0] * k
    total = 0
    rounds = 0
    while total < length:
        arm = policy.select()
        y = env.pull(arm, rng)
        policy.update(arm, y)
        pulls[arm] += 1
        total += y
        rounds += 1
    return rounds, pulls


def stopping_time_regret(make_policy: Callable[[int], object], env, length: int,
                         seeds: Sequence[int]) -> Tuple[float, float]:
    """Monte-Carlo estimate of BanditSpec Eq. 1-2: ``E[ST(ALG)] - E[ST(ALG_{i*})]`` (same seeds for
    both, so the reward streams are coupled). Returns (regret, E[ST(ALG_{i*})])."""
    best = env.best_arm()
    st_alg = [run_stopping_time(make_policy(s), env, length, seed=s)[0] for s in seeds]
    st_best = [run_stopping_time(FixedArm(best), env, length, seed=s)[0] for s in seeds]
    return (sum(st_alg) - sum(st_best)) / len(seeds), sum(st_best) / len(seeds)
