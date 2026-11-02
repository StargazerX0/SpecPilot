"""Relaxed (approximate, lossy) acceptance rules for greedy DFlash2 chain verification.

Reference implementation for the ``approximate`` contract family
(``research/knowledge/search_space/02_relaxed_verification.md``). Pure PyTorch, batched,
vectorised: no host syncs, no ``.item()``/``.tolist()``, no data-dependent shapes, Python
control flow only on the (static) config. Every output has a shape fixed by ``[bs, n]``,
so the functions can run inside the eager ``_accept_block`` of a timed run or be captured
in a CUDA graph.

Layout (identical to the engine's verify block, ``EXECUTION_PATH.md`` section 1 step 10)::

    candidates    [bs, n]      int   column 0 = anchor (previous bonus), columns 1..n-1 = drafts
    target_logits [bs, n, V]   float row j = target logits after candidates[:, :j+1]
                  (or [bs*n, V], the engine's ``next_token_logits``)

Draft slot ``j`` (0 <= j < D = n-1) is token ``candidates[:, j+1]``, judged against row ``j``.
Row ``n-1`` only supplies the bonus when every draft is accepted.

Stock rule (exact greedy, ``_dflash_accept_bonus_contig_kernel``)::

    accept_len = longest prefix with candidates[:, 1+j] == argmax(target_logits[:, j])
    bonus      = argmax(target_logits[:, accept_len])
    out_tokens = candidates[:, 1:] with bonus written at accept_len   commit_len = accept_len + 1

Relaxed rules keep that skeleton and only enlarge the per-slot "ok" set: an exact match is
always ok; a *mismatch* is ok when the configured rule marks it eligible and the block cap,
the per-request budget and the suffix policy allow it. The accepted length is the longest
all-ok prefix (contiguity is essential: target row j is conditioned on every earlier
drafted token, so a gap would desynchronise KV / GDN state from the emitted text). The
bonus is the target argmax at row ``accept_len``, which is conditioned on the realized
(possibly relaxed) prefix, so the engine's existing commit path (``_commit_accept``,
``commit_lens - 1`` GDN scatter, prefix-valid draft-KV append) stays consistent unchanged.

Rules (``RelaxedConfig.rule``), all evaluated on mismatching slots only:

* ``"greedy"``   exact token-id greedy (stock). Never relaxes.
* ``"lenience"`` Leviathan et al. 2023, App. A.5 (greedy variant): accept x if
                 ``p(x) >= lam * max_v p(v)`` with ``p = softmax(z / T)``; equivalently
                 ``(z_max - z_x) / T <= -log(lam)``. ``lam >= 1`` is the strict limit (off).
* ``"typical"``  Medusa typical acceptance (Cai et al. 2024, section 2.3.1): accept x if
                 ``p(x) > min(eps, delta * exp(-H(p)))``. Strict limit: ``eps >= 1`` with
                 ``delta = inf`` (a mismatch always has ``p(x) <= 0.5``).
* ``"topk"``     top-k membership + margin: accept x if its tie-broken rank in the target row is
                 ``< top_k`` and ``z_max - z_x <= max_gap`` and ``log p(x) >= min_logprob``.
                 ``top_k = 1`` is exactly greedy (rank 0 <=> x == argmax, first-index ties).
* ``"mars"``     MARS (Song et al. 2026, Alg. 1): accept x only if it is the target runner-up
                 (tie-broken rank 1) and the raw-logit ratio ``z_2 / z_1 > theta`` with ``z_1 > 0``.
                 ``theta >= 1`` is the strict limit. Not shift-invariant in the logits.
* ``"asd"``      ASD regret gate (Feng et al. 2026, Eq. 7): accept x if
                 ``(z_max - z_x) / (D - j) <= g`` (remaining-proposal-length normalisation).
                 Combine with ``cost="regret"``, ``block_cap`` and a ``budget`` ledger for full
                 ASD (Eqs. 9-12). Budget ``<= 0`` dispatches the row to exact greedy.

Shared gates (any rule): target-entropy window, slot window, protected token ids (never
relax when the target argmax or the drafted token is protected, e.g. EOS / ``</think>`` /
``<|im_end|>``), per-block cap on relaxed accepts, per-request budget (error ledger) with
optional per-token refill (window scope), and suffix retention.

Suffix retention (``suffix_min_run = m``): a relaxed accept at slot j is retained only if it
is followed by at least ``m`` consecutive *exact* matches (slots j+1..j+m, evaluated
under the realized prefix, which is what the teacher-forced verify rows are). If the exact
run reaches the end of the block before m matches, ``suffix_end_ok`` decides. ``m = 0`` is
ASD-style "longest feasible prefix" (a block may end on a relaxed token). With ``m >= 1`` a
relaxation is only ever spent when it unlocks at least m additional target-greedy tokens,
which is the "confidence-gated suffix retention" policy of the search-space contract.

Nothing here is lossless except ``rule="greedy"`` (and every rule at its strict limit).
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional, Union

import msgspec
import torch

Number = Union[float, int, torch.Tensor]

RULES = ("greedy", "lenience", "typical", "topk", "mars", "asd")
COSTS = ("count", "regret")


class RelaxedConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Static configuration of a relaxed acceptance rule.

    Float thresholds may also be tensors broadcastable to ``[D]`` (a per-slot schedule) or
    ``[bs, D]`` (per-request / per-slot); they are never read on the host.
    Defaults are the strict limit of every rule, i.e. exact greedy.
    """

    rule: str = "greedy"
    # Temperature used to form p = softmax(z / T) for probability-based rules and gates.
    # The emitted tokens stay greedy; T only shapes the judging distribution.
    temperature: float = 1.0
    # lenience (Leviathan A.5): accept if p(x) >= lam * max p.
    lenience: Number = 1.0
    # typical (Medusa): accept if p(x) > min(eps, delta * exp(-H)).
    typical_eps: Number = 1.0
    typical_delta: Number = float("inf")
    # topk: rank(x) < top_k and z_max - z_x <= max_gap and log p(x) >= min_logprob.
    top_k: int = 1
    max_gap: Number = float("inf")
    min_logprob: Number = float("-inf")
    # mars: x is the target runner-up and z_2 / z_1 > mars_theta (with z_1 > 0).
    mars_theta: Number = 1.0
    # asd: (z_max - z_x) / (D - j) <= asd_gate.
    asd_gate: Number = float("inf")
    # Shared gates.
    min_entropy: Number = float("-inf")  # relax only where H(p) >= min_entropy (nats)
    max_entropy: Number = float("inf")   # ... and H(p) <= max_entropy
    min_slot: int = 0                    # relax only slots j in [min_slot, max_slot)
    max_slot: Optional[int] = None
    block_cap: Optional[int] = None      # max relaxed accepts per block (ASD's M)
    cost: str = "count"                  # ledger cost of one relaxed accept: 1 or z_max - z_x
    suffix_min_run: int = 0              # retain a relaxed accept only if >= m exact matches follow
    suffix_end_ok: bool = True           # ... or the exact run reaches the end of the block
    refill_per_token: float = 0.0        # budget += refill * commit_len after each block
    budget_cap: float = float("inf")     # ... clipped to this value

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"unknown rule {self.rule!r}; expected one of {RULES}")
        if self.cost not in COSTS:
            raise ValueError(f"unknown cost {self.cost!r}; expected one of {COSTS}")
        if not (isinstance(self.temperature, (int, float)) and self.temperature > 0):
            raise ValueError("temperature must be a positive float")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.suffix_min_run < 0:
            raise ValueError("suffix_min_run must be >= 0")
        if self.block_cap is not None and self.block_cap < 0:
            raise ValueError("block_cap must be >= 0 or None")


GREEDY = RelaxedConfig()


class SlotSignals(NamedTuple):
    """Per-slot verify signals, all ``[bs, D]`` except ``top1`` (``[bs, n]``).

    ``probs_valid`` is False when the probability fields were not computed (they are then
    zero-filled placeholders of the right shape).
    """

    top1: torch.Tensor        # int64 [bs, n] target argmax per row (first index on ties)
    match: torch.Tensor       # bool  [bs, D] draft == argmax
    regret: torch.Tensor      # f32   [bs, D] z_max - z_x >= 0 (= log-prob gap at T=1)
    rank: torch.Tensor        # int64 [bs, D] tie-broken rank of x (0 <=> x == argmax)
    logp_x: torch.Tensor      # f32   [bs, D] log p(x) at cfg.temperature
    entropy: torch.Tensor     # f32   [bs, D] H(p) in nats at cfg.temperature
    top2_gap: torch.Tensor    # f32   [bs, D] z_top1 - z_top2 (target decisiveness)
    zmax: torch.Tensor        # f32   [bs, D] z_top1 (raw logit; MARS needs the scale)
    probs_valid: bool


class RelaxedAcceptResult(NamedTuple):
    accept_len: torch.Tensor     # int32 [bs]   accepted drafts (excludes anchor and bonus)
    commit_lens: torch.Tensor    # int32 [bs]   accept_len + 1
    bonus: torch.Tensor          # int64 [bs]   argmax at row accept_len
    out_tokens: torch.Tensor     # int64 [bs, n] accepted drafts, bonus at accept_len (engine layout)
    relaxed_mask: torch.Tensor   # bool  [bs, D] accepted slots whose token != target argmax
    num_relaxed: torch.Tensor    # int32 [bs]
    spent: torch.Tensor          # f32   [bs]   ledger cost spent in this block
    budget_out: Optional[torch.Tensor]  # f32 [bs] remaining budget (None when no budget given)
    strict_accept_len: torch.Tensor     # int32 [bs] what stock greedy would have accepted


# --------------------------------------------------------------------------------------
# Helpers


def _as_3d(target_logits: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be [bs, n], got {tuple(candidates.shape)}")
    bs, n = candidates.shape
    if n < 2:
        raise ValueError("a verify block needs the anchor plus at least one draft (n >= 2)")
    if target_logits.ndim == 2:
        if target_logits.shape[0] != bs * n:
            raise ValueError(
                f"2-D logits must be [bs*n, V] = [{bs * n}, V], got {tuple(target_logits.shape)}"
            )
        return target_logits.view(bs, n, -1)
    if target_logits.ndim != 3 or tuple(target_logits.shape[:2]) != (bs, n):
        raise ValueError(f"logits must be [bs, n, V], got {tuple(target_logits.shape)}")
    return target_logits


def _slotwise(v: Number, like: torch.Tensor) -> torch.Tensor:
    """Float threshold or tensor -> float32 tensor broadcastable against ``like`` [bs, D]."""
    if isinstance(v, torch.Tensor):
        return v.to(device=like.device, dtype=torch.float32)
    return torch.full((), float(v), device=like.device, dtype=torch.float32)


def _is_trivial(v: Number, trivial: float) -> bool:
    """True only for a Python number equal to ``trivial`` (a tensor is never trivial)."""
    return not isinstance(v, torch.Tensor) and float(v) == trivial


def _needs_probs(cfg: RelaxedConfig) -> bool:
    if cfg.rule == "typical":
        return True
    return not (
        _is_trivial(cfg.min_logprob, float("-inf"))
        and _is_trivial(cfg.min_entropy, float("-inf"))
        and _is_trivial(cfg.max_entropy, float("inf"))
    )


def commit_tokens(candidates: torch.Tensor, accept_len: torch.Tensor, bonus: torch.Tensor):
    """``_commit_accept`` / ``_dflash_accept_bonus_contig_kernel`` layout: drafts shifted left,
    bonus at the accept boundary, last column 0 unless the bonus lands there. Entries after the
    bonus keep the shifted drafts; the scheduler reads only the first ``commit_lens`` entries."""
    bs, n = candidates.shape
    cols = torch.arange(n, device=candidates.device)
    out = torch.zeros((bs, n), dtype=torch.int64, device=candidates.device)
    out[:, :-1] = candidates[:, 1:].to(torch.int64)
    acc = accept_len.to(torch.int64)[:, None]
    out = torch.where(cols[None, :] == acc, bonus.to(torch.int64)[:, None], out)
    return out, accept_len.to(torch.int32) + 1


def _leading_true(mask: torch.Tensor) -> torch.Tensor:
    """Length of the all-True prefix of each row of a bool [bs, D] mask (int64 [bs])."""
    return mask.to(torch.int64).cumprod(dim=-1).sum(dim=-1)


def _run_after(match: torch.Tensor) -> torch.Tensor:
    """run[b, j] = number of consecutive exact matches at slots j+1, j+2, ... (int64)."""
    bs, d = match.shape
    idx = torch.arange(d, device=match.device).expand(bs, d)
    # next_false[i] = smallest k >= i with match[k] False, else d.
    nf = torch.where(match, torch.full_like(idx, d), idx)
    next_false = torch.cummin(nf.flip(-1), dim=-1).values.flip(-1)
    run_from = next_false - idx  # consecutive matches starting at i
    pad = torch.zeros((bs, 1), dtype=run_from.dtype, device=match.device)
    return torch.cat([run_from[:, 1:], pad], dim=-1)


# --------------------------------------------------------------------------------------
# Signals and rules


def slot_signals(
    target_logits: torch.Tensor,
    candidates: torch.Tensor,
    *,
    temperature: float = 1.0,
    with_probs: bool = True,
) -> SlotSignals:
    """Per-slot statistics of the drafted tokens under the target's verify rows.

    Everything is computed from the verify logits the engine already has
    (``next_token_logits`` after penalties / grammar masking); no extra forward.
    """
    z_all = _as_3d(target_logits, candidates)
    bs, n, vocab = z_all.shape
    d = n - 1
    top1 = torch.argmax(z_all, dim=-1)  # [bs, n]; first index among ties, like the engine
    z = z_all[:, :d, :].float()
    x = candidates[:, 1:].to(torch.int64)
    zx = z.gather(-1, x[..., None]).squeeze(-1)
    top2 = torch.topk(z, k=min(2, vocab), dim=-1).values
    zmax = top2[..., 0]
    top2_gap = (top2[..., 0] - top2[..., 1]) if vocab > 1 else torch.full_like(zmax, float("inf"))
    regret = zmax - zx
    vid = torch.arange(vocab, device=z.device)
    rank = (z > zx[..., None]).sum(-1) + (
        (z == zx[..., None]) & (vid < x[..., None])
    ).sum(-1)
    match = x == top1[:, :d]
    if with_probs:
        logp = torch.log_softmax(z / float(temperature), dim=-1)
        logp_x = logp.gather(-1, x[..., None]).squeeze(-1)
        entropy = torch.special.entr(logp.exp()).sum(-1)
    else:
        logp_x = torch.zeros_like(regret)
        entropy = torch.zeros_like(regret)
    return SlotSignals(top1, match, regret, rank, logp_x, entropy, top2_gap, zmax, with_probs)


def eligible(sig: SlotSignals, cfg: RelaxedConfig) -> torch.Tensor:
    """bool [bs, D]: slots whose token the rule would accept *as a relaxed (non-argmax) token*.

    Only rule thresholds and the per-slot gates are applied here; caps, budget and suffix
    retention act in ``relaxed_accept``. Exact matches are never "eligible" (they are
    accepted unconditionally and cost nothing).
    """
    mismatch = ~sig.match
    bs, d = mismatch.shape
    t = float(cfg.temperature)
    if cfg.rule == "greedy":
        return torch.zeros_like(mismatch)
    if cfg.rule == "lenience":
        lam = _slotwise(cfg.lenience, sig.regret)
        # p(x)/p_max = exp(-regret/T) >= lam, and lam >= 1 means "off" (strict limit).
        ok = (-sig.regret / t >= torch.log(lam.clamp_min(0.0))) & (lam < 1.0)
    elif cfg.rule == "typical":
        if not sig.probs_valid:
            raise ValueError("typical acceptance needs probabilities")
        eps = _slotwise(cfg.typical_eps, sig.regret)
        delta = _slotwise(cfg.typical_delta, sig.regret)
        thr = torch.minimum(eps.expand_as(sig.entropy), delta * torch.exp(-sig.entropy))
        ok = sig.logp_x.exp() > thr
    elif cfg.rule == "topk":
        ok = (sig.rank < int(cfg.top_k)) & (sig.regret <= _slotwise(cfg.max_gap, sig.regret))
    elif cfg.rule == "mars":
        z1 = sig.zmax
        ratio = (z1 - sig.top2_gap) / torch.where(z1 > 0, z1, torch.ones_like(z1))
        ok = (sig.rank == 1) & (z1 > 0) & (ratio > _slotwise(cfg.mars_theta, sig.regret))
    elif cfg.rule == "asd":
        remaining = (d - torch.arange(d, device=sig.regret.device)).to(torch.float32)  # q_i = K-i+1
        ok = sig.regret / remaining <= _slotwise(cfg.asd_gate, sig.regret)
    else:  # pragma: no cover - guarded by RelaxedConfig
        raise ValueError(cfg.rule)
    if not _is_trivial(cfg.min_logprob, float("-inf")):
        ok = ok & (sig.logp_x >= _slotwise(cfg.min_logprob, sig.regret))
    if not _is_trivial(cfg.min_entropy, float("-inf")):
        ok = ok & (sig.entropy >= _slotwise(cfg.min_entropy, sig.regret))
    if not _is_trivial(cfg.max_entropy, float("inf")):
        ok = ok & (sig.entropy <= _slotwise(cfg.max_entropy, sig.regret))
    j = torch.arange(d, device=mismatch.device)
    window = j >= int(cfg.min_slot)
    if cfg.max_slot is not None:
        window = window & (j < int(cfg.max_slot))
    # A non-finite regret means the drafted token is masked (-inf: grammar / bias): never relax.
    return ok & mismatch & window[None, :] & torch.isfinite(sig.regret)


# --------------------------------------------------------------------------------------
# Accept


def greedy_accept(target_logits: torch.Tensor, candidates: torch.Tensor) -> RelaxedAcceptResult:
    """Exact stock rule (``compute_dflash_correct_drafts_and_bonus`` + ``_commit_accept``)."""
    z = _as_3d(target_logits, candidates)
    bs, n = candidates.shape
    top1 = torch.argmax(z, dim=-1)
    match = candidates[:, 1:].to(torch.int64) == top1[:, :-1]
    accept_len = _leading_true(match)
    bonus = top1.gather(1, accept_len[:, None]).squeeze(1)
    out, commit = commit_tokens(candidates, accept_len, bonus)
    zeros_f = torch.zeros(bs, dtype=torch.float32, device=candidates.device)
    return RelaxedAcceptResult(
        accept_len.to(torch.int32), commit, bonus.to(torch.int64), out,
        torch.zeros_like(match), torch.zeros(bs, dtype=torch.int32, device=candidates.device),
        zeros_f, None, accept_len.to(torch.int32),
    )


def relaxed_accept(
    target_logits: torch.Tensor,
    candidates: torch.Tensor,
    cfg: RelaxedConfig = GREEDY,
    *,
    budget: Optional[torch.Tensor] = None,
    protect_ids: Optional[torch.Tensor] = None,
    signals: Optional[SlotSignals] = None,
) -> RelaxedAcceptResult:
    """Accept length, bonus and committed tokens under a relaxed greedy rule.

    ``target_logits`` are the verify logits after penalties / grammar masking, ``[bs, n, V]``
    or ``[bs*n, V]``. ``budget`` is the optional per-request remaining error budget ``[bs]``
    (the ledger, e.g. gathered from a per-request buffer by ``req_pool_indices``); rows with
    budget ``<= 0`` are verified exactly, ``cfg.cost`` sets what one relaxed accept costs, and
    the updated ledger comes back in ``budget_out`` for the caller to scatter. ``protect_ids``
    ``[P]`` are token ids never relaxed, as target argmax or as drafted token (EOS, stop and
    think-close ids). ``signals`` may be precomputed by ``slot_signals`` (with probabilities
    if the rule or a gate needs them). All outputs have static shapes.
    """
    z = _as_3d(target_logits, candidates)
    bs, n = candidates.shape
    d = n - 1
    dev = candidates.device
    if signals is None:
        signals = slot_signals(z, candidates, temperature=cfg.temperature, with_probs=_needs_probs(cfg))
    sig = signals
    match = sig.match
    strict_len = _leading_true(match)

    elig = eligible(sig, cfg)
    if protect_ids is not None and protect_ids.numel() > 0:
        pid = protect_ids.to(device=dev, dtype=torch.int64)
        x = candidates[:, 1:].to(torch.int64)
        prot = torch.isin(x, pid) | torch.isin(sig.top1[:, :d], pid)
        elig = elig & ~prot
    if budget is not None:
        if budget.shape != (bs,):
            raise ValueError(f"budget must be [bs]={bs}, got {tuple(budget.shape)}")
        budget_f = budget.to(device=dev, dtype=torch.float32)
        elig = elig & (budget_f > 0)[:, None]  # ASD: B - s == 0 -> exact greedy dispatch
    if cfg.suffix_min_run > 0:
        run = _run_after(match)
        j = torch.arange(d, device=dev)
        reaches_end = (j[None, :] + 1 + run) >= d
        keep = run >= int(cfg.suffix_min_run)
        if cfg.suffix_end_ok:
            keep = keep | reaches_end
        elig = elig & keep

    ok = match | elig
    mismatch = ~match
    feasible = ok
    if cfg.block_cap is not None:
        n_relaxed_prefix = torch.cumsum(mismatch.to(torch.int64), dim=-1)
        feasible = feasible & (n_relaxed_prefix <= int(cfg.block_cap))
    if cfg.cost == "regret":
        unit = torch.where(torch.isfinite(sig.regret), sig.regret, torch.zeros_like(sig.regret))
    else:
        unit = torch.ones_like(sig.regret)
    cost = torch.where(mismatch, unit, torch.zeros_like(unit))
    if budget is not None:
        spent_prefix = torch.cumsum(cost, dim=-1)
        # A mismatch also needs budget left *before* it (ASD repo, dspark torch_rule), so a
        # zero-regret tie is not free once an earlier token exhausted the budget exactly.
        spent_before = spent_prefix - cost
        affordable = match | (
            (spent_prefix <= budget_f[:, None]) & (spent_before < budget_f[:, None])
        )
        feasible = feasible & affordable

    accept_len = _leading_true(feasible)
    cols = torch.arange(d, device=dev)
    accepted = cols[None, :] < accept_len[:, None]
    relaxed_mask = accepted & mismatch
    spent = torch.where(relaxed_mask, cost, torch.zeros_like(cost)).sum(-1)
    bonus = sig.top1.gather(1, accept_len[:, None]).squeeze(1)
    out, commit = commit_tokens(candidates, accept_len, bonus)
    budget_out = None
    if budget is not None:
        budget_out = budget_f - spent
        if cfg.refill_per_token != 0.0:
            budget_out = budget_out + float(cfg.refill_per_token) * commit.to(torch.float32)
        if math.isfinite(cfg.budget_cap):
            budget_out = budget_out.clamp_max(float(cfg.budget_cap))
    return RelaxedAcceptResult(
        accept_len.to(torch.int32),
        commit,
        bonus.to(torch.int64),
        out,
        relaxed_mask,
        relaxed_mask.sum(-1).to(torch.int32),
        spent,
        budget_out,
        strict_len.to(torch.int32),
    )


def asd_config(g: float = 0.25, block_cap: int = 2, **kw) -> RelaxedConfig:
    """Full ASD (Feng et al. 2026, Alg. 1): regret gate g, block cap M, regret-cost ledger.

    Pass the request budget B - s as ``budget=`` to ``relaxed_accept``. The paper's frozen
    operating point (B, g, M) = (8, 0.25, 2) was tuned for Qwen3-14B + DSpark-14B; it is a
    starting point for dev-split tuning here, not a validated setting.
    """
    return RelaxedConfig(rule="asd", asd_gate=g, block_cap=block_cap, cost="regret", **kw)


__all__ = [
    "RULES",
    "COSTS",
    "RelaxedConfig",
    "GREEDY",
    "SlotSignals",
    "RelaxedAcceptResult",
    "slot_signals",
    "eligible",
    "greedy_accept",
    "relaxed_accept",
    "commit_tokens",
    "asd_config",
]
