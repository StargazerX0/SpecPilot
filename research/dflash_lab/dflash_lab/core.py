"""Shared plumbing for dflash_lab experiments (one DFlash worker per process).

* ``hook(target, fn, kind)``: registers with SGLang's ``HookRegistry``
  (``replace`` / ``around`` / ``before`` / ``observe``) and logs the hook, so the boot
  gate records what a candidate patched. ``observe`` is an AFTER hook whose return value
  is discarded, so it cannot change results. Any target may be hooked: the plugin is part
  of the candidate snapshot, and what a candidate changes is judged by its numerical
  contract and validation obligations, not by a hook allowlist.
* A proposal pipeline on ``DFlashWorkerV2._refine_draft_tokens``: experiments
  append refiners ``fn(ctx)``; they run in registration order and may rewrite
  ``ctx.draft_tokens[:, 1:]`` of greedy rows.
* Optional per-process buffers, allocated before CUDA-graph capture:
  ``unary``/``cand`` (the drafter's top-K unary logits and candidate ids per
  draft slot, written inside the captured draft graph and by the eager
  fallback) and ``hist`` (per-request prompt + committed token ids, indexed by
  ``req_pool_indices`` and token position; positions ``<= anchor`` are always
  written by the current request, later positions may hold junk).
* Parameter fingerprints of the drafter and target at boot, and optionally
  every ``DFLASH_LAB_FINGERPRINT_EVERY`` decode steps (host sync; untimed runs).

Nothing in the decode path here syncs with the host.
"""
from __future__ import annotations

import os
import pkgutil
import sys

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

_W = "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2"
REFINE = f"{_W}._refine_draft_tokens"
STEP = f"{_W}.forward_batch_generation"
INIT = f"{_W}.init_cuda_graphs"
ACCEPT = f"{_W}._accept_block"
CANDIDATES = "sglang.srt.models.dflash.DFlash2DraftModel.compute_candidates"
LATTICE = "sglang.srt.models.dflash.CandidateSelector.build_lattice"
WALK = "sglang.srt.models.dflash.CandidateSelector.sample_path"

_KINDS = {"replace": HookType.REPLACE, "around": HookType.AROUND, "before": HookType.BEFORE,
          "observe": HookType.AFTER}


def log(msg: str) -> None:
    print(f"[dflash_lab] {msg}", file=sys.stderr, flush=True)


def hook(target: str, fn, kind: str) -> None:
    if kind not in _KINDS:
        raise ValueError(f"unknown hook kind {kind}")
    obj_path, attr = target.rsplit(".", 1)
    getattr(pkgutil.resolve_name(obj_path), attr)
    if kind == "observe":
        inner = fn

        def fn(result, *args, **kwargs):  # noqa: F811
            inner(result, *args, **kwargs)
            return None

    HookRegistry.register(target, fn, _KINDS[kind])
    log(f"hooked {kind} {target}")


class RefineContext:
    """What a refiner sees for one decode step (views of worker buffers)."""

    __slots__ = ("worker", "batch", "draft_tokens", "positions", "rows", "anchor_pos", "bs", "greedy")

    def __init__(self, worker, batch, draft_tokens, positions):
        self.worker = worker
        self.batch = batch
        self.draft_tokens = draft_tokens
        self.positions = positions
        self.rows = batch.req_pool_indices
        self.anchor_pos = positions[:, 0]
        self.bs = draft_tokens.shape[0]
        info = batch.sampling_info
        self.greedy = info is None or info.is_all_greedy


class LabState:
    def __init__(self):
        self.worker = None
        self.refiners = []
        self.init_callbacks = []
        self.step_observers = []
        self.pre_step = []
        self.want_unary = False
        self.want_hist = False
        self.unary = None
        self.cand = None
        self.hist = None
        self.cap_rows = 0
        self.decode_steps = 0
        self.last_refine = None
        self.fingerprint = None
        self.counters = {}
        self.extend_ids = None


STATE = LabState()
_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True
    hook(INIT, _before_init_cuda_graphs, "before")
    hook(REFINE, _refine, "replace")
    hook(STEP, _around_step, "around")


def add_refiner(fn) -> None:
    STATE.refiners.append(fn)


def on_init(fn) -> None:
    STATE.init_callbacks.append(fn)


def add_step_observer(fn) -> None:
    STATE.step_observers.append(fn)


def add_pre_step(fn) -> None:
    """``fn(worker=, batch=)`` before each decode step's draft (eager, before the draft graph
    replay); for staging per-row values into buffers the captured graph reads."""
    STATE.pre_step.append(fn)


def need_unary() -> None:
    if not STATE.want_unary:
        STATE.want_unary = True
        hook(CANDIDATES, _capture_candidates, "observe")


def add_counters(name: str, tensor: torch.Tensor, labels) -> None:
    """GPU int64 accumulators, printed (with a sync) only at fingerprint checks."""
    STATE.counters[name] = (tensor, list(labels))


def need_hist() -> None:
    STATE.want_hist = True


def unary_top1_conf(bs: int) -> torch.Tensor:
    """[bs, S] softmax mass of each draft slot's top-1 unary candidate (top-K normalized)."""
    num_slots = STATE.worker.block_size - 1
    return torch.softmax(STATE.unary[: bs * num_slots], dim=-1)[:, 0].view(bs, num_slots)


# --- core hooks -------------------------------------------------------------------------

def _before_init_cuda_graphs(worker, *args, **kwargs):
    STATE.worker = worker
    from sglang.srt.arg_groups.overrides import resolving_view

    device = worker.device
    server_args = resolving_view(worker.server_args)
    if STATE.want_hist and not server_args.disable_radix_cache:
        raise RuntimeError("dflash_lab token history requires --disable-radix-cache")
    k = worker.draft_model.candidate_selector.top_k
    mrr = int(server_args.max_running_requests or 64)
    STATE.cap_rows = (worker.block_size - 1) * max(64, mrr)
    if STATE.want_unary:
        STATE.unary = torch.zeros((STATE.cap_rows, k), dtype=torch.float32, device=device)
        STATE.cand = torch.zeros((STATE.cap_rows, k), dtype=torch.int64, device=device)
    if STATE.want_hist:
        rtt = worker.model_runner.req_to_token_pool.req_to_token
        STATE.hist = torch.zeros(tuple(rtt.shape), dtype=torch.int32, device=device)
        log(f"token history {tuple(rtt.shape)} int32 ({STATE.hist.numel() * 4 / 2**20:.1f} MiB)")
    for fn in STATE.init_callbacks:
        fn(worker)
    STATE.fingerprint = _fingerprints(worker)
    log("fingerprint " + " ".join(f"{k}={v}" for k, v in STATE.fingerprint.items()))
    return None


def _capture_candidates(result, *args, **kwargs):
    ids, vals = result
    n = ids.shape[0]
    if STATE.unary is not None and n <= STATE.cap_rows:
        STATE.unary[:n].copy_(vals)
        STATE.cand[:n].copy_(ids)


def _refine(worker, *, draft_tokens, batch, positions):
    ctx = RefineContext(worker, batch, draft_tokens, positions)
    if STATE.hist is not None:
        STATE.hist[ctx.rows.long(), ctx.anchor_pos.long()] = draft_tokens[:, 0].to(torch.int32)
    for fn in STATE.refiners:
        fn(ctx)
    STATE.last_refine = ctx
    return None


def _around_step(original, worker, batch=None, *args, **kwargs):
    """Returns the worker's result unchanged. The batch is classified before the call:
    the decode path rewrites ``batch.forward_mode`` to TARGET_VERIFY during verify."""
    if batch is None:
        batch = kwargs.pop("batch")
    if batch.forward_mode.is_idle():
        return original(worker, batch, *args, **kwargs)
    is_extend = batch.forward_mode.is_extend() or batch.is_extend_in_batch
    extend_ids = batch.input_ids if is_extend else None
    if not is_extend:
        for fn in STATE.pre_step:
            fn(worker=worker, batch=batch)
    result = original(worker, batch, *args, **kwargs)
    _after_step(result, worker, batch, is_extend, extend_ids)
    return result


def _after_step(result, worker, batch, is_extend, extend_ids):
    ctx = None
    if not is_extend:
        ctx, STATE.last_refine = STATE.last_refine, None
        if ctx is None or ctx.batch is not batch:
            raise RuntimeError("dflash_lab: decode step without a matching refine call")
        STATE.decode_steps += 1
    STATE.extend_ids = extend_ids
    if STATE.hist is not None:
        if is_extend:
            _hist_extend(batch, result, extend_ids)
        else:
            _hist_commit(ctx, result)
    for fn in STATE.step_observers:
        fn(worker=worker, batch=batch, result=result, ctx=ctx, is_extend=is_extend)
    every = int(os.environ.get("DFLASH_LAB_FINGERPRINT_EVERY", "0"))
    if every and not is_extend and STATE.decode_steps % every == 0:
        now = _fingerprints(worker)
        if now != STATE.fingerprint:
            log(f"FATAL: parameter fingerprint changed at step {STATE.decode_steps}: {now}")
            os._exit(4)
        log(f"fingerprint-ok step={STATE.decode_steps}")
        for name, (t, labels) in STATE.counters.items():
            log(f"activity {name} step={STATE.decode_steps} " + " ".join(f"{k}={v}" for k, v in zip(labels, t.tolist())))


def _hist_extend(batch, result, extend_ids):
    hist = STATE.hist
    device = hist.device
    ext = [int(x) for x in batch.extend_lens]
    total = sum(ext)
    ext_t = torch.tensor(ext, dtype=torch.int64, device=device)
    pre_t = torch.tensor([int(x) for x in batch.prefix_lens], dtype=torch.int64, device=device)
    rows = batch.req_pool_indices.long()
    seg = torch.repeat_interleave(torch.arange(len(ext), device=device), ext_t, output_size=total)
    starts = torch.cumsum(ext_t, 0) - ext_t
    pos = pre_t[seg] + torch.arange(total, device=device) - starts[seg]
    if extend_ids.shape[0] != total:
        raise RuntimeError(f"dflash_lab: extend input_ids {tuple(extend_ids.shape)} != sum(extend_lens) {total}")
    hist[rows[seg], pos.clamp_max(hist.shape[1] - 1)] = extend_ids.to(torch.int32)
    bonus_pos = (pre_t + ext_t).clamp_max(hist.shape[1] - 1)
    hist[rows, bonus_pos] = result.next_token_ids.view(-1)[: len(ext)].to(torch.int32)


def _hist_commit(ctx, result):
    hist = STATE.hist
    block = ctx.positions.shape[1]
    out = result.next_token_ids.view(ctx.bs, block)
    # All block entries, unconditionally: those past commit_len sit beyond the next
    # anchor and are overwritten before any read.
    pos = (ctx.anchor_pos.long()[:, None] + 1 + torch.arange(block, device=hist.device)).clamp_max(hist.shape[1] - 1)
    hist[ctx.rows.long()[:, None], pos] = out.to(torch.int32)


def _fingerprints(worker) -> dict:
    target = worker.target_worker.model_runner.model
    # The worker attaches the target lm_head to the draft model inside init_cuda_graphs (after
    # the boot fingerprint); shared storage is covered by the target fingerprint.
    shared = {p.data_ptr() for p in target.parameters()}
    return {"draft": _fp(worker.draft_model, skip=shared), "target": _fp(target)}


@torch.no_grad()
def _fp(module, skip=frozenset()) -> str:
    acc = None
    params = [(n, p) for n, p in module.named_parameters() if p.data_ptr() not in skip]
    for _, p in sorted(params, key=lambda kv: kv[0]):
        t = p.detach().reshape(-1)
        size = t.element_size()
        v = t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[size])
        for chunk in v.split(1 << 24):
            c = chunk.to(torch.int64)
            s = torch.stack([c.sum(), (c * c).sum()])
            acc = s if acc is None else acc + s
    return "none" if acc is None else "%x:%x" % tuple(int(x) & (2**64 - 1) for x in acc.tolist())
