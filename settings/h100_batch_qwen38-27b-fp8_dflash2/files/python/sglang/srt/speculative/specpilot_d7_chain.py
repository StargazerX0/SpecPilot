"""SPECPILOT_D7: prefill as a teacher-forced verify chain (DFlash2 tree verify, greedy).

A newly admitted request with a short prompt ``p[0..P-1]`` skips the extend forward. Its prefill batch returns a
synthetic result (no target / draft forward): KV committed 0, pending root = ``p[0]``. It joins the running decode
batch at once, and every following verify step its tree slot is a forced chain of ``L <= N`` prompt tokens:
root ``p[s]`` (the previous step's bonus), nodes ``1..L-1`` = ``p[s+1..s+L-1]``. The chain is accepted as a whole:
the target predictions of the forced parents are rewritten to the forced tokens before the greedy tree accept, and
nodes ``>= L`` are masked. GDN / conv / KV commit follow the accepted chain, so the committed state is the prompt's
state. A non-final chunk's bonus is overridden with the next prompt token ``p[s+L]``; the final chunk's bonus is the
target argmax after ``p[P-1]``: the request's first output token. The scheduler emits nothing for chain steps and
the first output token at the final chunk (host ``kv_committed_len`` follows the device sequence length).

Knobs (read once; SPECPILOT_D7_CHAIN unset = the parent's behaviour and graphs):
  SPECPILOT_D7_CHAIN          1 = on (needs the DDTree verify, greedy requests)
  SPECPILOT_D7_CHAIN_MAXP     longest prompt (tokens) fed as a chain; longer prompts keep the extend (default 384)
  SPECPILOT_D7_CHAIN_WHEN     delayed (default): only admissions the min-free-slots delayer would hold back (the
                              running batch is near the request cap) enter as chains, at once; always: every eligible
                              admission (measured worse at c16/c20: the chain's slot-steps and tier rows cost more than
                              the extend it saves there)
  SPECPILOT_D7_CHAIN_SHARE    share of the step's extra verify rows (tier minus one root per request) the forced chains
                              may take when every other row is a normal request (default 0.5); the normal rows keep
                              (1 - share) pro rata to their count; every forced node must be verified, so chunks are cut
"""
from __future__ import annotations

import functools
import os
from typing import List, Optional

import msgspec
import torch
import triton
import triton.language as tl


class ChainConfig(msgspec.Struct, frozen=True):
    maxp: int
    always: bool
    share: float


class ChainState(msgspec.Struct):
    """Host cursor of one request's forced chain: prompt tokens already scheduled into verify steps."""

    prompt_len: int
    sched: int = 0


@functools.lru_cache(maxsize=None)
def chain_config() -> Optional[ChainConfig]:
    raw = os.environ.get("SPECPILOT_D7_CHAIN", "").strip().lower()
    if raw in ("", "0", "off", "none"):
        return None
    return ChainConfig(
        maxp=int(os.environ.get("SPECPILOT_D7_CHAIN_MAXP", "384")),
        always=os.environ.get("SPECPILOT_D7_CHAIN_WHEN", "delayed").strip().lower() == "always",
        share=float(os.environ.get("SPECPILOT_D7_CHAIN_SHARE", "0.5")),
    )


def chain_enabled() -> bool:
    return chain_config() is not None


def _prompt_fits(req) -> bool:
    cfg = chain_config()
    n = len(req.origin_input_ids)
    return cfg is not None and 2 <= n <= cfg.maxp


def req_eligible(req) -> bool:
    """A fresh greedy request whose whole prompt is this extend (no prefix, no chunking, no logprobs/grammar)."""
    if not _prompt_fits(req):
        return False
    sp = req.sampling_params
    greedy = sp.temperature == 0 or sp.top_k == 1
    rng = req.extend_range
    return (
        greedy
        and (chain_config().always or req.d7_chain_ok)
        and not req.return_logprob
        and not req.return_hidden_states
        and req.grammar is None
        and req.multimodal_inputs is None
        and rng is not None
        and rng.start == 0
        and rng.end == len(req.origin_input_ids)
    )


def batch_eligible(batch) -> bool:
    if not chain_enabled() or not batch.reqs or batch.decoding_reqs:
        return False
    return all(req_eligible(r) for r in batch.reqs)


def admit_delayed(waiting_queue) -> bool:
    """The min-free-slots delayer would hold the admission back: a chain admission costs no forward, so the
    waiting requests that fit enter now as chains (tagged); returns whether to admit."""
    cfg = chain_config()
    if cfg is None or not waiting_queue or not _prompt_fits(waiting_queue[0]):
        return False
    for req in waiting_queue:
        if _prompt_fits(req):
            req.d7_chain_ok = True
    return True


# ------------------------------------------------------------------------------------------------
# per-step forced-chain buffers (static: addresses baked into the captured draft graph)
# ------------------------------------------------------------------------------------------------


class ChainBuffers:
    RING = 4

    def __init__(self, max_bs: int, n_nodes: int, device):
        self.max_bs, self.N = int(max_bs), int(n_nodes)
        self.forced_len = torch.zeros((self.max_bs,), dtype=torch.int32, device=device)
        self.forced_tok = torch.zeros((self.max_bs, self.N), dtype=torch.int64, device=device)
        self.forced_next = torch.zeros((self.max_bs,), dtype=torch.int64, device=device)
        self.final = torch.zeros((self.max_bs,), dtype=torch.bool, device=device)
        # pinned host staging ring: a slot is reused only RING steps later (the host runs at most one step ahead)
        pin = torch.cuda.is_available()
        self._host = [
            tuple(
                torch.zeros(shape, dtype=dt, pin_memory=pin)
                for shape, dt in (
                    ((self.max_bs,), torch.int32),
                    ((self.max_bs, self.N), torch.int64),
                    ((self.max_bs,), torch.int64),
                    ((self.max_bs,), torch.bool),
                )
            )
            for _ in range(self.RING)
        ]
        self._slot = 0
        self._dirty = False
        self.active = False

    def stage(self, reqs, extra_rows: Optional[int] = None) -> bool:
        """Write this step's forced chains (rows = batch order) and advance the host cursors; returns whether any row
        carries a chain. ``extra_rows``: verify rows beyond one root per request this step (None = every node of every
        request is verified); the forced nodes of all chains fit in ``share`` of them (the budget policy ranks forced
        nodes first, so each is verified and the device commits exactly the chunk the host scheduled).
        No host sync: pinned non-blocking copies on the current stream."""
        assert len(reqs) <= self.max_bs
        cfg = chain_config()
        n_chain = sum(1 for r in reqs if r.d7_chain is not None and r.d7_chain.sched < r.d7_chain.prompt_len)
        if extra_rows is None:
            quota = self.N * len(reqs)
        else:  # the normal rows keep (1 - share) of the extra rows, pro rata to their count
            quota = int(extra_rows * (1.0 - (1.0 - cfg.share) * (len(reqs) - n_chain) / max(1, len(reqs))))
        rows = []
        for i, req in enumerate(reqs):
            st = req.d7_chain
            if st is None or st.sched >= st.prompt_len:
                continue
            rows.append((i, req, st))
        self.active = bool(rows)
        if not rows and not self._dirty:
            return False
        h_len, h_tok, h_next, h_fin = self._host[self._slot]
        self._slot = (self._slot + 1) % self.RING
        h_len.zero_()
        h_tok.zero_()
        h_next.zero_()
        h_fin.zero_()
        for i, req, st in rows:
            s = st.sched
            L = min(self.N, st.prompt_len - s, 1 + max(0, quota))
            quota -= L - 1
            h_len[i] = L
            h_tok[i, :L] = torch.as_tensor(list(req.origin_input_ids[s : s + L]), dtype=torch.int64)
            if s + L < st.prompt_len:
                h_next[i] = int(req.origin_input_ids[s + L])
            else:
                h_fin[i] = True
            st.sched = s + L
        self.forced_len.copy_(h_len, non_blocking=True)
        self.forced_tok.copy_(h_tok, non_blocking=True)
        self.forced_next.copy_(h_next, non_blocking=True)
        self.final.copy_(h_fin, non_blocking=True)
        self._dirty = self.active
        return self.active


MAX_BS = 256  # rows of the static buffers (above the decode graphs' max bs); arbitrary
_BUFS: Optional[ChainBuffers] = None


def buffers(n_nodes: int, device) -> ChainBuffers:
    global _BUFS
    if _BUFS is None:
        _BUFS = ChainBuffers(MAX_BS, n_nodes, device)
    assert _BUFS.N == int(n_nodes)
    return _BUFS


# ------------------------------------------------------------------------------------------------
# in-graph: forced rows of the best-first tree become chains (draft-graph tail hook, before the budget policy)
# ------------------------------------------------------------------------------------------------


@triton.jit
def _chain_tree_kernel(
    forced_len_ptr,
    forced_tok_ptr,
    tokens_ptr,
    parents_ptr,
    depth_ptr,
    next_tok_ptr,
    next_sib_ptr,
    mask_ptr,
    score_ptr,
    N: tl.constexpr,
    NP: tl.constexpr,
    HAS_SCORE: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    L = tl.load(forced_len_ptr + b)
    if L > 0:
        j = tl.arange(0, NP)
        valid = j < N
        base = b * N
        root = tl.load(tokens_ptr + base)
        ftok = tl.load(forced_tok_ptr + base + j, mask=valid, other=0)
        # nodes >= L carry the root token (a valid id); accept masks them, the budget policy never selects them
        tok = tl.where(j == 0, root, tl.where(j < L, ftok, root))
        tl.store(tokens_ptr + base + j, tok, mask=valid)
        tl.store(parents_ptr + base + j, (j - 1).to(tl.int64), mask=valid)
        tl.store(depth_ptr + base + j, j.to(tl.int64), mask=valid)
        tl.store(next_tok_ptr + base + j, tl.where(j + 1 < N, j + 1, -1).to(tl.int32), mask=valid)
        tl.store(next_sib_ptr + base + j, tl.full([NP], -1, tl.int32), mask=valid)
        ii = j[:, None]
        jj = j[None, :]
        tl.store(
            mask_ptr + b * N * N + ii * N + jj,
            (jj <= ii).to(tl.int8),
            mask=(ii < N) & (jj < N),
        )
        if HAS_SCORE:
            sc = tl.where(j == 0, 0.0, tl.where(j < L, 1.0, float("-inf")))
            tl.store(score_ptr + base + j, sc, mask=valid)


def apply_tree(tree: dict, bufs: ChainBuffers) -> None:
    """Overwrite the forced rows of ``tree`` (static buffers, first bs rows) with their chains."""
    bs, N = tree["tokens"].shape
    has_score = "score" in tree
    _chain_tree_kernel[(bs,)](
        bufs.forced_len,
        bufs.forced_tok,
        tree["tokens"],
        tree["parents"],
        tree["depth"],
        tree["next_token"],
        tree["next_sibling"],
        tree["mask"].view(torch.int8),
        tree["score"] if has_score else tree["depth"],
        N=N,
        NP=max(16, triton.next_power_of_2(N)),
        HAS_SCORE=has_score,
    )


# ------------------------------------------------------------------------------------------------
# eager, around the greedy tree accept
# ------------------------------------------------------------------------------------------------


def pre_accept(target_predict: torch.Tensor, candidates: torch.Tensor, bufs: ChainBuffers):
    """Forced rows: pred[j-1] := forced token of node j (1 <= j < L) and every node >= L masked, so the greedy tree
    accept takes exactly the chain; its bonus is the target argmax after the last forced token."""
    bs, N = candidates.shape
    L = bufs.forced_len[:bs, None].to(torch.int64)
    j = torch.arange(N, device=candidates.device)[None, :]
    nxt = torch.cat([bufs.forced_tok[:bs, 1:], bufs.forced_tok[:bs, :1]], dim=1)
    pred = torch.where((L > 0) & (j + 1 < L), nxt, target_predict)
    cand = torch.where((L > 0) & (j >= L), torch.full_like(candidates, -1), candidates)
    return pred, cand


def post_accept(tree_acc: dict, bufs: ChainBuffers, bs: int) -> None:
    """Non-final forced rows: the bonus (next root) is the next prompt token, in place in the accept buffers."""
    nf = (bufs.forced_len[:bs] > 0) & ~bufs.final[:bs]
    bonus = tree_acc["bonus"][:bs]
    bonus.copy_(torch.where(nf, bufs.forced_next[:bs], bonus))
    out = tree_acc["out_tokens"][:bs]
    idx = tree_acc["accept_len"][:bs].to(torch.int64)[:, None]
    out.scatter_(1, idx, torch.where(nf[:, None], bufs.forced_next[:bs, None], out.gather(1, idx)))


# ------------------------------------------------------------------------------------------------
# host: admission and result accounting
# ------------------------------------------------------------------------------------------------


def start_chains(reqs) -> List[int]:
    """Prefill batch taken by the chain: KV committed 0 (the device starts at seq_len 0); returns the roots p[0]."""
    roots = []
    for req in reqs:
        req.d7_chain = ChainState(prompt_len=len(req.origin_input_ids))
        req.kv.kv_committed_len = 0
        roots.append(int(req.origin_input_ids[0]))
    return roots


def resolve_tokens(req, accept_tokens: list) -> list:
    """Decode result of a chain step: the committed run (forced tokens + bonus) advances the host KV length; only the
    final chunk emits a token (its bonus, the first output token). Clears the chain at the final chunk."""
    req.kv.kv_committed_len += len(accept_tokens)
    if req.kv.kv_committed_len < req.d7_chain.prompt_len:
        return []
    req.d7_chain = None
    return accept_tokens[-1:]
