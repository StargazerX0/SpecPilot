"""SPECPILOT_D8R3_RBLOCK: per-request drafter block lengths packed into <= 256 drafter rows (LINE E).

At graph batch sizes 17..24 the seed drafter runs bs x 16 rows (272..384), past the DeepGEMM M=256 tile
cliff of the 5-layer FP8 stack and the FP8 candidate head. Here every request gets its own block length
L_b in {16, 12, 8}, chosen on the GPU from the previous steps (mix = prev2 / 5 + pconf / 3: mean of the
last two commit counts and the greedy 7-slot chain log-prob of the request's previous lattice); the
table entry for bs fixes how many requests get 16 / 12 / 8 rows so that the packed row count M is a
static <= 256 per captured graph. The drafter then runs once on the packed rows:

- rows of request b: anchor (bonus) + L_b - 1 mask tokens at positions P_b + j, KV written to
  req_to_token[rpi_b, P_b + j] (the parent's verify locations), per-row in-block position for the
  grouped conv (``rowpos >= tap`` instead of ``arange % 16 >= tap``);
- attention: direct FA3 varlen (cu_seqlens_q from the block offsets, cache_seqlens P_b + L_b,
  non-causal, the drafter's (window, window)), the same kernel/arguments as the draft backend's
  TARGET_VERIFY path except for the ragged query lengths;
- head + top-K on the M - bs non-anchor rows; candidates, unary logits and hidden rows scattered into
  the dense [bs, 15, ...] lattice inputs (slots >= L_b - 1 hold stale finite values) and the tree
  builder caps each request's depth at L_b - 1 slots (``build_tree(slots=...)``), so no node is ever
  taken from a slot beyond its request's block. Verification is unchanged (the drafter only proposes).

bs <= 16 (and any bs not in the table) keeps the parent's draft graph; only a one-launch state
update runs there (commit tracking). Knob unset = parent (nothing is built or launched).

Knob (read once): SPECPILOT_D8R3_RBLOCK = '1' (default table below) or 'bs:n16,n12;bs:n16,n12;...'
(n8 = bs - n16 - n12; 16 n16 + 12 n12 + 8 n8 <= 256).
"""
from __future__ import annotations

import functools
import logging
import os
import types
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.utils.common import get_compiler_backend

logger = logging.getLogger(__name__)

BLOCK = 16
MAX_ROWS = 256
# bs -> (n16, n12); n8 = bs - n16 - n12; every entry packs to exactly 256 rows.
DEFAULT_TABLE = {17: (13, 4), 18: (10, 8), 19: (7, 12), 20: (6, 12), 21: (5, 12), 22: (4, 12),
                 23: (4, 10), 24: (4, 8)}
PCONF_SLOTS = 7  # greedy chain depth of the policy signal (every block has >= 7 slots)
PCONF_INIT = -3.0
PREV_INIT = 5.0


def parse_table(raw: str) -> Optional[Dict[int, Tuple[int, int]]]:
    raw = (raw or "").strip()
    if raw in ("", "0", "off"):
        return None
    if raw == "1":
        table = dict(DEFAULT_TABLE)
    else:
        table = {}
        for ent in raw.split(";"):
            ent = ent.strip()
            if not ent:
                continue
            bs, rest = ent.split(":")
            n16, n12 = (int(x) for x in rest.split(","))
            table[int(bs)] = (n16, n12)
    for bs, (n16, n12) in table.items():
        n8 = bs - n16 - n12
        if bs <= 0 or n16 < 0 or n12 < 0 or n8 < 0:
            raise ValueError(f"SPECPILOT_D8R3_RBLOCK: bad entry {bs}:{n16},{n12}")
        if rows(bs, (n16, n12)) > MAX_ROWS:
            raise ValueError(f"SPECPILOT_D8R3_RBLOCK: {bs}:{n16},{n12} packs {rows(bs, (n16, n12))} > {MAX_ROWS} rows")
    return table


def rows(bs: int, ent: Tuple[int, int]) -> int:
    n16, n12 = ent
    return 16 * n16 + 12 * n12 + 8 * (bs - n16 - n12)


@functools.lru_cache(maxsize=None)
def table() -> Optional[Dict[int, Tuple[int, int]]]:
    return parse_table(os.environ.get("SPECPILOT_D8R3_RBLOCK", ""))


def enabled() -> bool:
    return table() is not None


GRAPH_OUT = types.SimpleNamespace(can_run_graph=True, logits_output=None)


def _np2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


# ------------------------------------------------------------------------------------------------
# plan: block lengths, packed row maps, state update (one program; eager before the graph replay)
# ------------------------------------------------------------------------------------------------


@triton.jit
def _rb_plan_kernel(
    seq_lens_ptr,  # [bs] committed prefix P (any int dtype)
    rpi_ptr,  # [bs] req pool indices
    block_ids_ptr,  # [*, block_ids_stride] int64: column 0 = bonus
    block_ids_stride,
    r2t_ptr,  # req_to_token [R, r2t_stride] int32
    r2t_stride,
    state_ptr,  # [R, 4] float32: P_last, c_last, n_seen (0/1/2), pconf
    rpi_out,  # [BS] int64
    bonus_out,  # [BS] int64
    len_out,  # [BS] int32
    slots_out,  # [BS] int32 (L - 1)
    cuq_out,  # [BS + 1] int32
    cuk_out,  # [BS + 1] int32
    cache_seqlens_out,  # [BS] int32 (P + L)
    ids_out,  # [M] int64
    pos_out,  # [M] int64
    loc_out,  # [M] int64
    rowpos_out,  # [M] int32
    head_src_out,  # [M - BS] int64 packed row of each lattice row (parent: rows 1..L-1; ROW0: rows 0..L-2)
    head_dst_out,  # [M - BS] int64 dense slot row b * (BLOCK - 1) + j - 1 (ROW0: + j)
    mask_id,
    BS: tl.constexpr,
    M: tl.constexpr,
    N16: tl.constexpr,
    N12: tl.constexpr,
    BB: tl.constexpr,
    BM: tl.constexpr,
    TRACK_ONLY: tl.constexpr,
    PREV0: tl.constexpr,
    PCONF0: tl.constexpr,
    BLK: tl.constexpr,
    ROW0: tl.constexpr = 0,  # SPECPILOT_D1_LD: 1 = DSpark (rows 0..L-2 feed slots 0..L-2)
):
    bi = tl.arange(0, BB)
    bm = bi < BS
    P = tl.load(seq_lens_ptr + bi, mask=bm, other=0).to(tl.int64)
    rpi = tl.load(rpi_ptr + bi, mask=bm, other=0).to(tl.int64)
    st = state_ptr + rpi * 4
    p_last = tl.load(st + 0, mask=bm, other=0.0)
    c_last = tl.load(st + 1, mask=bm, other=0.0)
    n_seen = tl.load(st + 2, mask=bm, other=0.0)
    pconf = tl.load(st + 3, mask=bm, other=0.0)
    c = P.to(tl.float32) - p_last
    new0 = n_seen < 0.5
    prev2 = tl.where(new0, PREV0, tl.where(n_seen < 1.5, c, 0.5 * (c + c_last)))
    key = prev2 / 5.0 + tl.where(new0, PCONF0, pconf) / 3.0
    tl.store(st + 0, P.to(tl.float32), mask=bm)
    tl.store(st + 1, tl.where(new0, c_last, c), mask=bm)
    tl.store(st + 2, tl.minimum(n_seen + 1.0, 2.0), mask=bm)
    if not TRACK_ONLY:
        # rank by key (descending; ties -> lower batch row first)
        kj = key[None, :]
        kb = key[:, None]
        jj = bi[None, :]
        ii = bi[:, None]
        beats = ((kj > kb) | ((kj == kb) & (jj < ii))) & (jj < BS)
        rank = tl.sum(beats.to(tl.int32), 1)
        L = tl.where(rank < N16, 16, tl.where(rank < N16 + N12, 12, 8))
        L = tl.where(bm, L, 0).to(tl.int32)
        incl = tl.cumsum(L, 0)
        off = incl - L
        bonus = tl.load(block_ids_ptr + bi.to(tl.int64) * block_ids_stride, mask=bm, other=0).to(tl.int64)
        cs = (P + L.to(tl.int64)).to(tl.int32)
        cs = tl.where(bm, cs, 0)
        csum = tl.cumsum(cs, 0)
        tl.store(rpi_out + bi, rpi, mask=bm)
        tl.store(bonus_out + bi, bonus, mask=bm)
        tl.store(len_out + bi, L, mask=bm)
        tl.store(slots_out + bi, L - 1, mask=bm)
        tl.store(cache_seqlens_out + bi, cs, mask=bm)
        tl.store(cuq_out + bi, off, mask=bm)
        tl.store(cuq_out + bi + 1, incl, mask=bm)
        tl.store(cuk_out + bi + 1, csum, mask=bm)
        tl.store(cuk_out + bi, csum * 0, mask=bi == 0)
        # rows
        r = tl.arange(0, BM)
        rm = r < M
        ge = (off[None, :] <= r[:, None]) & (bi[None, :] < BS)
        rb = tl.sum(ge.to(tl.int32), 1) - 1
        oh = bi[None, :] == rb[:, None]
        off_r = tl.sum(tl.where(oh, off[None, :], 0), 1)
        p_r = tl.sum(tl.where(oh, P[None, :], 0), 1)
        rpi_r = tl.sum(tl.where(oh, rpi[None, :], 0), 1)
        bonus_r = tl.sum(tl.where(oh, bonus[None, :], 0), 1)
        j = r - off_r
        pos = p_r + j.to(tl.int64)
        loc = tl.load(r2t_ptr + rpi_r * r2t_stride + pos, mask=rm, other=0).to(tl.int64)
        ids = tl.where(j == 0, bonus_r, mask_id)
        tl.store(ids_out + r, ids, mask=rm)
        tl.store(pos_out + r, pos, mask=rm)
        tl.store(loc_out + r, loc, mask=rm)
        tl.store(rowpos_out + r, j.to(tl.int32), mask=rm)
        if ROW0:  # SPECPILOT_D1_LD: DSpark samples from the anchor row; the block's last row feeds no slot
            l_r = tl.sum(tl.where(oh, L[None, :], 0), 1)
            hm = rm & (j <= l_r - 2)
            h = r - rb
            tl.store(head_src_out + h, r.to(tl.int64), mask=hm)
            tl.store(head_dst_out + h, (rb * (BLK - 1) + j).to(tl.int64), mask=hm)
        else:
            hm = rm & (j >= 1)
            h = r - rb - 1
            tl.store(head_src_out + h, r.to(tl.int64), mask=hm)
            tl.store(head_dst_out + h, (rb * (BLK - 1) + j - 1).to(tl.int64), mask=hm)


@triton.jit
def _rb_pconf_kernel(
    scores_ptr,  # [bs, S, K, K] float (contiguous)
    rpi_ptr,  # [bs] int64
    state_ptr,  # [R, 4] float32
    S: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    kidx = tl.arange(0, K)
    base = scores_ptr + b * (S * K * K)
    row = tl.zeros([], dtype=tl.int64)
    acc = tl.zeros([], dtype=tl.float32)
    for d in tl.static_range(D):
        v = tl.load(base + d * K * K + row * K + kidx).to(tl.float32)
        m = tl.max(v, 0)
        lse = m + tl.log(tl.sum(tl.exp(v - m), 0))
        acc += m - lse
        row = tl.min(tl.where(v == m, kidx, K), 0).to(tl.int64)
    rpi = tl.load(rpi_ptr + b)
    tl.store(state_ptr + rpi * 4 + 3, acc)


def plan(rb: "RBlockDrafter", bs: int, seq_lens: torch.Tensor, req_pool_indices: torch.Tensor,
         block_ids: torch.Tensor, track_only: bool) -> None:
    ent = rb.table.get(bs, (0, 0))
    M = rows(bs, ent) if not track_only else 1
    _rb_plan_kernel[(1,)](
        seq_lens, req_pool_indices, block_ids, block_ids.stride(0),
        rb.req_to_token, rb.req_to_token.stride(0), rb.state,
        rb.rpi, rb.bonus, rb.blen, rb.slots, rb.cuq, rb.cuk, rb.cache_seqlens,
        rb.ids, rb.pos, rb.loc, rb.rowpos, rb.head_src, rb.head_dst,
        int(rb.mask_id),
        BS=bs, M=M, N16=ent[0], N12=ent[1], BB=_np2(bs + 1), BM=_np2(M), TRACK_ONLY=track_only,
        PREV0=PREV_INIT, PCONF0=PCONF_INIT, BLK=BLOCK, ROW0=int(rb.row0),
        num_warps=8,
    )


def reset_requests(state: torch.Tensor, req_pool_indices: torch.Tensor) -> None:
    """A request's first draft step sees n_seen = 0 (prev2 = 5, pconf = -3)."""
    idx = req_pool_indices.to(torch.int64).view(-1)
    state.index_copy_(0, idx, state.new_tensor([0.0, 0.0, 0.0, PCONF_INIT]).expand(idx.numel(), 4).contiguous())


# ------------------------------------------------------------------------------------------------
# packed drafter (stand-ins swapped in while the graph is captured)
# ------------------------------------------------------------------------------------------------


@torch.compile(dynamic=True, backend=get_compiler_backend())
def _grouped_conv_rows(hidden_states, delta, base, rowpos, num_groups, group_size, taps):
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    out = coefficients[:, 0] * blocks
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        out = out + coefficients[:, tap] * shifted * (rowpos >= tap).view(-1, 1, 1)
    return out.flatten(-2)


def _conv_rows(conv, rowpos):
    def _convolve(hidden_states, delta, side: int) -> torch.Tensor:
        torch._dynamo.mark_static(hidden_states, 1)
        torch._dynamo.mark_static(delta, 1)
        torch._dynamo.mark_static(delta, 2)
        return _grouped_conv_rows(hidden_states, delta, conv.base_kernel[side], rowpos, conv.num_groups,
                                  conv.group_size, conv.taps)

    return _convolve


class _PackedAttn(torch.nn.Module):
    """Stand-in for the drafter's RadixAttention on packed rows: KV write + FA3 varlen over the pool."""

    def __init__(self, layer, rb: "RBlockDrafter", bs: int):
        super().__init__()
        self.layer = layer
        self.rb = rb
        self.bs = bs

    def forward(self, q, k, v, forward_batch=None, **kw):
        L, rb, bs = self.layer, self.rb, self.bs
        nh, kvh, d, dv = L.tp_q_head_num, L.tp_k_head_num, L.head_dim, L.v_head_dim
        M = q.shape[0]
        k = k.view(M, kvh, d)
        v = v.view(M, kvh, dv)
        rb.write_kv(L, rb.loc[:M], k, v)
        kc, vc = rb.kv_cache(L)
        window, causal = rb.window_causal(L)
        o = rb.fa(
            q=q.contiguous().view(M, nh, d), k_cache=kc, v_cache=vc, page_table=rb.page_table[:bs],
            cache_seqlens=rb.cache_seqlens[:bs], cu_seqlens_q=rb.cuq[: bs + 1], cu_seqlens_k_new=rb.cuk[: bs + 1],
            max_seqlen_q=BLOCK, softmax_scale=L.scaling, causal=causal, window_size=window,
            softcap=L.logit_cap, num_splits=rb.num_splits)
        if isinstance(o, (tuple, list)):
            o = o[0]
        return o.reshape(M, nh * dv)


class RBlockDrafter:
    """Owns the static buffers and the per-bs CUDA graphs of the packed drafter.

    draft_model: DFlash2 drafter (layers with self_attn.attn / attention_conv / mlp_conv, forward,
    compute_candidates, candidate_selector); embed_fn(ids [M]) -> scaled noise embeddings [M, H];
    finish_fn(scores, cand, bonus, slots, bs): the parent's in-graph tree tail on the dense lattice.
    write_kv(layer, loc, k, v) / kv_cache(layer) -> (k_cache, v_cache) paged views; fa: FA3 entry.
    """

    def __init__(self, *, table_, draft_model, embed_fn, finish_fn, write_kv, kv_cache, fa, num_splits,
                 req_to_token, page_table_width, mask_id, hidden_size, top_k, device, max_reqs=None, pad_bs=None,
                 row0: int = 0, wide_m: int = 0, union_builder=None):
        self.table = dict(table_)
        self.row0 = int(row0)  # SPECPILOT_D1_LD: 1 = lattice rows start at the anchor row (DSpark)
        self.dm = draft_model
        self.embed_fn = embed_fn
        self.finish_fn = finish_fn
        self.write_kv = write_kv
        self.kv_cache = kv_cache
        self.fa = fa
        self.num_splits = num_splits
        self.req_to_token = req_to_token
        self.mask_id = int(mask_id)
        self.K = int(top_k)
        self.device = device
        # Verify graph bs of each bs: the tree tail (tree, RP budgets, out) covers its padding rows as the parent's
        # padded draft graph does (rp_valid marks them root-only; the verify reads n_buf for every graph row).
        self.pad_bs = {bs: int((pad_bs or {}).get(bs, bs)) for bs in self.table}
        assert all(g >= bs for bs, g in self.pad_bs.items())
        max_bs = max(self.pad_bs.values())
        self.max_bs = max_bs
        max_m = max(rows(bs, e) for bs, e in self.table.items())
        R = int(max_reqs if max_reqs is not None else req_to_token.shape[0])
        i32, i64 = torch.int32, torch.int64
        z = functools.partial(torch.zeros, device=device)
        self.state = z((R, 4), dtype=torch.float32)
        self.state[:, 3] = PCONF_INIT
        self.rpi, self.bonus = z((max_bs,), dtype=i64), z((max_bs,), dtype=i64)
        self.blen = z((max_bs,), dtype=i32)
        self.slots = torch.full((max_bs,), BLOCK - 1, dtype=i32, device=device)  # rows >= bs: graph padding
        self.cuq, self.cuk = z((max_bs + 1,), dtype=i32), z((max_bs + 1,), dtype=i32)
        self.cache_seqlens = z((max_bs,), dtype=i32)
        self.ids, self.pos, self.loc = z((max_m,), dtype=i64), z((max_m,), dtype=i64), z((max_m,), dtype=i64)
        self.rowpos = z((max_m,), dtype=i32)
        self.head_src, self.head_dst = z((max_m,), dtype=i64), z((max_m,), dtype=i64)
        self.page_table = z((max_bs, int(page_table_width)), dtype=i32)
        S = BLOCK - 1
        self.dcand = z((max_bs, S, self.K), dtype=i64)
        self.dunary = z((max_bs, S, self.K), dtype=torch.float32)
        self.dhid = z((max_bs, S, int(hidden_size)), dtype=torch.bfloat16)
        # SPECPILOT_D1_LD wide: dense per-slot shortlists of the packed rows (vals, full ids, tail)
        self.wide_m = int(wide_m)
        self.union = None  # SPECPILOT_D1_LD union: d4's builder + dense subset logits of the packed rows
        if union_builder is not None:
            self.union = union_builder
            self.dlog = z((max_bs, S, union_builder.Vs), dtype=torch.bfloat16)
        elif self.wide_m:
            self.dsv = z((max_bs, S, self.wide_m), dtype=torch.float32)
            self.dsi = z((max_bs, S, self.wide_m), dtype=i64)
            self.dst = z((max_bs, S), dtype=torch.float32)
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.out_scores: Dict[int, torch.Tensor] = {}
        self.out_hidden: Dict[int, torch.Tensor] = {}

    @staticmethod
    def window_causal(layer):
        """(window_size, causal) exactly as the FA3 backend's extend/verify path derives them."""
        from sglang.srt.layers.radix_attention import AttentionType

        causal = not (getattr(layer, "is_cross_attention", False) or layer.attn_type in (
            AttentionType.ENCODER_ONLY, AttentionType.DECODER_BIDIRECTIONAL))
        sw = layer.sliding_window_size
        if sw is not None and sw > -1:
            return (sw, 0 if causal else sw), causal
        return (-1, -1), causal

    # ---- eager pieces ------------------------------------------------------------------------
    def active(self, bs: int) -> bool:
        return bs in self.graphs

    def stage(self, bs, seq_lens, req_pool_indices, block_ids):
        plan(self, bs, seq_lens, req_pool_indices, block_ids, track_only=bs not in self.table)

    def body(self, bs: int):
        """The packed drafter step for an exact batch size (captured into the graph)."""
        from sglang.kernels.ops.kvcache.trtllm_mha_page_table import build_trtllm_mha_page_table

        M = rows(bs, self.table[bs])
        Mh = M - bs
        S = BLOCK - 1
        build_trtllm_mha_page_table(req_to_token=self.req_to_token, req_pool_indices=self.rpi[:bs],
                                    cache_seqlens=self.cache_seqlens[:bs], page_table=self.page_table[:bs],
                                    page_size=1)
        emb = self.embed_fn(self.ids[:M])
        with self.swapped(bs):
            hidden = self.dm.forward(input_ids=None, positions=self.pos[:M], forward_batch=None,
                                     input_embeds=emb).hidden_states
        hh = hidden.index_select(0, self.head_src[:Mh])
        dst = self.head_dst[:Mh]
        G = self.pad_bs[bs]  # rows bs..G-1: graph padding (stale finite lattice inputs)
        if self.union is not None:  # SPECPILOT_D1_LD union: dense subset logits -> d4 builder (slot caps L_b - 1)
            from sglang.srt.speculative import specpilot_d1_ld as _d1ld

            logits = _d1ld.subset_logits_raw(self.dm, hh)
            self.dlog[:bs].view(bs * S, -1).index_copy_(0, dst, logits.to(self.dlog.dtype))
            cand_d, scores = self.union.lattice(self.dlog[:G], self.bonus[:G], slots=self.slots[:G])
            self.dcand[:G].copy_(cand_d)
            cand_d = self.dcand[:G]
        elif self.wide_m:  # SPECPILOT_D1_LD wide: shortlist the packed rows, Viterbi beam on the dense slots
            from sglang.srt.speculative import specpilot_d1_ld as _d1ld

            logits, ids_map = _d1ld.subset_logits(self.dm, hh)
            sv, si, st = _d1ld.shortlist(logits, ids_map, self.wide_m)
            self.dsv[:bs].view(bs * S, -1).index_copy_(0, dst, sv)
            self.dsi[:bs].view(bs * S, -1).index_copy_(0, dst, si)
            self.dst[:bs].view(bs * S).index_copy_(0, dst, st)
            cand_d, scores = self.dm.candidate_selector.lattice_from_shortlist(
                self.dsv[:G], self.dsi[:G], self.dst[:G], self.bonus[:G])
            self.dcand[:G].copy_(cand_d)
            cand_d = self.dcand[:G]
        else:
            cand, unary = self.dm.compute_candidates(hh)
            self.dcand[:bs].view(bs * S, self.K).index_copy_(0, dst, cand.long())
            self.dunary[:bs].view(bs * S, self.K).index_copy_(0, dst, unary.float())
            self.dhid[:bs].view(bs * S, -1).index_copy_(0, dst, hh.to(self.dhid.dtype))
            cand_d, unary_d, hid_d = self.dcand[:G], self.dunary[:G], self.dhid[:G]
            scores = self.dm.candidate_selector.build_lattice(candidate_ids=cand_d, unary_logits=unary_d,
                                                              hidden_states=hid_d, anchor_token_ids=self.bonus[:G])
        scores = scores.contiguous()
        _rb_pconf_kernel[(bs,)](scores, self.rpi, self.state, S=S, K=self.K, D=PCONF_SLOTS)
        self.finish_fn(scores, cand_d, self.bonus[:G], self.slots[:G], G)
        return scores, hidden

    class _Swap:
        def __init__(self, rb, bs):
            self.rb, self.bs = rb, bs

        def __enter__(self):
            self.saved = []
            for layer in self.rb.dm.layers:
                attn = layer.self_attn.attn
                convs = [c for c in (layer.attention_conv, layer.mlp_conv) if c is not None]
                self.saved.append((layer, attn, convs))
                layer.self_attn.attn = _PackedAttn(attn, self.rb, self.bs)
                for c in convs:
                    c._convolve = _conv_rows(c, self.rb.rowpos[: rows(self.bs, self.rb.table[self.bs])])
            return self

        def __exit__(self, *exc):
            for layer, attn, convs in self.saved:
                layer.self_attn.attn = attn
                for c in convs:
                    c.__dict__.pop("_convolve", None)
            return False

    def swapped(self, bs):
        return RBlockDrafter._Swap(self, bs)

    # ---- graphs ------------------------------------------------------------------------------
    def capture(self, stage_fn: Callable[[int], None], pool=None, warmup: int = 2):
        """stage_fn(bs): writes a valid plan for bs (dummy inputs) before the warmups/capture."""
        pool = pool if pool is not None else torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        for bs in sorted(self.table, reverse=True):
            stage_fn(bs)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    self.body(bs)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool, stream=stream):
                scores, hidden = self.body(bs)
            torch.cuda.synchronize()
            self.graphs[bs] = g
            self.out_scores[bs] = scores
            self.out_hidden[bs] = hidden
        self.pool = pool

    def replay(self, bs: int, seq_lens, req_pool_indices, block_ids):
        self.stage(bs, seq_lens, req_pool_indices, block_ids)
        self.graphs[bs].replay()


# ------------------------------------------------------------------------------------------------
# worker glue
# ------------------------------------------------------------------------------------------------


def build_for_worker(worker) -> Optional[RBlockDrafter]:
    """Called after the parent's draft graphs are captured. None (with a warning) if unsupported."""
    tab = table()
    if tab is None:
        return None
    from sglang.srt.mem_cache.memory_pool import KVWriteLoc
    from sglang.srt.speculative import specpilot_d1_ld as _d1ld  # SPECPILOT_D1_LD
    from sglang.srt.speculative import specpilot_db_rp as _dbrp
    from sglang.srt.speculative import specpilot_tree as _dtree

    sampler = getattr(worker, "_draft_sampler", None)
    why = None
    if int(worker.block_size) != BLOCK:
        why = f"block_size {worker.block_size} != {BLOCK}"
    elif sampler is None or getattr(sampler, "tree_out", None) is None:
        why = "needs the in-graph selector tree sampler (SPECPILOT_TREE_MODE)"
    elif worker.use_compact_draft_cache:
        why = "compact draft cache"
    elif int(worker.page_size) != 1:
        why = "page_size != 1"
    if why is None:
        backend = worker.draft_model_runner.attn_backend
        pool = worker.draft_model_runner.token_to_kv_pool
        if getattr(backend, "fa_impl_ver", None) != 3 or getattr(backend, "use_sliding_window_kv_pool", False):
            why = "draft attention backend is not plain FA3"
        tr = getattr(pool, "translator", None) or getattr(pool, "index_translator", None)
        if tr is not None and getattr(tr, "is_translating", False):
            why = "translating draft KV pool"
    if why is not None:
        logger.warning("SPECPILOT_D8R3_RBLOCK disabled: %s", why)
        return None
    dm = worker.draft_model
    req_to_token = worker.model_runner.req_to_token_pool.req_to_token
    tvm = getattr(backend, "target_verify_metadata", None)
    width = int(tvm["page_table"].shape[1]) if isinstance(tvm, dict) and "page_table" in tvm else int(req_to_token.shape[1])
    full_embed = worker._full_embed_gpu
    target_model = worker.target_worker.model_runner.model
    from sglang.srt.speculative.dflash_worker_v2 import _resolve_dflash_embedding_module, unwrap_lora_layer

    embed_module = unwrap_lora_layer(_resolve_dflash_embedding_module(dm, target_model))
    scale = float(worker._noise_embed_scale)

    def embed_fn(ids):
        e = F.embedding(ids, full_embed) if full_embed is not None else embed_module(ids)
        return e * scale if scale != 1.0 else e

    def write_kv(layer, loc, k, v):
        pool.set_kv_buffer(layer, KVWriteLoc(loc), k, v, layer.k_scale, layer.v_scale)

    def kv_cache(layer):
        return backend.get_paged_mha_kv_cache(layer)

    def finish_fn(scores, cand, bonus, slots, bs):
        tree = _dtree.slice_buffers(sampler.tree_out, bs)
        _dtree.build_tree(scores, cand, bonus, n_nodes=BLOCK, temp=sampler.tree_inv_temps, out=tree, slots=slots,
                          norm=_d1ld.tree_norm(dm))  # SPECPILOT_D1_LD
        if sampler.d7_chain is not None:
            from sglang.srt.speculative import specpilot_d7_chain as _d7chain

            _d7chain.apply_tree(tree, sampler.d7_chain)
        if sampler.rp_cfg is not None:
            _dbrp.apply_policy(tree, sampler.rp_cfg, valid=sampler.rp_valid)
        sampler.out[: bs * (BLOCK - 1)].copy_(tree["tokens"][:, 1:].reshape(-1))

    from sglang.srt.speculative import specpilot_rv as _drv

    runner = worker.target_worker.model_runner.decode_cuda_graph_runner
    cap = runner.capture_bs if runner is not None else None
    rb = RBlockDrafter(
        pad_bs={bs: _drv.graph_bs(bs, cap) for bs in tab},
        table_=tab, draft_model=dm, embed_fn=embed_fn, finish_fn=finish_fn, write_kv=write_kv, kv_cache=kv_cache,
        fa=backend.flash_attn_with_kvcache, num_splits=backend.num_splits, req_to_token=req_to_token,
        page_table_width=width, mask_id=worker._mask_token_id, hidden_size=int(dm.config.hidden_size),
        top_k=int(dm.candidate_selector.top_k), device=worker.device,
        row0=1 - _d1ld.row_offset(dm),  # SPECPILOT_D1_LD
        wide_m=_d1ld.shortlist_size() if _d1ld.wide(dm) else 0,
        union_builder=dm.candidate_selector.union_builder(dm) if _d1ld.union(dm) else None)
    ids_buf = torch.zeros((rb.max_bs, BLOCK), dtype=torch.int64, device=worker.device)
    ids_buf[:, 0] = 1

    def stage_fn(bs):
        seq = torch.full((bs,), 1, dtype=torch.int64, device=worker.device)
        rpi = torch.zeros((bs,), dtype=torch.int64, device=worker.device)
        rb.stage(bs, seq, rpi, ids_buf)

    with torch.inference_mode():
        rb.capture(stage_fn)
    rb.state.zero_()
    rb.state[:, 3] = PCONF_INIT
    logger.info("SPECPILOT_D8R3_RBLOCK: packed drafter graphs for bs %s (rows %s, tree tail at graph bs %s)",
                sorted(rb.graphs), [rows(b, tab[b]) for b in sorted(rb.graphs)], [rb.pad_bs[b] for b in sorted(rb.graphs)])
    return rb

