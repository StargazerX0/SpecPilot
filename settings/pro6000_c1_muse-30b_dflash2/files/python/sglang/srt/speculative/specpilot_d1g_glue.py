"""SPECPILOT_D1G: fused hand-off glue of the two-level verify cycle (SPECPILOT_D8R_TLV).

The exact step assembles its tree from Q's chain and the second draft's best-first subtree
(``specpilot_d8_tlv.compose_chain_tree``: ~45 eager torch ops), then copies the tokens into the draft-token
buffer, rebuilds the verify positions (prefix + i) and the verify cache locations (req_to_token rows). With
SPECPILOT_D1G_FUSE=1 one Triton kernel per step does all of it. Every output is integer/bool and bit-identical
to the eager reference (tests/test_d1g_glue.py), so the verify inputs and the emitted tokens do not change.

Knobs (read once at import; unset = parent behaviour):
  SPECPILOT_D1G_FUSE   1 = fused chain + subtree assembly and verify-layout kernel
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_ENABLED = (os.environ.get("SPECPILOT_D1G_FUSE", "") or "0").strip() not in ("", "0")


def enabled() -> bool:
    return _ENABLED


@triton.jit
def _d1g_compose_kernel(
    chain_ptr,  # [bs, N] int64
    chain_len_ptr,  # [bs] int32
    sub_tok_ptr,  # [*, N] int64
    sub_par_ptr,  # [*, N] int64
    sub_dep_ptr,  # [*, N] int64
    sub_mask_ptr,  # [*, N, N] uint8 (bool view)
    out_tok_ptr,  # [*, N] int64
    out_par_ptr,  # [*, N] int64
    out_dep_ptr,  # [*, N] int64
    out_mask_ptr,  # [*, N, N] uint8 (bool view)
    out_next_ptr,  # [*, N] int32
    out_sib_ptr,  # [*, N] int32
    draft_tok_ptr,  # [bs, N] int64 (draft-token buffer, row stride draft_stride)
    draft_stride,
    prefix_ptr,  # [bs] int64 (committed prefix lengths)
    positions_ptr,  # [bs, N] int64 (row stride pos_stride)
    pos_stride,
    req_pool_ptr,  # [bs] int64
    req_to_token_ptr,  # [pool, max_len] int32
    r2t_stride,
    cache_loc_ptr,  # [bs, N] int64 (row stride loc_stride)
    loc_stride,
    N: tl.constexpr,
):
    b = tl.program_id(0)
    r = tl.arange(0, N)  # node index (rows)
    c = tl.arange(0, N)  # node index (columns)
    C = tl.load(chain_len_ptr + b).to(tl.int64)

    # node tables, computed in the row orientation and again in the column orientation (same values)
    is_chain = r < C
    s = tl.maximum(r - C, 0)
    sub_t = tl.load(sub_tok_ptr + b * N + s)
    sub_p = tl.load(sub_par_ptr + b * N + s)
    sub_d = tl.load(sub_dep_ptr + b * N + s)
    tok = tl.where(is_chain, tl.load(chain_ptr + b * N + r), sub_t)
    par = tl.where(is_chain, r - 1, tl.where(sub_p < 0, C - 1, sub_p + C))
    dep = tl.where(is_chain, r.to(tl.int64), sub_d + C)

    sc = tl.maximum(c - C, 0)
    sub_pc = tl.load(sub_par_ptr + b * N + sc)
    par_c = tl.where(c < C, c - 1, tl.where(sub_pc < 0, C - 1, sub_pc + C))

    # ancestor mask: chain rows see the chain prefix; subtree rows see the whole chain + their sub-mask row
    ii = r[:, None]
    jj = c[None, :]
    row_chain = ii < C
    col_chain = jj < C
    sm = tl.load(sub_mask_ptr + b * N * N + s[:, None] * N + sc[None, :]) != 0
    mask = tl.where(row_chain, col_chain & (jj <= ii), col_chain | (sm & (jj >= C)))

    # first child (smallest j > 0 with parent i) and next sibling (smallest j > i with the same parent)
    big = tl.full((N, N), N, tl.int64)
    jj64 = jj.to(tl.int64) + tl.zeros((N, N), tl.int64)
    is_child = (par_c[None, :] == ii) & (jj > 0)
    first = tl.min(tl.where(is_child, jj64, big), axis=1)
    first = tl.where(first >= N, -1, first)
    same = (par_c[None, :] == par[:, None]) & (par[:, None] >= 0) & (jj > ii)
    nsib = tl.min(tl.where(same, jj64, big), axis=1)
    nsib = tl.where(nsib >= N, -1, nsib)

    tl.store(out_tok_ptr + b * N + r, tok)
    tl.store(out_par_ptr + b * N + r, par)
    tl.store(out_dep_ptr + b * N + r, dep)
    tl.store(out_mask_ptr + b * N * N + ii * N + jj, mask.to(tl.uint8))
    tl.store(out_next_ptr + b * N + r, first.to(tl.int32))
    tl.store(out_sib_ptr + b * N + r, nsib.to(tl.int32))
    tl.store(draft_tok_ptr + b * draft_stride + r, tok)

    # verify layout relative to the committed prefix: positions prefix + i, cache locs req_to_token[rpi, prefix + i]
    pre = tl.load(prefix_ptr + b).to(tl.int64)
    tl.store(positions_ptr + b * pos_stride + r, pre + r)
    rpi = tl.load(req_pool_ptr + b).to(tl.int64)
    loc = tl.load(req_to_token_ptr + rpi * r2t_stride + pre + r)
    tl.store(cache_loc_ptr + b * loc_stride + r, loc.to(tl.int64))


def compose_and_layout(
    chain_tokens: torch.Tensor,
    chain_len: torch.Tensor,
    sub: dict,
    out: dict,
    draft_tokens: torch.Tensor,
    prefix_lens: torch.Tensor,
    positions_2d: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_loc_2d: torch.Tensor,
) -> dict:
    """Fused ``compose_chain_tree`` + draft_tokens.copy_(tokens) + positions_2d = prefix + i +
    cache_loc_2d = assign_extend_cache_locs(prefix, prefix + N). Same outputs as the eager path, bit for bit."""
    bs, N = chain_tokens.shape
    assert chain_tokens.is_contiguous() and N & (N - 1) == 0
    for t in (draft_tokens, positions_2d, cache_loc_2d):
        assert t.dtype == torch.int64 and t.stride(1) == 1
    assert req_to_token.stride(1) == 1
    cl = chain_len if chain_len.dtype == torch.int32 else chain_len.to(torch.int32)
    _d1g_compose_kernel[(bs,)](
        chain_tokens,
        cl,
        sub["tokens"],
        sub["parents"],
        sub["depth"],
        sub["mask"].view(torch.uint8),
        out["tokens"],
        out["parents"],
        out["depth"],
        out["mask"].view(torch.uint8),
        out["next_token"],
        out["next_sibling"],
        draft_tokens,
        draft_tokens.stride(0),
        prefix_lens,
        positions_2d,
        positions_2d.stride(0),
        req_pool_indices,
        req_to_token,
        req_to_token.stride(0),
        cache_loc_2d,
        cache_loc_2d.stride(0),
        N=N,
        num_warps=4,
    )
    return {k: v[:bs] for k, v in out.items()}
