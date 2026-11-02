"""SPECPILOT_D8R3_TRUNC=<dir>: untimed truncation-agreement dump for LINE E (per-request drafter block).

After every decode draft forward (graph replay unchanged), re-run the drafter eagerly for the same batch with the
block truncated to B in BLOCKS (16 = reference) and dump each variant's selector lattice. The reference reads the
committed context K/V of every request from the draft KV pool (positions [0, P), sliding window as FA3: key position
>= query position - (window - 1)) and computes the block's own K/V itself (the pool is never written), so the only
difference between variants is the block length seen by the non-causal block attention (the grouped conv is causal
inside the block). Sanity: the B16 reference tree tokens vs the engine's in-graph tree tokens.

Files in <dir>: trunc_<pid>_<n>.npz (per chunk: rid index, P, prompt_len, match, scores_<B> f16 [rows, B-1, K, K],
cand_<B> i32 [rows, B-1, K]); seqs_<pid>.jsonl ({rid, seq} once a request leaves the batch). Knob unset = no-op.
The dump syncs the host every step (diagnostic only, never timed).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch import nn

OUT = os.environ.get("SPECPILOT_D8R3_TRUNC", "")
ENABLED = bool(OUT)
BLOCKS = tuple(int(b) for b in os.environ.get("SPECPILOT_D8R3_TRUNC_BLOCKS", "16,12,10,8").split(","))
CHUNK = int(os.environ.get("SPECPILOT_D8R3_TRUNC_CHUNK", "32"))

_S = {"reqs": {}, "seq_f": None, "buf": [], "n": 0, "steps": 0}


class _RefAttn(nn.Module):
    """Stand-in for RadixAttention: SDPA of the block rows over (context K/V from the pool) + (the block's own K/V)."""

    def __init__(self, layer, ctx, block):
        super().__init__()
        self.layer = layer
        self.ctx = ctx  # list of (k [P, kvh, d], v [P, kvh, d], P)
        self.block = block

    def forward(self, q, k, v, forward_batch=None, **kw):
        L = self.layer
        nh, kvh, d = L.tp_q_head_num, L.tp_k_head_num, L.head_dim
        B = self.block
        bs = q.shape[0] // B
        q = q.view(bs, B, nh, d)
        k = k.view(bs, B, kvh, d)
        v = v.view(bs, B, kvh, d)
        win = L.sliding_window_size
        out = torch.empty_like(q)
        for i, (ck, cv, P) in enumerate(self.ctx):
            kk = torch.cat([ck, k[i]], 0).transpose(0, 1).float()  # [kvh, P+B, d]
            vv = torch.cat([cv, v[i]], 0).transpose(0, 1).float()
            qq = q[i].transpose(0, 1).float()  # [nh, B, d]
            rep = nh // kvh
            kk = kk.repeat_interleave(rep, 0)
            vv = vv.repeat_interleave(rep, 0)
            mask = None
            if win is not None and win > -1 and P + B > win:
                qpos = torch.arange(P, P + B, device=q.device)[:, None]
                kpos = torch.arange(P + B, device=q.device)[None, :]
                mask = kpos >= qpos - win  # FA3 window (win, win), win = sliding_window - 1
            o = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask, scale=L.scaling)
            out[i] = o.transpose(0, 1).to(q.dtype)
        return out.reshape(bs * B, nh * d)


def _ref_hidden(dm, pool, x, pos, ctx_locs, block):
    """Eager drafter backbone on [bs*block] rows; context K/V gathered per layer."""
    saved = []
    try:
        for layer in dm.layers:
            attn = layer.self_attn.attn
            ctx = [(pool.get_key_buffer(attn.layer_id)[locs], pool.get_value_buffer(attn.layer_id)[locs], P)
                   for locs, P in ctx_locs]
            saved.append((layer, attn, [(c, c.block_size) for c in (layer.attention_conv, layer.mlp_conv) if c is not None]))
            layer.self_attn.attn = _RefAttn(attn, ctx, block)
            for c, _ in saved[-1][2]:
                c.block_size = block
        out = dm.forward(input_ids=None, positions=pos, forward_batch=None, input_embeds=x)
        return out.hidden_states
    finally:
        for layer, attn, convs in saved:
            layer.self_attn.attn = attn
            for c, b in convs:
                c.block_size = b


def observe(worker, batch, forward_batch, block_ids, prefix_lens, bs):
    global ENABLED
    if bs <= 0:
        return
    try:
        _observe(worker, batch, forward_batch, block_ids, prefix_lens, bs)
    except Exception:  # diagnostic only: log once, stop dumping, keep serving
        import logging
        import traceback

        logging.getLogger(__name__).error("SPECPILOT_D8R3_TRUNC disabled:\n%s", traceback.format_exc())
        ENABLED = False


def _observe(worker, batch, forward_batch, block_ids, prefix_lens, bs):
    from sglang.srt.speculative import specpilot_tree as _dtree
    from sglang.srt.speculative.dflash_worker_v2 import _selector_lattice

    dm = worker.draft_model
    pool = worker.draft_model_runner.token_to_kv_pool
    req_to_token = worker.model_runner.req_to_token_pool.req_to_token
    full = int(worker.block_size)
    torch.cuda.current_stream().synchronize()
    P = prefix_lens[:bs].to("cpu", torch.int64).tolist()
    rpi = batch.req_pool_indices[:bs].to("cpu", torch.int64).tolist()
    reqs = list(batch.reqs)[:bs]
    ctx_locs = [(req_to_token[r, :p].long(), p) for r, p in zip(rpi, P)]
    emb = forward_batch.input_embeds.view(bs, full, -1)
    posv = forward_batch.positions.view(bs, full)
    ids = block_ids[:bs]
    anchor = ids[:, 0]
    rec = {}
    with torch.inference_mode():
        for B in BLOCKS:
            x = emb[:, :B].reshape(bs * B, -1).clone()
            pos = posv[:, :B].reshape(-1).contiguous()
            h = _ref_hidden(dm, pool, x, pos, ctx_locs, B).view(bs, B, -1)[:, 1:, :]
            cand, scores = _selector_lattice(dm, h, anchor)
            rec[B] = (cand, scores)
        match = np.zeros(bs, dtype=bool)
        extra = {}
        sampler = getattr(worker, "_draft_sampler", None)
        if full in rec and sampler is not None and getattr(sampler, "tree_out", None) is not None:
            tree = _dtree.alloc_tree_buffers(bs, full, ids.device, with_score=True)
            cand, scores = rec[full]
            _dtree.build_tree(scores, cand, anchor, n_nodes=full, temp=sampler.tree_inv_temps, out=tree)
            eng = sampler.out[: bs * (full - 1)].view(bs, full - 1)
            match = (tree["tokens"][:, 1:] == eng).all(1).cpu().numpy()
            extra["ref_tok"] = tree["tokens"][:, 1:].cpu().numpy().astype(np.int32)
            extra["ref_par"] = tree["parents"].cpu().numpy().astype(np.int8)
            extra["ref_score"] = tree["score"].float().cpu().numpy()
            eo = sampler.tree_out
            extra["eng_tok"] = eng.cpu().numpy().astype(np.int32)
            extra["eng_par"] = eo["parents"][:bs].cpu().numpy().astype(np.int8)
            if "score" in eo:
                extra["eng_score"] = eo["score"][:bs].float().cpu().numpy()
    row = {"rid": [r.rid for r in reqs], "P": np.array(P, dtype=np.int64),
           "prompt_len": np.array([len(r.origin_input_ids) for r in reqs], dtype=np.int64),
           "bonus": anchor.cpu().numpy().astype(np.int64), "pos0": posv[:, 0].cpu().numpy().astype(np.int64),
           "match": match, **extra}
    for B, (cand, scores) in rec.items():
        row[f"scores_{B}"] = scores.to(torch.float16).cpu().numpy()
        row[f"cand_{B}"] = cand.to(torch.int32).cpu().numpy()
    _S["buf"].append(row)
    live = set()
    for r in reqs:
        _S["reqs"][r.rid] = r
        live.add(r.rid)
    _flush_seqs(live)
    _S["steps"] += 1
    if len(_S["buf"]) >= CHUNK:
        _flush_chunk()


def _flush_seqs(live):
    if _S["seq_f"] is None:
        os.makedirs(OUT, exist_ok=True)
        _S["seq_f"] = open(os.path.join(OUT, f"seqs_{os.getpid()}.jsonl"), "a", buffering=1)
    for rid in [k for k in _S["reqs"] if k not in live]:
        r = _S["reqs"].pop(rid)
        _S["seq_f"].write(json.dumps({"rid": rid, "seq": list(r.origin_input_ids) + list(r.output_ids)}) + "\n")


def _flush_chunk():
    rows = _S["buf"]
    _S["buf"] = []
    cat = {}
    rid_tab = []
    for row in rows:
        rid_tab.extend(row["rid"])
    cat["rid"] = np.array(rid_tab)
    for k in rows[0]:
        if k != "rid":
            cat[k] = np.concatenate([row[k] for row in rows], 0)
    cat["step"] = np.concatenate([np.full(len(row["rid"]), _S["steps"] - len(rows) + j) for j, row in enumerate(rows)])
    np.savez(os.path.join(OUT, f"trunc_{os.getpid()}_{_S['n']:05d}.npz"), **cat)
    _S["n"] += 1
