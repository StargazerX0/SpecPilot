"""SPECPILOT_D7X_QLOG (stream Q, diagnostic only): Q-view vs exact argmax agreement under two-level verify.

At each exact step of SPECPILOT_D8R_TLV the composed tree is chain (nodes 0..C-1, node d at depth d) + Q's bonus
(node C) + subtree. The exact verify's greedy prediction at chain node d is compared with the token the Q pass chose
there (node d + 1; Q's bonus for d = C - 1): both passes saw the same tokens and positions on these rows, so the
match rate is the per-row Q-vs-exact argmax agreement on the Q-accepted path. At a mismatch the exact logit gap
logit[exact] - logit[Q token] (fp32 verify logits, after output_multiplier and softcap) is logged for tie judgement.

One JSON line per exact step: {"p": prefix_len, "C": chain_len, "m": "1101..." (match per chain row), "acc": exact
commit_len, "g": [[row, gap, logit_exact], ...]}. Host syncs: diagnostic runs only (untimed).

Knob (read once at import): SPECPILOT_D7X_QLOG=<path> (append). Unset = no-op (nothing imported, no call).
"""
from __future__ import annotations

import json
import os

import torch

_PATH = (os.environ.get("SPECPILOT_D7X_QLOG", "") or "").strip()


def enabled() -> bool:
    return bool(_PATH)


def record(tree_tokens: torch.Tensor, chain_len: torch.Tensor, pred: torch.Tensor, logits, prefix_lens: torch.Tensor,
           commit_lens: torch.Tensor) -> None:
    """tree_tokens / pred [bs, N]; chain_len, prefix_lens, commit_lens [bs]; logits [bs * N, V] fp32 or None."""
    bs, n = pred.shape
    C = chain_len.view(-1).to(torch.int64)
    rows = torch.arange(n - 1, device=pred.device).unsqueeze(0)
    match = pred[:, : n - 1] == tree_tokens[:, 1:n]
    valid = rows < C.unsqueeze(1)
    mm_rows = (valid & ~match).nonzero().tolist()
    gaps = {}
    if logits is not None and mm_rows:
        lg = logits.view(bs, n, -1)
        for b, r in mm_rows:
            le = float(lg[b, r, pred[b, r]])
            lq = float(lg[b, r, tree_tokens[b, r + 1]])
            gaps.setdefault(b, []).append([r, le - lq, le])
    Cl, pl, cl = C.tolist(), prefix_lens.view(-1).tolist(), commit_lens.view(-1).tolist()
    ml = match.tolist()
    with open(_PATH, "a") as f:
        for b in range(bs):
            c = int(Cl[b])
            f.write(json.dumps({"p": int(pl[b]), "C": c, "m": "".join("1" if x else "0" for x in ml[b][:c]),
                                "acc": int(cl[b]), "g": gaps.get(b, [])}) + "\n")
