"""Draft-tree construction, layout and greedy tree acceptance for a DFlash2 top-K lattice.

Reference implementation for the ``numerical`` tree family
(``research/knowledge/search_space/01_tree.md``). Pure PyTorch, device-agnostic, importable
without a GPU and without sglang. Not imported by the plugin.

Tree layout (identical to ``research/validation/branch_oracle.py``)::

    tokens  [B, N] int64   node 0 = root = the anchor (previous bonus, already chosen by the target)
    parents [B, N] int64   parents[:, 0] == -1; a padded node has parent PAD (-2);
                           every parent index is smaller than its child's index
    depth   [B, N] int64   root 0; padded nodes 0
    position offset        = depth: engine position = seq_len + depth (the root sits at seq_len, as
                           in the chain verify ``positions = seq_len + [0..7]``); branch_oracle's
                           ``positions(parents, prefix_len) = prefix_len - 1 + depth`` is the same
                           thing with prefix_len = seq_len + 1 (root counted as committed)

Lattice inputs (what ``DFlash2DraftModel.compute_candidates`` / ``CandidateSelector.build_lattice``
produce for the S = block-1 draft slots)::

    cand_ids  [B, S, K]     top-K candidate ids per slot (K = 16 in stock)
    marginal  [B, S, K]     per-slot log-probs of those candidates (log_softmax of the unary top-K
                            logits; a factorised / DDTree surrogate)
    edge      [B, S, K, K]  conditional log-probs edge[b, s, p, c] = log P(cand[s, c] | cand[s-1, p]);
                            for s = 0 the parent is the anchor and only row 0 is read (the engine's
                            lattice repeats the anchor K times). From the selector lattice:
                            ``edge = log_softmax(scores / T, -1)`` (path-conditional surrogate)

Functions (no host syncs, static shapes, Python control flow only on ints):

* ``frontier_tree``  - generic depth-synchronous frontier + global top-B selection (batched)
* ``ddtree``         - DDTree best-first optimal tree for a node budget (Ringel & Romano 2026,
                       arXiv:2604.12989, Alg. 1 / Prop. 2), equal to ``frontier_tree`` with width = budget
* ``eagle2_tree``    - EAGLE-2 expand-top-k-per-depth, then global rerank to ``total_tokens``
                       (Li et al. 2024, arXiv:2406.16858)
* ``sequoia_dp``     - Sequoia DP-optimal static tree shape from per-(depth, rank) acceptance
                       probabilities (Chen et al. 2024, arXiv:2402.12374); offline, CPU, pure Python
* ``instantiate_template`` - fill a static (rank) template from a batch lattice
* ``rank_acceptance_table`` - fit the Sequoia/Medusa (depth, rank) table from traced candidates + truth
* ``cost_aware_node_count`` / ``truncate_tree`` - CaDDTree/BASTION-style per-row budget choice
* helpers: ``depth_of``, ``ancestor_mask``, ``positions``, ``first_child_next_sibling``,
  ``dfs_preorder``, ``chain_tree``
* acceptance: ``greedy_tree_accept`` (batched; committed tokens equal ``branch_oracle.oracle_accept``),
  ``fcns_greedy_accept`` (first-child / next-sibling walk in the style of SGLang's EAGLE
  ``verify_tree_greedy``), ``tree_accept_len_from_truth`` (offline greedy screens)
* recurrent references for kernel tests: ``gdn_tree_wy_reference`` (ancestor-masked WY/UT
  solve, the algebra of Weaver / TreeWY / Bole) and ``tree_conv_window_indices`` /
  ``tree_causal_conv`` (parent-indexed short causal conv)

Every tree emitted by the builders is in best-first order (non-increasing cumulative score, ties
broken toward smaller depth), so any prefix ``[:n]`` is itself a valid prefix-closed tree: budget
truncation is a slice, not a rebuild.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import torch

PAD = -2
NEG_INF = float("-inf")


class Tree(NamedTuple):
    tokens: torch.Tensor   # [B, N] int64, node 0 = anchor
    parents: torch.Tensor  # [B, N] int64, -1 root, PAD padding
    depth: torch.Tensor    # [B, N] int64
    rank: torch.Tensor     # [B, N] int64, index of the node's token in cand_ids[:, depth-1]; -1 root/pad
    score: torch.Tensor    # [B, N] float32, cumulative surrogate log-prob; 0 root, -inf pad

    @property
    def valid(self) -> torch.Tensor:
        return self.parents != PAD


# ----------------------------------------------------------------------------------------------
# tree construction
# ----------------------------------------------------------------------------------------------

def _child_logp(lp: torch.Tensor, slot: int, parent_rank: Optional[torch.Tensor]) -> torch.Tensor:
    """Children log-probs at ``slot`` for each parent: [B, W, K] (or [B, 1, K] for the root)."""
    if lp.dim() == 3:                                   # marginal [B, S, K]: parent-independent
        row = lp[:, slot, :].unsqueeze(1)
        return row if parent_rank is None else row.expand(-1, parent_rank.shape[1], -1)
    if parent_rank is None:                             # edge [B, S, K, K], slot 0: anchor row 0
        return lp[:, slot, 0, :].unsqueeze(1)
    K = lp.shape[-1]
    idx = parent_rank.clamp(min=0).unsqueeze(-1).expand(-1, -1, K)
    return lp[:, slot].gather(1, idx)


def frontier_tree(logp: torch.Tensor, cand_ids: torch.Tensor, anchor: torch.Tensor, budget: int,
                  *, width: Optional[int] = None, children: Optional[int] = None,
                  max_depth: Optional[int] = None) -> Tree:
    """Depth-synchronous frontier expansion, then one global top-``budget`` over every generated node.

    At depth 1 the root generates its top-``children`` candidates. At depth d >= 2 the ``width``
    best generated nodes of depth d-1 (the "expanded" set) each generate their top-``children``
    candidates at slot d-1. All generated nodes compete in one global selection by cumulative
    log-prob. Because a child never scores above its parent (log-probs <= 0) and ties are broken
    toward smaller depth, the selection is prefix-closed and in best-first order.

    ``logp``: marginal [B, S, K] or edge [B, S, K, K] log-probabilities (must be <= 0 for the
    prefix-closure guarantee); ``cand_ids`` [B, S, K]; ``anchor`` [B]. Returns a Tree with
    N = budget + 1 nodes (padded with PAD when fewer nodes exist or have finite score).
    """
    B, S, K = cand_ids.shape
    D = S if max_depth is None else min(int(max_depth), S)
    W = budget if width is None else int(width)
    C = min(K, budget) if children is None else min(int(children), K)
    dev = cand_ids.device
    lp = logp.float()

    gen_score, gen_rank, gen_parent, gen_depth = [], [], [], []
    offsets, total = [], 0
    exp_score, exp_rank = None, None
    for d in range(1, D + 1):
        cl = _child_logp(lp, d - 1, exp_rank)                         # [B, Wp, K]
        top_lp, top_rank = torch.sort(cl, dim=-1, descending=True, stable=True)
        top_lp, top_rank = top_lp[..., :C], top_rank[..., :C]        # [B, Wp, C]
        base = torch.zeros(B, 1, 1, device=dev) if exp_score is None else exp_score.unsqueeze(-1)
        sc = (base + top_lp).reshape(B, -1)
        rk = top_rank.reshape(B, -1)
        Wp = top_lp.shape[1]
        par = torch.arange(Wp, device=dev).repeat_interleave(C).unsqueeze(0).expand(B, -1)
        sc, order = torch.sort(sc, dim=-1, descending=True, stable=True)
        rk, par = rk.gather(1, order), par.gather(1, order)
        if d == 1:
            par = torch.full_like(par, -1)
        offsets.append(total)
        total += sc.shape[1]
        gen_score.append(sc), gen_rank.append(rk), gen_parent.append(par)
        gen_depth.append(torch.full_like(rk, d))
        exp_score, exp_rank = sc[:, :W], rk[:, :W]

    score = torch.cat(gen_score, 1)
    rank = torch.cat(gen_rank, 1)
    depth = torch.cat(gen_depth, 1)
    # parent as an index into the concatenation (-1 = root)
    par_cat = torch.cat([p if d == 0 else torch.where(p >= 0, p + offsets[d - 1], p)
                         for d, p in enumerate(gen_parent)], 1)
    if total < budget:                                                # tiny lattices: pad
        pad = budget - total
        score = torch.cat([score, torch.full((B, pad), NEG_INF, device=dev)], 1)
        rank = torch.cat([rank, torch.zeros(B, pad, dtype=rank.dtype, device=dev)], 1)
        depth = torch.cat([depth, torch.ones(B, pad, dtype=depth.dtype, device=dev)], 1)
        par_cat = torch.cat([par_cat, torch.full((B, pad), -1, dtype=par_cat.dtype, device=dev)], 1)

    sel_score, sel = torch.sort(score, dim=-1, descending=True, stable=True)
    sel_score, sel = sel_score[:, :budget], sel[:, :budget]          # [B, budget]
    out_pos = torch.full_like(score, -1, dtype=torch.long)
    out_pos.scatter_(1, sel, torch.arange(budget, device=dev).unsqueeze(0).expand(B, -1))
    s_rank, s_depth, s_pcat = rank.gather(1, sel), depth.gather(1, sel), par_cat.gather(1, sel)
    s_parent = torch.where(s_pcat >= 0, out_pos.gather(1, s_pcat.clamp(min=0)) + 1,
                           torch.zeros_like(s_pcat))
    valid = torch.isfinite(sel_score)
    flat = ((s_depth - 1).clamp(min=0) * K + s_rank).clamp(max=S * K - 1)
    s_tok = cand_ids.reshape(B, S * K).gather(1, flat)

    tokens = torch.cat([anchor.reshape(B, 1).to(cand_ids.dtype), torch.where(valid, s_tok, 0)], 1)
    parents = torch.cat([torch.full((B, 1), -1, dtype=torch.long, device=dev),
                         torch.where(valid, s_parent, torch.full_like(s_parent, PAD))], 1)
    depth_o = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=dev),
                         torch.where(valid, s_depth, 0)], 1)
    rank_o = torch.cat([torch.full((B, 1), -1, dtype=torch.long, device=dev),
                        torch.where(valid, s_rank, -1)], 1)
    score_o = torch.cat([torch.zeros(B, 1, device=dev), sel_score], 1)
    return Tree(tokens.long(), parents, depth_o, rank_o, score_o)


def ddtree(logp: torch.Tensor, cand_ids: torch.Tensor, anchor: torch.Tensor, budget: int,
           max_depth: Optional[int] = None) -> Tree:
    """DDTree: the ``budget`` highest-probability prefixes of the (truncated) lattice.

    With marginal ``logp`` [B, S, K] this is DDTree exactly (Prop. 2: top-B prefixes form an
    optimal tree for the factorised surrogate E[accepted] = sum of node prefix probabilities;
    Alg. 1 finds them with a heap). The heap is replaced by a frontier of width ``budget``,
    which contains every top-B prefix because a prefix's parent always outranks it (cf.
    DARTree's Lemma 1, arXiv:2608.13524). With edge ``logp`` [B, S, K, K] the same rule gives
    the best-first tree for the first-order (selector-conditioned) surrogate.
    Note: the surrogate only sees the K lattice candidates (Lemma 1 of DDTree needs K >= budget
    for global optimality over the vocabulary).
    """
    return frontier_tree(logp, cand_ids, anchor, budget, width=budget, children=None,
                         max_depth=max_depth)


def eagle2_tree(logp: torch.Tensor, cand_ids: torch.Tensor, anchor: torch.Tensor, *, topk: int,
                depth: int, total_tokens: int) -> Tree:
    """EAGLE-2 dynamic tree on a lattice: expand the ``topk`` best nodes of each depth with their
    ``topk`` best children (value = cumulative confidence product = sum of log-probs), for
    ``depth`` layers, then rerank all generated nodes and keep the best ``total_tokens``
    (non-root nodes; EAGLE's ``total_tokens`` counts drafts). Unlike EAGLE-2 itself this needs no
    extra drafter forward per depth: every depth's candidates come from the one block pass."""
    return frontier_tree(logp, cand_ids, anchor, total_tokens, width=topk, children=topk,
                         max_depth=depth)


def chain_tree(chain_tokens: torch.Tensor, anchor: torch.Tensor) -> Tree:
    """The stock verify layout as a tree: anchor + S drafts, each the child of the previous."""
    B, S = chain_tokens.shape
    dev = chain_tokens.device
    ar = torch.arange(S + 1, device=dev)
    return Tree(torch.cat([anchor.reshape(B, 1), chain_tokens], 1).long(),
                (ar - 1).unsqueeze(0).expand(B, -1).clone(), ar.unsqueeze(0).expand(B, -1).clone(),
                torch.full((B, S + 1), -1, dtype=torch.long, device=dev),
                torch.zeros(B, S + 1, device=dev))


# ----------------------------------------------------------------------------------------------
# Sequoia: DP-optimal static tree shape (offline)
# ----------------------------------------------------------------------------------------------

def sequoia_dp(accept: Sequence, n_nodes: int, max_depth: Optional[int] = None) -> dict:
    """DP-optimal tree of ``n_nodes`` nodes (root included) for per-(depth, rank) acceptance.

    ``accept``: either a 1-D sequence p[r] (positional acceptance, Sequoia Alg. 1) or a 2-D
    sequence p[d][r] for children at depth d+1 (depth-indexed, Sequoia App. F). Expected commit
    length of a tree = sum over nodes of the product of p along the root path (root counts 1:
    the bonus), assuming acceptance depends only on (depth, rank) and sibling events are
    disjoint (true for greedy with distinct sibling tokens). Children may use any subset of
    ranks (no monotonicity assumption on p).

    Returns ``{"value": F(n_nodes), "curve": [F(1), ..., F(n_nodes)], "parents": [...],
    "depth": [...], "rank": [...]}`` with nodes in DFS preorder (parents precede children;
    rank of the root is -1). ``curve[n-1]`` is the optimal expected commit with n nodes:
    combine with a measured verify-time table t(n) to pick n (Sequoia sec. 4.1).
    Cost O(D * K * N^2) in pure Python: offline only.
    """
    if len(accept) and not isinstance(accept[0], (list, tuple)) and not hasattr(accept[0], "__len__"):
        D = int(max_depth) if max_depth is not None else n_nodes - 1
        P = [list(map(float, accept))] * max(D, 0)
    else:
        P = [list(map(float, row)) for row in accept]
        D = len(P) if max_depth is None else min(int(max_depth), len(P))
        P = P[:D]
    N = int(n_nodes)
    K = max((len(r) for r in P), default=0)
    NEG = float("-inf")
    # F[d][n]: best value of a subtree rooted at depth d with n nodes; H[d][i][m]: best forest of
    # m nodes as children (depth d+1) of a depth-d node using ranks >= i.
    F = [[NEG] * (N + 1) for _ in range(D + 1)]
    H = [[[NEG] * (N + 1) for _ in range(K + 1)] for _ in range(D)]
    Hc = [[[None] * (N + 1) for _ in range(K + 1)] for _ in range(D)]
    F[D][1] = 1.0
    for d in range(D - 1, -1, -1):
        Kd = len(P[d])
        for i in range(K, -1, -1):
            H[d][i][0] = 0.0
        for m in range(1, N + 1):
            for i in range(Kd - 1, -1, -1):
                best, arg = H[d][i + 1][m], 0                         # skip rank i
                p = P[d][i]
                for s in range(1, m + 1):
                    f, h = F[d + 1][s], H[d][i + 1][m - s]
                    if f == NEG or h == NEG:
                        continue
                    v = p * f + h
                    if v > best:
                        best, arg = v, s
                H[d][i][m], Hc[d][i][m] = best, arg
        for n in range(1, N + 1):
            h = H[d][0][n - 1]
            F[d][n] = NEG if h == NEG else 1.0 + h
    parents, depth, rank = [], [], []

    def build(d, n, parent, r):
        me = len(parents)
        parents.append(parent), depth.append(d), rank.append(r)
        if d == D:
            return
        i, m = 0, n - 1
        while m > 0 and i < len(P[d]):
            s = Hc[d][i][m]
            if s:
                build(d + 1, s, me, i)
                m -= s
            i += 1

    if N >= 1 and F[0][N] > NEG:
        build(0, N, -1, -1)
    return {"value": F[0][N], "curve": [F[0][n] for n in range(1, N + 1)], "parents": parents,
            "depth": depth, "rank": rank}


def template_value(parents: Sequence[int], rank: Sequence[int], accept: Sequence) -> float:
    """Expected commit length of a static (rank) template under a (depth, rank) acceptance table."""
    two_d = hasattr(accept[0], "__len__")
    val, prob = 0.0, [0.0] * len(parents)
    for i, p in enumerate(parents):
        if p == PAD:
            continue
        if p < 0:
            prob[i] = 1.0
        else:
            d = 0
            j = p
            while j >= 0:
                d += 1
                j = parents[j]
            row = accept[d - 1] if two_d else accept
            prob[i] = prob[p] * float(row[rank[i]])
        val += prob[i]
    return val


def instantiate_template(parents: Sequence[int], rank: Sequence[int], cand_ids: torch.Tensor,
                         anchor: torch.Tensor, logp: Optional[torch.Tensor] = None) -> Tree:
    """Fill a static template (parents, child rank) from a batch lattice, batched, static shapes.

    Without ``logp`` (or with marginal [B, S, K] log-probs) a rank-r child at depth d takes
    ``cand_ids[:, d-1, r]`` (Medusa / Sequoia: rank = position in the per-slot top-K, which is
    sorted). With edge log-probs [B, S, K, K] rank r means the r-th best child *given the parent's
    candidate* (a selector-conditioned template). Requires parents to precede children.
    """
    B, S, K = cand_ids.shape
    dev = cand_ids.device
    N = len(parents)
    par = torch.tensor([p if p >= 0 else 0 for p in parents], device=dev)
    dep_l = [0] * N
    for i, p in enumerate(parents):
        dep_l[i] = dep_l[p] + 1 if p >= 0 else 0
    dep = torch.tensor(dep_l, device=dev)
    rk = torch.tensor([max(r, 0) for r in rank], device=dev)
    valid = torch.tensor([p != PAD for p in parents], device=dev)
    cidx = torch.full((B, N), -1, dtype=torch.long, device=dev)       # candidate index per node
    score = torch.zeros(B, N, device=dev)
    lp = None if logp is None else logp.float()
    order = None
    if lp is not None and lp.dim() == 4:
        order = torch.sort(lp, dim=-1, descending=True, stable=True)
    for d in range(1, max(dep_l, default=0) + 1):
        nodes = [i for i in range(N) if dep_l[i] == d and parents[i] != PAD]
        if not nodes:
            continue
        nd = torch.tensor(nodes, device=dev)
        r = rk[nd].unsqueeze(0).expand(B, -1)
        if order is not None:
            prow = cidx[:, par[nd]].clamp(min=0) if d > 1 else torch.zeros(B, len(nodes), dtype=torch.long, device=dev)
            ci = order.indices[:, d - 1].gather(1, prow.unsqueeze(-1).expand(-1, -1, K)).gather(
                2, r.unsqueeze(-1))[..., 0]
            step = lp[:, d - 1].gather(1, prow.unsqueeze(-1).expand(-1, -1, K)).gather(2, ci.unsqueeze(-1))[..., 0]
        else:
            ci = r
            step = lp[:, d - 1].gather(1, ci) if lp is not None else torch.zeros(B, len(nodes), device=dev)
        cidx[:, nd] = ci
        score[:, nd] = score[:, par[nd]] + step
    flat = ((dep.unsqueeze(0) - 1).clamp(min=0) * K + cidx.clamp(min=0)).clamp(max=S * K - 1)
    tok = cand_ids.reshape(B, S * K).gather(1, flat.expand(B, -1))
    tok[:, 0] = anchor.reshape(B)
    tok = torch.where(valid.unsqueeze(0), tok, 0)
    parents_t = torch.tensor(list(parents), device=dev).unsqueeze(0).expand(B, -1).clone()
    score = torch.where(valid.unsqueeze(0), score, NEG_INF)
    return Tree(tok.long(), parents_t, torch.where(valid, dep, 0).unsqueeze(0).expand(B, -1).clone(),
                torch.where(valid.unsqueeze(0), cidx, -1), score)


def rank_acceptance_table(cand_ids: torch.Tensor, truth: torch.Tensor,
                          truth_valid: Optional[torch.Tensor] = None):
    """Per-(depth, rank) acceptance counts from traced lattices (``lattice_trace``/``trace cand=true``).

    ``cand_ids`` [M, S, K] top-K per slot, ``truth`` [M, S] the target's greedy tokens after the
    anchor along the recorded trajectory (``replay.py``: committed tokens at P+1..P+S).
    A step reaches depth d (1-based) when truth[j] is among cand[j] for every j < d-1 (a full
    rank-K tree would still be alive). Returns ``(reach [S], hit [S, K])``; p[d][r] = hit/reach
    is the Sequoia / Medusa table. Marginal ranks only (positional acceptance assumption).
    """
    M, S, K = cand_ids.shape
    eq = cand_ids == truth.unsqueeze(-1)                             # [M, S, K]
    inside = eq.any(-1)
    if truth_valid is not None:
        inside = inside & truth_valid
        eq = eq & truth_valid.unsqueeze(-1)
    alive = torch.cat([torch.ones(M, 1, dtype=torch.bool, device=cand_ids.device),
                       torch.cumprod(inside.long(), 1)[:, :-1].bool()], 1)
    if truth_valid is not None:
        alive = alive & truth_valid
    reach = alive.sum(0)
    hit = (eq & alive.unsqueeze(-1)).sum(0)
    return reach, hit


# ----------------------------------------------------------------------------------------------
# budget control
# ----------------------------------------------------------------------------------------------

def cost_aware_node_count(score: torch.Tensor, cost: torch.Tensor, draft_cost: float = 0.0,
                          allowed: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Per-row node count maximising expected commit per unit time (CaDDTree / BASTION rule).

    ``score`` [B, N]: cumulative log-probs of a best-first tree (root at column 0, -inf pads);
    ``cost`` [N + 1]: measured verify time for n nodes incl. the root, index n (index 0 unused);
    ``allowed``: captured graph sizes (node counts incl. root). Returns n* [B] (int64), the
    argmax over allowed n of (sum_{i<n} p_i) / (draft_cost + cost[n]) with p_0 = 1 for the root.
    Pure tensor ops: no host sync. Only meaningful if the surrogate is calibrated.
    """
    B, N = score.shape
    p = torch.exp(score.float())
    p[:, 0] = 1.0
    gain = torch.cumsum(p, 1)                                       # gain[:, n-1] = E[commit | n nodes]
    c = cost.float().to(score.device)[1:N + 1]
    rate = gain / (draft_cost + c).unsqueeze(0)
    if allowed is not None:
        mask = torch.zeros(N, dtype=torch.bool, device=score.device)
        mask[[a - 1 for a in allowed if 1 <= a <= N]] = True
        rate = torch.where(mask.unsqueeze(0), rate, NEG_INF)
    return rate.argmax(1) + 1


def truncate_tree(tree: Tree, n: torch.Tensor) -> Tree:
    """Keep the first n[b] nodes of each (best-first ordered) tree, PAD the rest; shapes unchanged."""
    B, N = tree.parents.shape
    keep = torch.arange(N, device=n.device).unsqueeze(0) < n.reshape(B, 1)
    return Tree(torch.where(keep, tree.tokens, 0), torch.where(keep, tree.parents, PAD),
                torch.where(keep, tree.depth, 0), torch.where(keep, tree.rank, -1),
                torch.where(keep, tree.score, NEG_INF))


# ----------------------------------------------------------------------------------------------
# layout helpers
# ----------------------------------------------------------------------------------------------

def _as_batch(parents) -> torch.Tensor:
    t = parents if torch.is_tensor(parents) else torch.tensor(list(parents))
    return t.long().unsqueeze(0) if t.dim() == 1 else t.long()


def depth_of(parents) -> torch.Tensor:
    """Depth per node [B, N] (root 0, PAD 0); parents precede children. O(N) pointer jumps."""
    p = _as_batch(parents)
    B, N = p.shape
    d = torch.zeros(B, N, dtype=torch.long, device=p.device)
    cur = p.clone()
    for _ in range(N):                                             # static bound: depth <= N-1
        live = cur >= 0
        d = d + live.long()
        cur = torch.where(live, p.gather(1, cur.clamp(min=0)), cur)
    return d


def ancestor_mask(parents, max_depth: Optional[int] = None) -> torch.Tensor:
    """mask[b, i, j] = node i may attend to node j (j == i or j an ancestor of i). Padded nodes see
    only themselves. ``max_depth`` bounds the pointer walk (static; default N-1). Graph-capturable:
    one one-hot scatter per hop, no boolean indexing."""
    p = _as_batch(parents)
    return ancestor_mask_static(p, p.shape[1] - 1 if max_depth is None else int(max_depth))


def ancestor_mask_static(parents, max_depth: int) -> torch.Tensor:
    p = _as_batch(parents)
    B, N = p.shape
    m = torch.eye(N, dtype=torch.bool, device=p.device).unsqueeze(0).repeat(B, 1, 1)
    cur = p.clone()
    for _ in range(int(max_depth)):
        live = cur >= 0
        hop = torch.zeros(B, N, N + 1, dtype=torch.bool, device=p.device)
        hop.scatter_(2, torch.where(live, cur, N).unsqueeze(-1), True)
        m = m | hop[..., :N]
        cur = torch.where(live, p.gather(1, cur.clamp(min=0)), cur)
    return m


def positions(parents, prefix_len) -> torch.Tensor:
    """branch_oracle convention: prefix_len - 1 + depth (root = last committed token). In the engine
    use ``seq_len + depth`` (root = bonus at seq_len)."""
    d = depth_of(parents)
    pl = prefix_len if torch.is_tensor(prefix_len) else torch.tensor(prefix_len)
    return pl.reshape(-1, 1).to(d.device) - 1 + d


def first_child_next_sibling(parents):
    """(first_child [B, N], next_sibling [B, N]), -1 where none; children in node-index order.

    This is the link structure SGLang's EAGLE tree verify walks (``retrive_next_token`` = first
    child, ``retrive_next_sibling`` = next sibling; see research/knowledge/code/tree.md)."""
    p = _as_batch(parents)
    B, N = p.shape
    dev = p.device
    idx = torch.arange(N, device=dev).unsqueeze(0).expand(B, -1)
    big = N + 1
    # first child: min child index per parent
    pc = torch.where(p >= 0, p, N)
    fc = torch.full((B, N + 1), big, dtype=torch.long, device=dev)
    fc.scatter_reduce_(1, pc, idx, reduce="amin", include_self=True)
    first_child = fc[:, :N]
    first_child = torch.where(first_child >= big, -1, first_child)
    # next sibling: smallest j > i with the same parent
    same = (p.unsqueeze(2) == p.unsqueeze(1)) & (p.unsqueeze(2) >= 0)
    later = idx.unsqueeze(2) < idx.unsqueeze(1)
    cand = torch.where(same & later, idx.unsqueeze(1).expand(B, N, N), big)
    ns = cand.min(2).values
    next_sibling = torch.where(ns >= big, -1, ns)
    return first_child, next_sibling


def dfs_preorder(parents: Sequence[int]) -> list:
    """Permutation (list of old indices) putting a single tree in DFS preorder (children in index
    order), padded nodes last. Preorder makes every root-to-leaf path a set of chains that
    branch-local / chain-decomposed recurrent kernels can walk."""
    ch = {}
    for i, q in enumerate(parents):
        if q >= 0:
            ch.setdefault(q, []).append(i)
    out, stack = [], [0]
    while stack:
        i = stack.pop()
        out.append(i)
        stack.extend(reversed(ch.get(i, [])))
    return out + [i for i, q in enumerate(parents) if q == PAD]


def permute_tree(parents: Sequence[int], perm: Sequence[int]) -> list:
    """Parents after reordering nodes by ``perm`` (new i = old perm[i])."""
    inv = {old: new for new, old in enumerate(perm)}
    return [parents[o] if parents[o] < 0 else inv[parents[o]] for o in perm]


# ----------------------------------------------------------------------------------------------
# greedy tree acceptance
# ----------------------------------------------------------------------------------------------

class Accept(NamedTuple):
    accept_len: torch.Tensor   # [B] accepted drafts (excl. root and bonus)
    leaf: torch.Tensor         # [B] node index of the deepest accepted node (0 = root only)
    path: torch.Tensor         # [B, Dmax+1] accepted node indices root first, -1 padded
    committed: torch.Tensor    # [B, Dmax+1] accepted draft tokens then the bonus, 0 padded
    commit_len: torch.Tensor   # [B] accept_len + 1, cut by EOS-at-bonus semantics and budget


def greedy_tree_accept(tokens: torch.Tensor, parents: torch.Tensor, predictions: torch.Tensor, *,
                       eos: Optional[int] = None, budget=None, max_depth: Optional[int] = None) -> Accept:
    """Batched greedy acceptance on a tree; committed tokens equal ``branch_oracle.oracle_accept``.

    ``predictions[b, i]`` = target argmax given prefix + node i's root path (the verify logits row
    of node i). A node is accepted iff every node on its path matches its parent's prediction and
    no proper ancestor predicted ``eos`` (the walk stops at an EOS prediction, which becomes the
    bonus). The deepest accepted node wins; ties (duplicate siblings) go to the smallest node
    index - every tied path carries identical tokens, so committed tokens are unique (node ids
    are not). ``budget`` (int or [B]) caps commit_len. No host syncs; loop bounds are static.
    """
    tok, par, pred = tokens.long(), _as_batch(parents), predictions.long()
    B, N = tok.shape
    dev = tok.device
    D = N - 1 if max_depth is None else int(max_depth)
    valid = par != PAD
    pc = par.clamp(min=0)
    ppred = pred.gather(1, pc)
    ok = valid & (par >= 0) & (tok == ppred)
    if eos is not None:
        ok = ok & (ppred != eos)
    acc = par == -1
    for _ in range(D):
        acc = (par == -1) | (ok & acc.gather(1, pc))
    dep = depth_of(par)
    idx = torch.arange(N, device=dev).unsqueeze(0)
    key = torch.where(acc, dep * (N + 1) + (N - idx), -1)
    leaf = key.argmax(1)
    alen = dep.gather(1, leaf.unsqueeze(1))[:, 0]
    path = torch.full((B, D + 2), -1, dtype=torch.long, device=dev)   # column D+1 is a sink
    cur = leaf.clone()
    for _ in range(D + 1):
        live = cur >= 0
        col = torch.where(live, dep.gather(1, cur.clamp(min=0).unsqueeze(1))[:, 0], D + 1)
        path.scatter_(1, col.clamp(max=D + 1).unsqueeze(1), cur.unsqueeze(1))
        cur = torch.where(live, par.gather(1, cur.clamp(min=0).unsqueeze(1))[:, 0], cur)
    path = path[:, : D + 1]
    j = torch.arange(D + 1, device=dev).unsqueeze(0)
    path_tok = tok.gather(1, path.clamp(min=0))
    shifted = torch.cat([path_tok[:, 1:], torch.zeros(B, 1, dtype=torch.long, device=dev)], 1)
    bonus = pred.gather(1, leaf.unsqueeze(1))
    committed = torch.where(j < alen.unsqueeze(1), shifted, 0)
    committed = committed.scatter(1, alen.unsqueeze(1), bonus)
    clen = alen + 1
    if budget is not None:
        bud = budget if torch.is_tensor(budget) else torch.full((B,), int(budget), device=dev)
        clen = torch.minimum(clen, bud.to(dev).long())
        committed = torch.where(j < clen.unsqueeze(1), committed, 0)
    path = torch.where(j <= alen.unsqueeze(1), path, -1)
    return Accept(alen, leaf, path, committed, clen)


def fcns_greedy_accept(tokens: Sequence[int], first_child: Sequence[int], next_sibling: Sequence[int],
                       predictions: Sequence[int]) -> list:
    """Reference walk over first-child / next-sibling links (one tree, Python): from the root,
    scan the current node's children via next_sibling and descend into the first child whose
    token equals the current node's prediction. Returns the committed tokens. This is the
    *first-match* rule; it equals ``oracle_accept`` when siblings carry distinct tokens (any
    lattice-built tree), and may commit fewer tokens with duplicate siblings."""
    cur, out = 0, []
    while True:
        want = predictions[cur]
        c = first_child[cur]
        while c != -1 and tokens[c] != want:
            c = next_sibling[c]
        if c == -1:
            return out + [want]
        out.append(tokens[c])
        cur = c


def tree_accept_len_from_truth(tokens: torch.Tensor, parents: torch.Tensor, truth: torch.Tensor,
                               max_depth: Optional[int] = None) -> torch.Tensor:
    """Offline greedy screen: accepted drafts of a tree given the target's greedy continuation
    ``truth`` [B, S] (replay.py truth: committed tokens P+1..P+S). Exact at T=0 for the recorded
    anchor because an accepted node's prediction is truth[depth]."""
    par = _as_batch(parents)
    dep = depth_of(par)
    S = truth.shape[1]
    pred = torch.where(dep < S, truth.long().gather(1, dep.clamp(max=S - 1)), -1)
    return greedy_tree_accept(tokens, par, pred, max_depth=max_depth).accept_len


# ----------------------------------------------------------------------------------------------
# recurrent-layer references (for tree-verify kernel tests; fp64 recommended)
# ----------------------------------------------------------------------------------------------

def gdn_tree_wy_reference(parents: Sequence[int], q, k, v, g, beta, S0):
    """All node outputs and states of a gated-delta-rule layer on a tree, in one triangular solve.

    Same recurrence as ``branch_oracle.gdn_sequential`` (S_t = a S + beta k (v - (a S)^T k)^T,
    o_t = S_t^T q_t, a = exp(g)), evaluated along each node's root path, root included.
    With la_i = sum of g over i's path (log cumulative decay), strict ancestor set A(i):

        X[i, j] = beta_i exp(la_i - la_j) k_i.k_j        (j in A(i))
        (I + X) U = beta * V - beta * exp(la) * (K S0)    (unit lower-triangular: parents first)
        O[i]   = exp(la_i) S0^T q_i + sum_{j in A(i) + i} exp(la_i - la_j) (q_i.k_j) U[j]
        S[i]   = exp(la_i) S0 + sum_{j in A(i) + i} exp(la_i - la_j) k_j U[j]^T

    (the ancestor-masked chunk/WY form used by Weaver arXiv:2607.06763 eq. 9-12, TreeWY
    arXiv:2608.20961, Bole arXiv:2608.01651). Shapes: q,k [N,H,K], v [N,H,V], g,beta [N,H],
    S0 [H,K,V]. Returns (O [N,H,V], states [N,H,K,V], U [N,H,V]). Reference only (dense N x N).
    """
    par = [p for p in parents]
    N = len(par)
    dt = torch.float64
    q, k, v, g, beta, S0 = (x.to(dt) for x in (q, k, v, g, beta, S0))
    M = ancestor_mask(par)[0].to(q.device)                         # incl. self
    Ms = M & ~torch.eye(N, dtype=torch.bool, device=q.device)
    la = torch.einsum("ij,jh->ih", M.to(dt), g)                    # [N, H]
    diff = la.unsqueeze(1) - la.unsqueeze(0)                       # [N, N, H] la_i - la_j
    dec_s = torch.where(Ms.unsqueeze(-1), torch.exp(torch.where(Ms.unsqueeze(-1), diff, 0.0)), 0.0)
    dec_m = torch.where(M.unsqueeze(-1), torch.exp(torch.where(M.unsqueeze(-1), diff, 0.0)), 0.0)
    kk = torch.einsum("ihd,jhd->ijh", k, k)
    qk = torch.einsum("ihd,jhd->ijh", q, k)
    X = (beta.unsqueeze(1) * dec_s * kk).permute(2, 0, 1)          # [H, N, N]
    Y = (dec_m * qk).permute(2, 0, 1)
    a = torch.exp(la)                                              # [N, H]
    KS0 = torch.einsum("ihk,hkv->ihv", k, S0)
    rhs = (beta.unsqueeze(-1) * (v - a.unsqueeze(-1) * KS0)).permute(1, 0, 2)   # [H, N, V]
    I = torch.eye(N, dtype=dt, device=q.device).expand_as(X)
    U = torch.linalg.solve_triangular(I + X, rhs, upper=False, unitriangular=True)  # [H, N, V]
    QS0 = torch.einsum("ihk,hkv->ihv", q, S0).permute(1, 0, 2)
    O = a.T.unsqueeze(-1) * QS0 + Y @ U                             # [H, N, V]
    W = dec_m.permute(2, 0, 1)                                     # [H, i, j]
    states = a.T[..., None, None] * S0.unsqueeze(1) + torch.einsum("hij,jhk,hjv->hikv", W, k, U)
    return O.permute(1, 0, 2), states.permute(1, 0, 2, 3), U.permute(1, 0, 2)


def tree_conv_window_indices(parents: Sequence[int], width: int) -> torch.Tensor:
    """[N, width] input indices of each node's causal-conv window along its root path, oldest first:
    >= 0 is a tree node, -1 is the newest committed history entry, -2 the one before, and so on.
    This is the parent-indexed gather a tree-aware ``causal_conv1d`` must perform instead of
    reading flat neighbours (rows of a padded node are its own path from the root)."""
    N = len(parents)
    out = torch.empty(N, width, dtype=torch.long)
    for i in range(N):
        path, j = [], i
        while j >= 0:
            path.append(j)
            j = parents[j]
        path = path[::-1]
        if parents[i] == PAD:
            path = [i]
        seq = list(range(-(width - 1), 0)) + path
        out[i] = torch.tensor(seq[-width:])
    return out


def tree_causal_conv(parents: Sequence[int], x: torch.Tensor, weight: torch.Tensor,
                     history: torch.Tensor):
    """Depthwise causal conv over a tree via ``tree_conv_window_indices``. x [N, D], weight [D, W],
    history [W-1, D] (oldest first). Returns y [N, D] and per-node committed windows [N, W-1, D]
    (what a commit of that node must write as the new conv state)."""
    W = weight.shape[1]
    idx = tree_conv_window_indices(parents, W).to(x.device)
    src = torch.cat([history, x], 0)                               # history row h -> index h
    rows = torch.where(idx >= 0, idx + (W - 1), (W - 1) + idx)
    win = src[rows]                                                # [N, W, D]
    y = (win.float() * weight.T.float().unsqueeze(0)).sum(1)
    return y, win[:, 1:]
