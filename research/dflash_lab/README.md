# dflash_lab: DFlash2 lab hooks and diagnostics as an SGLang plugin

The plugin is part of every candidate snapshot (`<snapshot>/lab`) and is imported from there
(PYTHONPATH). SGLang calls `dflash_lab.plugin:register` in every server process before model load
and CUDA-graph capture, so hooks on captured code (draft backbone, conv, selector) are baked into
the graphs. `register` always prints one runtime-evidence line per process (the directories `sglang`
and the plugin came from) and refuses banned environment variables; with `DFLASH_LAB` unset it
installs no hooks. The venv's installed `dflash-lab` distribution only provides the entry point.

| experiment | hook | purpose |
|---|---|---|
| `noop` | refine pipeline + empty refiner | control: must be token-identical to stock |
| `trace` | refiner (first) + step observer | per-step drafts, confidence, candidates, commits, history tail for `research/autoresearch/replay.py` (diagnostic) |
| `verify_trace` | observe `DFlashWorkerV2._accept_block` | verify-shape top-k and emitted-token logprobs along the committed path (chain layout; diagnostic) |
| `lattice_trace` | copy of `build_lattice` + step observer | binary per-step selector lattice dump (scores, candidates; optional target argmax / unary / hiddens; diagnostic) |
| `accept_trace` | observe `_accept_block` | per-step commit-length histogram (diagnostic) |

Diagnostic experiments (`DIAGNOSTIC = True`) sync the host every step: untimed runs only
(`ar live --diag trace,verify_trace`).

Add an experiment: `dflash_lab/experiments/<name>.py` with `apply(params)` using `dflash_lab.core`
(`hook(target, fn, kind)`, `add_refiner` for greedy-row proposal edits after drafting, `need_unary`,
`need_hist`, `on_init` for buffers allocated before graph capture, `add_step_observer`,
`add_pre_step`, `add_counters`). Registration failures exit the server process. No host syncs,
`.item()`, `.tolist()` or data-dependent shapes on the decode path of timed runs. Params:
`DFLASH_LAB_PARAMS='{"<name>": {...}}'` (the controller sets both from the candidate spec).

Hooks suit proposal-side changes. Changes to the block/tree layout, verification kernels,
acceptance, state commit or scheduling belong in the candidate's engine tree (`<workspace>/python`).
Weights stay fixed; the target's greedy verification decides what is emitted.
