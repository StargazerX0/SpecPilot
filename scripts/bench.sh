#!/usr/bin/env bash
# Speed benchmark of a running server (scripts/serve.sh) with the paper's workload.
#   bash scripts/bench.sh <setting> <tag> [gpu] [port]
# Workload: first turns of the dev split (gsm8k, humaneval, math500, mbpp, mt-bench), greedy, thinking on, 2048 new
# tokens. Single-request settings run at concurrency 1 on the first 6 items of each suite (30 requests, as in the
# paper's single-request tables; ITEMS=20 for all 100); batching settings run closed loop at the setting's concurrency
# levels (16/20/24) over all 100 requests, each level preceded by an untimed warm-up at that concurrency.
# Output: $SPECPILOT_WORK/bench/<tag>.json (per-request tokens, latency, verify steps, output ids and text).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SETTING=${1:?usage: bench.sh <setting> <tag> [gpu] [port]}; TAG=${2:?tag}
GPU=${3:-0}; PORT=${4:-$((30000 + 10 * GPU))}
export SPECPILOT_SETTING=$SETTING
source "$REPO/research/env.sh" > /dev/null
OUT=$SPECPILOT_WORK/bench/$TAG.json; mkdir -p "$(dirname "$OUT")"
KEEP=()
python3 -c "import json,sys; sys.exit(0 if json.load(open('$REPO/settings/$SETTING/config.json'))['keep_special_tokens'] else 1)" && KEEP=(--keep-special-tokens)
if [ "$SPECPILOT_LOAD_LEVELS" = 1 ]; then
  LOAD=(--concurrency 1 --items-per-suite "${ITEMS:-6}")
else
  LOAD=(--levels "$SPECPILOT_LOAD_LEVELS" --min-requests-per-slot 4)
fi
cd "$REPO"
env -u PYTHONPATH python research/bench/bench.py --tag "$TAG" --port "$PORT" --split dev --first-turn-only \
  --max-new-tokens 2048 --repeats 1 --no-full-warmup "${LOAD[@]}" --gpu-index "$GPU" --tokenizer "$TARGET_MODEL" \
  "${KEEP[@]}" --out "$OUT"
echo "[bench] wrote $OUT"
