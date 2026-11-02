#!/usr/bin/env bash
# Start one server for a setting and block until it is ready.
#   bash scripts/serve.sh <setting> <ar|stock|final> [gpu] [port]
#     ar     target only (autoregressive decoding), stock engine
#     stock  the unmodified speculative decoder (DFlash2 or DSpark) at the setting's stock block, stock engine
#     final  SpecPilot: final engine + the final's env switches, block size and launch args (config.json `final`)
# gpu defaults to 0, port to 30000 + 10 * gpu (torch.distributed uses port + 1). Stop with scripts/stop.sh <port>.
# The engine tree is assembled once under $SPECPILOT_WORK/engines. The server starts from a clean environment whose
# PYTHONPATH holds only that tree, so nothing else on the machine can shadow it.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SETTING=${1:?usage: serve.sh <setting> <ar|stock|final> [gpu] [port]}; ARM=${2:?arm: ar|stock|final}
GPU=${3:-0}; PORT=${4:-$((30000 + 10 * GPU))}
case $ARM in ar|stock) KIND=stock ;; final) KIND=final ;; *) echo "unknown arm $ARM"; exit 2 ;; esac
[ "$ARM" = ar ] && MODE=ar || MODE=dflash2
export SPECPILOT_SETTING=$SETTING
source "$REPO/research/env.sh" > /dev/null
ENG=$SPECPILOT_WORK/engines/$SETTING-$KIND
[ -f "$ENG/TREE_SHA" ] || { rm -rf "$ENG"; python3 "$REPO/scripts/build_engine.py" "$SETTING" "$KIND" "$ENG"; }
LOG=$SPECPILOT_WORK/logs/server_${SETTING}_${ARM}_$PORT.log
mkdir -p "$(dirname "$LOG")"
# Engine import path, per-engine build caches, and for the final its env switches / block size / launch args.
EXPORTS=$(python3 - "$REPO" "$SETTING" "$KIND" "$ENG" "$XINFER_SHARED" <<'PY'
import json, shlex, sys
from pathlib import Path
repo, setting, kind, eng, shared = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5])
cfg = json.loads((repo / "settings" / setting / "config.json").read_text())
sha = (eng / "TREE_SHA").read_text().strip()
own = shared / "cache/engines" / sha[:16]
env = {"XINFER_ENGINE": eng / "python", "XINFER_LAB": eng / "lab", "PYTHONPATH": f"{eng / 'python'}:{eng / 'lab'}",
       "SGLANG_CACHE_DIR": own / "sglang", "SGLANG_JIT_CACHE_DIR": own / "sglang/jit",
       "TORCH_EXTENSIONS_DIR": own / "torch_extensions", "SNAPSHOT_SHA": sha}
launch = []
if kind == "final":
    f = cfg["final"]
    env["SPECPILOT_BLOCK_SIZE"] = f["block_size"]
    env["XINFER_CANDIDATE"] = f["name"]
    env.update(f["env"])
    launch = f["launch_args"]
for k, v in env.items():
    print(f"export {k}={shlex.quote(str(v))}")
print(f"LAUNCH=({' '.join(shlex.quote(x) for x in launch)})")
PY
)
PASS=(HOME="$HOME" USER="$USER" LOGNAME="$USER" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 TERM=dumb
      SPECPILOT_DATA="$SPECPILOT_DATA" XINFER_SHARED="$XINFER_SHARED" SPECPILOT_WORK="$SPECPILOT_WORK"
      SPECPILOT_SETTING="$SETTING" CUDA_VISIBLE_DEVICES="$GPU" SPECPILOT_NCCL_PORT=$((PORT + 1)))
[ -n "${HF_TOKEN:-}" ] && PASS+=(HF_TOKEN="$HF_TOKEN")
echo "[serve] $SETTING $ARM on GPU $GPU port $PORT (engine $ENG, log $LOG)"
env -i "${PASS[@]}" bash -c '
set -euo pipefail
eval "$1"; REPO=$2; MODE=$3; PORT=$4; LOG=$5
source "$REPO/research/env.sh" > /dev/null
"$REPO/research/scripts/serve.sh" "$MODE" "$PORT" "$LOG" "${LAUNCH[@]}"
' _ "$EXPORTS" "$REPO" "$MODE" "$PORT" "$LOG"
echo "[serve] ready: http://127.0.0.1:$PORT (stop with scripts/stop.sh $PORT)"
