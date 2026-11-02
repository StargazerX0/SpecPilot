#!/usr/bin/env bash
# serve.sh <ar|dflash2> [port] [log] [extra launch args...]: start launch.sh detached in its
# own process group, record the pgid, and block until the server is ready. The boot holds the machine-wide CUDA-init
# slot (research/autoresearch/cudaslot.py) from the launch until the server is healthy (SPECPILOT_CUDASLOT_DONE=1 skips it).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
MODE=$1; PORT=${2:-30000}; LOG=${3:-$INFER_WORK/logs/server_${MODE}_${PORT}.log}; shift $(( $# >= 3 ? 3 : $# ))
mkdir -p "$INFER_WORK/run" "$(dirname "$LOG")"
SLOT=
CUDASLOT="$INFER_REPO/research/autoresearch/cudaslot.py"
if [ "${SPECPILOT_CUDASLOT_DONE:-}" != 1 ]; then
  SLOT=$(python3 -I "$CUDASLOT" acquire --kind server --owner-pid $$ --what "serve.sh $MODE port $PORT") || SLOT=
  [ -n "$SLOT" ] && export SPECPILOT_CUDASLOT=$SLOT
fi
setsid "$(dirname "${BASH_SOURCE[0]}")/launch.sh" "$MODE" "$PORT" "$@" > "$LOG" 2>&1 < /dev/null &
echo $! > "$INFER_WORK/run/server_$PORT.pgid"
rc=0
"$(dirname "${BASH_SOURCE[0]}")/wait_ready.sh" "$LOG" "$PORT" || rc=$?
if [ -n "$SLOT" ]; then
  python3 -I "$CUDASLOT" release "$SLOT" --reason "$([ $rc = 0 ] && echo healthy || echo boot_failed)" > /dev/null || true
fi
exit $rc
