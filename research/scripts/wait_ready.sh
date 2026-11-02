#!/usr/bin/env bash
# wait_ready.sh <log> [port] [timeout_s]: block until /health_generate answers; exit 1 on crash.
LOG=$1; PORT=${2:-30000}; T=${3:-1800}
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
PG=$(cat "$INFER_WORK/run/server_$PORT.pgid" 2>/dev/null || echo "")
for i in $(seq 1 "$T"); do
  if grep -q "The server is fired up" "$LOG" 2>/dev/null && curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then echo "ready ($LOG)"; exit 0; fi
  if grep -q -E "Scheduler hit an exception|Received sigquit" "$LOG" 2>/dev/null || { [ -n "$PG" ] && ! kill -0 -- -"$PG" 2>/dev/null; }; then
    echo "server failed; see $LOG"; exit 1; fi
  sleep 1
done; echo "timeout waiting for $LOG"; exit 1
