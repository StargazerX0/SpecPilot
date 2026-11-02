#!/usr/bin/env bash
# stop.sh [port]: terminate the process group recorded by serve.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
PORT=${1:-30000}; F=$INFER_WORK/run/server_$PORT.pgid
[ -f "$F" ] || { echo "no server recorded for port $PORT"; exit 0; }
PG=$(cat "$F"); kill -TERM -- -"$PG" 2>/dev/null || true
for i in $(seq 1 60); do kill -0 -- -"$PG" 2>/dev/null || break; sleep 1; done
kill -KILL -- -"$PG" 2>/dev/null || true; rm -f "$F"; sleep 2; echo "stopped pgid $PG"
