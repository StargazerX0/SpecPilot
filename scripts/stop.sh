#!/usr/bin/env bash
# Stop the server that scripts/serve.sh started on a port.
#   bash scripts/stop.sh <port>
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SPECPILOT_SETTING=${SPECPILOT_SETTING:-h100_c1_qwen38-27b_dflash2}
exec "$REPO/research/scripts/stop.sh" "${1:?usage: stop.sh <port>}"
