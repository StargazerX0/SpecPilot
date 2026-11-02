#!/usr/bin/env bash
# Launch the SGLang server on the GPU in CUDA_VISIBLE_DEVICES (one GPU per server). Use scripts/serve.sh.
#   research/scripts/launch.sh ar      [port]   # target-only autoregressive baseline
#   research/scripts/launch.sh dflash2 [port]   # the setting's speculative decoder (DFlash2 or DSpark)
# Extra args after the port are forwarded to sglang.launch_server. Server flags come from the setting's config
# (`stock.server_flags`, exported as SPECPILOT_SERVER_FLAGS), the speculative algorithm and its fixed arguments from
# `stock.spec_flags`, the verify width (--speculative-num-draft-tokens) from SPECPILOT_BLOCK_SIZE. The torch.distributed
# store listens on SPECPILOT_NCCL_PORT or port+1, never on a random free port (servers booting together raced for one).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
MODE=${1:?usage: launch.sh ar|dflash2|spec [port] [extra args...]}; PORT=${2:-30000}; shift $(( $# >= 2 ? 2 : 1 ))
: "${SPECPILOT_SERVER_FLAGS:?SPECPILOT_SERVER_FLAGS unset: source research/env.sh with a valid SPECPILOT_SETTING}"
read -r -a SERVER_FLAGS <<< "$SPECPILOT_SERVER_FLAGS"
# `stock.server_env` of the setting (memory-management switches; never a semantics change)
for kv in ${SPECPILOT_SERVER_ENV:-}; do export "$kv"; done
COMMON=(
  --model-path "$TARGET_MODEL"
  "${SERVER_FLAGS[@]}"
  --random-seed 0
  --host 127.0.0.1 --port "$PORT" --nccl-port "${SPECPILOT_NCCL_PORT:-$((PORT + 1))}"
  --served-model-name "$SERVED_MODEL_NAME"
)
case $MODE in
  ar)      read -r -a SPEC <<< "${SPECPILOT_AR_FLAGS:-}" ;;   # `stock.ar_flags` (target-only memory split)
  dflash2|spec)
    read -r -a SPEC_FLAGS <<< "${SPECPILOT_SPEC_FLAGS:---speculative-algorithm DFLASH}"
    SPEC=("${SPEC_FLAGS[@]}" --speculative-draft-model-path "$DRAFT_MODEL" --speculative-num-draft-tokens "$SPECPILOT_BLOCK_SIZE")
    # DSpark: the verify width is gamma + 1 and SGLang requires both flags to agree, so gamma follows the block size
    case " ${SPEC_FLAGS[*]} " in *" DSPARK "*) SPEC+=(--speculative-dspark-block-size "$((SPECPILOT_BLOCK_SIZE - 1))") ;; esac ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
cd "$INFER_REPO"
exec python -m sglang.launch_server "${COMMON[@]}" "${SPEC[@]}" "$@"
