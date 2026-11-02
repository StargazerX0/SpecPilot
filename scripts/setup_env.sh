#!/usr/bin/env bash
# Build the Python environment once per machine (idempotent; re-run after any failure).
#   bash scripts/setup_env.sh
# Creates $XINFER_SHARED/venv (Python 3.11, the exact package set of research/requirements.lock.txt, this repo's stock
# engine installed editable) and, on drivers older than R580, NVIDIA's CUDA 13.0 forward-compat libcuda.
# XINFER_SHARED defaults to $SPECPILOT_DATA/shared, SPECPILOT_DATA to $HOME/specpilot_data (research/env.sh).
# Models go to $XINFER_SHARED/models (scripts/download_models.sh).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SPECPILOT_SETTING=${SPECPILOT_SETTING:-h100_c1_qwen38-27b_dflash2}
source "$REPO/research/env.sh"
say() { echo "[setup $(date +%H:%M:%S)] $*"; }
mkdir -p "$XINFER_SHARED"/{models,cache} "$SPECPILOT_WORK"/{logs,run}
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }

# 1. venv: locked packages, then the stock engine (python/) editable. The Rust extensions are not on this path.
[ -x "$INFER_VENV/bin/python" ] || uv venv -q -p 3.11 --seed "$INFER_VENV"
PY=$INFER_VENV/bin/python
uv pip install -q --python "$PY" -r "$REPO/research/requirements.lock.txt"
SGLANG_BUILD_RUST_EXTS=none uv pip install -q --python "$PY" --no-deps -e "$REPO/python"
# The JIT toolchain must match torch's CUDA 13.0 runtime headers and the 13.0 compat driver.
uv pip install -q --python "$PY" "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*" "huggingface_hub[hf_xet]"
uv pip install -q --python "$PY" -e "$REPO/research/dflash_lab"
CU=$($PY -c "import nvidia, os; print(os.path.join(nvidia.__path__[0], 'cu13'))")
mkdir -p "$CU/lib64"; ln -sf ../lib/libcudart.so.13 "$CU/lib64/libcudart.so"   # JIT links -lcudart from lib64

# 2. driver: torch 2.13+cu130 needs R580; older datacenter drivers use NVIDIA's forward-compat libs.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
if [ "${DRV%%.*}" -lt 580 ] && [ ! -e "$XINFER_SHARED/cuda-compat/usr/local/cuda-13.0/compat/libcuda.so.1" ]; then
  say "driver $DRV < 580: fetching cuda-compat-13-0 (580.178.04)"
  mkdir -p "$XINFER_SHARED/cuda-compat"
  curl -fsSL -o "$XINFER_SHARED/cuda-compat/compat.deb" \
    https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-0_580.178.04-1_amd64.deb
  (cd "$XINFER_SHARED/cuda-compat" && dpkg-deb -x compat.deb . && rm compat.deb)
fi
source "$REPO/research/env.sh"
$PY -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda ok on', torch.cuda.get_device_name(0))"
say "done. Next: scripts/download_models.sh <setting>"
