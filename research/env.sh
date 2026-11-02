# Source before every command:  SPECPILOT_SETTING=<setting> source research/env.sh
#
# Data lives outside the checkout under SPECPILOT_DATA (default $HOME/specpilot_data):
#   XINFER_SHARED  ($SPECPILOT_DATA/shared) venv, checkpoints (models/), HF cache, CUDA compat libs, build caches
#   SPECPILOT_WORK ($SPECPILOT_DATA/work)   built engine trees, server logs, benchmark results
# TARGET_MODEL / DRAFT_MODEL / SPECPILOT_BLOCK_SIZE / SPECPILOT_SERVER_FLAGS / ... come from
# settings/$SPECPILOT_SETTING/config.json (research/scripts/protocol_env.py) unless already set.
export INFER_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SPECPILOT_DATA=${SPECPILOT_DATA:-$HOME/specpilot_data}
export XINFER_SHARED=${XINFER_SHARED:-$SPECPILOT_DATA/shared}
export SPECPILOT_WORK=${SPECPILOT_WORK:-$SPECPILOT_DATA/work}
export XINFER_WORK=$SPECPILOT_WORK
export INFER_WORK=$SPECPILOT_WORK
export INFER_VENV=${INFER_VENV:-$XINFER_SHARED/venv}
export SPECPILOT_SETTING=${SPECPILOT_SETTING:-h100_c1_qwen38-27b_dflash2}
eval "$(python3 "$INFER_REPO/research/scripts/protocol_env.py")"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# torch 2.13+cu130 needs an R580 driver; on older drivers it runs via NVIDIA's forward-compat libcuda
# (installed by scripts/setup_env.sh).
_COMPAT=$XINFER_SHARED/cuda-compat/usr/local/cuda-13.0/compat
_CU13=$INFER_VENV/lib/python3.11/site-packages/nvidia/cu13
if [ -e "$_COMPAT/libcuda.so.1" ]; then
  case ":${LD_LIBRARY_PATH:-}:" in *":$_COMPAT:"*) ;; *) export LD_LIBRARY_PATH=$_COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH};; esac
fi
export CUDA_HOME=$_CU13 PATH=$INFER_VENV/bin:$_CU13/bin:$PATH VIRTUAL_ENV=$INFER_VENV
export HF_HOME=$XINFER_SHARED/hf HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# Build caches. scripts/serve.sh gives every engine tree its own SGLANG_CACHE_DIR, TORCH_EXTENSIONS_DIR and
# SGLANG_JIT_CACHE_DIR; the defaults below only fill gaps. Content-keyed caches (Triton, Inductor, FlashInfer, DeepGEMM,
# the CUDA driver cache) are shared.
export SGLANG_CACHE_DIR=${SGLANG_CACHE_DIR:-$XINFER_SHARED/cache/sglang} SGLANG_JIT_CACHE_DIR=${SGLANG_JIT_CACHE_DIR:-$XINFER_SHARED/cache/sglang/jit} TRITON_CACHE_DIR=$XINFER_SHARED/cache/triton
export FLASHINFER_WORKSPACE_BASE=$XINFER_SHARED/cache TORCHINDUCTOR_CACHE_DIR=$XINFER_SHARED/cache/inductor
export SGLANG_DG_CACHE_DIR=$XINFER_SHARED/cache/sglang/deep_gemm CUDA_CACHE_PATH=$XINFER_SHARED/cache/sglang/nv
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$XINFER_SHARED/cache/torch_extensions}
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# No DeepGEMM "compile all M" warmup at boot: kernels compile on first use into the shared on-disk cache instead.
# With it on, FP8 servers booting at the same time stalled for minutes inside CUDA-graph capture.
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}
# Machine-wide CUDA-init slot (research/autoresearch/cudaslot.py, taken by research/scripts/serve.sh): at most
# SPECPILOT_BOOT_CONCURRENCY servers on the machine in the heavy CUDA-init phase at once (0 disables), grants at least
# SPECPILOT_BOOT_STAGGER_S apart, held until healthy or SPECPILOT_BOOT_HOLD_MAX_S. Scheduling only.
export SPECPILOT_BOOT_CONCURRENCY=${SPECPILOT_BOOT_CONCURRENCY:-1}
export SPECPILOT_BOOT_STAGGER_S=${SPECPILOT_BOOT_STAGGER_S:-10}
export SPECPILOT_BOOT_HOLD_MAX_S=${SPECPILOT_BOOT_HOLD_MAX_S:-240}
# Which engine a process imports: scripts/serve.sh sets XINFER_ENGINE / XINFER_LAB to a built engine tree and starts
# the server with a PYTHONPATH that holds only that tree; otherwise this checkout's stock copies are used.
_ENGINE=${XINFER_ENGINE:-$INFER_REPO/python}
_LAB=${XINFER_LAB:-$INFER_REPO/research/dflash_lab}
case ":${PYTHONPATH:-}:" in *":$_LAB:"*) ;; *) export PYTHONPATH=$_LAB${PYTHONPATH:+:$PYTHONPATH};; esac
case ":${PYTHONPATH:-}:" in *":$_ENGINE:"*) ;; *) export PYTHONPATH=$_ENGINE${PYTHONPATH:+:$PYTHONPATH};; esac
unset _COMPAT _CU13 _ENGINE _LAB
