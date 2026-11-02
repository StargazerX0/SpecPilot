# Qwen3.8-27B + DFlash2, single request, RTX PRO 6000

Final: `k-001-d3qpdl`. Hardware: RTX PRO 6000 Blackwell Server (sm_120). Regime: single request (batch 1). Greedy decoding, lossless.

## Results (paper, dev split first turns, 2048 new tokens)

| | stock | SpecPilot final |
|---|---|---|
| speedup over AR (geometric mean over requests) | 4.46x | 10.87x |
| tau (tokens per verify step) | 5.23 | 12.80* |

Final vs stock: 2.44x. *Under TLV, tau counts tokens per exact-weight pass.

## Models

| role | Hugging Face | revision | local dir |
|---|---|---|---|
| target | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `$XINFER_SHARED/models/Qwen3.8-27B` |
| drafter | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | `$XINFER_SHARED/models/Qwen3.8-27B-DFlash2` |

Download: `bash scripts/download_models.sh pro6000_c1_qwen38-27b_dflash2`

## Stock (the baseline)

Unmodified DFlash2 speculative decoding on the stock engine, block 12.

```
--model-path <target> --dtype bfloat16 --kv-cache-dtype auto --mamba-ssm-dtype float32 --attention-backend triton --mem-fraction-static 0.90 --max-running-requests 4 --cuda-graph-max-bs-decode 4 --chunked-prefill-size 8192 --disable-radix-cache --random-seed 0
--speculative-algorithm DFLASH --speculative-draft-model-path <drafter> --speculative-num-draft-tokens 12
```

## Final

Engine = stock engine + `files/` (54 files). Launched with the stock server flags plus block size 32, and the 51 `SPECPILOT_*` environment switches in `config.json` (`final.env`).

Main mechanisms:

- W12 re-implemented on mma.sync for sm_120
- draft tree verify on triton attention
- TLV with split-KV verify attention
- scaled INT6/INT8 Q copy with its own stream-K kernel (D3Q)
- L2 time-shifted weight prefetch, PDL on the Q pass

## Run

```
bash scripts/download_models.sh pro6000_c1_qwen38-27b_dflash2
bash scripts/serve.sh pro6000_c1_qwen38-27b_dflash2 final 0       # or: stock, ar (fresh server per run)
bash scripts/bench.sh pro6000_c1_qwen38-27b_dflash2 final-1 0     # speed on the dev split (first run after install: run twice)
bash scripts/stop.sh 30000
python3 scripts/compare.py speed --ar ar-1.json --stock stock-1.json --final final-1.json   # in $SPECPILOT_WORK/bench
python3 scripts/score.py final-1.json        # benchmark scores
```

## Files

- `config.json`: models and revisions, stock server flags and block size, the final's block size, launch args and
  `SPECPILOT_*` environment switches, the paper's numbers
- `files/`: the engine files the final adds or changes (copied over the stock engine by `scripts/build_engine.py`)

Note: the CUDA kernels of this final are built for sm_120a (RTX PRO 6000 Blackwell) only.
