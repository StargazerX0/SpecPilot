# Qwen3.8-27B FP8 + DFlash2, batching, H100

Final: `int-210-full`. Hardware: H100 80GB SXM5. Regime: batching, closed loop, 16/20/24 concurrent requests. Greedy decoding, lossless.

## Results (paper, dev split first turns, 2048 new tokens)

| | stock | SpecPilot final |
|---|---|---|
| speedup over AR (geometric mean over requests) | 2.02x | 3.90x |
| tau (tokens per verify step) | 4.56 | 5.39 |

Final vs stock: 1.93x.

## Models

| role | Hugging Face | revision | local dir |
|---|---|---|---|
| target | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` | `$XINFER_SHARED/models/Qwen3.8-27B-FP8` |
| drafter | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | `$XINFER_SHARED/models/Qwen3.8-27B-DFlash2` |

Download: `bash scripts/download_models.sh h100_batch_qwen38-27b-fp8_dflash2`

## Stock (the baseline)

Unmodified DFlash2 speculative decoding on the stock engine, block 8.

```
--model-path <target> --dtype bfloat16 --kv-cache-dtype auto --mamba-ssm-dtype float32 --attention-backend fa3 --mem-fraction-static 0.92 --max-running-requests 24 --cuda-graph-max-bs-decode 24 --chunked-prefill-size 8192 --disable-radix-cache --random-seed 0
--speculative-algorithm DFLASH --speculative-draft-model-path <drafter> --speculative-num-draft-tokens 8
```

## Final

Engine = stock engine + `files/` (55 files). Launched with the stock server flags plus block size 16, extra args `--max-total-tokens=131072 --linear-attn-prefill-backend=triton`, and the 61 `SPECPILOT_*` environment switches in `config.json` (`final.env`).

Main mechanisms:

- ragged draft trees with snapshot-free GDN tree verify and a tiered verify budget
- FP8 GEMM routing and warp-specialised FP8 GEMMs
- fused GDN paths
- segment admission of prompts into the verify step
- per-request drafter block length

## Run

```
bash scripts/download_models.sh h100_batch_qwen38-27b-fp8_dflash2
bash scripts/serve.sh h100_batch_qwen38-27b-fp8_dflash2 final 0       # or: stock, ar (fresh server per run)
bash scripts/bench.sh h100_batch_qwen38-27b-fp8_dflash2 final-1 0     # speed on the dev split (first run after install: run twice)
bash scripts/stop.sh 30000
python3 scripts/compare.py speed --ar ar-1.json --stock stock-1.json --final final-1.json   # in $SPECPILOT_WORK/bench
python3 scripts/score.py final-1.json        # benchmark scores
```

## Files

- `config.json`: models and revisions, stock server flags and block size, the final's block size, launch args and
  `SPECPILOT_*` environment switches, the paper's numbers
- `files/`: the engine files the final adds or changes (copied over the stock engine by `scripts/build_engine.py`)
