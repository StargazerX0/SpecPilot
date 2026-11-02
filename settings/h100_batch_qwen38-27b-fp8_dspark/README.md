# Qwen3.8-27B FP8 + DSpark, batching, H100

Final: `d1-002-ldu`. Hardware: H100 80GB SXM5. Regime: batching, closed loop, 16/20/24 concurrent requests. Greedy decoding, lossless.

## Results (paper, dev split first turns, 2048 new tokens)

| | stock | SpecPilot final |
|---|---|---|
| speedup over AR (geometric mean over requests) | 1.94x | 3.93x |
| tau (tokens per verify step) | 4.37 | 5.42 |

Final vs stock: 2.03x.

## Models

| role | Hugging Face | revision | local dir |
|---|---|---|---|
| target | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` | `$XINFER_SHARED/models/Qwen3.8-27B-FP8` |
| drafter | [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) | `b9a5dbdf03bc999c6c73c426b19c2d9041cea393` | `$XINFER_SHARED/models/Qwen3.8-27B-DSpark` |

Download: `bash scripts/download_models.sh h100_batch_qwen38-27b-fp8_dspark`

## Stock (the baseline)

Unmodified DSpark speculative decoding on the stock engine, block 8 (DSpark gamma 7).

```
--model-path <target> --dtype bfloat16 --kv-cache-dtype auto --mamba-ssm-dtype float32 --attention-backend fa3 --mem-fraction-static 0.92 --max-running-requests 24 --cuda-graph-max-bs-decode 24 --chunked-prefill-size 8192 --disable-radix-cache --random-seed 0
--speculative-algorithm DSPARK --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-draft-model-path <drafter> --speculative-num-draft-tokens 8 --speculative-dspark-block-size 7
```

## Final

Engine = stock engine + `files/` (58 files). Launched with the stock server flags plus block size 16, extra args `--max-total-tokens=131072 --linear-attn-prefill-backend=triton`, and the 66 `SPECPILOT_*` environment switches in `config.json` (`final.env`).

Main mechanisms:

- the DSpark adapter on the FP8 batching lineage (all target-side wins of the DFlash2 batching final)
- a batched union-candidate lattice builder

## Run

```
bash scripts/download_models.sh h100_batch_qwen38-27b-fp8_dspark
bash scripts/serve.sh h100_batch_qwen38-27b-fp8_dspark final 0       # or: stock, ar (fresh server per run)
bash scripts/bench.sh h100_batch_qwen38-27b-fp8_dspark final-1 0     # speed on the dev split (first run after install: run twice)
bash scripts/stop.sh 30000
python3 scripts/compare.py speed --ar ar-1.json --stock stock-1.json --final final-1.json   # in $SPECPILOT_WORK/bench
python3 scripts/score.py final-1.json        # benchmark scores
```

## Files

- `config.json`: models and revisions, stock server flags and block size, the final's block size, launch args and
  `SPECPILOT_*` environment switches, the paper's numbers
- `files/`: the engine files the final adds or changes (copied over the stock engine by `scripts/build_engine.py`)
