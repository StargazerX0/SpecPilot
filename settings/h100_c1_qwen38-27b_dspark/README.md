# Qwen3.8-27B + DSpark, single request, H100

Final: `d6-001-qcopy`. Hardware: H100 80GB SXM5. Regime: single request (batch 1). Greedy decoding, lossless.

## Results (paper, dev split first turns, 2048 new tokens)

| | stock | SpecPilot final |
|---|---|---|
| speedup over AR (geometric mean over requests) | 3.74x | 9.41x |
| tau (tokens per verify step) | 4.42 | 12.85* |

Final vs stock: 2.52x. *Under TLV, tau counts tokens per exact-weight pass.

## Models

| role | Hugging Face | revision | local dir |
|---|---|---|---|
| target | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `$XINFER_SHARED/models/Qwen3.8-27B` |
| drafter | [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) | `b9a5dbdf03bc999c6c73c426b19c2d9041cea393` | `$XINFER_SHARED/models/Qwen3.8-27B-DSpark` |

Download: `bash scripts/download_models.sh h100_c1_qwen38-27b_dspark`

## Stock (the baseline)

Unmodified DSpark speculative decoding on the stock engine, block 8 (DSpark gamma 7).

```
--model-path <target> --dtype bfloat16 --kv-cache-dtype auto --mamba-ssm-dtype float32 --attention-backend fa3 --mem-fraction-static 0.90 --max-running-requests 4 --cuda-graph-max-bs-decode 4 --chunked-prefill-size 8192 --disable-radix-cache --random-seed 0
--speculative-algorithm DSPARK --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-draft-model-path <drafter> --speculative-num-draft-tokens 8 --speculative-dspark-block-size 7
```

## Final

Engine = stock engine + `files/` (53 files). Launched with the stock server flags plus block size 32, and the 51 `SPECPILOT_*` environment switches in `config.json` (`final.env`).

Main mechanisms:

- DSpark adapter into the DFlash2-lineage worker
- Markov-conditioned draft tree over a union candidate lattice, fused tree builder
- vocabulary-restricted draft head and candidate-restricted Markov walk
- INT6 Q copy for two-level verification (built at load by round-to-nearest)
- the target-side stack of the DFlash2 single-request final (W12, tree verify, certified head)

## Run

```
bash scripts/download_models.sh h100_c1_qwen38-27b_dspark
bash scripts/serve.sh h100_c1_qwen38-27b_dspark final 0       # or: stock, ar (fresh server per run)
bash scripts/bench.sh h100_c1_qwen38-27b_dspark final-1 0     # speed on the dev split (first run after install: run twice)
bash scripts/stop.sh 30000
python3 scripts/compare.py speed --ar ar-1.json --stock stock-1.json --final final-1.json   # in $SPECPILOT_WORK/bench
python3 scripts/score.py final-1.json        # benchmark scores
```

## Files

- `config.json`: models and revisions, stock server flags and block size, the final's block size, launch args and
  `SPECPILOT_*` environment switches, the paper's numbers
- `files/`: the engine files the final adds or changes (copied over the stock engine by `scripts/build_engine.py`)
