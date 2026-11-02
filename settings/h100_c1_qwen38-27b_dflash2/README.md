# Qwen3.8-27B + DFlash2, single request, H100

Final: `int-015-tlvcbn32`. Hardware: H100 80GB SXM5. Regime: single request (batch 1). Greedy decoding, lossless.

## Results (paper, dev split first turns, 2048 new tokens)

| | stock | SpecPilot final |
|---|---|---|
| speedup over AR (geometric mean over requests) | 3.95x | 8.86x |
| tau (tokens per verify step) | 4.65 | 13.08* |

Final vs stock: 2.24x. *Under TLV, tau counts tokens per exact-weight pass.

## Models

| role | Hugging Face | revision | local dir |
|---|---|---|---|
| target | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `$XINFER_SHARED/models/Qwen3.8-27B` |
| drafter | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | `$XINFER_SHARED/models/Qwen3.8-27B-DFlash2` |

Download: `bash scripts/download_models.sh h100_c1_qwen38-27b_dflash2`

## Stock (the baseline)

Unmodified DFlash2 speculative decoding on the stock engine, block 8.

```
--model-path <target> --dtype bfloat16 --kv-cache-dtype auto --mamba-ssm-dtype float32 --attention-backend fa3 --mem-fraction-static 0.90 --max-running-requests 4 --cuda-graph-max-bs-decode 4 --chunked-prefill-size 8192 --disable-radix-cache --random-seed 0
--speculative-algorithm DFLASH --speculative-draft-model-path <drafter> --speculative-num-draft-tokens 8
```

## Final

Engine = stock engine + `files/` (49 files). Launched with the stock server flags plus block size 32, and the 46 `SPECPILOT_*` environment switches in `config.json` (`final.env`).

Main mechanisms:

- wide draft tree verify (32 nodes over a 16-token draft block) with snapshot-free GDN tree kernels
- W12: lossless 12-bit weight streaming
- two-level verification (TLV) on the nested 8-bit view of W12
- certified exact-argmax LM head (INT8 screen, BF16 rescore)
- tail skip, low-precision drafter weights, fused glue kernels and PDL

## Run

```
bash scripts/download_models.sh h100_c1_qwen38-27b_dflash2
bash scripts/serve.sh h100_c1_qwen38-27b_dflash2 final 0       # or: stock, ar (fresh server per run)
bash scripts/bench.sh h100_c1_qwen38-27b_dflash2 final-1 0     # speed on the dev split (first run after install: run twice)
bash scripts/stop.sh 30000
python3 scripts/compare.py speed --ar ar-1.json --stock stock-1.json --final final-1.json   # in $SPECPILOT_WORK/bench
python3 scripts/score.py final-1.json        # benchmark scores
```

## Files

- `config.json`: models and revisions, stock server flags and block size, the final's block size, launch args and
  `SPECPILOT_*` environment switches, the paper's numbers
- `files/`: the engine files the final adds or changes (copied over the stock engine by `scripts/build_engine.py`)
