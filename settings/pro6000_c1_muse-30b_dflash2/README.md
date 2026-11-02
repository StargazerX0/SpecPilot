# Muse-Glimmer-30B + DFlash2, single request, RTX PRO 6000

Final: `int-003`. Hardware: RTX PRO 6000 Blackwell Server (sm_120). Regime: single request (batch 1). Greedy decoding, lossless.

## Results (paper, dev split first turns, 2048 new tokens)

| | stock | SpecPilot final |
|---|---|---|
| speedup over AR (geometric mean over requests) | 4.69x | 10.63x |
| tau (tokens per verify step) | 5.16 | 12.64* |

Final vs stock: 2.27x. *Under TLV, tau counts tokens per exact-weight pass.

## Models

| role | Hugging Face | revision | local dir |
|---|---|---|---|
| target | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | `a4e59da52a7bc87ae7251dd5545c0dd437c44b68` | `$XINFER_SHARED/models/Muse-Glimmer-30B` |
| drafter | [incoai/Muse-Glimmer-30B-DFlash2](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2) | `8336acb8dc9b8bf9c25f12d7785ee6df26703119` | `$XINFER_SHARED/models/Muse-Glimmer-30B-DFlash2` |

Download: `bash scripts/download_models.sh pro6000_c1_muse-30b_dflash2`

## Stock (the baseline)

Unmodified DFlash2 speculative decoding on the stock engine, block 16.

```
--model-path <target> --dtype bfloat16 --kv-cache-dtype auto --mamba-ssm-dtype float32 --attention-backend flashinfer --mem-fraction-static 0.88 --max-running-requests 4 --cuda-graph-max-bs-decode 4 --chunked-prefill-size 8192 --disable-radix-cache --max-total-tokens 131072 --random-seed 0
--speculative-algorithm DFLASH --speculative-draft-model-path <drafter> --speculative-num-draft-tokens 16
```

## Final

Engine = stock engine + `files/` (64 files). Launched with the stock server flags plus block size 32, extra args `--attention-backend=triton --language-model-only`, and the 37 `SPECPILOT_*` environment switches in `config.json` (`final.env`).

Main mechanisms:

- W12 lossless 12-bit weights on sm_120 (also the LM head and output projections)
- two-level verification (TLV) with a scaled INT6 (MLP) / INT8 (other layers) Q copy built at load
- draft tree verify (top-32) on triton attention, split-KV verify attention
- L2 time-shifted weight prefetch, tail skip

## Run

```
bash scripts/download_models.sh pro6000_c1_muse-30b_dflash2
bash scripts/serve.sh pro6000_c1_muse-30b_dflash2 final 0       # or: stock, ar (fresh server per run)
bash scripts/bench.sh pro6000_c1_muse-30b_dflash2 final-1 0     # speed on the dev split (first run after install: run twice)
bash scripts/stop.sh 30000
python3 scripts/compare.py speed --ar ar-1.json --stock stock-1.json --final final-1.json   # in $SPECPILOT_WORK/bench
python3 scripts/score.py final-1.json        # benchmark scores
```

## Files

- `config.json`: models and revisions, stock server flags and block size, the final's block size, launch args and
  `SPECPILOT_*` environment switches, the paper's numbers
- `files/`: the engine files the final adds or changes (copied over the stock engine by `scripts/build_engine.py`)

Note: the CUDA kernels of this final are built for sm_120a (RTX PRO 6000 Blackwell) only.

Note: the Muse chat template writes the current date into the system prompt, so outputs change from day to day; run all arms of a comparison within one UTC day.
