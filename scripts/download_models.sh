#!/usr/bin/env bash
# Download the pinned target and drafter checkpoints of one setting (or all) into $XINFER_SHARED/models.
#   bash scripts/download_models.sh <setting>|all
# Revisions are the ones every paper measurement used (settings/<setting>/config.json). All repos are public; set HF_TOKEN only if you hit
# rate limits. A directory that already holds a config.json is kept as is.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WHICH=${1:?usage: download_models.sh <setting>|all}
export SPECPILOT_SETTING=${SPECPILOT_SETTING:-h100_c1_qwen38-27b_dflash2}
source "$REPO/research/env.sh"
python3 - "$REPO/settings" "$WHICH" <<'PY' | while read -r hf rev dir; do
import json, sys
from pathlib import Path
s = {p.parent.name: json.loads(p.read_text()) for p in Path(sys.argv[1]).glob("*/config.json")}
names = sorted(s) if sys.argv[2] == "all" else [sys.argv[2]]
seen = set()
for n in names:
    if n not in s:
        sys.exit(f"unknown setting {n}; choose from: {', '.join(sorted(s))}")
    for m in s[n]["models"].values():
        if m["dir"] not in seen:
            seen.add(m["dir"])
            print(m["hf"], m["revision"], m["dir"])
PY
  dst=$XINFER_SHARED/models/$dir
  if [ -f "$dst/config.json" ]; then echo "have $dst"; continue; fi
  echo "downloading https://huggingface.co/$hf (revision $rev) -> $dst"
  HF_HUB_OFFLINE=0 "$INFER_VENV/bin/hf" download "$hf" --revision "$rev" --local-dir "$dst"
done
