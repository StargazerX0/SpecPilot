#!/usr/bin/env python3
"""Prints shell exports for settings/$SPECPILOT_SETTING/config.json (sourced by research/env.sh).

The setting's config is the single source of truth for its target, drafter, stock block and server flags. Variables
already set in the environment win. Stdlib only.
"""
import json
import os
import shlex
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
name = os.environ.get("SPECPILOT_SETTING", "")
path = repo / "settings" / name / "config.json"
if not path.exists():
    sys.stderr.write(f"env.sh: no setting {name!r}; have {sorted(p.parent.name for p in (repo / 'settings').glob('*/config.json'))}\n")
    sys.exit(0)
c = json.loads(path.read_text())
models = Path(os.environ.get("XINFER_SHARED", "")) / "models"
st = c["stock"]
out = {"TARGET_PRECISION": c["precision"], "TARGET_MODEL": str(models / c["models"]["target"]["dir"]),
       "DRAFT_MODEL": str(models / c["models"]["drafter"]["dir"]), "SPECPILOT_BLOCK_SIZE": str(st["block_size"]),
       "SPECPILOT_SERVER_FLAGS": " ".join(st["server_flags"]),
       "SPECPILOT_SPEC_FLAGS": " ".join(st["spec_flags"]),
       "SPECPILOT_AR_FLAGS": " ".join(st["ar_flags"]),
       "SPECPILOT_SERVER_ENV": " ".join(f"{k}={v}" for k, v in sorted(st["server_env"].items())),
       "SPECPILOT_LOAD_LEVELS": ",".join(str(x) for x in c["concurrency_levels"]),
       "SERVED_MODEL_NAME": c["models"]["target"]["hf"].split("/")[-1].lower()}
for k, v in out.items():
    print(f"export {k}={shlex.quote(os.environ.get(k) or v)}")
