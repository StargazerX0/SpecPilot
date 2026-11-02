"""Entry point for the ``sglang.srt.plugins`` group (loaded in every server process).

Always: prints one runtime-evidence line per process (the directories ``sglang`` and this plugin
were imported from, and the candidate id), which the controller's boot gate checks against the
candidate snapshot, and refuses banned environment variables.

``DFLASH_LAB=name[,name...]`` selects experiments from ``dflash_lab.experiments``;
``DFLASH_LAB_PARAMS='{"name": {...}}'`` gives each experiment its parameters. Each module exposes
``apply(params)``, which registers hooks through ``dflash_lab.core``. Unset means no hooks.

Failures are fatal: an experiment that cannot register (unknown name, bad params, banned env)
exits the process with status 3 instead of serving without its hooks.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import traceback

BANNED_ENV = ("SGLANG_SIMULATE_ACC_LEN", "SGLANG_SIMULATE_ACC_METHOD", "SGLANG_SIMULATE_ACC_TOKEN_MODE")


def config_hash(names, params) -> str:
    blob = json.dumps({"experiments": names, "params": params}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _evidence() -> None:
    sg = sys.modules.get("sglang")
    sg_dir = os.path.dirname(os.path.abspath(sg.__file__)) if sg is not None and sg.__file__ else None
    lab_dir = os.path.dirname(os.path.abspath(__file__))
    # stderr, not logging: plugins load before SGLang configures its loggers.
    print(f"[xinfer] runtime pid={os.getpid()} candidate={os.environ.get('XINFER_CANDIDATE', '-')} "
          f"sglang={sg_dir} lab={lab_dir}", file=sys.stderr, flush=True)


def register() -> None:
    try:
        _evidence()
        banned = [k for k in BANNED_ENV if os.environ.get(k) not in (None, "", "0")]
        if banned:
            raise RuntimeError(f"banned env set in an evaluated run: {banned}")
        names = [n.strip() for n in os.environ.get("DFLASH_LAB", "").split(",") if n.strip()]
        if not names:
            return
        params = json.loads(os.environ.get("DFLASH_LAB_PARAMS") or "{}")
        unknown = sorted(set(params) - set(names))
        if unknown:
            raise ValueError(f"DFLASH_LAB_PARAMS names experiments that are not enabled: {unknown}")
        from dflash_lab import core

        core.install()
        for name in names:
            module = importlib.import_module(f"dflash_lab.experiments.{name}")
            module.apply(dict(params.get(name, {})))
        print(f"[dflash_lab] config experiments={','.join(names)} params_sha={config_hash(names, params)} "
              f"plugin={os.path.dirname(os.path.abspath(__file__))} pid={os.getpid()}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("[dflash_lab] FATAL: experiment registration failed", file=sys.stderr, flush=True)
        os._exit(3)
