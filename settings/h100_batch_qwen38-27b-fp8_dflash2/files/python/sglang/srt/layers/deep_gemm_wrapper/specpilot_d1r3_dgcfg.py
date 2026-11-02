"""SPECPILOT_D1R3_DGCFG: per-(N, K, M) DeepGEMM launch-config table for the dense FP8 GEMM (fp8_gemm_nt).

DeepGEMM's heuristic is non-monotonic in M at the M values the CUDA graphs capture (target verify rows, drafter rows
bs x 16, drafter head rows); a table entry sets deep_gemm.set_num_sms and/or set_block_size_multiple_of around that one
call and restores them after. The config is chosen at capture time, so graph replay pays nothing. Entries come from an
after-GEMM microbench of this campaign and are kept only where the output is bitwise the default config's (sm90 fp8
GEMM without split-K: one CTA owns an output tile and runs the same k loop and promotion whatever the tiling).

Table file: specpilot_d1r3_dgcfg.json next to this module,
  {"default_mult": [m, n], "entries": {"NxK": {"M": {"num_sms": int|null, "mult": [m, n]|null}}}}
"default_mult" is the block-size multiple restored after an entry that sets one (measured equal to DeepGEMM's own
default on every table shape). Knob (read once at import): SPECPILOT_D1R3_DGCFG=1 enables; unset = the parent path.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Tuple

ENABLED = os.environ.get("SPECPILOT_D1R3_DGCFG", "0") == "1"
TABLE_PATH = Path(__file__).with_name("specpilot_d1r3_dgcfg.json")

Entry = Tuple[Optional[int], Optional[Tuple[int, int]]]


def parse_table(raw: dict) -> Tuple[Tuple[int, int], Dict[Tuple[int, int, int], Entry]]:
    default_mult = tuple(int(v) for v in raw.get("default_mult", (1, 1)))
    table: Dict[Tuple[int, int, int], Entry] = {}
    for nk, per_m in raw.get("entries", {}).items():
        n, k = (int(v) for v in nk.split("x"))
        for m, e in per_m.items():
            num_sms = e.get("num_sms")
            mult = e.get("mult")
            entry = (int(num_sms) if num_sms else None, tuple(int(v) for v in mult) if mult else None)
            if entry != (None, None):
                table[(n, k, int(m))] = entry
    return default_mult, table


def load_table(path: Path = TABLE_PATH):
    if not path.exists():
        return (1, 1), {}
    return parse_table(json.loads(path.read_text()))


DEFAULT_MULT, TABLE = load_table() if ENABLED else ((1, 1), {})


def lookup(n: int, k: int, m: int) -> Optional[Entry]:
    return TABLE.get((n, k, m)) if ENABLED else None


@contextmanager
def applied(entry: Entry, *, default_mult: Tuple[int, int] = None):
    import deep_gemm

    num_sms, mult = entry
    restore_mult = default_mult or DEFAULT_MULT
    old_sms = deep_gemm.get_num_sms()
    if num_sms is not None:
        deep_gemm.set_num_sms(num_sms)
    if mult is not None:
        deep_gemm.set_block_size_multiple_of(tuple(mult))
    try:
        yield
    finally:
        if num_sms is not None:
            deep_gemm.set_num_sms(old_sms)
        if mult is not None:
            deep_gemm.set_block_size_multiple_of(tuple(restore_mult))
