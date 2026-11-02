#!/usr/bin/env python3
"""Frozen task graders for the protocol splits (policy version ``grade_v2``).

Extraction: the thinking model writes ``...</think>answer``. Only text after the
last ``</think>`` is graded; an output without it is ``no_final`` and counts as incorrect.
Gemma 4 records (special tokens kept, ``sampling.skip_special_tokens`` false) think in an optional
``<|channel>thought ...<channel|>`` channel: the text after the last ``<channel|>`` is graded, an
output that opens the channel and never closes it is ``no_final``, and one without it is all answer.
Every graded row also gets an ``outcome``: ``correct``, ``wrong`` (an answer was extracted
and failed), ``parse_fail`` (no final answer / no boxed / no code block) or ``truncated``
(finish_reason ``length`` without a correct answer), so regressions in termination and
formatting stay separate from wrong answers.

* gsm8k / math500: last ``\\boxed{...}`` of the final answer, light LaTeX
  normalization, numeric comparison when both sides parse as numbers.
* humaneval / mbpp: last fenced code block of the final answer that defines a
  function (else the last block, else the whole final answer), executed with the dataset tests in a fresh ``python -I``
  subprocess (10 s wall, 4 GB address space, empty env, temp cwd). The z-lab MBPP
  prompt never names the function, so when the tested name is undefined it is
  aliased to the last top-level function the answer defines (``mbpp_alias``).
* mt-bench: not graded (``graded=False``); quality claims there are limited to
  output length / termination.

  python research/bench/grade.py RESULT.json [--split-dir research/bench/splits/dev]
  python research/bench/grade.py --validate-alias [--split-dir research/bench/splits/quality]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GRADER_VERSION = "grade_v2"
HERE = Path(__file__).resolve().parent


# Where the user-facing answer starts: Qwen-style thinking ends at "</think>"; Muse Glimmer's ATEM
# template opens the user channel with a "to=user" header (the reasoning channel is "to=self").
_FINAL_MARKERS = ("</think>", "to=user")
# Gemma 4 (records generated with skip_special_tokens false): the reasoning is "<|channel>thought ...<channel|>" and
# thinking is optional (an output without the opener is all answer); the turn may end with "<turn|>" / "<eos>".
_CHANNEL_OPEN, _CHANNEL_CLOSE, _END_MARKERS = "<|channel>", "<channel|>", ("<turn|>", "<eos>")


def strip_end(text):
    """``text`` without trailing end-of-turn special tokens (kept when special tokens are not skipped)."""
    text = text or ""
    while text.rstrip().endswith(_END_MARKERS):
        text = text.rstrip()
        for m in _END_MARKERS:
            if text.endswith(m):
                text = text[: -len(m)]
    return text


def channel_format(record):
    """True for records whose texts keep special tokens (Gemma 4 channel markers)."""
    return (record.get("sampling") or {}).get("skip_special_tokens") is False


def final_answer(text, channel=False):
    if text is None:
        return None
    if channel:
        text = strip_end(text)
        if _CHANNEL_OPEN not in text:
            return text
        return text.rsplit(_CHANNEL_CLOSE, 1)[1] if _CHANNEL_CLOSE in text.rsplit(_CHANNEL_OPEN, 1)[1] else None
    for marker in _FINAL_MARKERS:
        if marker in text:
            return text.rsplit(marker, 1)[1]
    return None


def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(s):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
        k += 1
    return None


_STRIP = [r"\left", r"\right", r"\!", r"\,", r"\;", r"\:", r"\ ", "$", r"\displaystyle"]


def norm_math(s):
    s = s.strip()
    for t in _STRIP:
        s = s.replace(t, "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{([^}]*)\}", r"\1", s)
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace("^\\circ", "").replace("^{\\circ}", "")
    s = s.replace("\\%", "").replace("%", "")
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    if s.startswith("x=") or s.startswith("y="):
        s = s[2:]
    return s


def as_number(s):
    s = s.replace(",", "")
    m = re.fullmatch(r"-?\d+(\.\d+)?", s)
    if m:
        return float(s)
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}", s)
    if m and int(m.group(3)) != 0:
        return (-1 if m.group(1) else 1) * int(m.group(2)) / int(m.group(3))
    return None


def math_equal(pred, ref):
    p, r = norm_math(pred), norm_math(ref)
    if p == r:
        return True
    pn, rn = as_number(p), as_number(r)
    return pn is not None and rn is not None and abs(pn - rn) <= 1e-6 * max(1.0, abs(rn))


def extract_code(final):
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", final, flags=re.S)
    with_def = [b for b in blocks if re.search(r"^\s*def ", b, flags=re.M)]
    return with_def[-1] if with_def else (blocks[-1] if blocks else final)


def mbpp_alias(code, test_list):
    """``name = last_def`` when the tests call a function the code does not define."""
    import ast

    try:
        defs = [n.name for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)]
    except SyntaxError:
        return ""
    m = re.search(r"assert\s+\(?\s*(?:not\s+)?\(?\s*(?:math\.isclose\(|set\()?\s*([A-Za-z_]\w*)\s*\(", test_list[0])
    want = m.group(1) if m else None
    if not defs or want is None or want in defs:
        return ""
    return f"\n{want} = {defs[-1]}\n"


def run_python(source, timeout=10.0):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "prog.py"
        path.write_text(source)
        limits = "import resource; resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))"
        try:
            p = subprocess.run([sys.executable, "-I", "-c", f"{limits}\nexec(open('prog.py').read())"],
                               cwd=d, env={"PATH": "/usr/bin:/bin"}, capture_output=True, timeout=timeout,
                               stdin=subprocess.DEVNULL)
            return p.returncode == 0, (p.stderr.decode(errors="replace")[-300:] if p.returncode else "")
        except subprocess.TimeoutExpired:
            return False, "timeout"


def _outcome(g, finish_reason):
    if g["correct"]:
        return "correct"
    if finish_reason == "length":
        return "truncated"
    return "parse_fail" if g["reason"] in ("no_final", "no_boxed", "no_code") else "wrong"


def grade(item, text, finish_reason=None, channel=False):
    suite = item["suite"]
    if suite == "mt-bench":
        return {"graded": False, "correct": None, "reason": "ungraded", "outcome": None}
    g = _grade(item, text, channel)
    g["outcome"] = _outcome(g, finish_reason)
    return g


def _grade(item, text, channel=False):
    suite = item["suite"]
    final = final_answer(text, channel)
    if final is None:
        return {"graded": True, "correct": False, "reason": "no_final"}
    if suite in ("gsm8k", "math500"):
        box = last_boxed(final)
        if box is None:
            return {"graded": True, "correct": False, "reason": "no_boxed"}
        ok = math_equal(box, item["answer"])
        return {"graded": True, "correct": ok, "reason": "ok" if ok else f"wrong:{box[:40]}"}
    if "```" not in final and not re.search(r"^\s*def ", final, flags=re.M):
        return {"graded": True, "correct": False, "reason": "no_code"}
    code = extract_code(final)
    alias = ""
    if suite == "humaneval":
        src = f"{code}\n\n{item['test']}\n\ncheck({item['entry_point']})\n"
    else:
        alias = mbpp_alias(code, item["test_list"])
        src = mbpp_program(item, code, alias)
    ok, err = run_python(src)
    return {"graded": True, "correct": ok, "reason": "ok" if ok else f"fail:{err[-80:]}", "alias": bool(alias)}


def mbpp_program(item, code, alias):
    return ("\n".join(item.get("test_imports") or []) + f"\n{code}\n" + alias
            + "\n" + "\n".join(item["test_list"]) + "\n")


def validate_mbpp_alias(items):
    """Checks the alias heuristic against the original task tests: every canonical solution must pass
    as is, and again after its tested function is renamed (so only the alias can bind the name)."""
    rows = []
    for it in items.values():
        if it["suite"] != "mbpp" or not it.get("canonical"):
            continue
        code = it["canonical"]
        plain, _ = run_python(mbpp_program(it, code, ""))
        m = re.search(r"assert\s+\(?\s*(?:not\s+)?\(?\s*(?:math\.isclose\(|set\()?\s*([A-Za-z_]\w*)\s*\(", it["test_list"][0])
        want = m.group(1) if m else None
        renamed = re.sub(rf"\bdef\s+{re.escape(want)}\b", "def _renamed_solution", code) if want else code
        renamed = re.sub(rf"\b{re.escape(want)}\(", "_renamed_solution(", renamed) if want else renamed
        alias = mbpp_alias(renamed, it["test_list"])
        aliased, _ = run_python(mbpp_program(it, renamed, alias))
        rows.append({"id": it["id"], "canonical_passes": plain, "tested_name": want, "alias_restores": aliased})
    n = len(rows)
    return {"n": n, "canonical_pass": sum(r["canonical_passes"] for r in rows),
            "alias_restores": sum(r["alias_restores"] for r in rows if r["canonical_passes"]),
            "failures": [r for r in rows if r["canonical_passes"] and not r["alias_restores"]]}


def load_items(paths):
    items = {}
    for p in paths:
        for line in open(p):
            if line.strip():
                it = json.loads(line)
                items[it["id"]] = it
    return items


def grade_record(record, items, workers=16):
    """Grade the final turn of every item; returns per-row grades and per-suite accuracy."""
    texts = record["texts"]
    channel = channel_format(record)
    finish = {r["id"]: r.get("finish_reason") for r in record.get("per_request", [])}
    jobs = []
    for rid, text in texts.items():
        item_id, turn = rid.rsplit("#t", 1)
        it = items.get(item_id) or items.get(item_id.split("@c", 1)[0])   # load rows: <item>@c<level>~<copy>
        if it is None:
            raise KeyError(f"no split item for {rid}")
        if int(turn) == len(it["turns"]) - 1 or it["suite"] != "mt-bench":
            jobs.append((rid, it, text))
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda j: (j[0], j[1]["suite"], grade(j[1], j[2], finish.get(j[0]), channel)), jobs))
    per_suite = {}
    for rid, suite, g in res:
        if g["graded"]:
            s = per_suite.setdefault(suite, {"n": 0, "correct": 0, "wrong": 0, "parse_fail": 0, "truncated": 0})
            s["n"] += 1
            s["correct"] += int(g["correct"])
            if g["outcome"] != "correct":
                s[g["outcome"]] += 1
    for s in per_suite.values():
        s["acc"] = s["correct"] / s["n"]
    return {"grader": GRADER_VERSION, "per_request": {rid: g for rid, _, g in res}, "per_suite": per_suite}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", nargs="?")
    ap.add_argument("--split-dir", default=str(HERE / "splits" / "dev"))
    ap.add_argument("--validate-alias", action="store_true")
    a = ap.parse_args()
    items = load_items(sorted(Path(a.split_dir).glob("*.jsonl")))
    if a.validate_alias:
        print(json.dumps(validate_mbpp_alias(items), indent=1))
        return
    rec = json.load(open(a.result))
    g = grade_record(rec, items)
    print(json.dumps(g["per_suite"], indent=1))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
