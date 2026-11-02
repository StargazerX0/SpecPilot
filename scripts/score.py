#!/usr/bin/env python3
"""Benchmark scores of bench records: accuracy per dataset (gsm8k, math500, humaneval, mbpp; mt-bench is not graded).

    python3 scripts/score.py RESULT.json [RESULT2.json ...] [--split dev|quality]

Grading (research/bench/grade.py): only the text after the last </think> counts; gsm8k / math500 compare the last
\\boxed{} answer; humaneval / mbpp run the last code block against the dataset tests in a sandboxed subprocess. An
answer cut off at max_new_tokens without a correct answer counts as wrong and is reported as truncated.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research/bench"))
import grade  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    items = grade.load_items(sorted((REPO / "research/bench/splits" / a.split).glob("*.jsonl")))
    for p in a.results:
        g = grade.grade_record(json.load(open(p)), items)
        ps = g["per_suite"]
        n = sum(v["n"] for v in ps.values())
        c = sum(v["correct"] for v in ps.values())
        tr = sum(v["truncated"] for v in ps.values())
        cells = "  ".join(f"{s} {v['correct']}/{v['n']}" for s, v in sorted(ps.items()))
        print(f"{Path(p).stem:<20} {cells}  | all {c}/{n} = {c / n:.3f}  truncated {tr}")


if __name__ == "__main__":
    main()
