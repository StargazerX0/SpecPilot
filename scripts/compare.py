#!/usr/bin/env python3
"""Compare AR / stock / final runs of one setting: speed, tokens per verify step and output identity.

    python3 scripts/compare.py speed --ar AR.json [AR2.json ...] --stock S.json [...] --final F.json [...]

speed: bench records of scripts/bench.sh. Each arm may have several runs (fresh server each); a request's rate is the
geometric mean of its rates over the runs. Speedup = exp(mean over cells (suite x concurrency level) of the mean over
requests of log(rate_arm / rate_ref)), with a 95% bootstrap CI (2000 resamples of requests within cells). tok/s =
summed tokens / summed request latency. tau = tokens per verify step (under TLV: per exact-weight pass). Then the same
per dataset (gsm8k, humaneval, math500, mbpp, mt-bench), as in the paper's tables and the DFlash evaluation.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path


def load(paths):
    return [json.load(open(p)) for p in paths]


def rates(records):
    """{row id: (cell, geometric-mean rate over runs, tokens, latency, verify)} for rows present in every run."""
    per = {}
    for rec in records:
        for r in rec["per_request"]:
            per.setdefault(r["id"], []).append(r)
    out = {}
    for rid, rs in per.items():
        if len(rs) != len(records):
            continue
        r0 = rs[0]
        cell = (r0["suite"].split("@c")[0], r0.get("level"))
        rate = math.exp(sum(math.log(r["completion_tokens"] / r["latency_s"]) for r in rs) / len(rs))
        out[rid] = (cell, rate, sum(r["completion_tokens"] for r in rs) / len(rs), sum(r["latency_s"] for r in rs) / len(rs),
                    sum(r["spec_verify_ct"] for r in rs) / len(rs))
    return out


def speedup(num, den, boot=2000, seed=0):
    ids = sorted(set(num) & set(den))
    cells = {}
    for i in ids:
        cells.setdefault(num[i][0], []).append(math.log(num[i][1] / den[i][1]))
    point = math.exp(sum(sum(v) / len(v) for v in cells.values()) / len(cells))
    rng = random.Random(seed)
    if not boot:
        return point, None, None, len(ids)
    bs = []
    for _ in range(boot):
        tot = 0.0
        for v in cells.values():
            tot += sum(v[rng.randrange(len(v))] for _ in v) / len(v)
        bs.append(math.exp(tot / len(cells)))
    bs.sort()
    return point, bs[int(0.025 * boot)], bs[int(0.975 * boot) - 1], len(ids)


def arm_stats(r):
    toks = sum(v[2] for v in r.values())
    lat = sum(v[3] for v in r.values())
    ver = sum(v[4] for v in r.values())
    return toks / lat, (toks / ver if ver else None)


def identical(a, b):
    ids = sorted(set(a["outputs"]) & set(b["outputs"]))
    return sum(a["outputs"][i] == b["outputs"][i] for i in ids), len(ids)


def per_suite(r, arms):
    """Per dataset (DFlash-style): speedup of stock and final over AR, and tau of each arm."""
    suites = sorted({v[0][0] for v in next(iter(r.values())).values()})
    names = [k for k in ("stock", "final") if k in r]
    head = "dataset     " + "".join(f"{k + ' tau':>11}" for k in names)
    if "ar" in r:
        head += "".join(f"{k + '/AR':>11}" for k in names)
    if "stock" in r and "final" in r:
        head += f"{'final/stock':>13}"
    print(head)
    for s in suites + ["average"]:
        sel = {k: {i: v for i, v in r[k].items() if s == "average" or v[0][0] == s} for k in r}
        line = f"{s:<12}" + "".join(f"{arm_stats(sel[k])[1]:>11.2f}" for k in names)
        if "ar" in r:
            line += "".join(f"{speedup(sel[k], sel['ar'], boot=0)[0]:>10.2f}x" for k in names)
        if "stock" in r and "final" in r:
            line += f"{speedup(sel['final'], sel['stock'], boot=0)[0]:>12.2f}x"
        print(line)


def cmd_speed(a):
    arms = {k: load(v) for k, v in (("ar", a.ar), ("stock", a.stock), ("final", a.final)) if v}
    r = {k: rates(v) for k, v in arms.items()}
    print("arm      runs  requests  tok/s     tau")
    for k, v in r.items():
        tps, tau = arm_stats(v)
        print(f"{k:<8} {len(arms[k]):>4}  {len(v):>8}  {tps:8.2f}  {tau if tau is None else round(tau, 2)}")
    print()
    for num, den in (("stock", "ar"), ("final", "ar"), ("final", "stock")):
        if num in r and den in r:
            p, lo, hi, n = speedup(r[num], r[den])
            same, tot = identical(arms[num][0], arms[den][0])
            print(f"{num} vs {den}: {p:.3f}x [{lo:.3f}, {hi:.3f}] over {n} requests; identical outputs {same}/{tot}")
    print()
    per_suite(r, arms)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("speed")
    s.add_argument("--ar", nargs="*", default=[])
    s.add_argument("--stock", nargs="*", default=[])
    s.add_argument("--final", nargs="*", default=[])
    a = ap.parse_args()
    cmd_speed(a)


if __name__ == "__main__":
    main()
