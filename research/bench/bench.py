#!/usr/bin/env python3
"""Fixed-prompt benchmark against a running SGLang server (greedy, lossless check).

Every request is pre-tokenized client-side with the target's chat template, so
all configurations see identical input ids. Multi-turn items ("turns") feed the
model's own previous answer back, like z-lab/dflash's MT-Bench runner. Records
per-turn latency, output tokens, verification steps (speculative runs), exact
output ids (for greedy agreement), and NVML peak GPU memory of the measured
phase.

Items with a ``history`` field (canonical assistant answers for earlier turns,
frozen by ``ar history``) use that history instead
of the model's own answers, so every configuration sees identical later-turn
inputs ("fixed" rows); without it later turns are "live" and only comparable
while earlier turns match. Every repeat's rows, the input ids and their sha256
are kept in the record. Requests carry ``rid = <row id>|<pass label>``.

``--levels 8,16,24`` runs the load protocol (protocol ``load``): for every concurrency level ``c``, in
ascending order, an untimed warmup at ``c`` and then a closed-loop pass of ``c`` client threads over the
level's request stream (``load_stream``: the items repeated so that there are at least
``--min-requests-per-slot`` x c requests, interleaved by a fixed per-level permutation). Copies get row ids
``<item>@c<c>~<copy>#t<turn>`` and suites ``<suite>@c<c>``, so every downstream statistic treats a
(suite, level) pair as one cell. Rows record their start/end time in the pass; ``per_level`` holds the
steady-state wall-clock throughput (tokens completed per second while all ``c`` slots are busy).

``--ttft-probe N`` adds an untimed-for-R streaming pass over the first turn of N items (round-robin
over suites, 64 new tokens) after the measured pass: client time to first streamed token (TTFT) and
decode-only rate. ``--shuffle-seed`` permutes the item order (state/contamination checks).

  python research/bench/bench.py --tag X --port 30000 --out $XINFER_WORK/scratch/X.json
  python research/bench/bench.py --tag X --prompts research/bench/data/*.jsonl \
      --repeats 1 --no-full-warmup --max-new-tokens 2048 --out ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

import grade  # noqa: E402  (research/bench is on sys.path: script directory, or added by the controller)

HERE = Path(__file__).resolve().parent


def load_items(paths):
    items = []
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    it = json.loads(line)
                    it.setdefault("turns", [it.get("prompt")])
                    it.setdefault("suite", Path(path).stem)
                    items.append(it)
    return items


class GpuMemSampler:
    """Polls NVML used memory of one physical GPU; reports the max."""

    def __init__(self, index: int, period: float = 0.05):
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._h = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.period = period
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _used(self):
        return self._nvml.nvmlDeviceGetMemoryInfo(self._h).used

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._used())
            time.sleep(self.period)

    def __enter__(self):
        self.peak = self._used()
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


class Client:
    def __init__(self, url, tokenizer_path, enable_thinking, sampling):
        from transformers import AutoTokenizer

        self.url = url
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
        self.enable_thinking = enable_thinking
        self.sampling = sampling

    def encode(self, messages):
        ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, enable_thinking=self.enable_thinking
        )
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        return list(ids)

    def _generate(self, input_ids, sampling, rid):
        t0 = time.perf_counter()
        r = requests.post(self.url + "/generate", json={"input_ids": input_ids, "sampling_params": sampling,
                                                        "rid": rid}, timeout=3600)
        dt = time.perf_counter() - t0
        r.raise_for_status()
        return r.json(), dt

    def stream_first_turn(self, item, max_new_tokens, label):
        """Streaming request: client TTFT and decode-only rate of the first turn."""
        ids = self.encode([{"role": "user", "content": item["turns"][0]}])
        body = {"input_ids": ids, "sampling_params": {**self.sampling, "max_new_tokens": max_new_tokens},
                "stream": True, "rid": f"{item['id']}#t0|{label}"}
        t0 = time.perf_counter()
        first, meta = None, {}
        with requests.post(self.url + "/generate", json=body, stream=True, timeout=3600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                j = json.loads(payload)
                meta = j.get("meta_info", meta)
                if first is None and meta.get("completion_tokens", 0) > 0:
                    first = time.perf_counter() - t0
        total = time.perf_counter() - t0
        n = meta.get("completion_tokens", 0)
        return {"id": f"{item['id']}#t0", "suite": item["suite"], "prompt_tokens": len(ids), "completion_tokens": n,
                "ttft_s": first, "total_s": total,
                "decode_tok_s": (n - 1) / (total - first) if first is not None and n > 1 and total > first else None}

    def run_item(self, item, sampling=None, label="p0", t_ref=None):
        """One conversation; one result row per turn (``t_start``/``t_end`` relative to ``t_ref``)."""
        sampling = sampling or self.sampling
        t_ref = time.perf_counter() if t_ref is None else t_ref
        history = item.get("history")
        messages, rows = [], []
        for turn, user in enumerate(item["turns"]):
            messages.append({"role": "user", "content": user})
            input_ids = self.encode(messages)
            row_id = f"{item['id']}#t{turn}"
            t_start = time.perf_counter() - t_ref
            j, dt = self._generate(input_ids, sampling, f"{row_id}|{label}")
            meta = j["meta_info"]
            n_out, verify = meta["completion_tokens"], meta.get("spec_verify_ct") or 0
            rows.append({
                "id": row_id, "item": item["id"], "turn": turn,
                "suite": item["suite"], "category": item["category"],
                "prompt_tokens": meta["prompt_tokens"], "completion_tokens": n_out,
                "latency_s": dt, "server_e2e_s": meta.get("e2e_latency"), "spec_verify_ct": verify,
                "accept_len": (n_out / verify) if verify else None,
                "finish_reason": (meta.get("finish_reason") or {}).get("type"),
                "history": "none" if turn == 0 else ("fixed" if history else "live"),
                **level_fields(item), "t_start": t_start, "t_end": t_start + dt,
                "input_ids": input_ids, "input_sha256": ids_sha256(input_ids),
                "output_ids": j.get("output_ids"), "text": j.get("text"),
            })
            answer = history[turn] if history and turn < len(history) else j.get("text", "")
            if sampling.get("skip_special_tokens") is False:
                answer = grade.strip_end(answer)   # the chat template closes the turn itself
            messages.append({"role": "assistant", "content": answer})
        return rows


def level_fields(item):
    """A row's ``level`` is its load level (concurrency; None outside ``--levels``); a dataset's own ``level`` field
    (math500 difficulty) goes to ``difficulty``. Records before this split carried the difficulty in ``level`` outside
    load runs."""
    return {"level": item.get("load_level"), "difficulty": item.get("level")}


def first_items(items, n):
    """The first ``n`` items of every suite, in file order."""
    seen, out = {}, []
    for it in items:
        if seen.get(it["suite"], 0) < n:
            seen[it["suite"]] = seen.get(it["suite"], 0) + 1
            out.append(it)
    return out


def subsplit_items(items, k, n):
    """Conversation-balanced sub-split ``k`` of ``n`` per suite: items sorted by id hash, every n-th from k."""
    by_suite = {}
    for it in items:
        by_suite.setdefault(it["suite"], []).append(it)
    keep = set()
    for suite, its in by_suite.items():
        ordered = sorted(its, key=lambda it: hashlib.sha256(it["id"].encode()).hexdigest())
        keep |= {it["id"] for it in ordered[k::n]}
    return [it for it in items if it["id"] in keep]


def shard_items(items, k, m):
    """Shard ``k`` of ``m`` (whole conversations, every suite spread over all shards): items sorted per suite by a
    salted id hash, every m-th from k. The union of the m shards is the input; used to split one untimed arm over
    several servers (outputs at concurrency 1 do not depend on which server produced them)."""
    by_suite = {}
    for it in items:
        by_suite.setdefault(it["suite"], []).append(it)
    keep, off = set(), 0
    for _, its in sorted(by_suite.items()):
        ordered = sorted(its, key=lambda it: hashlib.sha256(("shard:" + it["id"]).encode()).hexdigest())
        keep |= {it["id"] for i, it in enumerate(ordered) if (i + off) % m == k}
        off += len(ordered)       # rotate so the suites' remainders land on different shards
    return [it for it in items if it["id"] in keep]


def probe_items(items, n):
    by_suite = {}
    for it in items:
        by_suite.setdefault(it["suite"], []).append(it)
    out, k = [], 0
    while len(out) < n and any(k < len(v) for v in by_suite.values()):
        out += [v[k] for _, v in sorted(by_suite.items()) if k < len(v)]
        k += 1
    return out[:n]


def ids_sha256(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def load_stream(items, level, min_requests_per_slot):
    """The request stream of one load level (protocol ``load.arrival``): ``items`` repeated
    k = max(1, ceil(min_requests_per_slot * level / requests)) times, copies renamed ``<id>@c<level>~<copy>``
    with suite ``<suite>@c<level>``, shuffled by ``random.Random(level)``. Deterministic in its inputs."""
    import math
    import random

    n_req = sum(len(it["turns"]) for it in items) or 1
    k = max(1, math.ceil(min_requests_per_slot * level / n_req)) if min_requests_per_slot else 1
    out = []
    for j in range(k):
        for it in items:
            out.append({**it, "id": f"{it['id']}@c{level}~{j}", "suite": f"{it['suite']}@c{level}", "base_id": it["id"],
                        "base_suite": it["suite"], "load_level": level, "copy": j})
    random.Random(level).shuffle(out)
    return out


def steady_throughput(rows, level):
    """Tokens completed per second while all ``level`` slots are busy: the window runs from the level-th request
    start to the last request start (after it the queue is empty and slots drain); each request's tokens are
    spread uniformly over its own duration. None when the window is empty."""
    starts = sorted(r["t_start"] for r in rows)
    if len(starts) < level:
        return None
    w0, w1 = starts[level - 1], starts[-1]
    if w1 <= w0:
        return None
    toks = 0.0
    for r in rows:
        dur = r["t_end"] - r["t_start"]
        if dur > 0:
            toks += r["completion_tokens"] * max(0.0, min(r["t_end"], w1) - max(r["t_start"], w0)) / dur
    return {"tok_s": toks / (w1 - w0), "window_s": w1 - w0}


def run_pass(client, items, concurrency, sampling=None, label="p0"):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as ex:
        res = [r for rows in ex.map(lambda it: client.run_item(it, sampling, label, t0), items) for r in rows]
    return res, time.perf_counter() - t0


def aggregate(sel):
    toks = sum(r["completion_tokens"] for r in sel)
    lat = sum(r["latency_s"] for r in sel)
    verify = sum(r["spec_verify_ct"] for r in sel)
    return {
        "n": len(sel), "output_tokens": toks, "sum_latency_s": lat, "mean_latency_s": lat / len(sel),
        "output_tok_s": toks / lat,
        "accept_len": (toks / verify) if verify else None,
        "mean_request_accept_len": statistics.mean(r["accept_len"] for r in sel) if verify else None,
    }


def summarize(passes, wall_times):
    """Per-request median latency across repeats, then aggregate per suite/category/overall."""
    by_id = {}
    for p in passes:
        for r in p:
            by_id.setdefault(r["id"], []).append(r)
    rows = []
    for rid, rs in by_id.items():
        lat = statistics.median(r["latency_s"] for r in rs)
        rows.append({**{k: rs[0][k] for k in ("id", "suite", "category", "turn", "prompt_tokens",
                                              "completion_tokens", "accept_len", "spec_verify_ct",
                                              "finish_reason", "history", "input_sha256", "level", "difficulty")},
                     "latency_s": lat, "tok_s": rs[0]["completion_tokens"] / lat,
                     "latencies_s": [r["latency_s"] for r in rs],
                     "repeat_deterministic": all(r["output_ids"] == rs[0]["output_ids"] for r in rs)})
    group = lambda key: {v: aggregate([r for r in rows if r[key] == v]) for v in sorted({r[key] for r in rows})}
    wall_tok_s = [sum(r["completion_tokens"] for r in p) / w for p, w in zip(passes, wall_times)]
    return {"per_request": rows, "per_suite": group("suite"), "per_category": group("category"),
            "overall": aggregate(rows), "pass_wall_s": wall_times, "pass_wall_tok_s": wall_tok_s}


def server_info(url):
    try:
        return requests.get(url + "/get_server_info", timeout=30).json()
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def git_rev(repo):
    try:
        rev = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", repo, "status", "--porcelain", "--", "python",
                                         "research/dflash_lab", "research/bench", "research/scripts"], text=True)
        return rev + ("-dirty" if dirty.strip() else "")
    except Exception:  # noqa: BLE001
        return None


def physical_gpu(index_env):
    return int(index_env.split(",")[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--prompts", nargs="+", default=[str(HERE / "prompts.jsonl")])
    ap.add_argument("--split", help="use every suite file of splits/<SPLIT>/ (overrides --prompts)")
    ap.add_argument("--history-file", help="JSON {item_id: [assistant answers]} for fixed multi-turn inputs")
    ap.add_argument("--warmup", default=str(HERE / "warmup.jsonl"))
    ap.add_argument("--full-warmup", action=argparse.BooleanOptionalAction, default=True,
                    help="also run one untimed pass over all prompts before measuring")
    ap.add_argument("--tokenizer", default=os.environ.get("TARGET_MODEL"))
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--levels", default=None, help="comma-separated load levels (protocol load.levels); overrides --concurrency")
    ap.add_argument("--min-requests-per-slot", type=int, default=4, help="load stream length per level (protocol load)")
    ap.add_argument("--disable-thinking", action="store_true")
    ap.add_argument("--keep-special-tokens", action="store_true",
                    help="skip_special_tokens false (Gemma 4: the thinking channel markers stay in the text)")
    ap.add_argument("--gpu-index", type=int, default=physical_gpu(os.environ.get("CUDA_VISIBLE_DEVICES", "0")))
    ap.add_argument("--ttft-probe", type=int, default=0, help="streaming first-turn probe over N items")
    ap.add_argument("--shuffle-seed", type=int, default=None, help="permute the item order")
    ap.add_argument("--first-turn-only", action="store_true", help="drop later turns (history generation)")
    ap.add_argument("--items-per-suite", type=int, default=None, help="keep the first N items of every suite")
    ap.add_argument("--subsplit", default=None, help="k/n: conversation-balanced sub-split k of n per suite")
    ap.add_argument("--shard", default=None, help="k/m: shard k of m of the selected items (untimed arms split over servers)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    url = f"http://{a.host}:{a.port}"
    sampling = {"temperature": 0.0, "top_k": 1, "max_new_tokens": a.max_new_tokens}
    if a.keep_special_tokens:
        sampling["skip_special_tokens"] = False
    client = Client(url, a.tokenizer, not a.disable_thinking, sampling)
    if a.split:
        a.prompts = [str(p) for p in sorted((HERE / "splits" / a.split).glob("*.jsonl"))]
    items = load_items(a.prompts)
    if a.history_file:
        hist = json.load(open(a.history_file))
        hist = hist.get("history", hist)
        for it in items:
            if len(it["turns"]) > 1:
                it["history"] = hist[it["id"]]
    if a.subsplit:
        k, n = (int(x) for x in a.subsplit.split("/"))
        items = subsplit_items(items, k, n)
    if a.items_per_suite is not None:
        items = first_items(items, a.items_per_suite)
    if a.shard:
        k, m = (int(x) for x in a.shard.split("/"))
        items = shard_items(items, k, m)
    if a.first_turn_only:
        for it in items:
            it["turns"] = it["turns"][:1]
            it.pop("history", None)
    if a.shuffle_seed is not None:
        import random

        random.Random(a.shuffle_seed).shuffle(items)

    levels = [int(x) for x in a.levels.split(",")] if a.levels else None
    per_level = {}
    if levels:
        passes = [[] for _ in range(a.repeats)]
        walls = []
        with GpuMemSampler(a.gpu_index) as mem:
            for c in sorted(levels):
                stream = load_stream(items, c, a.min_requests_per_slot)
                run_pass(client, load_items([a.warmup]), c, {**sampling, "max_new_tokens": 128}, f"w0c{c}")
                lv = {"requests": sum(len(it["turns"]) for it in stream), "copies": max(it["copy"] for it in stream) + 1,
                      "pass_wall_s": [], "steady": []}
                for rep in range(a.repeats):
                    res, wall = run_pass(client, stream, c, label=f"p{rep}c{c}")
                    passes[rep] += res
                    walls.append(wall)
                    lv["pass_wall_s"].append(wall)
                    lv["steady"].append(steady_throughput(res, c))
                lv["output_tokens"] = sum(r["completion_tokens"] for r in passes[0] if r["level"] == c)
                per_level[str(c)] = lv
        summary = summarize(passes, walls)
        for c, lv in per_level.items():
            lv.update(aggregate([r for r in summary["per_request"] if r["level"] == int(c)]))
    else:
        run_pass(client, load_items([a.warmup]), a.concurrency, {**sampling, "max_new_tokens": 128}, "w0")
        if a.full_warmup:
            run_pass(client, items, a.concurrency, label="w1")
        passes, walls = [], []
        with GpuMemSampler(a.gpu_index) as mem:
            for rep in range(a.repeats):
                res, wall = run_pass(client, items, a.concurrency, label=f"p{rep}")
                passes.append(res)
                walls.append(wall)
        summary = summarize(passes, walls)
    ttft = [client.stream_first_turn(it, 64, f"s{i}") for i, it in enumerate(probe_items(items, a.ttft_probe))]
    info = server_info(url)
    record = {
        "tag": a.tag, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_rev": git_rev(os.environ.get("INFER_REPO", str(HERE.parents[1]))),
        "candidate": os.environ.get("XINFER_CANDIDATE"), "shuffle_seed": a.shuffle_seed, "ttft_probe": ttft,
        "subsplit": a.subsplit, "shard": a.shard, "items_per_suite": a.items_per_suite, "first_turn_only": a.first_turn_only,
        "host": platform.node(), "gpu_index": a.gpu_index, "prompts": a.prompts,
        "sampling": sampling, "concurrency": a.concurrency if not levels else None, "levels": levels,
        "min_requests_per_slot": a.min_requests_per_slot if levels else None, "per_level": per_level, "repeats": a.repeats,
        "full_warmup": a.full_warmup, "thinking": not a.disable_thinking,
        "peak_gpu_mem_gib": mem.peak / 2**30,
        "server_info": {k: info.get(k) for k in (
            "version", "model_path", "speculative_algorithm", "speculative_num_draft_tokens",
            "attention_backend", "mem_fraction_static", "dtype", "kv_cache_dtype", "quantization",
            "mamba_ssm_dtype", "disable_radix_cache", "max_running_requests", "max_total_num_tokens")},
        **summary,
        "history_file": a.history_file,
        "outputs": {r["id"]: r["output_ids"] for r in passes[0]},
        "texts": {r["id"]: r["text"] for r in passes[0]},
        "inputs": {r["id"]: r["input_ids"] for r in passes[0]},
        "repeats_rows": [[{k: r[k] for k in ("id", "latency_s", "server_e2e_s", "completion_tokens",
                                             "spec_verify_ct", "finish_reason", "input_sha256", "level", "t_start", "t_end")}
                          | {"output_sha256": ids_sha256(r["output_ids"] or [])} for r in p] for p in passes],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(record, f, indent=1)
    o = summary["overall"]
    print(json.dumps({"tag": a.tag, **{k: o[k] for k in ("n", "output_tokens", "mean_latency_s", "output_tok_s",
                                                          "accept_len")},
                      "peak_gpu_mem_gib": round(record["peak_gpu_mem_gib"], 2)}, indent=1))


if __name__ == "__main__":
    main()
