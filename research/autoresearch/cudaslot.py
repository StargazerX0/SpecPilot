"""Machine-wide CUDA-init semaphore ("cuda slot"): at most ``SPECPILOT_BOOT_CONCURRENCY`` processes (default 1) are in
the heavy CUDA phase (context init, weight load, JIT module loads, CUDA-graph capture) at once on the machine, whichever
checkout, campaign or user starts them.

Overlapping CUDA context inits left processes in D state in the NVIDIA driver (os_acquire_rwlock_write) and stalled
nvidia-smi machine-wide for minutes (10-17, 10-18): two sglang boots 30 s apart, and pytest workers + a microbench +
nvidia-smi probes at once. Every CUDA init the framework starts takes a slot first:

* a server boot (``jobs.Server.start``, scorer boots, ``research/scripts/serve.sh``) holds its slot from the launch
  until the server is healthy, at most ``SPECPILOT_BOOT_HOLD_MAX_S`` (default 240) s;
* a short init (``ar microbench``'s worker, the GPU unit-layer pytest, ``research/scripts/cudaslot <class> -- cmd``,
  ``with cudaslot.slot(...)``) holds it until its first CUDA work is done (``signal_ready()``), the command exits, or
  ``SPECPILOT_CUDA_INIT_HOLD_S`` (default 30) s;
* two grants are at least ``SPECPILOT_BOOT_STAGGER_S`` (default 10, 0 allowed) s apart.

Waiters are served by class, then in arrival order: final > confirm > pretime > screen/live/null > sweep/profile >
microbench/unit/explore. ``ar`` sets the class per stage (``SPECPILOT_BOOT_CLASS``); ``SPECPILOT_BOOT_PRIORITY``
overrides it. A waiter moves up one class per ``SPECPILOT_BOOT_AGING_S`` (default 600) s of waiting, so no class starves.

State: ``$SPECPILOT_BOOT_LOCK_DIR`` (default ``/tmp/specpilot_cudaslot``, one per machine, world-writable) holds
``state.json`` (holders, waiters, the last grant, recent releases), changed only under a short ``flock`` on ``lock``; no
lock is held while waiting. Holders and waiters are recorded with PID + process start time: a dead holder's slot is
reclaimed at once by the next poll, a dead waiter leaves the queue, an expired holder (max hold) counts as released.
A ``sglang.launch_server`` younger than ``SPECPILOT_BOOT_FOREIGN_S`` (default 90) s that runs under no slot (another
framework, an older checkout, another user's process) or whose slot holder died counts as a holder (soft guard).

Scheduling only: nothing here is part of a measurement or deps identity (``identity.INFRA_FILES``). Stdlib only;
``serve.sh`` runs it as a script (``python3 -I cudaslot.py acquire|release|run|status``).
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

RANK = {"final": 0, "confirm": 1, "pretime": 2, "screen": 3, "live": 3, "null": 3, "sweep": 4, "profile": 4,
        "microbench": 5, "unit": 5, "explore": 5}
DEFAULT_CLASS = "explore"
# `ar <command>` -> class; `ar validate <layer>` by layer (absent: the screen class)
COMMAND_CLASS = {"final": "final", "confirm": "confirm", "pretime": "pretime", "sweep": "sweep", "profile": "profile",
                 "microbench": "microbench", "live": "live", "smoke": "screen", "screen": "screen", "null": "null",
                 "null-candidate": "null"}
VALIDATE_CLASS = {"cpu": "unit", "unit": "unit", "quality": "confirm"}

DEFAULTS = {"SPECPILOT_BOOT_CONCURRENCY": 1, "SPECPILOT_BOOT_STAGGER_S": 10.0, "SPECPILOT_BOOT_HOLD_MAX_S": 240.0,
            "SPECPILOT_CUDA_INIT_HOLD_S": 30.0, "SPECPILOT_BOOT_AGING_S": 600.0, "SPECPILOT_BOOT_FOREIGN_S": 90.0}
DEFAULT_DIR = "/tmp/specpilot_cudaslot"
LAUNCH_MARK = "sglang.launch_server"
MARK_ENV = "SPECPILOT_CUDASLOT"            # slot id a process tree runs under (servers, wrapped commands)
READY_ENV = "SPECPILOT_CUDASLOT_READY"     # file a wrapped command creates once its first CUDA work is done
DONE_ENV = "SPECPILOT_CUDASLOT_DONE"       # "1": the caller holds the boot's slot (controller -> serve.sh)
RECENT = 50
POLL_S = 1.0


def _env_num(name: str) -> float:
    try:
        return max(0.0, float(os.environ.get(name, DEFAULTS[name])))
    except ValueError:
        return float(DEFAULTS[name])


def settings() -> dict:
    return {"concurrency": int(_env_num("SPECPILOT_BOOT_CONCURRENCY")), "gap_s": _env_num("SPECPILOT_BOOT_STAGGER_S"),
            "hold_max_s": _env_num("SPECPILOT_BOOT_HOLD_MAX_S"), "short_hold_s": _env_num("SPECPILOT_CUDA_INIT_HOLD_S"),
            "aging_s": _env_num("SPECPILOT_BOOT_AGING_S"), "foreign_s": _env_num("SPECPILOT_BOOT_FOREIGN_S"),
            "dir": lock_dir()}


def lock_dir() -> str:
    return os.environ.get("SPECPILOT_BOOT_LOCK_DIR") or DEFAULT_DIR


def resolve_class(cls: str | None = None) -> str:
    """SPECPILOT_BOOT_PRIORITY, else the call's class, else the stage's (SPECPILOT_BOOT_CLASS), else explore."""
    if cls and cls not in RANK:
        raise ValueError(f"unknown cuda slot class {cls!r}; one of {sorted(RANK)}")
    for name, c in (("SPECPILOT_BOOT_PRIORITY", os.environ.get("SPECPILOT_BOOT_PRIORITY")), ("", cls),
                    ("SPECPILOT_BOOT_CLASS", os.environ.get("SPECPILOT_BOOT_CLASS"))):
        if c in RANK:
            return c
        if c:
            print(f"[cudaslot] ignoring {name}={c!r}: not one of {sorted(RANK)}", file=sys.stderr)
    return DEFAULT_CLASS


def set_stage(cmd: str, layer: str | None = None) -> str:
    """`ar` calls this once per invocation: the class every slot of the stage (servers, scorer, pytest) waits with."""
    cls = VALIDATE_CLASS.get(layer, "screen") if cmd == "validate" else COMMAND_CLASS.get(cmd, "screen")
    os.environ["SPECPILOT_BOOT_CLASS"] = cls
    return cls


# --- processes ---------------------------------------------------------------------------------------------------------

def _proc_start(pid) -> str | None:
    """Start time of ``pid`` in clock ticks since boot (a reused pid has another one); None when it is gone."""
    try:
        with open(f"/proc/{int(pid)}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError, ValueError, TypeError):
        return None


def _alive(pid, pstart) -> bool:
    st = _proc_start(pid)
    return st is not None and (pstart is None or st == pstart)


def _boot_time() -> float | None:
    try:
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("btime "):
                    return float(line.split()[1])
    except OSError:
        pass
    return None


def launch_servers(proc: str = "/proc") -> list:
    """[(pid, start epoch, slot id or None)] of every ``sglang.launch_server`` process on the machine; the slot id is
    the process's SPECPILOT_CUDASLOT (None when unset or unreadable, e.g. another user's process)."""
    btime, hz = _boot_time(), os.sysconf("SC_CLK_TCK")
    out = []
    if btime is None:
        return out
    for pid in os.listdir(proc):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            with open(f"{proc}/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
            if LAUNCH_MARK not in cmd:
                continue
            with open(f"{proc}/{pid}/stat") as f:
                start = btime + float(f.read().rsplit(")", 1)[1].split()[19]) / hz
        except (OSError, IndexError, ValueError):
            continue
        mark = None
        try:
            with open(f"{proc}/{pid}/environ", "rb") as f:
                for kv in f.read().split(b"\0"):
                    if kv.startswith(MARK_ENV.encode() + b"="):
                        mark = kv.split(b"=", 1)[1].decode(errors="replace") or None
        except OSError:
            pass
        out.append((int(pid), start, mark))
    return out


def foreign_boots(st: dict, procs: list, now: float, foreign_s: float) -> list:
    """The young servers that count as holders: no slot, or a slot whose holder died before the server was healthy."""
    died = {r.get("id") for r in st["recent"] if r.get("reason") == "holder_died"}
    return [{"pid": pid, "age_s": round(now - start, 1), "slot": mark} for pid, start, mark in procs
            if now - foreign_s < start <= now + 1 and mark not in st["holders"] and (mark is None or mark in died)]


# --- state -------------------------------------------------------------------------------------------------------------

def _empty() -> dict:
    return {"holders": {}, "waiters": {}, "last_grant": 0.0, "recent": []}


def _load(path: str) -> dict:
    try:
        with open(path) as f:
            st = json.load(f)
        if not (isinstance(st.get("holders"), dict) and isinstance(st.get("waiters"), dict)):
            raise ValueError("bad shape")
        st["last_grant"] = float(st.get("last_grant") or 0.0)
        st["recent"] = list(st.get("recent") or [])
        return st
    except (OSError, ValueError, TypeError, AttributeError):
        return _empty()


def _save(d: str, st: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".state.")
    with os.fdopen(fd, "w") as f:
        json.dump(st, f)
    os.chmod(tmp, 0o666)
    os.replace(tmp, os.path.join(d, "state.json"))


@contextlib.contextmanager
def _state(write: bool = True):
    """The state under the machine-wide flock (held only for this read-modify-write)."""
    d = lock_dir()
    old = os.umask(0)
    try:
        os.makedirs(d, mode=0o777, exist_ok=True)
        fd = os.open(os.path.join(d, "lock"), os.O_RDWR | os.O_CREAT, 0o666)
    finally:
        os.umask(old)
    with os.fdopen(fd, "r+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        st = _load(os.path.join(d, "state.json"))
        before = json.dumps(st, sort_keys=True)
        yield st
        if write and json.dumps(st, sort_keys=True) != before:
            _save(d, st)


def _note(st: dict, sid: str, h: dict, reason: str, now: float) -> dict:
    rec = {"id": sid, "what": h.get("what"), "class": h.get("class"), "kind": h.get("kind"), "pid": h.get("pid"),
           "waited_s": h.get("waited_s"), "held_s": round(now - h.get("granted", now), 1), "reason": reason,
           "released": now}
    st["recent"] = (st["recent"] + [rec])[-RECENT:]
    return rec


def _prune(st: dict, now: float) -> None:
    """Dead holders' slots are reclaimed, expired holders released, dead waiters leave the queue."""
    for sid, h in list(st["holders"].items()):
        if not _alive(h.get("pid"), h.get("pstart")):
            _note(st, sid, h, "holder_died", now)
            del st["holders"][sid]
        elif now > float(h.get("deadline") or 0):
            _note(st, sid, h, "max_hold", now)
            del st["holders"][sid]
    for wid, w in list(st["waiters"].items()):
        if not _alive(w.get("pid"), w.get("pstart")):
            del st["waiters"][wid]


def effective_rank(w: dict, now: float, aging_s: float) -> int:
    rank = RANK.get(w.get("class"), RANK[DEFAULT_CLASS])
    if aging_s > 0:
        since = w.get("since")
        rank -= int(max(0.0, now - (now if since is None else float(since))) // aging_s)
    return max(0, rank)


def queue(st: dict, now: float, aging_s: float) -> list:
    """Waiter ids in service order: effective class, then arrival."""
    return sorted(st["waiters"], key=lambda w: (effective_rank(st["waiters"][w], now, aging_s),
                                                float(st["waiters"][w].get("since") or 0), w))


def grantable(st: dict, wid: str, now: float, *, concurrency: int, gap_s: float, aging_s: float, foreign: int = 0) -> bool:
    free = concurrency - len(st["holders"]) - foreign
    last = st["last_grant"] if st["last_grant"] <= now + 3600 else 0.0      # a clock step is not a gap
    return free > 0 and now >= last + gap_s and wid in queue(st, now, aging_s)[:free]


# --- slots -------------------------------------------------------------------------------------------------------------

class Slot:
    """One granted slot (``id`` None when the semaphore is off or its directory is unusable: nothing to release)."""

    def __init__(self, *, sid, cls, what, kind, hold_max_s, waited_s=0.0, granted=None, error=None):
        self.id, self.cls, self.what, self.kind, self.hold_max_s = sid, cls, what, kind, hold_max_s
        self.waited_s, self.granted, self.error = waited_s, granted or time.time(), error
        self.reason = None          # set by the holder before release: healthy, boot_failed, ready, exited, ...
        self.released = None        # {"held_s", "reason"} once released

    def env(self) -> dict:
        return {MARK_ENV: self.id} if self.id else {}

    def record(self) -> dict:
        rec = {"class": self.cls, "kind": self.kind, "waited_s": self.waited_s, "hold_max_s": self.hold_max_s}
        if self.error:
            rec["error"] = self.error
        if self.released:
            rec.update(self.released)
        return rec


def _hold_max(kind: str, hold_max_s: float | None, s: dict) -> float:
    if hold_max_s is not None:
        return hold_max_s
    return s["hold_max_s"] if kind == "server" else s["short_hold_s"]


def acquire(what: str = "", cls: str | None = None, *, kind: str = "short", hold_max_s: float | None = None,
            owner_pid: int | None = None, log=None, poll_s: float | None = None) -> Slot:
    """Waits for a slot and returns it (held by ``owner_pid``, default this process). Never raises for an unusable lock
    directory: the CUDA init then goes ahead without a slot (``Slot.error``)."""
    cls, s = resolve_class(cls), settings()
    hold = _hold_max(kind, hold_max_s, s)
    t0 = time.time()
    if s["concurrency"] <= 0:
        return Slot(sid=None, cls=cls, what=what, kind=kind, hold_max_s=hold)
    owner = owner_pid or os.getpid()
    owner_start = _proc_start(owner)
    wid = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    me = {"pid": os.getpid(), "pstart": _proc_start(os.getpid()), "class": cls, "what": what, "kind": kind, "since": t0,
          "owner": owner}
    told = 0.0
    try:
        while True:
            if owner != os.getpid() and not _alive(owner, owner_start):
                raise RuntimeError(f"cuda slot: owner pid {owner} exited while waiting")
            procs = launch_servers()
            with _state() as st:
                now = time.time()
                _prune(st, now)
                st["waiters"].setdefault(wid, me)
                foreign = foreign_boots(st, procs, now, s["foreign_s"])
                if grantable(st, wid, now, concurrency=s["concurrency"], gap_s=s["gap_s"], aging_s=s["aging_s"],
                             foreign=len(foreign)):
                    del st["waiters"][wid]
                    waited = round(now - t0, 1)
                    st["holders"][wid] = {"pid": owner, "pstart": owner_start, "class": cls, "what": what, "kind": kind,
                                          "since": t0, "granted": now, "deadline": now + hold, "waited_s": waited}
                    st["last_grant"] = now
                    break
                pos = queue(st, now, s["aging_s"]).index(wid) + 1
                busy = [h.get("what") for h in st["holders"].values()] + [f"pid {f['pid']} (no slot)" for f in foreign]
            if log is not None and now - told >= 60:
                print(f"[cudaslot] {what or 'CUDA init'} [{cls}]: waiting for the machine-wide CUDA-init slot "
                      f"(queue position {pos}; held by {', '.join(map(str, busy)) or 'nobody, min gap'})",
                      file=log, flush=True)
                told = now
            time.sleep(POLL_S if poll_s is None else poll_s)
    except OSError as e:
        with contextlib.suppress(OSError), _state() as st:
            st["waiters"].pop(wid, None)
        return Slot(sid=None, cls=cls, what=what, kind=kind, hold_max_s=hold, waited_s=round(time.time() - t0, 1),
                    error=f"{type(e).__name__}: {e}")
    except BaseException:
        with contextlib.suppress(OSError), _state() as st:
            st["waiters"].pop(wid, None)
        raise
    if log is not None and waited >= 1:
        print(f"[cudaslot] {what or 'CUDA init'} [{cls}]: slot granted after {waited:.0f} s", file=log, flush=True)
    return Slot(sid=wid, cls=cls, what=what, kind=kind, hold_max_s=hold, waited_s=waited, granted=now)


def release(slot: Slot | str, reason: str | None = None) -> dict:
    """Gives the slot back; {"held_s", "reason"} (reason ``max_hold``/``holder_died`` when it had been taken back)."""
    sid = slot.id if isinstance(slot, Slot) else slot
    reason = reason or (slot.reason if isinstance(slot, Slot) else None) or "released"
    out = {"held_s": None, "reason": reason}
    if sid:
        try:
            with _state() as st:
                now = time.time()
                h = st["holders"].pop(sid, None)
                if h is not None:
                    rec = _note(st, sid, h, reason, now)
                    out = {"held_s": rec["held_s"], "reason": reason}
                else:
                    prev = [r for r in st["recent"] if r.get("id") == sid]
                    out = {k: prev[-1][k] for k in ("held_s", "reason")} if prev else {"held_s": None, "reason": "gone"}
        except OSError as e:
            out = {"held_s": None, "reason": reason, "error": f"{type(e).__name__}: {e}"}
    if isinstance(slot, Slot):
        slot.released = out
    return out


@contextlib.contextmanager
def slot(cls: str | None = None, what: str = "", *, kind: str = "short", hold_max_s: float | None = None,
         log=sys.stderr):
    """``with cudaslot.slot("explore", "my kernel test"):`` around a script's first CUDA work (context init, first
    allocation/graph capture). Released at the end of the block (or by max hold / the process's exit)."""
    s = acquire(what, cls, kind=kind, hold_max_s=hold_max_s, log=log)
    try:
        yield s
    finally:
        release(s)


def signal_ready() -> None:
    """Called by a command run under ``run()`` once its first CUDA work is done: its slot is released early."""
    path = os.environ.get(READY_ENV)
    if path:
        with contextlib.suppress(OSError):
            open(path, "a").close()


def run(cmd, cls: str | None = None, what: str = "", *, kind: str = "short", hold_max_s: float | None = None,
        log=sys.stderr, timeout: float | None = None, **popen_kw):
    """Runs ``cmd`` (Popen arguments) under a slot, released when the command calls ``signal_ready()``, exits, or the
    max hold passes. Returns (CompletedProcess, Slot); ``timeout`` kills the command (subprocess.TimeoutExpired)."""
    s = acquire(what or (cmd if isinstance(cmd, str) else " ".join(map(str, cmd)))[:200], cls, kind=kind,
                hold_max_s=hold_max_s, log=log)
    fd, ready = tempfile.mkstemp(prefix="cudaslot_ready.")
    os.close(fd)
    os.unlink(ready)
    env = {**(popen_kw.pop("env", None) or os.environ), **s.env(), READY_ENV: ready}
    stop = threading.Event()
    try:
        p = subprocess.Popen(cmd, env=env, **popen_kw)
    except BaseException:
        release(s, "launch_failed")
        raise

    def watch():
        end = s.granted + s.hold_max_s
        while not stop.is_set():
            if os.path.exists(ready):
                return release(s, "ready")
            if p.poll() is not None:
                return release(s, "exited")
            if time.time() >= end:
                return release(s, "max_hold")
            stop.wait(0.2)
        release(s, "exited" if p.poll() is not None else "interrupted")

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    try:
        out, err = p.communicate(timeout=timeout)
    except BaseException:
        p.kill()
        p.wait()
        raise
    finally:
        stop.set()
        th.join(5)
        with contextlib.suppress(OSError):
            os.unlink(ready)
    return subprocess.CompletedProcess(p.args, p.returncode, out, err), s


# --- visibility --------------------------------------------------------------------------------------------------------

def status(now: float | None = None) -> dict:
    s = settings()
    procs = launch_servers()
    try:
        with _state() as st:
            now = time.time() if now is None else now
            _prune(st, now)
            foreign = foreign_boots(st, procs, now, s["foreign_s"])
            holders = [{"id": k, **{f: h.get(f) for f in ("class", "kind", "what", "pid", "waited_s")},
                        "held_s": round(now - float(h.get("granted") or now), 1),
                        "releases_in_s": round(float(h.get("deadline") or now) - now, 1)} for k, h in st["holders"].items()]
            waiters = [{"id": w, "position": i + 1, **{f: st["waiters"][w].get(f) for f in ("class", "kind", "what", "pid")},
                        "effective_class": [c for c, r in RANK.items()
                                            if r == effective_rank(st["waiters"][w], now, s["aging_s"])][0],
                        "waiting_s": round(now - float(st["waiters"][w].get("since") or now), 1)}
                       for i, w in enumerate(queue(st, now, s["aging_s"]))]
            recent = st["recent"][-8:]
    except OSError as e:
        return {"settings": s, "error": f"{type(e).__name__}: {e}"}
    return {"settings": s, "holders": holders, "waiters": waiters, "foreign_boots": foreign, "recent": recent}


def text(st: dict | None = None, brief: bool = False) -> str:
    st = st or status()
    s = st["settings"]
    if st.get("error"):
        return f"cuda slot: state unreadable ({st['error']})"
    if brief and not (st["holders"] or st["waiters"] or st["foreign_boots"]):
        return ""
    lines = [f"cuda slot ({s['dir']}): concurrency {s['concurrency']}, min gap {s['gap_s']:g} s, hold max "
             f"{s['hold_max_s']:g} s server / {s['short_hold_s']:g} s short, aging {s['aging_s']:g} s/class"]
    lines += [f"  holding [{h['class']}] {h['what']} (pid {h['pid']}, {h['kind']}): held {h['held_s']:.0f} s after waiting "
              f"{h['waited_s']:.0f} s; max hold in {h['releases_in_s']:.0f} s" for h in st["holders"]]
    lines += [f"  foreign boot (no slot): pid {f['pid']}, {f['age_s']:.0f} s old" for f in st["foreign_boots"]]
    lines += [f"  waiting {w['position']}. [{w['class']}" + (f"->{w['effective_class']}" if w["effective_class"] != w["class"]
              else "") + f"] {w['what']} (pid {w['pid']}): {w['waiting_s']:.0f} s" for w in st["waiters"]]
    if not brief:
        lines += [f"  released [{r.get('class')}] {r.get('what')}: waited {r.get('waited_s') or 0:.0f} s, held "
                  f"{r.get('held_s') or 0:.0f} s ({r.get('reason')})" for r in st["recent"]]
    if len(lines) == 1:
        lines.append("  free")
    return "\n".join(lines)


# --- CLI (serve.sh, research/scripts/cudaslot) -------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("acquire", help="wait for a slot; print its id (nothing when the semaphore is off)")
    p.add_argument("--what", default="")
    p.add_argument("--class", dest="cls", default=None)
    p.add_argument("--kind", default="server", choices=["server", "short"])
    p.add_argument("--owner-pid", type=int, default=None, help="process that holds the slot (default: the caller's parent)")
    p = sub.add_parser("release")
    p.add_argument("id")
    p.add_argument("--reason", default=None)
    p = sub.add_parser("run", help="run a command under a slot: run [--class C] [--kind K] [--hold-max S] -- cmd ...")
    p.add_argument("--what", default="")
    p.add_argument("--class", dest="cls", default=None)
    p.add_argument("--kind", default="short", choices=["server", "short"])
    p.add_argument("--hold-max", type=float, default=None)
    p.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("status")
    p.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "acquire":
        s = acquire(a.what, a.cls, kind=a.kind, owner_pid=a.owner_pid or os.getppid(), log=sys.stderr)
        if s.error:
            print(f"[cudaslot] no slot taken: {s.error}", file=sys.stderr)
        print(s.id or "")
        return 0
    if a.cmd == "release":
        print(json.dumps(release(a.id, a.reason)))
        return 0
    if a.cmd == "status":
        st = status()
        print(json.dumps(st, indent=1) if a.json else text(st))
        return 0
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not cmd:
        ap.error("run: no command")
    for sig in (signal.SIGTERM, signal.SIGHUP):          # end the wrapped command with the wrapper
        signal.signal(sig, lambda n, _f: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        p, s = run(cmd, a.cls, a.what, kind=a.kind, hold_max_s=a.hold_max, log=sys.stderr)
    except KeyboardInterrupt:
        return 130
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
