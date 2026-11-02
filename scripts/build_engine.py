#!/usr/bin/env python3
"""Assemble the stock or final engine tree of one setting.

    python3 scripts/build_engine.py SETTING {stock,final} OUT_DIR

The tree is <OUT_DIR>/{python,lab}: the stock engine (python/ and research/dflash_lab of this repo), plus
overlays/dspark_stock for the DSpark settings, plus settings/SETTING/files for the final. It prints the tree's content
hash (sha256 of the canonical JSON {relative path: sha256 of file}); scripts/serve.sh keys the build caches by it.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", "build", "dist", ".mypy_cache"}
SKIP_SUFFIX = (".pyc", ".pyo", ".egg-info")


def _skip(rel: Path) -> bool:
    return any(p in SKIP_DIRS or p.endswith(SKIP_SUFFIX) for p in rel.parts)


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha(root: Path) -> str:
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if _skip(rel):
            continue
        if p.is_symlink():
            out[str(rel)] = "symlink:" + str(p.readlink())
        elif p.is_file():
            out[str(rel)] = _sha_file(p)
    return hashlib.sha256(json.dumps(out, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def overlay(src: Path, dst: Path) -> None:
    for p in sorted(src.rglob("*")):
        if p.is_file() and not _skip(p.relative_to(src)):
            t = dst / p.relative_to(src)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, t)
            t.chmod(0o755 if p.stat().st_mode & 0o100 else 0o644)


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[2] not in ("stock", "final"):
        sys.exit(__doc__)
    setting, which, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]).resolve()
    cfg_path = REPO / "settings" / setting / "config.json"
    if not cfg_path.exists():
        sys.exit(f"unknown setting {setting}; choose from: "
                 f"{', '.join(sorted(p.parent.name for p in (REPO / 'settings').glob('*/config.json')))}")
    cfg = json.loads(cfg_path.read_text())
    if out.exists():
        sys.exit(f"{out} exists; remove it or pick another directory")
    out.mkdir(parents=True)
    ignore = shutil.ignore_patterns(*SKIP_DIRS, "*.pyc")
    shutil.copytree(REPO / "python", out / "python", symlinks=True, ignore=ignore)
    shutil.copytree(REPO / "research" / "dflash_lab", out / "lab", symlinks=True, ignore=ignore)
    if cfg["dspark"]:
        overlay(REPO / "overlays" / "dspark_stock", out)
    if which == "final":
        overlay(REPO / "settings" / setting / "files", out)
    sha = tree_sha(out)
    (out / "TREE_SHA").write_text(sha + "\n")
    print(f"{setting} {which}: {out} (tree sha {sha[:16]})")


if __name__ == "__main__":
    main()
