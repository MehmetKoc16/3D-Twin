"""Idempotent shallow + sparse fetch of the pinned MakeHuman / MPFB2 data into .cache/ (gitignored)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from config import CACHE_DIR, SOURCES

REQUIRED_FILES = {
    "mh": [
        "makehuman/data/3dobjs/base.obj",
        "makehuman/data/targets/macrodetails/universal-male-young-averagemuscle-maxweight.target",
        "LICENSE.md",
    ],
    "mpfb2": [
        "src/mpfb/data/rigs/standard/rig.game_engine.json",
        "src/mpfb/data/rigs/standard/weights.game_engine.json",
        "LICENSE.md",
    ],
}


def _git(cwd: Path, *args: str, check: bool = True) -> str:
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
    res = subprocess.run(
        ["git", *args], cwd=str(cwd), env=env, capture_output=True, text=True, encoding="utf-8"
    )
    if check and res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {cwd}:\n{res.stderr}")
    return res.stdout.strip()


def _complete(key: str, directory: Path) -> bool:
    return all((directory / f).exists() for f in REQUIRED_FILES[key])


def _head(directory: Path) -> str | None:
    if not (directory / ".git").exists():
        return None
    out = _git(directory, "rev-parse", "HEAD", check=False)
    return out or None


def fetch_source(key: str) -> None:
    src = SOURCES[key]
    directory: Path = src["dir"]
    sha: str = src["sha"]
    if _head(directory) == sha and _complete(key, directory):
        print(f"[fetch] {key}: already at {sha[:10]}")
        return
    if directory.exists() and not (directory / ".git").exists() and any(directory.iterdir()):
        raise RuntimeError(f"{directory} exists but is not a git checkout; remove it and retry")
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / ".git").exists():
        _git(directory, "init", "--quiet")
        _git(directory, "remote", "add", "origin", src["url"])
    _git(directory, "sparse-checkout", "set", "--no-cone", *src["sparse"])
    print(f"[fetch] {key}: fetching {sha[:10]} from {src['url']} (shallow, sparse)")
    _git(directory, "fetch", "--depth", "1", "--filter=blob:none", "--quiet", "origin", sha)
    _git(directory, "-c", "advice.detachedHead=false", "checkout", "--quiet", "--force", sha)
    if _head(directory) != sha or not _complete(key, directory):
        raise RuntimeError(f"{key}: checkout at {sha} is incomplete")


def fetch_all() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for key in SOURCES:
        fetch_source(key)


def ensure_present() -> None:
    """--no-fetch mode: verify the cache is complete and at the pinned commits."""
    for key, src in SOURCES.items():
        if not _complete(key, src["dir"]):
            raise RuntimeError(f"{key} data missing in {src['dir']}; run without --no-fetch")
        head = _head(src["dir"])
        if head != src["sha"]:
            print(f"[fetch] WARNING: {key} is at {head}, pinned {src['sha']}", file=sys.stderr)


if __name__ == "__main__":
    fetch_all()
