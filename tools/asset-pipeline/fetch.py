"""Idempotent shallow + sparse fetch of the pinned MakeHuman / MPFB2 data into .cache/ (gitignored)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import hashlib
import urllib.request

from config import (
    CACHE_DIR, GARMENT_ASSETS, GARMENT_CACHE, GARMENT_PACK_URL, MEDIAPIPE, PART_ASSETS, PART_CACHE, PART_PACK_URL, SOURCES,
)

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


def _mediapipe_path(rel: str) -> Path:
    return MEDIAPIPE["dir"] / MEDIAPIPE["sha"] / rel


def _mediapipe_ok(rel: str) -> bool:
    p = _mediapipe_path(rel)
    return p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == MEDIAPIPE["files"][rel]


def fetch_mediapipe() -> None:
    """Download the pinned MediaPipe files (Apache-2.0) via raw.githubusercontent.com; sha256-verified, idempotent."""
    sha = MEDIAPIPE["sha"]
    for rel, digest in MEDIAPIPE["files"].items():
        if _mediapipe_ok(rel):
            continue
        url = f"https://raw.githubusercontent.com/{MEDIAPIPE['repo']}/{sha}/{rel}"
        print(f"[fetch] mediapipe: {rel} @ {sha[:10]}")
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (fixed https URL)
            data = resp.read()
        got = hashlib.sha256(data).hexdigest()
        if got != digest:
            raise RuntimeError(f"{rel}: sha256 {got} != pinned {digest}")
        dest = _mediapipe_path(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)


# ---------------------------------------------------------------------------------------------------------------
# Garment assets (MakeHuman community asset packs): single files out of remote zips, sha256-verified, idempotent.
# ---------------------------------------------------------------------------------------------------------------


class _HttpRangeFile:
    """Read-only seekable file over HTTP range requests (enough for zipfile to read single members)."""

    def __init__(self, url: str) -> None:
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (fixed https URL)
            self.size = int(resp.headers["Content-Length"])
            if resp.headers.get("Accept-Ranges", "").lower() != "bytes":
                raise RuntimeError(f"{url}: server does not support range requests")
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            data = resp.read()
        self.pos += len(data)
        return data


def garment_dir(gid: str) -> Path:
    return GARMENT_CACHE / gid


def _garment_names(gid: str) -> dict[str, str]:
    """File name -> sha256 for one template, including the pack json under the key `pack.json`."""
    a = GARMENT_ASSETS[gid]
    return {**a["files"], "pack.json": a["packJson"]}


def _garment_ok(gid: str) -> bool:
    d = garment_dir(gid)
    return all(
        (d / n).exists() and hashlib.sha256((d / n).read_bytes()).hexdigest() == h
        for n, h in _garment_names(gid).items()
    )


def fetch_garments() -> None:
    import zipfile

    zips: dict[str, zipfile.ZipFile] = {}
    for gid, a in GARMENT_ASSETS.items():
        if _garment_ok(gid):
            continue
        pack = a["pack"]
        if pack not in zips:
            name = pack.split("_")[0]
            print(f"[fetch] garments: opening {pack}.zip (range reads)")
            zips[pack] = zipfile.ZipFile(_HttpRangeFile(GARMENT_PACK_URL.format(name=name, pack=pack)))  # type: ignore[arg-type]
        zf = zips[pack]
        d = garment_dir(gid)
        d.mkdir(parents=True, exist_ok=True)
        for fname, digest in _garment_names(gid).items():
            member = f"packs/{pack.split('_')[0]}.json" if fname == "pack.json" else f"clothes/{a['dir']}/{fname}"
            data = zf.read(member)
            got = hashlib.sha256(data).hexdigest()
            if got != digest:
                raise RuntimeError(f"{gid}: {member} sha256 {got} != pinned {digest}")
            (d / fname).write_bytes(data)
        print(f"[fetch] garments: {gid} ({a['dir']}) ok")


def ensure_garments_present() -> None:
    for gid in GARMENT_ASSETS:
        if not _garment_ok(gid):
            raise RuntimeError(f"garment asset {gid} missing or not at the pinned sha256; run without --no-fetch")


# ---------------------------------------------------------------------------------------------------------------
# Body-part assets (MakeHuman system assets pack): same mechanism, members keyed by their path in the zip.
# ---------------------------------------------------------------------------------------------------------------


def part_dir(pid: str) -> Path:
    return PART_CACHE / pid


def _pack_dir(pack: str) -> str:
    """`makehuman_system_assets_cc0` -> `makehuman_system_assets` (the folder of the pack on the server)."""
    return pack.rsplit("_", 1)[0]


def _part_names(pid: str) -> dict[str, str]:
    """Cached file name -> sha256 of one part, including the pack json under the key `pack.json`."""
    a = PART_ASSETS[pid]
    return {**{Path(m).name: h for m, h in a["files"].items()}, "pack.json": a["packJson"]}


def _part_ok(pid: str) -> bool:
    d = part_dir(pid)
    return all(
        (d / n).exists() and hashlib.sha256((d / n).read_bytes()).hexdigest() == h
        for n, h in _part_names(pid).items()
    )


def fetch_parts() -> None:
    import zipfile

    zips: dict[str, zipfile.ZipFile] = {}
    for pid, a in PART_ASSETS.items():
        if _part_ok(pid):
            continue
        pack = a["pack"]
        if pack not in zips:
            print(f"[fetch] parts: opening {pack}.zip (range reads)")
            zips[pack] = zipfile.ZipFile(_HttpRangeFile(PART_PACK_URL.format(name=_pack_dir(pack), pack=pack)))  # type: ignore[arg-type]
        zf = zips[pack]
        d = part_dir(pid)
        d.mkdir(parents=True, exist_ok=True)
        wanted = {**a["files"], f"packs/{_pack_dir(pack)}.json": a["packJson"]}
        for member, digest in wanted.items():
            data = zf.read(member)
            got = hashlib.sha256(data).hexdigest()
            if got != digest:
                raise RuntimeError(f"{pid}: {member} sha256 {got} != pinned {digest}")
            (d / ("pack.json" if member.startswith("packs/") else Path(member).name)).write_bytes(data)
        print(f"[fetch] parts: {pid} ok")


def ensure_parts_present() -> None:
    for pid in PART_ASSETS:
        if not _part_ok(pid):
            raise RuntimeError(f"part asset {pid} missing or not at the pinned sha256; run without --no-fetch")


def fetch_all() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for key in SOURCES:
        fetch_source(key)
    fetch_mediapipe()
    fetch_garments()
    fetch_parts()


def ensure_present() -> None:
    """--no-fetch mode: verify the cache is complete and at the pinned commits."""
    for key, src in SOURCES.items():
        if not _complete(key, src["dir"]):
            raise RuntimeError(f"{key} data missing in {src['dir']}; run without --no-fetch")
        head = _head(src["dir"])
        if head != src["sha"]:
            print(f"[fetch] WARNING: {key} is at {head}, pinned {src['sha']}", file=sys.stderr)
    for rel in MEDIAPIPE["files"]:
        if not _mediapipe_ok(rel):
            raise RuntimeError(f"mediapipe file {rel} missing or not at the pinned sha256; run without --no-fetch")
    ensure_garments_present()
    ensure_parts_present()


if __name__ == "__main__":
    fetch_all()
