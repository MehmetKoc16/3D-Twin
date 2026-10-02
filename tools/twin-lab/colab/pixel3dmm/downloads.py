"""Official model sources only; resumable installation uses atomic file copies."""

from pathlib import Path
import shutil
import subprocess
import http.client
import time
import urllib.request

HF_REVISION = "98608f33a2d8af6eed387ac354db6af2030a8f44"
HF_BASE = "https://huggingface.co/SimonGiebenhain/Pixel3dmm/resolve/" + HF_REVISION + "/"
INSIGHTFACE_BASE = "https://github.com/deepinsight/insightface/releases/download/v0.7/"
DRIVE_WEIGHT_FOLDER = "/content/drive/MyDrive/flame/weights"


def drive_url(identifier: str) -> str:
    return "https://drive.google.com/file/d/" + identifier + "/view"


# Each pair is (transport, official URL). No tokens or third-party mirrors.
WEIGHTS = {
    "uv.ckpt": {"minimum": 100_000_000, "sources": [
        ("https", HF_BASE + "uv.ckpt"), ("gdown", drive_url("1SDV_8_qWTe__rX_8e4Fi-BE3aES0YzJY"))]},
    "normals.ckpt": {"minimum": 100_000_000, "sources": [
        ("https", HF_BASE + "normals.ckpt"), ("gdown", drive_url("1KYYlpN-KGrYMVcAOT22NkVQC0UAfycMD"))]},
    "antelopev2.zip": {"minimum": 1_000_000, "sources": [
        ("https", INSIGHTFACE_BASE + "antelopev2.zip"), ("gdown", drive_url("16PWKI_RjjbE4_kqpElG-YFqe8FpXjads"))]},
    "buffalo_l.zip": {"minimum": 1_000_000, "sources": [
        ("https", INSIGHTFACE_BASE + "buffalo_l.zip"), ("gdown", drive_url("1navJMy0DTr1_DHjLWu1i48owCPvXWfYc"))]},
    "mica.tar": {"minimum": 1_000_000, "sources": [
        ("gdown", drive_url("1bYsI_spptzyuFmfLYqYkcJA6GZWZViNt"))]},
    "epoch59.pth": {"minimum": 1_000_000, "sources": [
        ("gdown", drive_url("1nVkaSbxy3NeqblwMTGvLg4nF49cI_99C"))]},
}


def plausible(path: Path, minimum: int) -> bool:
    if not path.is_file() or path.stat().st_size <= minimum:
        return False
    with path.open("rb") as stream:
        prefix = stream.read(512).lstrip().lower()
    return not prefix.startswith((b"<!doctype html", b"<html", b"<?xml", b"{\"error"))


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def copy_drive_weights(cache: Path, folder: Path = Path(DRIVE_WEIGHT_FOLDER)) -> list:
    """Called only while Drive is mounted; reads six fixed weight filenames."""
    copied = []
    for name, spec in WEIGHTS.items():
        source = folder / name
        if source.is_file():
            if not plausible(source, spec["minimum"]):
                raise ValueError("Drive weight is too small or an HTML error page: " + name)
            destination = cache / "drive_weights" / name
            if not plausible(destination, spec["minimum"]):
                atomic_copy(source, destination)
            copied.append(name)
    return copied


def fetch_https(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "dt-pixel3dmm-colab"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)


def download_weight(name: str, destination: Path, cache: Path, fetchers: dict,
                    log, attempts: int = 3, sleep=time.sleep) -> None:
    """Prefer copied Drive weights, then cached files, then ordered network sources.

    fetchers map https/gdown to (URL, temporary_path) functions. A sub-step is
    marked done only after a plausible file replaces the destination atomically.
    """
    spec = WEIGHTS[name]
    local = cache / "drive_weights" / name
    if plausible(local, spec["minimum"]):
        if not plausible(destination, spec["minimum"]):
            atomic_copy(local, destination)
        log("Using user-supplied Drive copy: " + name)
        return
    if plausible(destination, spec["minimum"]):
        log("Reusing completed download: " + name)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    for transport, url in spec["sources"]:
        for attempt in range(1, attempts + 1):
            try:
                temporary.unlink(missing_ok=True)
                log(f"Downloading {name}: {transport}, attempt {attempt}/{attempts}; {url}")
                fetchers[transport](url, temporary)
                if not plausible(temporary, spec["minimum"]):
                    raise ValueError("Downloaded file is too small or an HTML error page")
                temporary.replace(destination)
                return
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, http.client.HTTPException) as error:
                log(f"{name} download attempt failed: {type(error).__name__}: {error}")
            finally:
                temporary.unlink(missing_ok=True)
            if attempt < attempts:
                sleep(min(2 ** (attempt - 1), 8))
    official = spec["sources"][0][1]
    message = (f"Missing weight: {name}. Official source: {official}. "
               f"Put it in MyDrive/flame/weights/{name}, then re-run install-and-fit.")
    log(message)
    raise RuntimeError(message)
