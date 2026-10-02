"""Original archive/privacy helpers; no third-party model code or data embedded."""

from pathlib import Path
import gc
import shutil
import stat
import zipfile


def safe_extract(archive: Path, destination: Path) -> None:
    """Reject traversal and symlinks before extracting a user-supplied model zip."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination) or "\\" in item.filename:
                raise ValueError("Unsafe zip member path")
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Zip symlinks are not allowed")
        bundle.extractall(destination)


def stage_flame(archive: Path, assets: Path, version: str) -> None:
    if version not in {"2020", "2023"}:
        raise ValueError("FLAME_VERSION must be 2020 or 2023")
    unpacked = assets / ("unpacked-" + version)
    safe_extract(archive, unpacked)
    filename = "generic_model.pkl" if version == "2020" else "flame2023_no_jaw.pkl"
    candidates = list(unpacked.rglob(filename))
    if len(candidates) != 1:
        raise ValueError(f"FLAME {version} zip must contain exactly one {filename}")
    target = assets / ("FLAME" + version)
    target.mkdir(parents=True, exist_ok=True)
    # Copy the model's sibling files, preserving upstream-supplied landmark assets.
    shutil.copytree(candidates[0].parent, target, dirs_exist_ok=True)
    shutil.rmtree(unpacked)
    if version == "2020":
        required = ["generic_model.pkl", "landmark_embedding.npy", "FLAME_masks/FLAME_masks.pkl"]
        missing = [name for name in required if not (target / name).is_file()]
        if missing:
            raise ValueError("FLAME 2020 model/landmark/mask assets missing: " + ", ".join(missing))


def upload_views(names) -> dict:
    result = {}
    for name in names:
        path = Path(name)
        view = path.stem.lower()
        if view not in {"front", "left", "right", "back"} or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("Use front/left/right/back with .png, .jpg or .jpeg extensions")
        if view in result:
            raise ValueError("Only one image per view is allowed")
        result[view] = path.name
    if not {"front", "left", "right"}.issubset(result):
        raise ValueError("Front, left profile and right profile are required; back is optional")
    return result


def cleanup_session(path: Path, parent: Path = Path("/content")) -> None:
    resolved = path.resolve()
    if resolved.parent != parent.resolve() or not resolved.name.startswith("dt-pixel3dmm-session-"):
        raise ValueError("Refusing cleanup outside the private Pixel3DMM session")
    if resolved.exists():
        shutil.rmtree(resolved)
    gc.collect()


def download_and_wait(path: Path) -> None:
    """Same completion convention as the TRELLIS notebook: await browser memory."""
    import IPython.display
    from google.colab import files, output
    from unittest.mock import patch

    original = IPython.display.display
    instrumented = False

    def display_with_completion(*objects, **kwargs):
        nonlocal instrumented
        changed = []
        for item in objects:
            if isinstance(item, IPython.display.Javascript) and item.data.strip().startswith("download("):
                item = IPython.display.Javascript("window.dtHeadDownload = " + item.data)
                instrumented = True
            changed.append(item)
        return original(*changed, **kwargs)

    with patch.object(IPython.display, "display", display_with_completion):
        files.download(str(path))
    if not instrumented:
        raise RuntimeError("Colab download API changed; completion wrapper needs updating")
    output.eval_js("window.dtHeadDownload.then(() => true)", timeout_sec=600)
