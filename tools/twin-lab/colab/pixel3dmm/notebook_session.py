"""Separate persistent software/model cache from every private photo session."""

from pathlib import Path
import gc
import json
import shutil
import tempfile

from diagnostics import print_diagnostics, print_warnings, step_context
from io_utils import cleanup_session, download_and_wait, upload_views
from setup_runtime import apply_runtime_patches, install, require_cache, run, runtime_environment


def private_config(path: Path, cache: Path, parent: Path = Path("/content")) -> bool:
    if not path.is_file():
        return False
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            try:
                values[key] = json.loads(value)
            except json.JSONDecodeError:
                return False
    if values.get("PIXEL3DMM_CODE_BASE") != str(cache / "pixel3dmm"):
        return False
    for key in ("PIXEL3DMM_PREPROCESSED_DATA", "PIXEL3DMM_TRACKING_OUTPUT"):
        value = values.get(key)
        if not isinstance(value, str):
            return False
        directory = Path(value)
        session = directory.parent
        if session.parent != parent or not session.name.startswith("dt-pixel3dmm-session-"):
            return False
    return True


def copy_flame_archives(cache: Path, selected: dict) -> None:
    from google.colab import drive

    record = cache / "selected_archives.json"
    previous = json.loads(record.read_text()) if record.is_file() else {}
    missing = {version: filename for version, filename in selected.items()
               if not (cache / ("FLAME" + version + ".zip")).is_file() or previous.get(version) != filename}
    if not missing:
        print("Reusing copied FLAME archives; no Drive mount needed.")
        return
    drive.mount("/content/drive")
    try:
        for version, filename in missing.items():
            # Preserve the lead's FUSE-safe check, relative paths and default flame/ folder.
            raw = str(filename).strip().strip('"').strip("'")
            model_zip = Path(raw) if raw.startswith("/") else Path("/content") / raw
            if not str(model_zip).startswith("/content/drive/") or ".." in model_zip.parts or model_zip.suffix.lower() != ".zip":
                raise ValueError(f"Select a FLAME .zip inside /content/drive (got {raw!r})")
            if not model_zip.is_file():
                raise FileNotFoundError("FLAME zip not found; check the parameter cell")
            # Read-only Drive usage: the only Drive file operation is this copy.
            destination = cache / ("FLAME" + version + ".zip")
            temporary = destination.with_suffix(".zip.partial")
            try:
                shutil.copyfile(model_zip, temporary)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            previous[version] = filename
            # A changed user model must be restaged even with a ready environment.
            stage_model(cache, version)
        record.write_text(json.dumps(previous), encoding="utf-8")
    finally:
        drive.flush_and_unmount()


def stage_model(cache: Path, version: str) -> None:
    from io_utils import stage_flame
    assets = cache / "pixel3dmm/src/pixel3dmm/preprocessing/MICA/data"
    if assets.is_dir():
        stage_flame(cache / ("FLAME" + version + ".zip"), assets, version)


def run_session(cache: Path, selected: dict, version: str, architecture: str, iters: int,
                global_iters: int, max_fit_batch_size: int, prepare: bool = False) -> None:
    from google.colab import files

    if version not in {"2020", "2023"} or any(not isinstance(value, int) or value < 1
                                              for value in (iters, global_iters, max_fit_batch_size)):
        raise ValueError("Invalid model variant or iteration/batch parameters")

    session = Path(tempfile.mkdtemp(prefix="dt-pixel3dmm-session-", dir="/content"))
    uploaded = {}
    config_created = False
    env_file = Path.home() / ".config/pixel3dmm/.env"
    try:
        step_context(session, "installation" if prepare else "cache_validation")
        if prepare:
            copy_flame_archives(cache, selected)
            python, worker_env = install(cache, architecture, session)
        else:
            require_cache(cache, architecture, version)
            apply_runtime_patches(cache)
            python = cache / "env/bin/python"
            worker_env = runtime_environment(cache, architecture, session)
            print("Fit-only retry: reusing the installed VM cache and copied FLAME files.")
        require_cache(cache, architecture, version)
        if env_file.exists():
            if private_config(env_file, cache):
                env_file.unlink()
            else:
                raise RuntimeError("Unrelated Pixel3DMM config found; use a fresh runtime")
        env_file.parent.mkdir(parents=True, exist_ok=True)
        paths = {"PIXEL3DMM_CODE_BASE": str(cache / "pixel3dmm"),
                 "PIXEL3DMM_PREPROCESSED_DATA": str(session / "preprocessed"),
                 "PIXEL3DMM_TRACKING_OUTPUT": str(session / "tracking")}
        env_file.write_text("".join(key + "=" + json.dumps(value) + "\n" for key, value in paths.items()), encoding="utf-8")
        config_created = True
        worker_env.update(paths)
        (session / "config.json").write_text(json.dumps({
            "flame_version": version, "iters": iters, "global_iters": global_iters,
            "max_fit_batch_size": max_fit_batch_size,
        }), encoding="utf-8")
        uploads = session / "uploads"
        uploads.mkdir()
        step_context(session, "photo_upload")
        print("Upload front and optional left/right/back head photos together.")
        uploaded = files.upload(target_dir=str(uploads))
        upload_views(uploaded.keys())
        uploaded.clear()
        print("Preprocessing each view, then fitting shared identity with independent cameras.")
        run([python, cache / "helpers/worker.py", session], session,
            cwd=cache / "pixel3dmm", env=worker_env, step="worker")
        print_warnings(session)
        step_context(session, "archive_download")
        download_and_wait(session / "head_fit.zip")
        print("head_fit.zip transferred to your browser. Save it to user-data/twin/head/flame/.")
    except BaseException:
        # These diagnostics must run while per-step logs still exist.
        print_warnings(session)
        print_diagnostics(session)
        raise
    finally:
        uploaded.clear()
        try:
            if config_created:
                env_file.unlink(missing_ok=True)
        finally:
            cleanup_session(session)
            print("Privacy cleanup: uploads, derived outputs, logs and archive deleted; VM model cache retained for retry.")


def cleanup_everything(cache: Path, parent: Path = Path("/content"), config_file=None) -> None:
    if cache.resolve().parent != parent.resolve() or cache.name != "dt-pixel3dmm-cache" or cache.is_symlink():
        raise ValueError("Refusing unsafe cache cleanup")
    for session in parent.glob("dt-pixel3dmm-session-*"):
        cleanup_session(session, parent)
    config = Path.home() / ".config/pixel3dmm/.env" if config_file is None else config_file
    if private_config(config, cache, parent):
        config.unlink()
    if cache.exists():
        shutil.rmtree(cache)
    gc.collect()
    print("Final privacy cleanup: all session files, installed environment, sources, weights and FLAME removed.")
