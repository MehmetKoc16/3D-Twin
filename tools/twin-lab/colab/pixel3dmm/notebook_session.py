"""Separate persistent software/model cache from every private photo session."""

from pathlib import Path
import gc
import json
import shutil
import tempfile
import traceback

from diagnostics import print_diagnostics, print_warnings, step_context
from io_utils import cleanup_session, download_and_wait, upload_views
from setup_runtime import apply_runtime_patches, install, require_cache, run, runtime_environment
from downloads import copy_drive_weights


def result_dir(parent: Path) -> Path:
    return parent / "dt-pixel3dmm-result"


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
    # An unfinished install mounts again so the user can add a missing weight
    # without recopied FLAME or a fresh runtime. Ready-cache fits need no mount.
    if not missing and (cache / "install_complete.json").is_file():
        print("Reusing copied FLAME archives; no Drive mount needed.")
        return
    drive.mount("/content/drive")
    try:
        copy_drive_weights(cache)
        for version, filename in missing.items():
            # Preserve the lead's FUSE-safe check, relative paths and default flame/ folder.
            raw = str(filename).strip().strip('"').strip("'")
            model_zip = Path(raw) if raw.startswith("/") else Path("/content") / raw
            if not str(model_zip).startswith("/content/drive/") or ".." in model_zip.parts or model_zip.suffix.lower() != ".zip":
                raise ValueError(f"Select a FLAME .zip inside /content/drive (got {raw!r})")
            if not model_zip.is_file():
                raise FileNotFoundError("FLAME zip not found; check the parameter cell")
            # Read-only Drive usage: copy model archives/optional weights only.
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
    # Explicit boundary: no general log tail once installation/cache checks end.
    installation_finished = not prepare
    env_file = Path.home() / ".config/pixel3dmm/.env"
    try:
        step_context(session, "install: copy Drive model files" if prepare else "cache_validation",
                     log_name="install-drive-copy-all.log" if prepare else None)
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
        installation_finished = True
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
        # Browsers may silently block files.download(); keep a copy outside the
        # session until the final privacy cell so the fit is never lost.
        kept = result_dir(session.parent) / "head_fit.zip"
        kept.parent.mkdir(exist_ok=True)
        shutil.copyfile(session / "head_fit.zip", kept)
        print(f"Result kept at {kept} until the final privacy cell. If no browser download appears, "
              "open the Files panel, refresh, and use the file's menu > Download.")
        download_and_wait(session / "head_fit.zip")
        print("head_fit.zip transferred to your browser. Save it to user-data/twin/head/flame/.")
    except BaseException:
        # These diagnostics must run while per-step logs still exist.
        if not installation_finished:
            context = json.loads((session / "current_step.json").read_text())
            name = context.get("log")
            if isinstance(name, str) and name.startswith("install-") and Path(name).name == name:
                (session / "logs").mkdir(exist_ok=True)
                with (session / "logs" / name).open("a", encoding="utf-8") as stream:
                    traceback.print_exc(file=stream)
        print_warnings(session)
        print_diagnostics(session, allow_install_logs=not installation_finished)
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
    kept = result_dir(parent)
    if kept.is_dir() and not kept.is_symlink():
        shutil.rmtree(kept)
    config = Path.home() / ".config/pixel3dmm/.env" if config_file is None else config_file
    if private_config(config, cache, parent):
        config.unlink()
    if cache.exists():
        shutil.rmtree(cache)
    gc.collect()
    print("Final privacy cleanup: all session files, installed environment, sources, weights and FLAME removed.")
