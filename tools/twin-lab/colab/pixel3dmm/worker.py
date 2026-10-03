"""User-operated Colab multi-photo fit. Never run on personal data during QA."""

from pathlib import Path
import json
import os
import re
import shutil
import sys
import subprocess
from time import perf_counter

from io_utils import upload_views, VIEW_ORDER, OPTIONAL_VIEWS
from fit_policy import fit_budget
from setup_runtime import run
from diagnostics import step_context


class DetectionError(RuntimeError):
    """A view has no usable face/landmarks; optional views may be omitted."""


def is_detection_failure(log: Path) -> bool:
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    exceptions = [line.strip() for line in text.splitlines()
                  if re.match(r"^[\w.]+(?:Error|Exception):", line.strip())]
    # A prior detection warning must never mask a later CUDA/import exception.
    terminal = exceptions[-1] if exceptions else ""
    return any(message in terminal for message in (
        "IndexError: list index out of range", "ValueError: need at least one array to stack",
        "Found face with too low detections confidence", "Face not detected",
    ))


def skip_optional(root: Path, view: str, reason: str) -> dict:
    context = json.loads((root / "current_step.json").read_text())
    if view == "front":
        raise DetectionError("Front face required; retry with a clear frontal photo")
    if view not in OPTIONAL_VIEWS:
        raise ValueError("Unknown optional view")
    item = {"view": view, "reason": reason, "step": context["step"]}
    path = root / "warnings.json"
    warnings = json.loads(path.read_text()) if path.is_file() else []
    warnings.append(item)
    path.write_text(json.dumps(warnings), encoding="utf-8")
    return item


def run_step(root: Path, script: Path, args: list, step: str, view: str = "all", cwd=None) -> None:
    step_context(root, step, view)
    wrapper = Path(__file__).parent / "runtime_compat.py"
    env = os.environ.copy()
    env["DT_PROFILE_CROP_RETRY"] = "1" if step == "cropping_landmarks_retry" else "0"
    try:
        run([sys.executable, wrapper, script, *args], root, cwd=cwd, step=step, view=view, env=env)
    except subprocess.CalledProcessError as error:
        if step in {"cropping_landmarks", "cropping_landmarks_retry"} and is_detection_failure(root / f"logs/{step}-{view}.log"):
            raise DetectionError("No usable crop or facial landmarks") from error
        raise


def crop_failure(folder: Path) -> str:
    import numpy as np

    crop = folder / "cropped/00000.jpg"
    landmark = folder / "PIPnet_landmarks/00000.npy"
    bounds = folder / "crop_ymin_ymax_xmin_xmax.npy"
    if not all(path.is_file() for path in (crop, landmark, bounds)):
        return "crop_or_landmarks_missing"
    try:
        lm = np.load(landmark, allow_pickle=False)
        box = np.load(bounds, allow_pickle=False)
    except (OSError, ValueError, EOFError):
        return "invalid_landmark_or_crop_file"
    if lm.shape != (98, 2) or not np.isfinite(lm).all() or not lm.any():
        return "invalid_facial_landmarks"
    if box.shape != (4,) or not np.isfinite(box).all() or box[1] <= box[0] or box[3] <= box[2]:
        return "invalid_crop_bounds"
    return ""


def crop_view(root: Path, source: Path, input_dir: Path, view: str) -> str:
    """One relaxed confidence retry; errors unrelated to detection still abort."""
    if view not in VIEW_ORDER:
        raise ValueError("Unknown crop view")
    folder = root / "preprocessed" / view
    for attempt, step in enumerate(("cropping_landmarks", "cropping_landmarks_retry")):
        try:
            run_step(root, source / "scripts/run_cropping.py", ["--video_or_images_path", input_dir],
                     step, view, cwd=source)
            reason = crop_failure(folder)
        except DetectionError:
            reason = "no_usable_crop_or_landmarks"
        if not reason:
            if attempt:
                path = root / "warnings.json"
                warnings = json.loads(path.read_text()) if path.is_file() else []
                warnings.append({"view": view, "step": step, "reason": "relaxed_landmark_confidence",
                                 "action": "used_relaxed_crop"})
                path.write_text(json.dumps(warnings))
            return "faceboxes_landmark_0.75_retry" if attempt else "faceboxes_landmark_0.99"
        if attempt:
            raise DetectionError(reason)
        # run_cropping.py skips a populated cropped/ folder. Remove generated
        # artifacts before retry so it also recomputes missing/invalid landmarks.
        if folder.exists():
            shutil.rmtree(folder)


def preprocess(root: Path) -> list:
    step_context(root, "detector_initialization")
    import numpy as np
    from PIL import Image, ImageOps
    from insightface.app import FaceAnalysis

    source = Path(os.environ["DT_CACHE_ROOT"]) / "pixel3dmm"
    data = root / "preprocessed"
    merged = data / "head"
    views = upload_views(path.name for path in (root / "uploads").iterdir())
    included, skipped = [], []
    detector = FaceAnalysis(name="antelopev2", root=os.environ["DT_INSIGHTFACE_ROOT"],
                            allowed_modules=["detection"], providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=-1, det_size=(512, 512))
    for view in VIEW_ORDER:
        if view not in views:
            skipped.append({"view": view, "reason": "not_uploaded"})
            continue
        step_context(root, "face_detection", view)
        input_dir = root / "inputs" / view
        input_dir.mkdir(parents=True)
        with Image.open(root / "uploads" / views[view]) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            original_size = list(normalized.size)
            normalized.save(input_dir / "input.png")
            # Insightface receives BGR, matching its documented API.
            if len(detector.get(np.asarray(normalized)[:, :, ::-1].copy())) == 0:
                skipped.append(skip_optional(root, view, "no_face_detected"))
                continue
        # run_preprocessing.py uses unchecked os.system. Execute its three
        # component scripts with checked subprocesses, independently per view.
        try:
            crop_method = crop_view(root, source, input_dir, view)
        except DetectionError as error:
            skipped.append(skip_optional(root, view, str(error)))
            continue
        folder = data / view
        crop = folder / "cropped/00000.jpg"
        landmark = folder / "PIPnet_landmarks/00000.npy"
        bounds = folder / "crop_ymin_ymax_xmin_xmax.npy"
        run_step(root, source / "src/pixel3dmm/preprocessing/MICA/demo.py",
                 ["-video_name", view, "-a", folder / "arcface"], "mica", view,
                 cwd=source / "src/pixel3dmm/preprocessing/MICA")
        identity = folder / "mica/00000/identity.npy"
        if not identity.is_file():
            skipped.append(skip_optional(root, view, "mica_face_output_missing"))
            continue
        run_step(root, source / "scripts/run_facer_segmentation.py", ["--video_name", view],
                 "segmentation", view, cwd=source)
        segment = folder / "seg_og/00000.png"
        if not segment.is_file():
            skipped.append(skip_optional(root, view, "face_segmentation_output_missing"))
            continue
        index = len(included)
        for path, destination in [
            (crop, merged / f"cropped/{index:05d}.jpg"),
            (landmark, merged / f"PIPnet_landmarks/{index:05d}.npy"),
            (segment, merged / f"seg_og/{index:05d}.png"),
        ]:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        shutil.copytree(identity.parent, merged / f"mica/{index:05d}")
        included.append({"view": view, "frame": index, "originalSizeWH": original_size,
                         "cropMethod": crop_method,
                         "cropBoundsYminYmaxXminXmax": np.load(bounds, allow_pickle=False).tolist()})
    if not any(item["view"] == "front" for item in included):
        raise DetectionError("Front must preprocess successfully")
    (root / "view_map.json").write_text(json.dumps({"included": included, "skipped": skipped}, indent=2))
    for prediction in ("normals", "uv_map"):
        run_step(root, source / "scripts/network_inference.py",
                 ["model.prediction_type=" + prediction, "video_name=head", "viz_uv_mesh=False"],
                 prediction + "_prediction", cwd=source)
        for item in included:
            path = merged / f"p3dmm/{prediction}/{item['frame']:05d}.png"
            if not path.is_file():
                step_context(root, prediction + "_prediction", item["view"])
                raise RuntimeError("Pixel3DMM prediction missing; upstream may have caught an inference error")
    return included


def fit(root: Path, included: list, config: dict) -> None:
    step_context(root, "tracking")
    import torch
    from omegaconf import OmegaConf
    from pixel3dmm.tracking.tracker import Tracker

    source = Path(os.environ["DT_CACHE_ROOT"]) / "pixel3dmm"
    budget = fit_budget(len(included), config["iters"], config["global_iters"], config["max_fit_batch_size"])
    overrides = {
        "video_name": "head", "num_views": 1, "batch_size": budget["batchSize"],
        "iters": config["iters"], "global_iters": budget["jointIters"],
        "is_discontinuous": True, "global_camera": False, "include_neck": False,
        "use_flame2023": config["flame_version"] == "2023",
        "ignore_mica": config["flame_version"] == "2023",
        "w_exp": 0.1, "use_mouth_lmk": False, "w_shape": 0.01, "w_shape_general": 0.001,
        "normal_super": 2000.0, "sil_super": 1000.0,
        "save_meshes": True, "save_landmarks": False, "delete_preprocessing": False,
        "size": 256, "image_size": [256, 256],
    }
    cfg = OmegaConf.merge(OmegaConf.load(source / "configs/tracking.yaml"), overrides)
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    tracker = Tracker(cfg)
    optimize = tracker.optimize_color
    timings = {"budget": budget, "online": [], "jointSeconds": 0.0}

    def timed_optimize(*args, **kwargs):
        joint = kwargs.get("is_joint", False)
        view = "all" if joint else included[len(timings["online"])]["view"]
        step_context(root, "tracking_joint" if joint else "tracking_online", view)
        torch.cuda.synchronize()
        phase_start = perf_counter()
        result = optimize(*args, **kwargs)
        torch.cuda.synchronize()
        elapsed = perf_counter() - phase_start
        if joint:
            timings["jointSeconds"] = elapsed
        else:
            timings["online"].append({"view": view, "seconds": elapsed})
        return result

    tracker.optimize_color = timed_optimize
    tracker.run()
    # Explicitly save every view AFTER joint optimization. canonical.ply saved
    # during online initialization is stale and must not be used as neutral export.
    # Upstream only calls save_checkpoint under no_grad; it calls .numpy() on graph tensors.
    for item in included:
        index = item["frame"]
        with torch.no_grad(): tracker.save_checkpoint(index, selected_frames=torch.tensor([index], device="cuda"))
    (root / "fit_path.txt").write_text(str(Path(tracker.output_folder)))
    (root / "fit_config.json").write_text(json.dumps(OmegaConf.to_container(cfg, resolve=True)))
    timings.update({"trackingSeconds": perf_counter() - started,
                    "torchPeakAllocatedMiB": torch.cuda.max_memory_allocated() / 1024 ** 2,
                    "torchPeakReservedMiB": torch.cuda.max_memory_reserved() / 1024 ** 2,
                    "memoryNote": "PyTorch counters exclude some native CUDA/driver allocations"})
    (root / "fit_runtime.json").write_text(json.dumps(timings))
    tracker.writer.close()


def main(root: Path) -> None:
    os.environ["DT_WORKER_ACTIVE"] = "1"
    step_context(root, "runtime_configuration")
    from runtime_compat import configure
    configure()
    import torch
    with torch.autocast(device_type="cuda", enabled=False):
        config = json.loads((root / "config.json").read_text())
        started = perf_counter()
        included = preprocess(root)
        preprocessing_seconds = perf_counter() - started
        fit(root, included, config)
        runtime_path = root / "fit_runtime.json"
        runtime = json.loads(runtime_path.read_text())
        runtime["preprocessingSeconds"] = preprocessing_seconds
        runtime_path.write_text(json.dumps(runtime))
        step_context(root, "export")
        from export_fit import export_fit
        export_fit(root, config)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
