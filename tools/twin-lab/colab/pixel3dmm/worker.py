"""User-operated Colab multi-photo fit. Never run on personal data during QA."""

from pathlib import Path
import json
import os
import shutil
import sys

from io_utils import upload_views
from setup_runtime import run


def preprocess(root: Path) -> list:
    import numpy as np
    from PIL import Image, ImageOps
    from insightface.app import FaceAnalysis

    source = root / "pixel3dmm"
    data = root / "preprocessed"
    merged = data / "head"
    views = upload_views(path.name for path in (root / "uploads").iterdir())
    included, skipped = [], []
    # Only back may be skipped. Detection/preprocessing failure for a required
    # view stops the run; it must never silently create a zero-landmark input.
    detector = FaceAnalysis(name="antelopev2", root=os.environ["DT_INSIGHTFACE_ROOT"],
                            allowed_modules=["detection"], providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=-1, det_size=(512, 512))
    for view in ("front", "left", "right", "back"):
        if view not in views:
            skipped.append({"view": view, "reason": "not_uploaded"})
            continue
        input_dir = root / "inputs" / view
        input_dir.mkdir(parents=True)
        with Image.open(root / "uploads" / views[view]) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            original_size = list(normalized.size)
            normalized.save(input_dir / "input.png")
            # Insightface receives BGR, matching its documented API.
            if view == "back" and len(detector.get(np.asarray(normalized)[:, :, ::-1].copy())) == 0:
                skipped.append({"view": view, "reason": "no_face_detected"})
                continue
        # run_preprocessing.py uses unchecked os.system. Execute its three
        # component scripts with checked subprocesses, independently per view.
        run([sys.executable, source / "scripts/run_cropping.py", "--video_or_images_path", input_dir], root, cwd=source)
        folder = data / view
        crop = folder / "cropped/00000.jpg"
        landmark = folder / "PIPnet_landmarks/00000.npy"
        bounds = folder / "crop_ymin_ymax_xmin_xmax.npy"
        for path in (crop, landmark, bounds):
            if not path.is_file():
                raise RuntimeError(f"Cropping/landmarks incomplete for {view}")
        lm = np.load(landmark, allow_pickle=False)
        if lm.shape != (98, 2) or not np.isfinite(lm).all() or not lm.any():
            raise RuntimeError(f"Invalid facial landmarks for {view}")
        run([sys.executable, "demo.py", "-video_name", view, "-a", folder / "arcface"], root,
            cwd=source / "src/pixel3dmm/preprocessing/MICA")
        identity = folder / "mica/00000/identity.npy"
        if not identity.is_file():
            raise RuntimeError(f"MICA face detection failed for {view}; use a less extreme profile")
        run([sys.executable, source / "scripts/run_facer_segmentation.py", "--video_name", view], root, cwd=source)
        segment = folder / "seg_og/00000.png"
        if not segment.is_file():
            raise RuntimeError(f"Face segmentation incomplete for {view}")
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
                         "cropBoundsYminYmaxXminXmax": np.load(bounds, allow_pickle=False).tolist()})
    if len(included) < 3:
        raise RuntimeError("Front and both profiles must preprocess successfully")
    (root / "view_map.json").write_text(json.dumps({"included": included, "skipped": skipped}, indent=2))
    for prediction in ("normals", "uv_map"):
        run([sys.executable, source / "scripts/network_inference.py",
             "model.prediction_type=" + prediction, "video_name=head", "viz_uv_mesh=False"], root, cwd=source)
        for item in included:
            path = merged / f"p3dmm/{prediction}/{item['frame']:05d}.png"
            if not path.is_file():
                raise RuntimeError("Pixel3DMM prediction missing; upstream may have caught an inference error")
    return included


def fit(root: Path, included: list, config: dict) -> None:
    import torch
    from omegaconf import OmegaConf
    from pixel3dmm.tracking.tracker import Tracker

    source = root / "pixel3dmm"
    overrides = {
        "video_name": "head", "num_views": 1, "batch_size": len(included),
        "iters": config["iters"], "global_iters": config["global_iters"],
        "is_discontinuous": True, "global_camera": False, "include_neck": False,
        "use_flame2023": config["flame_version"] == "2023",
        "ignore_mica": config["flame_version"] == "2023",
        "w_exp": 0.1, "use_mouth_lmk": False, "w_shape": 0.01, "w_shape_general": 0.001,
        "normal_super": 2000.0, "sil_super": 1000.0,
        "save_meshes": True, "save_landmarks": False, "delete_preprocessing": False,
        "size": 256, "image_size": [256, 256],
    }
    cfg = OmegaConf.merge(OmegaConf.load(source / "configs/tracking.yaml"), overrides)
    tracker = Tracker(cfg)
    tracker.run()
    # Explicitly save every view AFTER joint optimization. canonical.ply saved
    # during online initialization is stale and must not be used as neutral export.
    for item in included:
        index = item["frame"]
        tracker.save_checkpoint(index, selected_frames=torch.tensor([index], device="cuda"))
    (root / "fit_path.txt").write_text(str(Path(tracker.output_folder)))
    (root / "fit_config.json").write_text(json.dumps(OmegaConf.to_container(cfg, resolve=True)))
    tracker.writer.close()


def main(root: Path) -> None:
    os.environ["DT_WORKER_ACTIVE"] = "1"
    config = json.loads((root / "config.json").read_text())
    included = preprocess(root)
    fit(root, included, config)
    from export_fit import export_fit
    export_fit(root, config)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
