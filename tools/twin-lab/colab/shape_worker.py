"""Isolated Colab worker; embedded into the notebook by build_notebook.py.

Uses the pinned TRELLIS.2 shape substeps, without BRIA or NVIDIA renderers.
User data and all derived files are confined to the session directory.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import trimesh
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from rembg import new_session, remove

import meshops
from check_meta import validate_pair

MODEL_REPO = "microsoft/TRELLIS.2-4B"
MODEL_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"


def load_pipeline(config: dict):
    from trellis2 import models
    from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
    from trellis2.pipelines import Trellis2ImageTo3DPipeline, samplers

    path = hf_hub_download(MODEL_REPO, "pipeline.json", revision=MODEL_REVISION)
    arguments = json.loads(Path(path).read_text())["args"]
    names = ["sparse_structure_flow_model", "sparse_structure_decoder", "shape_slat_decoder", "shape_slat_flow_model_512"]
    if config["pipeline_type"] != "512":
        names.append("shape_slat_flow_model_1024")
    loaded = {}
    for name in names:
        checkpoint = arguments["models"][name]
        if checkpoint.startswith("microsoft/"):
            # The upstream sparse decoder belongs to the original MIT TRELLIS model.
            repo, model = "/".join(checkpoint.split("/")[:2]), "/".join(checkpoint.split("/")[2:])
            revision = config["sparse_decoder_revision"]
        else:
            repo, model, revision = MODEL_REPO, checkpoint, MODEL_REVISION
        local = None
        for extension in (".json", ".safetensors"):
            local = hf_hub_download(repo, model + extension, revision=revision)
        loaded[name] = models.from_pretrained(str(Path(local).with_suffix("")))
    sparse_spec = arguments["sparse_structure_sampler"]
    shape_spec = arguments["shape_slat_sampler"]
    # Construct directly: upstream from_pretrained also creates BRIA RMBG-2.0.
    pipeline = Trellis2ImageTo3DPipeline(
        models=loaded,
        sparse_structure_sampler=getattr(samplers, sparse_spec["name"])(**sparse_spec["args"]),
        shape_slat_sampler=getattr(samplers, shape_spec["name"])(**shape_spec["args"]),
        sparse_structure_sampler_params=sparse_spec["params"],
        shape_slat_sampler_params=shape_spec["params"],
        shape_slat_normalization=arguments["shape_slat_normalization"],
        image_cond_model=DinoV3FeatureExtractor(model_name=snapshot_download(
            arguments["image_cond_model"]["args"]["model_name"], revision=config["dino_revision"],
            allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"],
        )),
        rembg_model=None,
        low_vram=True,
        default_pipeline_type=config["pipeline_type"],
    )
    pipeline.cuda()
    return pipeline


def prepare_cutouts(session_dir: Path) -> tuple[dict, dict, dict]:
    session = new_session("u2net_human_seg", providers=["CPUExecutionProvider"])
    cutouts, masks, inputs = {}, {}, {}
    for view in meshops.VIEW_YAW_DEG:
        path = session_dir / "uploads" / f"{view}.png"
        if not path.exists():
            continue
        with Image.open(path) as source:
            source.load()
            cutout = remove(source.convert("RGB"), session=session).convert("RGBA")
        alpha = np.asarray(cutout)[:, :, 3]
        mask = (alpha > 127).astype(np.uint8) * 255
        if not mask.any() or not (alpha > 204).any():
            raise ValueError(f"No foreground found in {view}; use a clear full-body image.")
        out = session_dir / "out"
        cutout.save(out / f"cutout_{view}.png")
        Image.fromarray(mask).save(out / f"mask_{view}.png")
        cutouts[view], masks[view] = cutout, mask
        inputs[view] = {
            "file": f"{view}.png",
            "backgroundRemoval": {"library": "rembg", "model": "u2net_human_seg", "threshold": 127},
            "cutout": f"cutout_{view}.png",
            "mask": f"mask_{view}.png",
            "modelInput": None,
        }
    return cutouts, masks, inputs


def infer_shape(pipeline, image: Image.Image, config: dict):
    # The RGBA image already has alpha; preprocess_image skips its rembg_model.
    prepared = pipeline.preprocess_image(image)
    torch.manual_seed(config["seed"])
    parameters = {"steps": config["steps"]}
    with torch.inference_mode():
        condition_512 = pipeline.get_cond([prepared], 512)
        coords = pipeline.sample_sparse_structure(condition_512, 32, 1, parameters)
        if config["pipeline_type"] == "512":
            latent = pipeline.sample_shape_slat(condition_512, pipeline.models["shape_slat_flow_model_512"], coords, parameters)
            resolution = 512
        else:
            condition_1024 = pipeline.get_cond([prepared], 1024)
            latent, resolution = pipeline.sample_shape_slat_cascade(
                condition_512, condition_1024,
                pipeline.models["shape_slat_flow_model_512"], pipeline.models["shape_slat_flow_model_1024"],
                512, 1024, coords, parameters, max_num_tokens=config["max_num_tokens"],
            )
        generated, _ = pipeline.decode_shape_slat(latent, resolution)
    return generated[0], prepared.size, resolution


def detection_for(vertices: np.ndarray, config: dict) -> dict:
    detection = meshops.detect_frame(vertices)
    for label, parameter in (("up", "source_up"), ("forward", "source_forward")):
        axis = config[parameter]
        if axis != "auto":
            detection[f"{label}Axis"] = "XYZ".index(axis[1])
            detection[f"{label}Sign"] = 1 if axis[0] == "+" else -1
    if detection["upAxis"] == detection["forwardAxis"]:
        raise ValueError("Source up and forward must be different axes; set both overrides.")
    if config["flip_forward"]:
        detection["forwardSign"] *= -1
    detection["overrides"] = {key: config[key] for key in ("source_up", "source_forward", "flip_forward")}
    detection["note"] = "Heuristic orientation; anatomical front must be checked locally. Re-run with explicit axis overrides if needed."
    return detection


def main(session_dir: Path) -> None:
    config = json.loads((session_dir / "config.json").read_text())
    sys.path.insert(0, config["trellis_source"])
    pipeline = load_pipeline(config)
    cutouts, masks, inputs = prepare_cutouts(session_dir)
    generated, crop_size, resolution = infer_shape(pipeline, cutouts["front"], config)
    inputs["front"]["modelInput"] = {
        "preprocessing": "upstream alpha > 0.8 bbox, square crop, black alpha composite, DINOv3 resize",
        "cropSize": list(crop_size), "conditioningResolution": 512,
    }
    raw = trimesh.Trimesh(vertices=generated.vertices.cpu().numpy(), faces=generated.faces.cpu().numpy(), process=False)
    mesh, hires, cleanup = meshops.cleanup(raw, target_faces=config["target_faces"])
    detection = detection_for(np.asarray(mesh.vertices), config)
    vertices, norm = meshops.normalize(np.asarray(mesh.vertices), detection, config["height_cm"] / 100)
    mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
    trimesh.repair.fix_normals(mesh)
    _ = mesh.vertex_normals
    out = session_dir / "out"
    mesh.export(out / "mesh.glb")
    cameras = {
        view: meshops.fit_orthographic_camera(np.asarray(mesh.vertices), np.asarray(mesh.faces), masks[view], yaw)
        for view, yaw in meshops.VIEW_YAW_DEG.items() if view in masks
    }
    bounds = norm.final_bbox
    meta = {
        "schema": "twin-shape/1", "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": "trellis2_colab",
        "coordinateSystem": {
            "units": "meters", "handedness": "right", "up": "+Y", "forward": "+Z (the character faces +Z)",
            "characterLeft": "+X",
            "origin": "x,z = middle of the bbox of the lowest 3 % of the mesh (feet contact patch); y = 0 at the lowest vertex (soles)",
        },
        "inputs": inputs,
        "model": {
            "repo": MODEL_REPO, "revision": MODEL_REVISION, "sourceRevision": config["trellis_revision"],
            "multiView": False, "viewsUsedForShape": ["front"], "pipelineType": config["pipeline_type"],
            "resolutionUsed": resolution, "seed": config["seed"], "steps": config["steps"], "offload": "low_vram",
        },
        "normalization": {
            "sourceFrameDetection": detection, "rotationSourceToStandard": norm.rotation.tolist(),
            "scaleMetersPerSourceUnit": norm.scale, "translationMeters": norm.translation.tolist(),
            "matrix4x4RowMajor": norm.matrix().tolist(), "formula": "p_final = (R @ p_source) * scale + translation",
            "heightCm": config["height_cm"], "heightNote": "total mesh height, including soles and hair",
            "sourceBBox": norm.source_bbox,
            "finalBBox": {**bounds, "sizeMeters": (np.array(bounds["max"]) - bounds["min"]).tolist()},
        },
        "cameras": {
            "assumption": "orthographic; x' = x*cos(yaw) - z*sin(yaw); u = originPx[0] + pxPerMeter*x'; v = originPx[1] - pxPerMeter*y. Final world metres to ORIGINAL input pixels, v downwards; yaw front/left/back/right = 0/90/180/270. Fitted by silhouette IoU; perspective photos require refinement.",
            "views": cameras,
        },
        "mesh": {"faces": len(mesh.faces), "vertices": len(mesh.vertices), "watertight": bool(mesh.is_watertight), "hiresFaces": len(hires.faces), "cleanup": cleanup},
        "files": {"glb": "mesh.glb", "obj": None, "hiresGlb": None, "previews": None},
        "licenseNote": "TRELLIS.2 code and 4B weights: MIT; required DINOv3 encoder: Meta DINOv3 custom license. Not a permissive-only pipeline. BRIA RMBG-2.0 and NVIDIA renderers are not used.",
        "provenance": {"shape": "Microsoft TRELLIS.2-4B", "license": "MIT + DINOv3 custom license", "createdAt": datetime.now(timezone.utc).isoformat()},
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
    validate_pair(out / "mesh.glb", out / "meta.json")
    print("Shape exported and numeric convention checks passed. Check anatomical front locally.")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
