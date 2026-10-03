"""Export fitted identity, neutral geometry, native cameras and alignment overlays."""

from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import zipfile

import numpy as np

from camera import camera_matrices, projection_matrix
from setup_runtime import PIXEL_REVISION, MICA_REVISION, FACER_REVISION, PIPNET_REVISION


def expression_indicators(checkpoints: list) -> list:
    """Within-person neutral preference; coefficient magnitude is no smile classifier."""
    norms, rms = [], []
    for checkpoint in checkpoints:
        expression = np.asarray(checkpoint["flame"]["exp"], dtype=np.float64).reshape(-1)
        if expression.shape != (100,) or not np.isfinite(expression).all():
            raise ValueError("Invalid fitted expression coefficients")
        norms.append(float(np.linalg.norm(expression)))
        rms.append(float(np.sqrt(np.mean(expression ** 2))))
    return [{"magnitudeL2": norm, "magnitudeRMS": rms[index],
             "neutralityRank": 1 + sum(other < norm for other in norms),
             "mouthTexturePreference": 1.0 / (1.0 + norm),
             "smileProxyOnly": True} for index, norm in enumerate(norms)]


def camera_document(skipped: list) -> dict:
    return {"schema": "dt-flame-head-cameras/1", "model": "OpenGL perspective pinhole",
            "grid": "256x256 independent face crop; pixel centers at (column+0.5,row+0.5)",
            "projection": "q = worldToCamera @ [x,y,z,1]; depth=-q.z; u=fx*q.x/depth+cx; v=cy-fy*q.y/depth",
            "meshSpace": "native FLAME metres, unnormalized; extrinsics apply to fitted_views meshes with local neck/jaw/expression already applied",
            "expressionIndicator": {"source": "100 fitted FLAME expression coefficients",
                                    "interpretation": "Lower magnitude/rank prefers neutral mouth texture within this fit; smileProxyOnly is not a smile classification",
                                    "mouthTexturePreference": "1/(1+magnitudeL2); heuristic, not a probability"},
            "views": {}, "skipped": skipped}


def camera_view(item: dict, checkpoint: dict, expression: dict) -> dict:
    intrinsics, extrinsics, side = camera_matrices(checkpoint)
    joint_transforms = np.asarray(checkpoint["joint_transforms"])
    view = item["view"]
    return {**item, "imageSizeWH": [side, side], "intrinsics": intrinsics.tolist(),
            "worldToCamera": extrinsics.tolist(),
            "headCentricWorldToCamera": (extrinsics @ joint_transforms[0, 1]).tolist(),
            "headCentricNote": "Use only with the matching fitted mesh after undoing joint_transforms[0,1]; neutral expression changes geometry",
            "native": {key: np.asarray(value).tolist() for key, value in checkpoint["camera"].items()},
            "projectionMatrix": projection_matrix(intrinsics, side).tolist(),
            "overlay": f"overlays/{view}.png", "fittedMesh": f"fitted_views/{view}.ply",
            "expression": expression}


def render_overlay(vertices, faces, camera, image, destination: Path) -> None:
    import torch
    import nvdiffrast.torch as dr
    from PIL import Image
    import trimesh

    intrinsics, extrinsics, side = camera
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    points = torch.tensor(np.column_stack([vertices, np.ones(len(vertices))]), dtype=torch.float32, device="cuda")
    matrix = torch.tensor(projection_matrix(intrinsics, side) @ extrinsics, dtype=torch.float32, device="cuda")
    clips = (points @ matrix.T)[None].contiguous()
    triangles = torch.tensor(faces.astype(np.int32), device="cuda").contiguous()
    context = dr.RasterizeCudaContext()
    raster, _ = dr.rasterize(context, clips, triangles, resolution=[side, side])
    normals = torch.tensor(np.asarray(mesh.vertex_normals).copy(), dtype=torch.float32, device="cuda")[None]
    normal_image, _ = dr.interpolate(normals.contiguous(), raster, triangles)
    shading = (0.35 + 0.65 * normal_image[..., 2:3].abs()).clamp(0, 1)
    color = shading * torch.tensor([0.2, 0.7, 1.0], device="cuda")
    mask = (raster[..., 3:4] > 0).float()
    # The upstream projection matrix has negative fy; array row 0 is already top.
    rendered = (color * mask)[0].cpu().numpy()
    alpha = mask[0].cpu().numpy() * 0.5
    background = np.asarray(Image.open(image).convert("RGB").resize((side, side)), dtype=np.float32) / 255
    Image.fromarray(np.clip((background * (1 - alpha) + rendered * 0.5) * 255, 0, 255).astype(np.uint8)).save(destination)


def export_fit(root: Path, config: dict) -> None:
    import torch
    import trimesh
    from omegaconf import OmegaConf
    from pixel3dmm.tracking.flame.FLAME import FLAME

    fit_path = Path((root / "fit_path.txt").read_text())
    view_map = json.loads((root / "view_map.json").read_text())
    out = root / "export"
    (out / "overlays").mkdir(parents=True)
    (out / "fitted_views").mkdir()
    checkpoints = [torch.load(fit_path / f"checkpoint/{item['frame']:05d}.frame", map_location="cpu", weights_only=False)
                   for item in view_map["included"]]
    shape = np.asarray(checkpoints[0]["flame"]["shape"], dtype=np.float32).reshape(1, -1)
    if shape.shape != (1, 300) or not np.isfinite(shape).all():
        raise ValueError("Invalid shared shape coefficients")
    for checkpoint in checkpoints:
        if not np.allclose(shape, checkpoint["flame"]["shape"]):
            raise ValueError("Final per-view checkpoints do not share identity")
    flame = FLAME(OmegaConf.create({"num_shape_params": 300, "num_exp_params": 100,
                                   "use_flame2023": config["flame_version"] == "2023"})).cuda()
    with torch.no_grad():
        vertices = flame(shape_params=torch.tensor(shape, device="cuda"), cameras=None)[0][0].cpu().numpy()
    faces = flame.faces.cpu().numpy()
    neutral = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    neutral.export(out / "head_neutral.obj")
    neutral.export(out / "head_neutral.ply")
    identity6d = [1, 0, 0, 0, 1, 0]
    neutral_parameters = {"shape": shape[0].tolist(), "exp": [0.0] * 100,
                          "R": identity6d, "t": [0.0] * 3, "neck": identity6d,
                          "jaw": identity6d, "eyes": identity6d * 2, "eyelids": [0.0, 0.0]}
    parameters = {"schema": "dt-flame-head-parameters/1", "flameVersion": config["flame_version"],
                  "rotationEncoding": "6D: first two ROWS of a rotation matrix (Pixel3DMM/PyTorch3D)",
                  "neutral": neutral_parameters, "views": {}, "viewMetadata": {}}
    arrays = {"neutral_" + key: np.asarray(value, dtype=np.float32) for key, value in neutral_parameters.items()}
    cameras = camera_document(view_map["skipped"])
    indicators = expression_indicators(checkpoints)
    for item, checkpoint, expression in zip(view_map["included"], checkpoints, indicators):
        view, index = item["view"], item["frame"]
        fitted = {key: np.asarray(value) for key, value in checkpoint["flame"].items()}
        if not all(np.isfinite(value).all() for value in fitted.values()):
            raise ValueError("Nonfinite fitted parameters")
        parameters["views"][view] = {key: value.tolist() for key, value in fitted.items()}
        parameters["viewMetadata"][view] = {**item, "expression": expression}
        arrays.update({view + "_" + key: value for key, value in fitted.items()})
        joint_transforms = np.asarray(checkpoint["joint_transforms"])
        arrays[view + "_joint_transforms"] = joint_transforms
        intrinsics, extrinsics, side = camera_matrices(checkpoint)
        cameras["views"][view] = camera_view(item, checkpoint, expression)
        for key in ("magnitudeL2", "magnitudeRMS", "neutralityRank", "mouthTexturePreference"):
            arrays[view + "_expression_" + key] = np.asarray(expression[key])
        arrays[view + "_intrinsics"] = intrinsics
        arrays[view + "_world_to_camera"] = extrinsics
        source_mesh = fit_path / f"mesh/{index:05d}.ply"
        mesh = trimesh.load(source_mesh, process=False, force="mesh")
        shutil.copyfile(source_mesh, out / f"fitted_views/{view}.ply")
        render_overlay(np.asarray(mesh.vertices), np.asarray(mesh.faces), (intrinsics, extrinsics, side),
                       root / f"preprocessed/head/cropped/{index:05d}.jpg", out / f"overlays/{view}.png")
    (out / "parameters.json").write_text(json.dumps(parameters, indent=2, allow_nan=False))
    np.savez_compressed(out / "parameters.npz", **arrays)
    (out / "cameras.json").write_text(json.dumps(cameras, indent=2, allow_nan=False))
    runtime_path = root / "fit_runtime.json"
    if runtime_path.is_file():
        shutil.copyfile(runtime_path, out / "fit_runtime.json")
    provenance = {"createdAt": datetime.now(timezone.utc).isoformat(), "flameVersion": config["flame_version"],
                  "sources": {"pixel3dmm": PIXEL_REVISION, "MICA": MICA_REVISION, "facer": FACER_REVISION, "PIPNet": PIPNET_REVISION},
                  "license": "Personal non-commercial research only; Pixel3DMM CC BY-NC 4.0 plus separately accepted FLAME/MICA/insightface/nvdiffrast terms. Not cleared for app distribution.",
                  "isDiscontinuous": True, "globalCamera": False, "neutralExpression": True, "neutralJaw": True,
                  "includedViews": [item["view"] for item in view_map["included"]], "skippedViews": view_map["skipped"],
                  "fitSettings": json.loads((root / "fit_config.json").read_text()),
                  "limitations": ["No hair reconstruction", "Unseen back anatomy follows FLAME prior", "Scale from model prior; no metric calibration", "Expression magnitude is not a smile classifier", "Eyeglasses can bias fit and texture", "Not the rigged twin.glb bundle"]}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False))
    with zipfile.ZipFile(root / "head_fit.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(out).as_posix())
