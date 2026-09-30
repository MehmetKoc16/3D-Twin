"""End-to-end texturing pipeline: mesh -> UV -> views -> bake -> fill -> GLB + previews."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from . import align, meshio
from .bake import BakeConfig, run_bake, texel_attributes
from .camera import OrthoCamera
from .compose import compose_atlas
from .delight import remove_shading
from .estimate import estimate_gains
from .export import encode_texture, write_glb
from .preview import render_sheet
from .unwrap import unwrap
from .views import View, build_view, load_view_image

VIEW_NAMES = ("front", "back", "left", "right")


@dataclass
class PipelineConfig:
    mesh_path: Path
    views_dir: Path
    out_dir: Path
    view_files: dict[str, Path] = field(default_factory=dict)  # overrides / extra views
    size: int = 4096
    padding: int = 8
    max_faces: int = 150_000
    up: str = "+y"
    front: str = "+z"
    use_flow: bool = True
    allow_aspect: bool = False
    harmonize: bool = True
    delight: float = 0.0
    mirror: bool = True
    hair_prior: bool = True
    sharpness: float = 4.0
    face_boost: float = 8.0
    texture_format: str = "jpeg"
    jpeg_quality: int = 93
    previews: bool = True
    debug_maps: bool = True
    auto_side: bool = True
    shape_cutouts: bool = True  # prefer the shape agent's cutout_<view>.png alpha (semantic rembg) over our matting


def discover_views(views_dir: Path, overrides: dict[str, Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for n in VIEW_NAMES:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = views_dir / f"{n}{ext}"
            if p.exists():
                found[n] = p
                break
    found.update({k: Path(v) for k, v in overrides.items()})
    return found


def prepare_mesh(cfg: PipelineConfig, log=print) -> tuple[trimesh.Trimesh, np.ndarray]:
    mesh = meshio.load_mesh(cfg.mesh_path)
    log(f"mesh: {len(mesh.vertices)} v, {len(mesh.faces)} f, bounds {np.round(mesh.bounds, 3).tolist()}")
    mesh = meshio.orient(mesh, cfg.up, cfg.front)
    mesh = meshio.clean_mesh(mesh)
    mesh = meshio.ensure_outward(mesh)
    if cfg.max_faces and len(mesh.faces) > cfg.max_faces:
        mesh = meshio.decimate(mesh, cfg.max_faces)
        mesh = meshio.clean_mesh(mesh)
        mesh = meshio.ensure_outward(mesh)
        log(f"decimated to {len(mesh.faces)} faces")
    normals = meshio.smooth_vertex_normals(mesh, iterations=2)
    return mesh, normals


def choose_side_camera(name: str, verts: np.ndarray, faces: np.ndarray, alpha: np.ndarray, log=print) -> str:
    """A side view is either seen from the character's left or right; pick the mirror with the better IoU."""
    other = {"left": "right", "right": "left"}[name]
    scores = {}
    for n in (name, other):
        cam0 = OrthoCamera.axis(n)
        cam0.width, cam0.height = alpha.shape[1], alpha.shape[0]
        _, _, iou1 = align.fit_similarity(cam0, verts, faces, alpha)
        scores[n] = iou1
    best = max(scores, key=scores.get)
    log(f"  side view '{name}': IoU as {name}={scores[name]:.3f}, as {other}={scores[other]:.3f} -> using {best}")
    return best


def run(cfg: PipelineConfig, log=print) -> dict:
    t0 = time.time()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"mesh": str(cfg.mesh_path), "size": cfg.size}

    mesh, vnormals_src = prepare_mesh(cfg, log)
    report["mesh_faces"] = int(len(mesh.faces))
    log("unwrap (xatlas) ...")
    t = time.time()
    unw = unwrap(mesh, size=cfg.size, padding=cfg.padding, cache_dir=out / "cache")
    log(f"  {unw.chart_count} charts, utilisation {unw.utilization:.2f}, {time.time() - t:.1f}s")
    report.update(charts=unw.chart_count, utilization=unw.utilization, uv_vertices=int(len(unw.vertices)))
    vnormals = vnormals_src[unw.vmapping]
    verts_w = np.asarray(mesh.vertices, dtype=np.float64)
    faces_w = np.asarray(mesh.faces, dtype=np.int64)

    meta = meshio.load_shape_meta(Path(cfg.mesh_path).parent)
    meta_cams = (meta.get("cameras") or {}).get("views") or {}
    files = discover_views(cfg.views_dir, cfg.view_files)
    if not files:
        raise SystemExit(f"no view images found in {cfg.views_dir} (expected front/back/left/right .png)")
    log(f"views: {', '.join(f'{k}={v.name}' for k, v in files.items())}")
    views: list[View] = []
    used: set[str] = set()
    for name, path in files.items():
        rgb, alpha = load_view_image(path)
        cutout = Path(cfg.mesh_path).parent / f"cutout_{name}.png"
        if cfg.shape_cutouts and cutout.exists():
            from PIL import Image as _Image

            cut = np.array(_Image.open(cutout).convert("RGBA"))
            if cut.shape[:2] == alpha.shape:
                alpha = cut[..., 3].astype(np.float32) / 255.0
        cam_name = name
        if cfg.auto_side and name in ("left", "right"):
            cam_name = choose_side_camera(name, verts_w, faces_w, alpha, log)
            if cam_name in used:
                log(f"  WARNING: camera '{cam_name}' already used, keeping '{name}'")
                cam_name = name
        used.add(cam_name)
        cam0 = OrthoCamera.axis(cam_name)
        init = None
        mc = meta_cams.get(cam_name)
        if isinstance(mc, dict) and "pxPerMeter" in mc and "originPx" in mc and cfg.up == "+y" and cfg.front == "+z":
            init = (float(mc["pxPerMeter"]), float(mc["originPx"][0]), float(mc["originPx"][1]))
        v = build_view(cam_name, rgb, alpha, verts_w, faces_w, cam0, use_flow=cfg.use_flow,
                       allow_aspect=cfg.allow_aspect, init=init)
        log(f"  view {cam_name}: IoU {v.stats['iou_initial']:.3f} -> {v.stats['iou_similarity']:.3f} (similarity)"
            f" -> {v.stats['iou_final']:.3f} (flow), {v.image_lin.shape[1]}x{v.image_lin.shape[0]}px,"
            f" {v.stats['scale_px_per_m']:.0f} px/m")
        if cfg.delight > 0:
            v.image_lin = remove_shading(v.image_lin, v.alpha, cfg.delight)
        views.append(v)
        report.setdefault("views", {})[cam_name] = {"file": path.name, **{k: float(x) for k, x in v.stats.items()}}

    bcfg = BakeConfig(size=cfg.size, sharpness=cfg.sharpness, face_boost=cfg.face_boost)
    ymin, ymax = float(unw.vertices[:, 1].min()), float(unw.vertices[:, 1].max())
    if cfg.harmonize and len(views) > 1:
        small = max(cfg.size // 4, 512)
        rr, cc, P, N = texel_attributes(unw, vnormals, small, 0, small)
        gains = estimate_gains(views, P, N, bcfg, ymin, ymax)
        for v in views:
            v.image_lin = v.image_lin * gains[v.name]
        report["gains"] = {k: [round(float(x), 3) for x in g] for k, g in gains.items()}
        log(f"  colour gains vs front: { {k: np.round(g, 3).tolist() for k, g in gains.items()} }")

    log("bake ...")
    t = time.time()
    bake = run_bake(unw, vnormals, views, bcfg, log)
    report["coverage"] = bake.coverage
    log(f"  coverage {bake.coverage}  ({time.time() - t:.1f}s)")

    log("compose / fill ...")
    x_mid = float(np.average(unw.vertices[:, 0]))
    atlas, info = compose_atlas(bake, bcfg, x_mid, mirror=cfg.mirror, padding=cfg.padding, views=views,
                                hair=cfg.hair_prior, log=log)
    report["fill"] = info

    log("export ...")
    data, mime = encode_texture(atlas, cfg.texture_format, cfg.jpeg_quality)
    ext = "jpg" if "jpeg" in mime else "png"
    Image.fromarray(atlas).save(out / "baseColor.png", optimize=False, compress_level=3)
    extras = {
        "twintex": {
            "views": sorted(v.name for v in views),
            "note": "baseColor is sRGB; untextured/unobserved areas are diffusion-filled (see report.json)",
        }
    }
    glb = out / "textured.glb"
    write_glb(glb, unw.vertices, vnormals.astype(np.float32), unw.uv, unw.faces, data, mime, extras=extras)
    report["glb"] = str(glb)
    report["glb_mb"] = round(glb.stat().st_size / 1e6, 2)
    del ext

    if cfg.debug_maps:
        prev = out / "previews"
        prev.mkdir(parents=True, exist_ok=True)
        from .debugviz import save_debug_maps

        save_debug_maps(prev, atlas, bake, [v.name for v in views])
        for v in views:
            from .debugviz import save_alignment_overlay

            save_alignment_overlay(prev / f"debug_align_{v.name}.png", v, verts_w, faces_w)
    if cfg.previews:
        log("previews ...")
        report["previews"] = render_sheet(unw.vertices, unw.faces, unw.uv, vnormals, atlas, out / "previews")
    report["seconds"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"done in {report['seconds']}s -> {glb}")
    return report
