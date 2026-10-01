"""The refine stage: textured.glb -> refined.glb (face relief + armpit separation + face texture re-projection)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from mh import MHModel  # noqa: E402  (rig stage)
from PIL import Image

from . import REPO, log
from .armpit import ArmpitParams, apex_height, separate_arms
from .bodyfit import arm_weight, fit_body, make_welded, skin_weights
from .meshops import Corners, from_corners, to_corners
from .scan import Scan, load_scan, save_scan, welded_vertex_normals


@dataclass
class RefineConfig:
    inp: Path
    out: Path
    front: Path | None = None
    face: Path | None = None
    fit_cache: Path | None = None  # body-fit pickle (default <out dir>/cache/fit.pkl)
    armpits: bool = True
    face_relief: bool = True
    rebake: bool = True
    previews: bool = True
    preview_dir: Path | None = None
    gap_mm: float = 4.0
    min_webbing_edges: int = 40  # armpit separation only when the T-pose stretches at least this many edges > 2x
    face_edge_mm: float = 2.2
    face_res_mm: float = 0.5
    jpeg_quality: int = 95
    armpit: ArmpitParams = field(default_factory=ArmpitParams)
    detector: object | None = None  # callable(rgb, alpha, full_frame) -> FaceLandmarks | None (tests inject synthetic landmarks)


def default_photo(inp: Path, name: str) -> Path | None:
    """``user-data/twin/<name>`` next to the stage folders (``.../twin/out/texture/textured.glb``)."""
    for base in (inp.resolve().parent.parent.parent, REPO / "user-data" / "twin"):
        p = base / name
        if p.exists():
            return p
    return None


def run(cfg: RefineConfig) -> dict:
    t0 = time.time()
    out_dir = Path(cfg.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"input": str(cfg.inp), "output": str(cfg.out)}
    scan = load_scan(cfg.inp)
    log(f"scan: {scan.n} vertices, {len(scan.faces)} faces, atlas {None if scan.atlas is None else scan.atlas.shape}")
    mc0 = to_corners(scan.verts, scan.faces, scan.uv)
    mc = mc0.copy()
    atlas = None if scan.atlas is None else scan.atlas.copy()
    model = MHModel()
    fit_cache = cfg.fit_cache or (out_dir / "cache" / "fit.pkl")
    body = None
    need_body = cfg.armpits or cfg.face_relief
    if need_body:
        inv, uverts, ufaces = make_welded(scan.verts, scan.faces)
        res = fit_body(model, uverts, ufaces, cache=fit_cache)
        report["body_fit"] = {k: res.stats.get(k) for k in ("template_to_scan_cm", "scan_to_template_cm")}
        body = res

    # ---- the front photo as the texture stage saw it (camera + flow), from the unmodified geometry
    fv = None
    if cfg.face_relief:
        front = cfg.front or default_photo(cfg.inp, "front.png")
        if front is None or not Path(front).exists():
            log("no front photo found: face relief skipped (pass --front)")
            report["face"] = {"status": "skipped: no front photo"}
        else:
            from .frontview import build_front

            fv = build_front(Path(front), mc0.P, mc0.F, Path(cfg.inp))

    # ---- armpits
    stretch_before = stretch_after = None
    pm_before = None
    if cfg.armpits and body is not None:
        from .posecheck import pose_mesh, webbing_stats

        W = skin_weights(model, body, mc0.P, mc0.F)
        aw_l, aw_r = arm_weight(model, W, "l"), arm_weight(model, W, "r")
        prm = cfg.armpit
        prm.gap_mm = cfg.gap_mm
        pm_before = pose_mesh(model, body, mc0.P, mc0.F)
        stretch_before = webbing_stats(pm_before)
        if stretch_before["over2"] < cfg.min_webbing_edges:
            log(f"armpits: only {stretch_before['over2']} edges stretch > 2x in the T-pose near the armpits: nothing to separate")
            report["armpits"] = {"status": "skipped: no webbing", "tpose_edge_stretch_before": stretch_before}
        else:
            mc, atlas, arep = separate_arms(mc, atlas, aw_l, aw_r, apex_height(model, body), prm)
            report["armpits"] = {"y_top": arep.y_top, **arep.sides, "tpose_edge_stretch_before": stretch_before}
            pm_after = pose_mesh(model, body, mc.P, mc.F)
            stretch_after = webbing_stats(pm_after)
            report["armpits"]["tpose_edge_stretch_after"] = stretch_after
            log(f"armpits: T-pose edges stretched > 2x near the armpits {stretch_before['over2']} -> {stretch_after['over2']}")

    # ---- face
    fd = None
    if cfg.face_relief and fv is not None and body is not None:
        mc, atlas, fd, frep = face_stage(cfg, scan, mc, atlas, fv, body, model)
        report["face"] = frep

    verts, faces, uv = from_corners(mc)
    normals = welded_vertex_normals(verts, faces)
    refined = Scan(verts, faces, uv, atlas, scan.mime, scan.scene, scan.name)
    save_scan(cfg.out, refined, normals, cfg.jpeg_quality)
    report["output_vertices"] = int(len(verts))
    report["output_faces"] = int(len(faces))
    log(f"wrote {cfg.out} ({Path(cfg.out).stat().st_size / 1e6:.1f} MB), {len(verts)} vertices, {len(faces)} faces")

    if cfg.previews:
        from .previews import make_previews

        pdir = cfg.preview_dir or (out_dir / "previews")
        report["previews"] = make_previews(pdir, scan, mc0, refined, model, body, pm_before)
    report["seconds"] = round(time.time() - t0, 1)
    (out_dir / "refine_report.json").write_text(json.dumps(report, indent=2, default=_json), encoding="utf-8")
    return report


def _json(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def face_stage(cfg: RefineConfig, scan: Scan, mc: Corners, atlas, fv, body, model):
    """Landmarks -> MakeHuman face fit -> depth relief -> mesh refinement and displacement -> texture re-projection."""
    from twintex.raster import zbuffer

    from . import relief
    from .facedepth import Grid, build_face_depth
    from .facefit import FaceModel, fit_face, load_face_map
    from .landmarks import detect_landmarks
    from .rebake import front_source, portrait_source, rebake_face

    detect = cfg.detector or (lambda rgb, alpha, full_frame=False: detect_landmarks(rgb, alpha, full_frame=full_frame))
    rep: dict = {}
    try:
        lm = detect(fv.rgb, fv.alpha, False)
    except Exception as exc:  # mediapipe missing / model unreadable: the rest of the stage still runs
        log(f"landmark detection failed ({exc}): face relief skipped")
        return mc, atlas, None, {"status": f"skipped: {exc}"}
    if lm is None:
        log("no face found in the front photo: face relief skipped")
        return mc, atlas, None, {"status": "skipped: no face detected"}
    L = fv.photo_to_world_xy(lm.xy[:468])
    fm = load_face_map()
    fmodel = FaceModel(model, body, fm)
    fit = fit_face(fmodel, L)
    rep["landmark_rms_mm"] = {"before": fit.rms_neutral_mm, "after": fit.rms_mm}
    rep["modifiers"] = {m: float(v) for m, v in zip(fit.modifiers, fit.theta, strict=True)}
    grid = Grid.around(L, 0.05, cfg.face_res_mm * 1e-3)
    px, py = grid.to_px(mc.P[:, 0], mc.P[:, 1])
    depth, fid = zbuffer(np.stack([px, py, -mc.P[:, 2]], axis=1), mc.F, grid.W, grid.H)
    zscan = np.where(fid >= 0, -depth, np.nan).astype(np.float32)
    from .hairmask import refine_oval

    fd = build_face_depth(fmodel, fit, L, zscan, grid, fm, oval_fn=lambda g, ov: refine_oval(fv, g, L, fm, ov),
                          log=lambda s: log(str(s)))
    rep["depth"] = fd.info
    mcr = relief.refine_face_region(mc, fd, edge_mm=cfg.face_edge_mm)
    log(f"face region refined: {len(mc.F)} -> {len(mcr.F)} faces")
    mcd, rrep = relief.apply_relief(mcr, fd)
    mcd = relief.smooth_band(mcd, fd)
    mcd, nfold = relief.heal_folds(mcr, mcd, fd)
    log(f"face relief: {nfold} fold-over faces relaxed")
    rep["relief"] = {"faces_before": len(mc.F), "faces_after": len(mcd.F), "moved": rrep.moved_vertices,
                     "max_move_mm": rrep.max_move_mm, "mean_move_mm": rrep.mean_move_mm}
    if cfg.rebake and atlas is not None:
        sources = [front_source(fv)]
        face_path = cfg.face or default_photo(cfg.inp, "face.png")
        if face_path is not None and Path(face_path).exists():
            frgb = np.array(Image.open(face_path).convert("RGB"))
            flm = detect(frgb, None, True)
            if flm is None:
                log(f"no face found in {face_path}: using the front photo only")
            else:
                sources.insert(0, portrait_source(frgb, flm.xy, L))
                rep["face_png"] = {"file": str(face_path), "size": list(frgb.shape[1::-1])}
                log(f"face texture from {Path(face_path).name} ({frgb.shape[1]}x{frgb.shape[0]}) + front photo")
        atlas, brep = rebake_face(mcd, atlas, fd, sources)
        rep["rebake"] = {"texels": brep.texels, "mean_weight": brep.mean_weight, "gains": brep.gains,
                         "sources": brep.sources}
    return mcd, atlas, fd, rep
