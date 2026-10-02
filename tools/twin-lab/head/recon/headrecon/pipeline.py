"""The head stage: refined.glb + four real head photos -> head.glb."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from twinrefine.scan import Scan, load_scan, save_scan, welded_vertex_normals

from . import log
from .bake import BakeParams, bake_head
from .calibrate import HeadFitMesh, calibrate_view
from .facelm import front_landmarks
from .photos import NOMINAL_YAW, VIEWS, load_photos
from .reshape import STABLE_LM, ReshapeParams, reshape_head
from .twinhead import analyse_head

PITCH_RANGE = {"front": 40.0, "back": 25.0, "profile_nose_right": 30.0, "profile_nose_left": 30.0}
YAW_RANGE = {"front": 12.0, "back": 12.0, "profile_nose_right": 14.0, "profile_nose_left": 14.0}


@dataclass
class HeadConfig:
    inp: Path
    photos: Path
    out: Path
    use_clean: bool = False
    reshape: bool = True
    texture: bool = True
    previews: bool = True
    preview_dir: Path | None = None
    reshape_params: ReshapeParams = field(default_factory=ReshapeParams)
    bake_params: BakeParams = field(default_factory=BakeParams)
    detector: object | None = None  # callable(rgb) -> FaceLandmarks | None (tests inject synthetic landmarks)
    jpeg_quality: int = 95


def flipped_faces(v0: np.ndarray, v1: np.ndarray, faces: np.ndarray, gate: np.ndarray) -> int:
    """Number of head triangles whose normal turned by more than 90 degrees (fold-overs created by the deformation)."""
    t0, t1 = v0[faces], v1[faces]
    n0 = np.cross(t0[:, 1] - t0[:, 0], t0[:, 2] - t0[:, 0])
    n1 = np.cross(t1[:, 1] - t1[:, 0], t1[:, 2] - t1[:, 0])
    head = (gate[faces] > 0.05).any(axis=1)
    return int(((np.einsum("ij,ij->i", n0, n1) < 0) & head).sum())


def run(cfg: HeadConfig) -> dict:
    t0 = time.time()
    out_dir = Path(cfg.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"input": str(cfg.inp), "output": str(cfg.out)}
    scan = load_scan(cfg.inp)
    verts0 = scan.verts
    n0 = welded_vertex_normals(verts0, scan.faces)
    log(f"scan: {scan.n} vertices, {len(scan.faces)} faces")
    th = analyse_head(verts0, scan.faces, n0, scan.uv, scan.atlas, cfg.detector)
    photos = load_photos(cfg.photos, out_dir / "cache", cfg.use_clean)
    report["clean_photos"] = [v for v in VIEWS if photos[v].clean]
    flm = front_landmarks(photos["front"], cfg.detector)
    use_lm = flm is not None and th.lm3d is not None
    report["landmarks"] = {"front_photo": flm is not None, "twin": th.lm3d is not None}

    def calib(verts, init=None, y_cut=None, lm3d=None):
        hm = HeadFitMesh(verts, scan.faces, th.chin_y + 0.02 if y_cut is None else y_cut)
        res = {}
        for v in VIEWS:
            kw = {}
            if v == "front" and use_lm:
                kw = {"lm3d": th.lm3d[STABLE_LM] if lm3d is None else lm3d, "lm_px": flm[STABLE_LM]}
            res[v] = calibrate_view(photos[v], NOMINAL_YAW[v], hm, th.ymax, th.pivot, yaw_range=YAW_RANGE[v], pitch_range=PITCH_RANGE[v],
                                    init=None if init is None else init[v].cam, **kw)
        return res

    calibs = calib(verts0)
    report["calibration_before"] = {v: {"iou": round(c.iou, 3), "yaw": round(c.cam.yaw, 1), "pitch": round(c.cam.pitch, 1),
                                        "dist_m": round(c.cam.dist, 3)} for v, c in calibs.items()}
    verts, y_cut, gate = verts0, th.chin_y + 0.02, np.zeros(len(verts0))
    if cfg.reshape:
        res = reshape_head(verts0, scan.faces, th, photos, calibs, recalibrate=None,
                           front_landmarks=flm if use_lm else None, prm=cfg.reshape_params)
        verts, gate, y_cut = res.verts, res.gate, res.y_cut
        report["reshape"] = {"rounds": res.stats["rounds"], "max_move_mm": float(np.abs(res.disp).max() * 1000),
                             "flipped_faces": flipped_faces(verts0, verts, scan.faces, gate)}
        log(f"reshape: max move {report['reshape']['max_move_mm']:.1f} mm, flipped faces {report['reshape']['flipped_faces']}")
        calibs = calib(verts, res.calibs, y_cut, res.lm_pos)
        report["calibration_after"] = {v: {"iou": round(c.iou, 3), "yaw": round(c.cam.yaw, 1), "pitch": round(c.cam.pitch, 1),
                                           "dist_m": round(c.cam.dist, 3)} for v, c in calibs.items()}
    normals = welded_vertex_normals(verts, scan.faces)
    atlas = scan.atlas
    if cfg.texture and atlas is not None:
        atlas, brep = bake_head(verts, scan.faces, scan.uv, normals, atlas, th, photos, calibs, cfg.bake_params)
        report["bake"] = {"texels": brep.texels, "covered": brep.covered_frac, "view_share": brep.view_share,
                          "gains": brep.gains, "seam_gain": brep.seam_gain}
    out = Scan(verts, scan.faces, scan.uv, atlas, scan.mime, scan.scene, scan.name)
    save_scan(cfg.out, out, normals, cfg.jpeg_quality)
    log(f"wrote {cfg.out} ({Path(cfg.out).stat().st_size / 1e6:.1f} MB)")

    if cfg.previews:
        from .previews import photo_comparisons, standard_views

        pdir = cfg.preview_dir or (out_dir / "previews")
        center = th.pivot
        ymin = th.chin_y - 0.10
        report["previews"] = {
            "compare": photo_comparisons(pdir, verts, scan.faces, normals, scan.uv, atlas, photos, calibs, center, ymin),
            "views": standard_views(
                pdir, [("before", verts0, scan.faces, n0, scan.atlas), ("after", verts, scan.faces, normals, atlas),
                       ("after clay", verts, scan.faces, normals, None)], scan.uv, th.ymax),
        }
    report["seconds"] = round(time.time() - t0, 1)
    (out_dir / "head_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
