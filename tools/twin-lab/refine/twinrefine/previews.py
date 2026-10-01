"""Before / after preview renders (lit, orthographic): head sheets and T-pose full body + armpit close-ups."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from twintex.camera import OrthoCamera  # noqa: E402  (texture stage)

from . import log
from .meshops import Corners, from_corners, to_corners
from .render import montage, render, render_head_sheet
from .scan import Scan, welded_vertex_normals


def _save(path: Path, img: np.ndarray) -> str:
    Image.fromarray(img).save(path)
    return str(path)


def _cam(az: float, center, half: float, size: tuple[int, int]) -> OrthoCamera:
    W, H = size
    cam = OrthoCamera.azimuth("p", az)
    s = H / (2 * half)
    u = np.asarray(center) @ cam.right
    v = np.asarray(center) @ cam.up
    cam.scale, cam.aspect = float(s), 1.0
    cam.tx, cam.ty = W / 2 - s * u, H / 2 + s * v
    cam.width, cam.height = W, H
    return cam


def posed_scan_arrays(scan: Scan, mc: Corners, posed_welded: np.ndarray):
    """Index-space posed vertices + smooth normals for a scan whose welded vertices were posed."""
    verts, faces, uv, wid = from_corners(mc, True)
    pv = posed_welded[wid]
    return pv, faces, uv, welded_vertex_normals(pv, faces)


def make_previews(pdir: Path, scan0: Scan, mc0: Corners, refined: Scan, model, body, pm_before=None) -> dict:
    pdir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    n0 = welded_vertex_normals(scan0.verts, scan0.faces)
    n1 = welded_vertex_normals(refined.verts, refined.faces)
    out.update({f"before_{k}": v for k, v in render_head_sheet(
        scan0.verts, scan0.faces, n0, scan0.uv, scan0.atlas, pdir, prefix="before_").items()})
    out.update({f"after_{k}": v for k, v in render_head_sheet(
        refined.verts, refined.faces, n1, refined.uv, refined.atlas, pdir, prefix="after_").items()})
    # head comparison montage
    names = ["head_front", "head_threequarter", "head_side", "head_clay_front", "head_clay_threequarter"]
    rows = []
    for pre in ("before_", "after_"):
        rows.append(montage([np.array(Image.open(pdir / f"{pre}{n}.png").convert("RGB").resize((420, 420))) for n in names], len(names)))
    out["head_compare"] = _save(pdir / "head_compare.png", np.concatenate(rows, axis=0))

    if body is not None:
        from .posecheck import pose_mesh

        mc1 = to_corners(refined.verts, refined.faces, refined.uv)
        pm0 = pm_before or pose_mesh(model, body, mc0.P, mc0.F)
        pm1 = pose_mesh(model, body, mc1.P, mc1.F)
        tiles = []
        arm_tiles = []
        for tag, scan, mc, pm in (("before", scan0, mc0, pm0), ("after", refined, mc1, pm1)):
            pv, faces, uv, nrm = posed_scan_arrays(scan, mc, pm.posed)
            full = []
            for az in (0.0, 35.0, 90.0):
                cam = _cam(az, (0.0, 0.95, 0.0), 0.98, (700, 700))
                full.append(render(pv, faces, nrm, cam, 700, 700, uv, scan.atlas))
            tiles.append(np.concatenate(full, axis=1))
            out[f"tpose_{tag}"] = _save(pdir / f"tpose_{tag}.png", tiles[-1])
            arms = []
            for az, c in ((0.0, (0.30, 1.22, 0.0)), (-55.0, (0.30, 1.22, 0.0)), (150.0, (0.30, 1.22, 0.0))):
                cam = _cam(az, c, 0.17, (520, 520))
                arms.append(render(pv, faces, nrm, cam, 520, 520, uv, scan.atlas))
            arm_tiles.append(np.concatenate(arms, axis=1))
            out[f"armpit_{tag}"] = _save(pdir / f"armpit_{tag}.png", arm_tiles[-1])
        out["tpose_compare"] = _save(pdir / "tpose_compare.png", np.concatenate(tiles, axis=0))
        out["armpit_compare"] = _save(pdir / "armpit_compare.png", np.concatenate(arm_tiles, axis=0))
    log(f"previews -> {pdir}")
    return out
