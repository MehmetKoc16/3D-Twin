"""App-facing export of a rigged twin: `twin.json` + `mh2twin.bin` next to `rigged.glb`.

The web app (apps/web/src/features/twin) shows the rigged scan with our poses, zoom and wardrobe. For that it needs
(1) the fitted MakeHuman body, to solve the hidden body in the browser with the SAME avatar-core morph model
    (`fittedMacros` + `fittedModifiers`; the skeleton and the garments come from that body),
(2) the measurements of that body (`measurementsCm`, shown read-only in the body panel),
(3) a vertex mapping twin vertex -> nearest MakeHuman render vertex (`mh2twin.bin`), so twin triangles covered by a
    worn garment can be hidden.

Everything here is derived from the fit; nothing personal is added. The files stay in `user-data/` for a real person.

twin.json (version 1)
  version, generator, glb, source {scan, options}
  mapping {file, format "uint32le", twinVertexCount, renderVertexCount}
  boneOrder                 the 53 bone names in rig.json order (= glTF skin joint order)
  fittedMacros              {gender, muscle, weight, height}  (macro variables, 0..1)
  fittedModifiers           {modifier id: value}, NET values (incr - decr), only |v| > 1e-6
  measurementsRawCm         measure id -> cm on the fitted body, or bodyfix's achieved scan landmarks
  clothingAllowanceCm       cm the scan's clothes / hair / shoes add per measure (documented below)
  measurementsCm            raw - allowance
  restHeadsM                bone name -> [x, y, z] rest head in the frame of rigged.glb (metres, feet on y = 0)
  fit                       fit quality numbers (informational)

Canonical fit: the fitter minimises with an incr AND a decr column per modifier, but avatar-core (and the web app)
apply the NET value with one of the two targets. `canonicalize_fit` therefore rebuilds the rest body from the net values
(rounded to 6 decimals, exactly what twin.json stores), so the browser reproduces the body the rig was made from.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from mh import BODY_DIR, MHModel

TWIN_VERSION = 1
MAPPING_FILE = "mh2twin.bin"

# What the scan's clothes / hair / shoes add to each measurement of the fitted (clothed-silhouette) MakeHuman body.
# Rough constants for a person in a fitted T-shirt, jeans and sneakers (the default scan of this lab), in cm:
#   height     hair volume (~1 cm) + sneaker sole (~2 cm): the fit aligns the body with the scan's lowest point
#   neck       collar
#   shoulder   sleeve seams and fabric over the shoulder line
#   chest      T-shirt ease and fabric
#   waist      T-shirt hem, jeans waistband / belt
#   hip        jeans
#   thigh      jeans ease and fabric
#   upperArm   sleeve
#   armLength  none (measured along the arm surface from the shoulder to the wrist)
#   inseam     sneaker sole (the crotch height is measured from the floor) + jeans seam
#   footLength sneaker toe box and heel
# They are estimates, not measurements: edit twin.json (`measurementsCm`) if you know your real values.
CLOTHING_ALLOWANCE_CM: dict[str, float] = {
    "height": 3.0,
    "neck": 0.5,
    "shoulder": 1.0,
    "chest": 3.0,
    "waist": 3.0,
    "hip": 2.5,
    "thigh": 2.0,
    "upperArm": 2.5,
    "armLength": 0.0,
    "inseam": 2.5,
    "footLength": 2.5,
}

MACRO_KEYS = ("gender", "muscle", "weight", "height")


# ---------------------------------------------------------------------------------------------------- measures
# numpy port of packages/avatar-core measure() (reference: tools/asset-pipeline/measures.py::evaluate)


def _hull_perimeter(pts: np.ndarray) -> float:
    p = sorted(set(map(tuple, np.round(pts, 9).tolist())))
    if len(p) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: list = []
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    h = np.array(lower[:-1] + upper[:-1])
    return float(np.linalg.norm(h - np.roll(h, -1, axis=0), axis=1).sum())


def _plane_basis(normal: np.ndarray) -> np.ndarray:
    n = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(n, u)], axis=1)


def _circumference(points: np.ndarray) -> float:
    n = np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)  # Newell normal
    if np.linalg.norm(n) < 1e-12:
        c = points - points.mean(axis=0)
        n = np.linalg.eigh(c.T @ c)[1][:, 0]
    return _hull_perimeter((points - points.mean(axis=0)) @ _plane_basis(n))


def evaluate_measure(defn: dict, positions: np.ndarray, render_count: int) -> float:
    """Metres of one measures.json definition on combined positions (render vertices ++ joint points)."""
    t = defn["type"]
    if t == "circumference":
        return _circumference(positions[defn["verts"]])
    if t == "distance":
        a, b = positions[defn["verts"][0]], positions[defn["verts"][1]]
        if "axis" in defn:
            return float(abs(a["xyz".index(defn["axis"])] - b["xyz".index(defn["axis"])]))
        return float(np.linalg.norm(a - b))
    if t == "polyline":
        pts = positions[defn["verts"]]
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    if t == "height":
        y = positions[:render_count, 1]
        return float(y.max() - y.min())
    if t == "vertexHeight":
        return float(positions[defn["vert"], 1] - positions[:render_count, 1].min())
    raise ValueError(f"unknown measure type {t}")


def measure_body(model: MHModel, positions: np.ndarray) -> dict[str, float]:
    """All measures.json measures in cm on a combined-space body."""
    with open(os.path.join(BODY_DIR, "measures.json"), encoding="utf8") as fh:
        defs = json.load(fh)["measures"]
    return {d["id"]: evaluate_measure(d, positions, model.nr) * 100.0 for d in defs}


# ---------------------------------------------------------------------------------------------------- canonical fit


def canonicalize_fit(model: MHModel, res: Any, *, quantize: bool = True) -> None:
    """Rebuild from browser-compatible net values; bodyfix retains full precision with quantize=False."""
    if quantize:
        res.macro = {k: round(float(res.macro[k]), 6) for k in MACRO_KEYS}
        res.mods = {k: round(float(v), 6) for k, v in res.mods.items() if abs(v) > 1e-6}
    old = res.rest_positions
    pos = model.shape(res.macro, res.mods, ground=False)
    heads = model.rest_heads(pos)
    rot = {b: Rotation.from_rotvec(v).as_matrix() for b, v in res.pose_rotvec.items()}
    rots, posed = model.fk(heads, rot, res.root_t)
    res.rest_positions = pos
    res.posed_vertices = model.lbs(pos[: model.nr], model.skin_j, model.skin_w, heads, rots, posed)
    dev = np.linalg.norm(pos[: model.nr] - old[: model.nr], axis=1)
    res.stats["canonical_shape_dev_mm"] = {"mean": float(dev.mean() * 1000), "max": float(dev.max() * 1000)}


# ---------------------------------------------------------------------------------------------------- mapping


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    import trimesh

    return np.asarray(trimesh.Trimesh(verts, faces, process=False).vertex_normals)


def nearest_render_vertices(
    mh_verts: np.ndarray,
    mh_faces: np.ndarray,
    twin_verts: np.ndarray,
    twin_normals: np.ndarray | None,
    k: int = 12,
) -> tuple[np.ndarray, dict[str, float]]:
    """For each twin vertex the nearest MakeHuman render vertex, preferring surfaces that face the same way.

    Among the k nearest candidates the score is `distance + 3 cm * opposing-normal penalty`, so a vertex of the
    scan's sleeve does not map onto the torso behind it. Returns (uint32 indices, stats).
    """
    tree = cKDTree(mh_verts)
    dist, idx = tree.query(twin_verts, k=k)
    score = dist
    if twin_normals is not None:
        mn = _vertex_normals(mh_verts, mh_faces)
        dots = np.einsum("nkj,nj->nk", mn[idx], twin_normals)
        score = dist + 0.03 * np.clip(0.6 - dots, 0.0, None) / 0.6
    pick = np.argmin(score, axis=1)
    sel = idx[np.arange(len(idx)), pick]
    chosen = dist[np.arange(len(idx)), pick]
    stats = {
        "median_cm": float(np.median(chosen) * 100),
        "p99_cm": float(np.percentile(chosen, 99) * 100),
        "max_cm": float(chosen.max() * 100),
    }
    return sel.astype("<u4"), stats


# ---------------------------------------------------------------------------------------------------- export


def write_twin_package(
    out_dir: str,
    model: MHModel,
    res: Any,
    twin_rest_verts: np.ndarray,
    twin_normals: np.ndarray | None,
    ground_shift_y: float,
    source: dict[str, Any],
    *, bodyfix: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Writes twin.json and mh2twin.bin. `twin_rest_verts` (n,3) are the rigged mesh's rest vertices in the frame of
    rigged.glb (feet on y = 0); `ground_shift_y` is the shift that put them there (the fit's rest frame minus it)."""
    shift = np.array([0.0, ground_shift_y, 0.0])
    positions = res.rest_positions - shift  # combined space in the rigged.glb frame
    raw = measure_body(model, positions)
    allowance = {k: CLOTHING_ALLOWANCE_CM.get(k, 0.0) for k in raw}
    measurements = {k: round(raw[k] - allowance[k], 2) for k in raw}
    if bodyfix is not None:
        # Scan-landmark measurements are authoritative; the proxy has a different surface.
        raw = dict(bodyfix["achievedRawCm"])
        allowance = {k: bodyfix["clothingAllowanceCm"].get(k, 0.0) for k in raw}
        measurements = dict(bodyfix["achievedCm"])
    mapping, mstats = nearest_render_vertices(positions[: model.nr], model.faces, twin_rest_verts, twin_normals)
    with open(os.path.join(out_dir, MAPPING_FILE), "wb") as fh:
        fh.write(mapping.tobytes())
    heads = model.rest_heads(res.rest_positions) - shift
    body_min_y = float(positions[: model.nr, 1].min())
    twin = {
        "version": TWIN_VERSION,
        "generator": "dijital-ikiz twin-lab rig",
        "glb": "rigged.glb",
        "source": source,
        "mapping": {
            "file": MAPPING_FILE,
            "format": "uint32le",
            "twinVertexCount": int(len(twin_rest_verts)),
            "renderVertexCount": int(model.nr),
        },
        "boneOrder": list(model.bone_names),
        "fittedMacros": ({k: float(v) for k, v in res.macro.items()} if bodyfix is not None
                         else {k: float(res.macro[k]) for k in MACRO_KEYS}),
        "fittedModifiers": {k: float(v) for k, v in sorted(res.mods.items())},
        "measurementsRawCm": raw if bodyfix is not None else {k: round(v, 2) for k, v in raw.items()},
        "clothingAllowanceCm": allowance,
        "measurementsCm": measurements,
        "restHeadsM": {n: [round(float(x), 6) for x in heads[i]] for i, n in enumerate(model.bone_names)},
        "fit": {
            "bodyLowestYM": round(body_min_y, 4),
            "twinHeightM": round(float(twin_rest_verts[:, 1].max() - twin_rest_verts[:, 1].min()), 4),
            "mappingCm": {k: round(v, 3) for k, v in mstats.items()},
            **{k: v for k, v in res.stats.items() if k in ("template_to_scan_cm", "scan_to_template_cm", "canonical_shape_dev_mm")},
        },
    }
    if bodyfix is not None:
        twin["bodyfix"] = bodyfix
    with open(os.path.join(out_dir, "twin.json"), "w", encoding="utf8", newline="\n") as fh:
        json.dump(twin, fh, indent=1)
    return twin
