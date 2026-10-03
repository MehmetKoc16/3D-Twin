"""Synthetic inputs built around the real CC0 MakeHuman template: a body solution and a FLAME-like fit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from hybridbody.headfit import FlameFit
from scipy.spatial.transform import Rotation


def bodyfix_solution(model):
    """A valid dtBodyfix dict (the rig stage schema) for a synthetic body; no personal measurements."""
    from twin_export import measure_body

    macro = {"gender": 0.7, "muscle": 0.4, "weight": 0.55, "height": 0.6}
    mods = {"measure/measure-upperarm-length": 0.2, "head/head-scale-horiz": 0.05}
    raw = measure_body(model, model.shape(macro, mods, ground=False))
    allowance = {key: 0.5 for key in raw}
    achieved = {key: value - allowance[key] for key, value in raw.items()}
    targets = {"height": achieved["height"], "neck": achieved["neck"]}
    return {
        "version": 1,
        "fittedMacros": macro,
        "fittedModifiers": mods,
        "achievedRawCm": raw,
        "achievedCm": achieved,
        "targetsCm": targets,
        "residualsCm": {key: achieved[key] - value for key, value in targets.items()},
        "clothingAllowanceCm": allowance,
        "measurementBasis": "synthetic",
    }


def write_bodyfix_glb(path: Path, model, solution: dict):
    """A GLB that only carries the dtBodyfix extras (the hybrid stage reads the solution, not the scan)."""
    from glbio import GlbScene, Prim, write_static_glb

    positions = model.shape(solution["fittedMacros"], solution["fittedModifiers"], ground=True)[: model.nr]
    prim = Prim(positions.astype(np.float32), model.faces.astype(np.uint32))
    write_static_glb(str(path), GlbScene([prim], extras={"dtBodyfix": solution}))


def mh_flame(template, face_map, folder, *, scale=0.97, euler=(4.0, 0.0, 0.0), shift=(0.0, -1.55, 0.12)):
    """A FLAME-like fit built from the MakeHuman head: widened, shifted, with eyeball spheres and region masks."""
    rotation = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
    translation = np.array(shift)
    base = template.base
    head = base[:, 1] > template.anchor_y - 0.04
    centre = base[head].mean(0)
    target = base.copy()
    target[:, 0] = (base[:, 0] - centre[0]) * 1.04 + centre[0]
    target[:, 2] += 0.004 * np.clip((base[:, 1] - template.anchor_y) / 0.2, 0, 1)

    def to_flame(p):
        return ((p - translation) / scale) @ rotation

    keep = np.flatnonzero(head)
    remap = np.full(len(base), -1)
    remap[keep] = np.arange(len(keep))
    faces = remap[template.faces_w[head[template.faces_w].all(1)]]
    vertices = to_flame(target[keep])
    eye_masks = {}
    top = target[keep][:, 1].max()
    for key, x in (("left_eyeball", 0.033), ("right_eyeball", -0.033)):
        ball = trimesh.creation.icosphere(subdivisions=2, radius=0.0125)
        eye_masks[key] = len(vertices) + np.arange(len(ball.vertices))
        placed = to_flame(np.asarray(ball.vertices) + np.array([x, top - 0.115, 0.131]))
        faces = np.vstack((faces, np.asarray(ball.faces) + eye_masks[key][0]))
        vertices = np.vstack((vertices, placed))
    wanted = [i for i in face_map["landmarks"] if all(head[template.inverse[t]] for t in i["tri"])][:105]
    points = np.array([np.asarray(i["bary"]) @ to_flame(target[template.inverse[i["tri"]]]) for i in wanted])
    ids = np.array([i["index"] for i in wanted])
    local = target[keep]
    masks = {
        k: np.zeros(0, np.int64)
        for k in (
            "nose",
            "lips",
            "eye_region",
            "forehead",
            "left_eye_region",
            "right_eye_region",
            "left_ear",
            "right_ear",
            "boundary",
        )
    }
    masks.update(eye_masks)
    masks["face"] = np.flatnonzero((local[:, 2] > 0.07) & (local[:, 1] > template.anchor_y + 0.01))
    masks["scalp"] = np.flatnonzero(local[:, 1] > top - 0.05)
    masks["neck"] = np.flatnonzero(local[:, 1] < template.anchor_y + 0.02)
    return FlameFit(vertices, faces, masks, ids, points, Path(folder), Path(folder))
