"""Make a NON-personal stand-in scan from our MakeHuman body.

Different body (macros + modifiers), different arm/leg pose than the template rest pose, re-meshed (welded +
subdivided, no skin, no UV), smooth low-frequency noise. Ground truth (macro/mods/pose/joints) goes to truth.json.
Usage: python make_standin.py [out_dir]   (default .cache/standin)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from glbio import GlbScene, Prim, write_static_glb
from mh import MHModel

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    hard = "--hard" in sys.argv
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_dir = argv[0] if argv else os.path.join(HERE, ".cache", "standin_hard" if hard else "standin")
    os.makedirs(out_dir, exist_ok=True)
    m = MHModel()
    macro = {"gender": 0.8, "muscle": 0.62, "weight": 0.66, "height": 0.55}
    mods = {
        "measure/measure-waist-circ": 0.4,
        "torso/torso-scale-horiz": 0.2,
        "measure/measure-upperarm-length": 0.3,
        "armslegs/upperleg-fat": 0.3,
        "hip/hip-scale-horiz": -0.2,
        "measure/measure-shoulder-dist": 0.3,
    }
    pos = m.shape(macro, mods)
    heads = m.rest_heads(pos)
    # pose: arms raised ~14 deg more than the template A-pose, elbows straighter, legs spread a little, slight twist
    pose_deg = {
        "upperarm_l": [0, 0, 26 if hard else 14],
        "upperarm_r": [0, 0, -26 if hard else -14],
        "lowerarm_l": [0, 0, 8],
        "lowerarm_r": [0, 0, -8],
        "thigh_l": [0, 0, -3],
        "thigh_r": [0, 0, 3],
        "spine_02": [0, 3, 0],
    }
    local = {k: R.from_euler("xyz", v, degrees=True).as_matrix() for k, v in pose_deg.items()}
    rots, posed = m.fk(heads, local)
    v = m.lbs(pos[: m.nr], m.skin_j, m.skin_w, heads, rots, posed)
    tm = trimesh.Trimesh(v, m.faces, process=False)
    tm.merge_vertices(digits_vertex=5)
    verts, faces = trimesh.remesh.subdivide(tm.vertices, tm.faces)
    rng = np.random.default_rng(7)
    # smooth noise: sum of a few low-frequency sinusoids, +-6 mm, along the normal
    tm2 = trimesh.Trimesh(verts, faces, process=False)
    nrm = tm2.vertex_normals
    noise = np.zeros(len(verts))
    for _ in range(6):
        k = rng.normal(size=3) * 9.0
        noise += np.sin(verts @ k + rng.uniform(0, 6.28))
    verts = verts + nrm * (0.006 * noise / 6.0 * 2.0)[:, None]
    if hard:
        # scan-like defects: fused (smoothed) fingers, a hair cap, baggy clothes on legs/torso, stronger noise
        tm3 = trimesh.Trimesh(verts, faces, process=False)
        tm3 = trimesh.smoothing.filter_taubin(tm3, iterations=25)
        verts = np.array(tm3.vertices)
        nrm = tm3.vertex_normals
        top = verts[:, 1].max()
        head = (verts[:, 1] > top - 0.13) & (np.abs(verts[:, 0]) < 0.12)
        verts = verts + nrm * (0.03 * np.clip((verts[:, 1] - (top - 0.13)) / 0.13, 0, 1) * head)[:, None]
        cloth = (verts[:, 1] < 0.95) & (verts[:, 1] > 0.08) & (np.abs(verts[:, 0]) < 0.3)
        torso = (verts[:, 1] > 0.95) & (verts[:, 1] < 1.45) & (np.abs(verts[:, 0]) < 0.22)
        verts = verts + nrm * (0.02 * cloth + 0.015 * torso)[:, None]
    shift = float(verts[:, 1].min())
    verts[:, 1] -= shift
    scene = GlbScene(prims=[Prim(verts.astype(np.float32), faces.astype(np.uint32), tm2.vertex_normals.astype(np.float32), None, 0, "standin")])
    write_static_glb(os.path.join(out_dir, "mesh.glb"), scene)
    truth = {
        "macro": macro,
        "mods": mods,
        "pose_deg": pose_deg,
        "ground_shift_y": float(tm.vertices[:, 1].min()),
        "posed_heads": {n: (posed[i] - [0, shift, 0]).tolist() for i, n in enumerate(m.bone_names)},
        "rest_heads": {n: heads[i].tolist() for i, n in enumerate(m.bone_names)},
        "vertices": int(len(verts)),
        "height": float(verts[:, 1].max()),
    }
    json.dump(truth, open(os.path.join(out_dir, "truth.json"), "w"), indent=1)
    print("stand-in", len(verts), "verts", len(faces), "tris, height", truth["height"])


if __name__ == "__main__":
    main()
