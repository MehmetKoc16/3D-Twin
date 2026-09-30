"""Create a stand-in body mesh for end-to-end validation when the real shape output is not available.

Takes the MakeHuman base body (apps/web/public/assets/body/base.glb, +Y up, +Z front, feet at y=0), lowers the arms
towards the torso (crude rigid rotation about the shoulders) so its silhouette is closer to a person standing with
arms down, and writes ``.cache/standin_mesh.glb`` (geometry only, no UV, no texture).
NOT personal data: it is the generic CC0 base body.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

from twintex import meshio

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2] / "apps" / "web" / "public" / "assets" / "body" / "base.glb"


def make(angle_deg: float = 26.0, shoulder_x: float = 0.2, shoulder_y: float = 1.38) -> trimesh.Trimesh:
    m = meshio.clean_mesh(meshio.load_mesh(BASE))
    v = m.vertices.copy()
    for side in (+1.0, -1.0):
        ax = side * v[:, 0]
        w = np.clip((ax - shoulder_x) / 0.10, 0, 1)
        w = w * w * (3 - 2 * w)
        w = np.where(v[:, 1] > 0.85, w, 0.0)
        th = -side * np.radians(angle_deg) * w  # rotate the +x arm clockwise (hand goes down / in)
        c, s = np.cos(th), np.sin(th)
        dx = v[:, 0] - side * shoulder_x
        dy = v[:, 1] - shoulder_y
        nx = c * dx - s * dy
        ny = s * dx + c * dy
        sel = w > 0
        v[sel, 0] = (side * shoulder_x + nx)[sel]
        v[sel, 1] = (shoulder_y + ny)[sel]
    v[:, 1] -= v[:, 1].min()
    return trimesh.Trimesh(vertices=v, faces=m.faces, process=False)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / ".cache" / "standin_mesh.glb"
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh = make()
    mesh.export(out)
    print(out, mesh.bounds.tolist())
