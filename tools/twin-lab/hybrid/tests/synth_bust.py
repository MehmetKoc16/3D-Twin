"""A synthetic Hunyuan3D-style bust built from the generic CC0 MakeHuman head: no person, no photo, no private data.

The bust is the generic head with a dark hair region (volume on top, short at the back), skin colour elsewhere, scaled
(``SCALE`` bust units per metre), turned a little, written Z-up with a +90 degree node rotation like the real generator's
glTF and with a base colour texture painted per texel from the 3-D position. Landmarks are not detected by MediaPipe (a
head without eyes is not a face for it): ``fake_detector`` returns the known landmark positions in the bust's frame.
"""

from __future__ import annotations

import functools
import io
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from glbio import GlbScene, Prim, _split, write_static_glb
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from twintex.raster import rasterize_uv

from hybridbody import BODY_ASSETS, PARTS_ASSETS
from hybridbody.hairdemo import generic_head
from hybridbody.hairgen import HairStyle, HeadSurface, measure_head
from hybridbody.headfit import read_face_map
from hybridbody.register import smoothstep
from hybridbody.template import load_part, weld

SCALE = 1.7  # bust units per metre (the real bust is about 1.7 times too big)
YAW_DEG = 4.0
SHIFT = np.array([0.2, -0.3, 0.1])
HAIR_RGB = (28, 30, 36)
SKIN_RGB = (205, 160, 130)
TEXTURE_SIZE = 256


@dataclass
class Kit:
    positions: np.ndarray  # (n, 3) twin head render vertices (metres)
    faces: np.ndarray
    uv: np.ndarray
    eye_y: float
    landmark_points: np.ndarray  # (468, 3) twin head
    landmark_index: np.ndarray  # (468,) MediaPipe ids
    frame: object  # HeadFrame
    model: object
    combined: np.ndarray


def hair_region(points: np.ndarray, frame, top: float = 0.035) -> np.ndarray:
    """Where the synthetic head has hair: the top (above ``top`` over the eye line) and a short back, never near the ears."""
    az = np.abs(np.degrees(np.arctan2(points[:, 0] - frame.x0, points[:, 2] - frame.zc)))
    above = points[:, 1] - frame.eye_y
    inside = (above > top) | ((az > 100) & (above > -0.04))
    ear = cKDTree(frame.ear_points).query(points)[0] > 0.014
    return inside & ear


@functools.cache
def build_kit() -> Kit:
    model, combined, _, head = generic_head(1.0)
    body = combined[: model.nr]
    positions = body[head.ids]
    eye_y = float(load_part(PARTS_ASSETS, "eyes-default").bind(combined)[:, 1].mean())
    surface = HeadSurface(positions, head.faces)
    frame = measure_head(surface, eye_y, HairStyle())
    face_map = read_face_map(BODY_ASSETS / "face-map.json")
    lm = np.array([np.asarray(i["bary"]) @ combined[i["tri"]] for i in face_map["landmarks"]])
    index = np.array([i["index"] for i in face_map["landmarks"]])
    return Kit(positions, head.faces, model.uv[head.ids], eye_y, lm, index, frame, model, combined)


def to_bust_frame(points: np.ndarray, kit: Kit) -> np.ndarray:
    """Twin frame (metres) -> the bust's world frame (+Y up, +Z front, ``SCALE`` units per metre, turned and shifted)."""
    centre = kit.positions.mean(0)
    turn = Rotation.from_euler("y", YAW_DEG, degrees=True).as_matrix()
    return ((points - centre) @ turn.T) * SCALE + SHIFT


def bust_geometry(kit: Kit, top: float = 0.035):
    """Bust vertices in its own frame (hair volume added), vertex normals, per-vertex hair flag."""
    surface = HeadSurface(kit.positions, kit.faces)
    inverse, _ = weld(kit.positions)
    normals = surface.normals[inverse]
    hair = hair_region(kit.positions, kit.frame, top)
    thickness = 0.003 + 0.022 * smoothstep((kit.positions[:, 1] - (kit.eye_y + 0.05)) / 0.04)
    moved = kit.positions + normals * (thickness * hair)[:, None]
    return moved, normals, hair


def paint_texture(kit: Kit, size: int = TEXTURE_SIZE, top: float = 0.035) -> np.ndarray:
    """Colour texture painted per texel from the 3-D position: dark hair where the region says so, skin elsewhere."""
    rng = np.random.default_rng(1)
    image = np.zeros((size, size, 3), np.uint8)
    image[:] = SKIN_RGB
    face_id, bary = rasterize_uv(kit.uv * size, kit.faces, size, size)
    ys, xs = np.nonzero(face_id >= 0)
    points = np.einsum("ij,ijk->ik", bary[ys, xs], kit.positions[kit.faces[face_id[ys, xs]]])
    hair = hair_region(points, kit.frame, top)
    noise = rng.normal(0, 3, (len(ys), 3))
    colour = np.where(hair[:, None], np.array(HAIR_RGB), np.array(SKIN_RGB)) + noise
    image[ys, xs] = np.clip(colour, 0, 255).astype(np.uint8)
    return image


def png(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def write_bust(path: Path, kit: Kit | None = None, *, normal_map: bool = True, top: float = 0.035) -> Path:
    """Write the synthetic bust like the generator does: Z-up raw geometry under a +90 degree X rotation of the node."""
    kit = kit or build_kit()
    moved, normals, _ = bust_geometry(kit, top)
    world = to_bust_frame(moved, kit)
    world_normals = normals @ Rotation.from_euler("y", YAW_DEG, degrees=True).as_matrix().T
    raw = np.stack([world[:, 0], world[:, 2], -world[:, 1]], axis=1)  # raw = R_x(-90) world: Z-up
    raw_normals = np.stack([world_normals[:, 0], world_normals[:, 2], -world_normals[:, 1]], axis=1)
    prim = Prim(
        raw.astype(np.float32),
        kit.faces.astype(np.uint32),
        raw_normals.astype(np.float32),
        kit.uv.astype(np.float32),
        0,
        "node_0",
    )
    material = {"name": "Material.001", "doubleSided": True, "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
    images = [{"data": png(paint_texture(kit, top=top)), "mimeType": "image/png"}]
    textures = [{"sampler": 0, "source": 0}]
    if normal_map:
        flat = np.zeros((64, 64, 3), np.uint8)
        flat[:] = (128, 128, 255)
        images.append({"data": png(flat), "mimeType": "image/png"})
        textures.append({"sampler": 0, "source": 1})
        material["normalTexture"] = {"index": 1}
    scene = GlbScene([prim], [material], textures, [{"magFilter": 9729, "minFilter": 9987}], images)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_static_glb(str(path), scene)
    rotate_node(path)
    return path


def rotate_node(path: Path) -> None:
    """Add the generator's +90 degree rotation about X to the first node of a GLB (the writer has no node transforms)."""
    data = Path(path).read_bytes()
    document, blob = _split(data)
    half = float(np.sqrt(0.5))
    document["nodes"][0]["rotation"] = [half, 0.0, 0.0, half]
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    padded = blob + b"\0" * (-len(blob) % 4)
    Path(path).write_bytes(
        struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(padded))
        + struct.pack("<I4s", len(encoded), b"JSON")
        + encoded
        + struct.pack("<I4s", len(padded), b"BIN\0")
        + padded
    )


def fake_detector(kit: Kit | None = None, *, upright_only: bool = False):
    """A stand-in for ``hy3d.detect_front_landmarks``: the known landmarks in the bust's frame (478 points, all valid).

    With ``upright_only`` it behaves like a face detector that only finds the face when the bust is +Y up and +Z front
    (the nose is the highest-z landmark and the forehead is above the chin); otherwise it always answers.
    """
    kit = kit or build_kit()
    bust_points = to_bust_frame(kit.landmark_points, kit)
    points = np.zeros((478, 3))
    for row, i in enumerate(kit.landmark_index):
        points[i] = bust_points[row]
    valid = np.zeros(478, bool)
    valid[kit.landmark_index] = True
    valid[468:] = True
    points[468:] = bust_points.mean(0)

    def detect(bust, size=1024, region=None):
        if upright_only:
            spread = np.ptp(bust.positions, axis=0)
            top = bust.positions[np.argmax(bust.positions[:, 1])]
            if spread[1] < spread[2] * 0.9 or top[1] < bust.positions[:, 1].mean():
                return None
        return points, valid

    return detect
