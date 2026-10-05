"""Generate a procedural thin-wire accessory, fitted in the twin's rest head frame."""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "bundle"))
from accessory_glb import accessor_array, read_glb, validate_glasses

DEFAULTS = {
    "lensShape": "round",
    "outerRadius": 0.024,
    "bridgeWidth": 0.018,
    "frameWidth": 0.135,
    "thickness": 0.0012,
    "colour": "#b9b4ad",
    "templeLength": 0.14,
    "verticalOffset": 0.0,
    "clearLenses": False,
}


def read_params(path: Path | None) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else {}
    if not isinstance(raw, dict):
        raise ValueError("Params must be a JSON object")
    aliases = {
        "lens_shape": "lensShape",
        "outer_radius": "outerRadius",
        "lensRadius": "outerRadius",
        "lens_radius": "outerRadius",
        "bridge_width": "bridgeWidth",
        "frame_width": "frameWidth",
        "wireThickness": "thickness",
        "wire_thickness": "thickness",
        "color": "colour",
        "temple_length": "templeLength",
        "vertical_offset": "verticalOffset",
        "clear_lenses": "clearLenses",
    }
    params = dict(DEFAULTS)
    for key, value in raw.items():
        canonical = aliases.get(key, key)
        if canonical in params:
            params[canonical] = value
    limits = {
        "outerRadius": (0.012, 0.04),
        "bridgeWidth": (0.006, 0.035),
        "frameWidth": (0.08, 0.19),
        "thickness": (0.0004, 0.004),
        "templeLength": (0.08, 0.20),
        "verticalOffset": (-0.04, 0.04),
    }
    for key, (low, high) in limits.items():
        value = params[key]
        if (
            type(value) not in (int, float)
            or not np.isfinite(value)
            or not low <= value <= high
        ):
            raise ValueError(f"Invalid {key}; expected metres in {low}..{high}")
    if params["lensShape"] != "round":
        raise ValueError("Only round lenses are supported")
    if not isinstance(params["colour"], str) or not re.fullmatch(
        r"#[0-9a-fA-F]{6}", params["colour"]
    ):
        raise ValueError("Colour must be #rrggbb")
    if type(params["clearLenses"]) is not bool:
        raise ValueError("clearLenses must be boolean")
    if params["frameWidth"] <= 4 * params["outerRadius"] + params["bridgeWidth"]:
        raise ValueError("Frame width must exceed both rims plus the bridge")
    params["colour"] = params["colour"].lower()
    return params


def node_worlds(document: dict) -> list[np.ndarray]:
    nodes = document["nodes"]
    parents = {
        child: i for i, node in enumerate(nodes) for child in node.get("children", [])
    }
    cache: dict[int, np.ndarray] = {}

    def world(i: int, active: set[int]) -> np.ndarray:
        if i in cache:
            return cache[i]
        if i in active:
            raise ValueError("Cyclic skeleton")
        node = nodes[i]
        if "matrix" in node:
            local = np.asarray(node["matrix"], dtype=float).reshape(4, 4).T
        else:
            x, y, z, w = node.get("rotation", [0, 0, 0, 1])
            local = np.eye(4)
            local[:3, :3] = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            ) @ np.diag(node.get("scale", [1, 1, 1]))
            local[:3, 3] = node.get("translation", [0, 0, 0])
        cache[i] = world(parents[i], active | {i}) @ local if i in parents else local
        return cache[i]

    return [world(i, set()) for i in range(len(nodes))]


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def landmark_placement(report: dict, inverse: np.ndarray) -> np.ndarray | None:
    """Accept only explicitly 3D/metre landmarks, never image-pixel landmarks."""
    block = report.get("landmarks3d", report.get("faceLandmarksM", {}))
    if not isinstance(block, dict):
        return None

    def point(*names):
        value = next((block[key] for key in names if key in block), None)
        if isinstance(value, dict):
            value = value.get("positionM")
        if not isinstance(value, list) or len(value) != 3:
            return None
        try:
            array = np.asarray(value, dtype=float)
        except (ValueError, TypeError):
            return None
        return array if np.isfinite(array).all() else None

    left = point(
        "leftEye",
        "left_eye",
        "eyeLeft",
        "leftEyeCentre",
        "leftEyeCenter",
        "left_eye_center",
    )
    right = point(
        "rightEye",
        "right_eye",
        "eyeRight",
        "rightEyeCentre",
        "rightEyeCenter",
        "right_eye_center",
    )
    nose = point("noseBridge", "nose_bridge", "noseBridgeM")
    if left is None or right is None:
        return None
    local = block.get("coordinateSpace") == "head-local"
    eyes = (
        np.array([left, right])
        if local
        else transform(np.array([left, right]), inverse)
    )
    centre = eyes.mean(axis=0)
    if nose is not None:
        nose = nose if local else transform(np.array([nose]), inverse)[0]
        centre[0] = nose[0]
        centre[2] = max(centre[2], nose[2])
    centre[2] += 0.004
    return centre


def estimate_placement(twin: Path, report_paths: list[Path]) -> tuple[np.ndarray, str]:
    document, blob = read_glb(twin)
    worlds = node_worlds(document)
    mesh_nodes = [
        (i, node)
        for i, node in enumerate(document["nodes"])
        if "mesh" in node and "skin" in node
    ]
    if len(mesh_nodes) != 1:
        raise ValueError("Twin must have one skinned scan mesh")
    node_index, node = mesh_nodes[0]
    skin = document["skins"][node["skin"]]
    names = [document["nodes"][i].get("name") for i in skin["joints"]]
    if "head" not in names:
        raise ValueError("Twin skin has no head bone")
    head_joint = names.index("head")
    inverse = np.linalg.inv(worlds[skin["joints"][head_joint]])
    for path in report_paths:
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(report, dict):
                centre = landmark_placement(report, inverse)
                if centre is not None:
                    return centre, "landmarks3d"
    positions, selected = [], []
    for primitive in document["meshes"][node["mesh"]]["primitives"]:
        attrs = primitive["attributes"]
        points = accessor_array(document, blob, attrs["POSITION"])
        joints = accessor_array(document, blob, attrs["JOINTS_0"])
        weights = accessor_array(document, blob, attrs["WEIGHTS_0"])
        points = transform(transform(points, worlds[node_index]), inverse)
        mask = np.sum(np.where(joints == head_joint, weights, 0), axis=1) > 0.5
        positions.append(points)
        selected.append(mask)
    points = np.concatenate(positions)
    head = points[np.concatenate(selected)]
    if len(head) < 8:
        # A conservative bone-relative region if skin weights are blended at the neck.
        head = points[
            (points[:, 1] > -0.02)
            & (points[:, 1] < 0.30)
            & (np.abs(points[:, 0]) < 0.13)
        ]
    if len(head) < 8 or not np.isfinite(head).all():
        raise ValueError("Insufficient finite head geometry for placement")
    low, high = np.percentile(head[:, 1], [2, 98])
    eye_y = low + 0.60 * (high - low)
    centre_x = float(np.median(head[:, 0]))
    band = head[
        (np.abs(head[:, 1] - eye_y) < max(0.012, (high - low) * 0.09))
        & (np.abs(head[:, 0] - centre_x) < 0.055)
    ]
    if not len(band):
        band = head[np.argsort(np.abs(head[:, 1] - eye_y))[: max(8, len(head) // 10)]]
    return np.array(
        [centre_x, eye_y, np.percentile(band[:, 2], 95) + 0.004]
    ), "head-geometry"


def tube(path: np.ndarray, radius: float, closed: bool = False, sides: int = 8):
    """Smooth normals on an eight-sided wire; closed rims and capped open arms."""
    path = np.asarray(path, dtype=float)
    tangent = np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)
    if not closed:
        tangent[0], tangent[-1] = path[1] - path[0], path[-1] - path[-2]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    reference = (
        np.array([0.0, 0.0, 1.0])
        if abs(tangent[0, 2]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = np.zeros_like(tangent)
    u[0] = np.cross(tangent[0], reference)
    u[0] /= np.linalg.norm(u[0])
    # Parallel transport prevents a sudden cross-section flip where an arm bends.
    for i in range(1, len(path)):
        u[i] = u[i - 1] - tangent[i] * np.dot(u[i - 1], tangent[i])
        u[i] /= np.linalg.norm(u[i])
    v = np.cross(tangent, u)
    angles = np.arange(sides) * 2 * np.pi / sides
    normals = (
        u[:, None, :] * np.cos(angles)[None, :, None]
        + v[:, None, :] * np.sin(angles)[None, :, None]
    )
    vertices = (path[:, None, :] + radius * normals).reshape(-1, 3)
    normals = normals.reshape(-1, 3)
    triangles = []
    for i in range(len(path) if closed else len(path) - 1):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            c, d = (
                ((i + 1) % len(path)) * sides + j,
                ((i + 1) % len(path)) * sides + (j + 1) % sides,
            )
            triangles.extend([[a, b, c], [b, d, c]])
    if not closed:
        # Separate flat cap vertices avoid smoothing the end faces into the wire.
        for end, sign in ((0, -1), (len(path) - 1, 1)):
            start = len(vertices)
            vertices = np.vstack(
                [vertices, path[end], vertices[end * sides : (end + 1) * sides]]
            )
            normals = np.vstack([normals, np.tile(sign * tangent[end], (sides + 1, 1))])
            for j in range(sides):
                tri = [start, start + 1 + j, start + 1 + (j + 1) % sides]
                triangles.append(tri[::-1] if sign < 0 else tri)
    return vertices, normals, np.asarray(triangles, dtype=np.uint32)


def nose_pad(centre: np.ndarray):
    """Small solid oval with smooth ellipsoid normals, flattened along face depth."""
    sides, stacks = 16, 8
    unit = [[0, 1, 0]]
    for i in range(1, stacks):
        theta = np.pi * i / stacks
        for j in range(sides):
            angle = 2 * np.pi * j / sides
            unit.append(
                [
                    np.sin(theta) * np.cos(angle),
                    np.cos(theta),
                    np.sin(theta) * np.sin(angle),
                ]
            )
    unit.append([0, -1, 0])
    unit = np.array(unit)
    radii = np.array([0.0022, 0.0034, 0.0007])
    normals = unit / radii
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    faces = []
    for j in range(sides):
        faces.append([0, 1 + (j + 1) % sides, 1 + j])
    for i in range(stacks - 2):
        for j in range(sides):
            a, b = 1 + i * sides + j, 1 + i * sides + (j + 1) % sides
            faces.extend([[a, b, a + sides], [b, b + sides, a + sides]])
    last = 1 + (stacks - 2) * sides
    for j in range(sides):
        faces.append([last + j, last + (j + 1) % sides, len(unit) - 1])
    return unit * radii + centre, normals, np.array(faces, dtype=np.uint32)


def frame_geometry(params: dict, centre: np.ndarray):
    r, bridge = params["outerRadius"], params["bridgeWidth"]
    wire, half = params["thickness"] / 2, params["frameWidth"] / 2
    distance = r + bridge / 2
    pieces = []
    angles = np.arange(64) * 2 * np.pi / 64
    for sign in (-1, 1):
        pieces.append(
            tube(
                np.column_stack(
                    [
                        sign * distance + r * np.cos(angles),
                        r * np.sin(angles),
                        np.zeros(64),
                    ]
                ),
                wire,
                True,
            )
        )
        # End pieces and thicker short hinge barrels at the total frame width.
        pieces.append(
            tube(
                np.array([[sign * (distance + r), 0, 0], [sign * half, 0, -0.003]]),
                wire,
            )
        )
        pieces.append(
            tube(
                np.array([[sign * half, -0.002, -0.003], [sign * half, 0.002, -0.003]]),
                wire * 1.5,
            )
        )
        length = params["templeLength"]
        path = [[sign * half, 0, -0.003], [sign * (half + 0.002), 0, -length * 0.65]]
        for t in np.linspace(0, np.pi / 2, 10)[1:]:
            path.append(
                [
                    sign * (half + 0.002),
                    -0.025 * (1 - np.cos(t)),
                    -length * 0.65 - length * 0.35 * np.sin(t),
                ]
            )
        pieces.append(tube(np.array(path), wire))
        # Short pad stalk and solid flattened oval nose pad.
        pieces.append(
            tube(
                np.array(
                    [
                        [sign * bridge / 2, -0.005, 0],
                        [sign * (bridge / 2 - 0.002), -0.011, -0.005],
                    ]
                ),
                wire,
            )
        )
        pieces.append(nose_pad(np.array([sign * (bridge / 2 - 0.002), -0.011, -0.005])))
    t = np.linspace(0, np.pi, 20)
    pieces.append(
        tube(
            np.column_stack([-bridge / 2 * np.cos(t), 0.006 * np.sin(t), np.zeros(20)]),
            wire,
        )
    )
    vertices, normals, triangles, offset = [], [], [], 0
    for p, n, f in pieces:
        vertices.append(p + centre)
        normals.append(n)
        triangles.append(f + offset)
        offset += len(p)
    return np.vstack(vertices), np.vstack(normals), np.vstack(triangles)


def encode_glasses(
    params: dict,
    centre: np.ndarray,
    method: str,
    *,
    geometry=None,
    lenses=None,
    placement: dict | None = None,
) -> bytes:
    """Encode default wires or a measured fit, keeping the same rigid head skin.

    ``geometry`` and ``lenses`` are (positions, normals, triangles) in head-local
    metres. Fits can use ellipse rims and ear-guided temples without changing the
    accessory/runtime contract. Materials are texture-free standard glTF PBR.
    """
    document = {
        "asset": {
            "version": "2.0",
            "extras": {
                "dtAccessory": {
                    "id": "glasses",
                    "bone": "head",
                    "coordinateSpace": "head-local",
                    "params": params,
                    "placement": {"method": method, **(placement or {})},
                }
            },
        },
        "scenes": [{"nodes": [0, 1]}],
        "scene": 0,
        "nodes": [{"name": "head"}, {"name": "glasses", "mesh": 0, "skin": 0}],
        "bufferViews": [],
        "accessors": [],
    }
    blob = bytearray()

    def add(array, component, kind, bounds=False):
        array = np.asarray(
            array, dtype={5126: "<f4", 5123: "<u2", 5125: "<u4"}[component]
        )
        blob.extend(b"\0" * (-len(blob) % 4))
        view = len(document["bufferViews"])
        document["bufferViews"].append(
            {"buffer": 0, "byteOffset": len(blob), "byteLength": array.nbytes}
        )
        blob.extend(array.tobytes())
        accessor = {
            "bufferView": view,
            "componentType": component,
            "count": len(array),
            "type": kind,
        }
        if bounds:
            accessor.update(
                min=array.min(axis=0).tolist(), max=array.max(axis=0).tolist()
            )
        document["accessors"].append(accessor)
        return len(document["accessors"]) - 1

    def primitive(p, n, f, material):
        return {
            "attributes": {
                "POSITION": add(p, 5126, "VEC3", True),
                "NORMAL": add(n, 5126, "VEC3"),
                "JOINTS_0": add(np.zeros((len(p), 4)), 5123, "VEC4"),
                "WEIGHTS_0": add(np.tile([1, 0, 0, 0], (len(p), 1)), 5126, "VEC4"),
            },
            "indices": add(f.ravel(), 5125, "SCALAR"),
            "material": material,
        }

    p, n, f = frame_geometry(params, centre) if geometry is None else geometry
    primitives = [primitive(p, n, f, 0)]
    # glTF baseColorFactor is linear RGB; the supplied hex is sRGB.
    colour = np.array([int(params["colour"][i : i + 2], 16) / 255 for i in (1, 3, 5)])
    colour = np.where(
        colour <= 0.04045, colour / 12.92, ((colour + 0.055) / 1.055) ** 2.4
    )
    document["materials"] = [
        {
            "name": "thin metal",
            "pbrMetallicRoughness": {
                "baseColorFactor": [*colour.tolist(), 1],
                "metallicFactor": 1,
                "roughnessFactor": 0.3,
            },
        }
    ]
    if params["clearLenses"]:
        document["materials"].append(
            {
                "name": "subtle clear lenses",
                "alphaMode": "BLEND",
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.9, 0.96, 1, 0.035],
                    "metallicFactor": 0,
                    "roughnessFactor": 0.08,
                },
            }
        )
        points, faces = [], []
        r = params["outerRadius"] - params["thickness"] / 2
        for sign in (-1, 1):
            c = centre + [
                sign * (params["outerRadius"] + params["bridgeWidth"] / 2),
                0,
                -0.0002,
            ]
            offset = len(points)
            points.append(c)
            for angle in np.arange(64) * 2 * np.pi / 64:
                points.append(c + [r * np.cos(angle), r * np.sin(angle), 0])
            faces.extend(
                [[offset, offset + 1 + i, offset + 1 + (i + 1) % 64] for i in range(64)]
            )
        p = np.array(points)
        lens_geometry = (
            (p, np.tile([0, 0, 1], (len(p), 1)), np.array(faces))
            if lenses is None
            else lenses
        )
        primitives.append(primitive(*lens_geometry, 1))
    document["meshes"] = [{"name": "glasses", "primitives": primitives}]
    document["skins"] = [
        {
            "joints": [0],
            "skeleton": 0,
            "inverseBindMatrices": add(np.eye(4).reshape(1, 16), 5126, "MAT4"),
        }
    ]
    document["buffers"] = [{"byteLength": len(blob)}]
    encoded = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
    encoded += b" " * (-len(encoded) % 4)
    blob.extend(b"\0" * (-len(blob) % 4))
    raw = (
        struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(blob))
        + struct.pack("<I4s", len(encoded), b"JSON")
        + encoded
        + struct.pack("<I4s", len(blob), b"BIN\0")
        + blob
    )
    validate_glasses(raw)
    return raw


def make_glasses(
    twin: Path, out: Path, params_path: Path | None = None, report: Path | None = None
) -> None:
    inputs = (
        [twin] + ([params_path] if params_path else []) + ([report] if report else [])
    )
    if out.resolve() in [p.resolve() for p in inputs]:
        raise ValueError("Output must not overwrite an input")
    private = LAB.parents[1] / "user-data"
    if any(
        p.resolve().is_relative_to(private) for p in inputs
    ) and not out.resolve().is_relative_to(private):
        raise ValueError("Personal output must stay under user-data/")
    params = read_params(params_path)
    reports = (
        [report]
        if report
        else [
            twin.parent / "refine/refine_report.json",
            twin.parent.parent / "refine/refine_report.json",
            twin.parent / "refine/landmarks3d.json",
        ]
    )
    centre, method = estimate_placement(twin, reports)
    centre[1] += params["verticalOffset"]
    raw = encode_glasses(params, centre, method)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path)
    parser.add_argument("--twin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--refine-report", type=Path)
    args = parser.parse_args()
    try:
        make_glasses(args.twin, args.out, args.params, args.refine_report)
    except (
        OSError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        np.linalg.LinAlgError,
    ):
        print(
            "Glasses failed: invalid input or insufficient head geometry; inspect inputs locally."
        )
        return 1
    print("Glasses validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
