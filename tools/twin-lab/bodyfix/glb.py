"""Update geometry while retaining the original GLB texture, UVs and metadata."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import trimesh

from glbio import _accessor, _node_matrix, _split


class Document:
    def __init__(self, path: Path) -> None:
        self.json, self.binary = _split(path.read_bytes())
        js = self.json
        if js.get("skins") or js.get("animations"):
            raise ValueError("bodyfix requires a static scan before rigging")
        if len(js.get("buffers", [])) != 1 or "uri" in js["buffers"][0]:
            raise ValueError("bodyfix requires an embedded single-buffer GLB")
        if any("uri" in image for image in js.get("images", [])):
            raise ValueError("scan textures must be embedded")
        self.parts = []
        seen = set()

        def visit(index: int, parent: np.ndarray) -> None:
            node = js["nodes"][index]
            world = parent @ _node_matrix(node)
            if "mesh" in node:
                if node["mesh"] in seen:
                    raise ValueError("instanced meshes are not supported")
                seen.add(node["mesh"])
                for prim in js["meshes"][node["mesh"]]["primitives"]:
                    if prim.get("mode", 4) != 4 or prim.get("targets") or prim.get("extensions"):
                        raise ValueError("bodyfix requires uncompressed triangle primitives without morph targets")
                    pos = _accessor(js, self.binary, prim["attributes"]["POSITION"])
                    if js["accessors"][prim["attributes"]["POSITION"]].get("sparse"):
                        raise ValueError("sparse positions are not supported")
                    positions = pos @ world[:3, :3].T + world[:3, 3]
                    faces = (_accessor(js, self.binary, prim["indices"]).reshape(-1, 3)
                             if "indices" in prim else np.arange(len(pos)).reshape(-1, 3))
                    self.parts.append((prim, world, positions, faces.astype(np.int64)))
            for child in node.get("children", []):
                visit(child, world)

        scene = js.get("scene", 0)
        for root in js["scenes"][scene]["nodes"]:
            visit(root, np.eye(4))
        if not self.parts:
            raise ValueError("scan contains no triangle geometry")
        self.vertices = np.concatenate([part[2] for part in self.parts])
        offsets = np.cumsum([0] + [len(part[2]) for part in self.parts[:-1]])
        self.faces = np.concatenate([part[3] + offset for part, offset in zip(self.parts, offsets)])
        if not np.isfinite(self.vertices).all() or np.ptp(self.vertices[:, 1]) < 0.1:
            raise ValueError("scan must have finite metre-scale, Y-up geometry")

    def write(self, path: Path, vertices: np.ndarray) -> None:
        js = self.json
        binary = bytearray(self.binary)

        def add(values: np.ndarray, bounds: bool = False) -> int:
            binary.extend(b"\0" * (-len(binary) % 4))
            view = len(js["bufferViews"])
            data = np.asarray(values, dtype="<f4").tobytes()
            js["bufferViews"].append({"buffer": 0, "byteOffset": len(binary),
                                      "byteLength": len(data), "target": 34962})
            binary.extend(data)
            accessor = {"bufferView": view, "componentType": 5126, "count": len(values), "type": "VEC3"}
            if bounds:
                accessor.update(min=values.min(axis=0).tolist(), max=values.max(axis=0).tolist())
            index = len(js["accessors"])
            js["accessors"].append(accessor)
            return index

        offset = 0
        for prim, world, old, faces in self.parts:
            pos = vertices[offset:offset + len(old)]
            offset += len(old)
            local = (pos - world[:3, 3]) @ np.linalg.inv(world[:3, :3]).T
            normal = trimesh.Trimesh(local, faces, process=False).vertex_normals
            prim["attributes"]["POSITION"] = add(local, True)
            prim["attributes"]["NORMAL"] = add(normal)
            # Existing tangents describe the old surface; regenerate them downstream if needed.
            prim["attributes"].pop("TANGENT", None)
        js["buffers"][0]["byteLength"] = len(binary)
        encoded = json.dumps(js, separators=(",", ":"), allow_nan=False).encode("utf8")
        encoded += b" " * (-len(encoded) % 4)
        binary.extend(b"\0" * (-len(binary) % 4))
        data = (struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(binary))
                + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
                + struct.pack("<II", len(binary), 0x004E4942) + binary)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
