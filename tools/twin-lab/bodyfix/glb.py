"""Update geometry while retaining the original GLB texture, UVs and metadata."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import scipy.sparse as sp
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

    def write(self, path: Path, vertices: np.ndarray, faces: np.ndarray | None = None,
              transfer: sp.spmatrix | None = None, extras: dict | None = None) -> None:
        """Write new positions. With `faces`/`transfer` the topology changes too (single-primitive scans only):
        `transfer` (new x old) interpolates every per-vertex attribute (UVs, colours) onto the new vertices."""
        js = self.json
        binary = bytearray(self.binary)
        if extras:  # e.g. {"dtScanHandsRemoved": true}: read by rig_scan.py
            js.setdefault("asset", {}).setdefault("extras", {}).update(extras)

        def add(values: np.ndarray, bounds: bool = False, kind: str | None = None) -> int:
            binary.extend(b"\0" * (-len(binary) % 4))
            view = len(js["bufferViews"])
            if kind == "SCALAR":
                data = np.asarray(values, dtype="<u4").tobytes()
                target, component = 34963, 5125
            else:
                data = np.asarray(values, dtype="<f4").tobytes()
                target, component = 34962, 5126
                kind = {2: "VEC2", 3: "VEC3", 4: "VEC4"}[values.shape[1]]
            js["bufferViews"].append({"buffer": 0, "byteOffset": len(binary),
                                      "byteLength": len(data), "target": target})
            binary.extend(data)
            accessor = {"bufferView": view, "componentType": component, "count": len(values), "type": kind}
            if bounds:
                accessor.update(min=values.min(axis=0).tolist(), max=values.max(axis=0).tolist())
            index = len(js["accessors"])
            js["accessors"].append(accessor)
            return index

        if faces is not None:
            if len(self.parts) != 1 or transfer is None or transfer.shape != (len(vertices), len(self.vertices)):
                raise ValueError("topology edits need a single-primitive scan and a matching transfer matrix")
            prim, world, _, _ = self.parts[0]
            local = (vertices - world[:3, 3]) @ np.linalg.inv(world[:3, :3]).T
            for key in list(prim["attributes"]):
                if key in ("POSITION", "NORMAL"):
                    continue
                if key == "TANGENT":
                    del prim["attributes"][key]
                elif key.startswith(("TEXCOORD_", "COLOR_")):
                    values = _accessor(js, self.binary, prim["attributes"][key])
                    prim["attributes"][key] = add(np.asarray(transfer @ values))
                else:
                    raise ValueError(f"cannot edit the topology of a scan with {key}")
            prim["indices"] = add(np.asarray(faces).reshape(-1), kind="SCALAR")
            normal = trimesh.Trimesh(local, faces, process=False).vertex_normals
            prim["attributes"]["POSITION"] = add(local, True)
            prim["attributes"]["NORMAL"] = add(normal)
            self._finish(path, js, binary)
            return

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
        self._finish(path, js, binary)

    @staticmethod
    def _finish(path: Path, js: dict, binary: bytearray) -> None:
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
