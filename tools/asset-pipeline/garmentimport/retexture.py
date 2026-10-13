"""Re-use the material of an already shipped garment GLB on a freshly imported geometry.

Re-importing a garment keeps its UV layout (same source, same atlas arguments), so an artist-made texture set (for
example the procedural cotton jersey of the shipped tee) can be carried over instead of the source texture:

    python -m garmentimport.retexture NEW.glb DONOR.glb OUT.glb

The geometry accessors (positions, normals, UVs, tangents, indices) come from NEW; samplers, images, textures and
the material (factors, normal scale, double-sided flag) come from DONOR, byte for byte.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pygltflib as gl


def transplant_material(geometry: Path, donor: Path, output: Path) -> dict:
    new, old = gl.GLTF2().load_binary(str(geometry)), gl.GLTF2().load_binary(str(donor))
    new_blob, old_blob = new.binary_blob(), old.binary_blob()
    if len(new.meshes) != 1 or len(new.meshes[0].primitives) != 1 or len(old.meshes) != 1:
        raise ValueError("Both GLBs must contain exactly one single-primitive mesh (the importer's layout)")
    blob = bytearray()
    views = []

    def view(data, target=None):
        blob.extend(b"\0"*((-len(blob)) % 4))
        views.append(gl.BufferView(buffer=0, byteOffset=len(blob), byteLength=len(data), target=target))
        blob.extend(data)
        return len(views)-1

    # Geometry: copy every buffer view an accessor uses, in accessor order, and remap.
    mapping = {}
    for accessor in new.accessors:
        source = new.bufferViews[accessor.bufferView]
        if accessor.bufferView not in mapping:
            start = source.byteOffset or 0
            mapping[accessor.bufferView] = view(new_blob[start:start+source.byteLength], source.target)
        accessor.bufferView = mapping[accessor.bufferView]
    # Material: images from the donor, embedded again.
    image_views = {}
    images = []
    for image in old.images:
        if image.bufferView is None:
            raise ValueError("Donor GLB images must be embedded in its binary buffer")
        if image.bufferView not in image_views:
            source = old.bufferViews[image.bufferView]
            start = source.byteOffset or 0
            image_views[image.bufferView] = view(old_blob[start:start+source.byteLength])
        images.append(gl.Image(bufferView=image_views[image.bufferView], mimeType=image.mimeType))
    new.bufferViews = views
    new.images = images
    new.textures = copy.deepcopy(old.textures)
    new.samplers = copy.deepcopy(old.samplers)
    new.materials = copy.deepcopy(old.materials)
    new.meshes[0].primitives[0].material = 0
    blob.extend(b"\0"*((-len(blob)) % 4))
    new.buffers = [gl.Buffer(byteLength=len(blob))]
    new.set_binary_blob(bytes(blob))
    output.parent.mkdir(parents=True, exist_ok=True)
    new.save_binary(str(output))
    return {"images": len(images), "bytes": output.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("donor", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(transplant_material(args.geometry, args.donor, args.output))
