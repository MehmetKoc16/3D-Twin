"""Convert authored MHCLO data; helper vertices are evaluated before runtime rebinding."""
from __future__ import annotations

import json
import numpy as np

import mhclo
import mh_obj
from config import MH_DATA
from mh_morph import MhMorpher
from targets import TargetPacker, build_catalog
from .geometry import binding, reconstruct, scale_refs
from .mesh_io import Part, load_parts


def convert_records(cl, full_body, first, body, faces):
    """Keep authored body indices; fold extrapolation into world-axis offsets exactly at neutral."""
    if not cl.count or cl.indices.min() < 0 or cl.indices.max() >= len(full_body):
        raise ValueError("MHCLO indices outside the hm08 base")
    if not np.isfinite(cl.weights).all() or not np.isfinite(cl.offsets).all():
        raise ValueError("Nonfinite MHCLO records")
    if not np.allclose(cl.weights.sum(axis=1), 1, atol=.005):
        raise ValueError("MHCLO weights must sum to one")
    if set(cl.scale) != set("xyz"):
        raise ValueError("MHCLO needs x/y/z scale references; legacy shear bindings need conversion")
    for a, b, distance in cl.scale.values():
        if min(a, b) < 0 or max(a, b) >= len(full_body) or not np.isfinite(distance) or distance <= 0:
            raise ValueError("Invalid MHCLO scale reference")
    scales = mhclo.axis_scales(cl.scale, full_body, unit=.1)
    if np.any(scales < 1e-8):
        raise ValueError("Degenerate MHCLO axis scale")
    points = mhclo.reconstruct(cl.indices, cl.weights, cl.offsets*.1, scales, full_body)
    helper = (cl.indices >= len(first)).any(axis=1)
    refs = {k: [int(first[a]), int(first[b]), float(distance*.1)]
            for k, (a, b, distance) in cl.scale.items()} if all(
                max(a, b) < len(first) for a, b, _ in cl.scale.values()) else scale_refs(body)
    target_scales = np.array([abs(body[refs[k][0], i]-body[refs[k][1], i])/refs[k][2]
                              for i, k in enumerate("xyz")])
    indices, weights, offsets = binding(points, body, faces)
    direct = ~helper
    indices[direct] = first[cl.indices[direct]]
    clipped = np.maximum(cl.weights[direct], 0)
    if np.any(clipped.sum(axis=1) <= 0):
        raise ValueError("MHCLO has a record with no positive influence")
    weights[direct] = clipped/clipped.sum(axis=1, keepdims=True)
    offsets = (points-(body[indices]*weights[..., None]).sum(axis=1))/target_scales
    result = indices, weights, offsets
    error = np.linalg.norm(reconstruct(result, body, refs)-points, axis=1).max()*1000
    return points, result, refs, {"mode": "authored MHCLO", "authoredBodyRows": int(direct.sum()),
        "helperReboundRows": int(helper.sum()), "negativeWeightRows": int((cl.weights < 0).any(axis=1).sum()),
        "conversionMaxMm": float(error), "warnings": [
            "Helper bindings reconstructed from full hm08 morphs, then rebound to runtime body triangles; helper deformation is not retained."
        ] if helper.any() else []}


def load_native(path, body):
    cl = mhclo.parse_mhclo(path.read_text(encoding="utf-8", errors="replace"))
    if cl.header.get("basemesh", "hm08") != "hm08":
        raise ValueError("Only hm08 MHCLO bindings are supported")
    obj_path = (path.parent/cl.header["obj_file"]).resolve()
    obj = mhclo.parse_garment_obj(obj_path.read_text(encoding="utf-8", errors="replace"))
    if len(obj.verts) != cl.count:
        raise ValueError("MHCLO/OBJ vertex counts differ")
    mesh = mh_obj.load_base_mesh(MH_DATA/"3dobjs/base.obj")
    if mesh.render_count != body.render_count:
        raise ValueError("hm08 render layout differs from the supplied runtime body")
    first = mesh.mh_to_render_ids[mesh.mh_to_render_start[:mh_obj.BODY_VERTEX_COUNT]]
    morpher = MhMorpher(mesh, TargetPacker(mesh, np.empty((0, 8), int)), build_catalog())
    full = morpher.positions()
    manifest = json.loads((body.directory/"manifest.json").read_text(encoding="utf-8"))
    full[:, 1] += manifest["provenance"]["groundOffsetY"]
    if np.abs(full[mesh.render_mh]-body.positions[:body.render_count]).max() > 5e-5:
        raise ValueError("Full hm08 neutral morphs do not match the runtime body")
    # Use the runtime float positions for direct records, so serialization is tested in its exact frame.
    full[:len(first)] = body.positions[first]
    points, bind, refs, report = convert_records(cl, full, first,
                                               body.positions[:body.render_count], body.faces)
    rv, uv, faces = mhclo.render_split(obj)
    uv[:, 1] = 1-uv[:, 1]
    material = load_parts(obj_path, mhmat_path=path.parent/cl.header.get("material", obj_path.with_suffix(".mhmat").name))[0].material
    report["objFile"] = obj_path.name
    valid_delete = cl.delete_verts[(cl.delete_verts>=0)&(cl.delete_verts<len(first))]
    delete = np.sort(np.concatenate([mesh.render_ids_of(int(v)) for v in valid_delete])) if len(valid_delete) else np.zeros(0,int)
    report["ignoredHelperDeleteVertices"] = int(len(cl.delete_verts)-len(valid_delete))
    return [Part(points[rv], faces, uv, material)], tuple(a[rv] for a in bind), points[rv], refs, report, delete
