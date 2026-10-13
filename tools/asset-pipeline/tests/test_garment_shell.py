"""Synthetic checks for seam-safe geometry stages and clean garment openings (no real assets, no body morphs).

Covers: welded topology ignores UV seams, decimation keeps UV seams closed, per-patch regularisation of the inner-layer
decision, thin-slit restoration, duplicate-band / sliver removal and light boundary-loop smoothing.
"""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")
pytest.importorskip("rtree")
pytest.importorskip("fast_simplification")

from garmentimport.geometry import outer_shell
from garmentimport.mesh_io import Part, clean, decimate
from garmentimport.shell import (boundary_loops, mesh_summary, regularize_removal, restore_bridges,
                                 sliver_components, smooth_openings, stacked_components, weld_ids)


def material():
    return {"base": None, "mr": None, "normal": None, "occlusion": None, "emissive": None,
            "factor": [1, 1, 1, 1], "roughness": .7, "metallic": .2, "emissiveFactor": [0, 0, 0],
            "alphaMode": "OPAQUE", "alphaCutoff": .5, "normalScale": 1., "occlusionStrength": 1.}


def tube(radius=.1, y0=0., y1=.3, around=48, rows=12, seam_copy=True, flip=False):
    """Open cylinder along Y (outward normals); the wrap-around column is a separate render vertex pair (UV seam)."""
    angle = np.linspace(0, 2*np.pi, around+1)
    height = np.linspace(y0, y1, rows+1)
    grid = np.array([[radius*np.cos(a), y, radius*np.sin(a)] for y in height for a in angle])
    uv = np.array([[a/(2*np.pi), (y-y0)/max(y1-y0, 1e-9)] for y in height for a in angle])
    faces = []
    for r in range(rows):
        for c in range(around):
            a, b = r*(around+1)+c, r*(around+1)+c+1
            d, e = (r+1)*(around+1)+c, (r+1)*(around+1)+c+1
            faces += [[a, e, b], [a, d, e]]
    faces = np.array(faces)
    if not seam_copy:  # weld the last column onto the first one
        index = np.arange(len(grid)).reshape(rows+1, around+1)
        index[:, -1] = index[:, 0]
        remap = np.zeros(len(grid), int)
        remap[index.reshape(-1)] = index.reshape(-1)
        faces = remap[faces]
    if flip:
        faces = faces[:, ::-1]
    return grid, faces, uv


def combine(*meshes):
    positions, faces, uvs, offset = [], [], [], 0
    for p, f, *rest in meshes:
        positions.append(p)
        faces.append(f+offset)
        uvs.append(rest[0] if rest else np.zeros((len(p), 2)))
        offset += len(p)
    return np.concatenate(positions), np.concatenate(faces), np.concatenate(uvs)


def test_welded_topology_ignores_uv_seams_and_counts_openings():
    p, f, _ = tube()
    summary = mesh_summary(p, f)
    assert summary["boundaryLoops"] == 2 and summary["nonManifoldEdges"] == 0
    assert summary["boundaryEdges"] == 2*48
    # The seam copies are distinct render vertices but one welded position.
    assert len(np.unique(weld_ids(p))) < len(p)
    loops, _, _ = boundary_loops(p, f)
    assert sorted(len(l) for l in loops) == [48, 48]


def test_decimation_keeps_uv_seams_closed():
    mesh = trimesh.creation.icosphere(subdivisions=4)
    v, f = mesh.vertices, mesh.faces
    upper = mesh.triangles_center[:, 1] >= 0
    positions, faces, uvs, offset = [], [], [], 0
    for flag in (True, False):  # two UV islands split along the equator
        selected = f[upper == flag]
        keep = np.unique(selected)
        remap = np.zeros(len(v), int)
        remap[keep] = np.arange(len(keep))
        positions.append(v[keep])
        faces.append(remap[selected]+offset)
        uvs.append((v[keep][:, [0, 2]]+1)/4+(.5 if flag else 0))
        offset += len(keep)
    part = clean(Part(np.concatenate(positions), np.concatenate(faces), np.concatenate(uvs), material()))
    assert mesh_summary(part.positions, part.faces)["boundaryEdges"] == 0
    output = decimate([part], 800)[0]
    assert len(output.faces) <= 800
    # Welding on position before decimation: no gap opens along the seam.
    assert mesh_summary(output.positions, output.faces)["boundaryEdges"] == 0
    assert output.uv.min() >= 0 and output.uv.max() <= 1 and np.isfinite(output.uv).all()
    # Both islands survive: seam vertices keep one render copy per island.
    assert len(np.unique(weld_ids(output.positions))) < len(output.positions)


def test_regularize_removal_makes_the_layer_decision_per_patch():
    outer = tube(.1)
    inner = tube(.0995, flip=True)
    p, f, _ = combine(outer, inner)
    half = len(outer[1])
    removed = np.zeros(len(f), bool)
    removed[half:] = True
    noisy = removed.copy()
    noisy[[5, 111, 230]] = True            # speckles in the outer layer
    noisy[half+np.array([7, 90, 201, 300])] = False  # speckles in the inner layer
    fixed, report = regularize_removal(p, f, noisy)
    assert (fixed == removed).all()
    assert report["trianglesChanged"] == 7


def box_tube(side=.2, chamfer=.002, height=.3):
    """Open square tube whose four corners are cut by narrow chamfer strips (each its own smooth patch)."""
    h = side/2
    corners = [np.array(c) for c in ((h, h), (-h, h), (-h, -h), (h, -h))]
    ring = []
    for i, corner in enumerate(corners):
        following = corners[(i+1) % 4]
        direction = (following-corner)/np.linalg.norm(following-corner)
        ring += [corner+direction*chamfer, following-direction*chamfer]  # side start, side end
    count = len(ring)
    positions = np.array([[q[0], y, q[1]] for y in (0., height) for q in ring])
    faces, chamfer_faces = [], []
    for i in range(count):
        j = (i+1) % count
        faces += [[i, count+j, j], [i, count+i, count+j]]
        if i % 2 == 1:  # between the end of one side and the start of the next: the chamfer strip
            chamfer_faces += [len(faces)-2, len(faces)-1]
    return positions, np.array(faces), np.array(chamfer_faces)


def test_restore_bridges_closes_a_thin_slit_but_keeps_wide_layers():
    p, f, chamfer = box_tube(chamfer=.002)
    removed = np.zeros(len(f), bool)
    removed[chamfer[:2]] = True  # one 2 mm strip between two kept panels
    restored, report = restore_bridges(p, f, removed)
    assert not restored.any() and report["restoredPatches"] == 1
    # A 30 mm strip is not a slit: it stays removed.
    p, f, chamfer = box_tube(chamfer=.03)
    removed = np.zeros(len(f), bool)
    removed[chamfer[:2]] = True
    restored, report = restore_bridges(p, f, removed)
    assert restored.sum() == 2 and report["restoredPatches"] == 0


def test_stacked_band_is_dropped_and_independent_parts_are_kept():
    wall = tube(.1, 0, .3)
    band = tube(.1005, .02, .05, rows=3)         # rolled hem band hugging the wall, own component
    other = tube(.03, 0.6, .7, around=24)         # a separate part elsewhere
    p, f, _ = combine(wall, band, other)
    drop, report = stacked_components(p, f)
    first_band = len(wall[1])
    assert drop[first_band:first_band+len(band[1])].all()
    assert not drop[:first_band].any() and not drop[first_band+len(band[1]):].any()
    assert report["droppedComponents"] == 1


def test_sliver_components_are_dropped_but_closed_pieces_stay():
    wall = tube(.1, 0, .3)
    positions = np.array([[0., 1, 0], [.15, 1, 0], [.15, 1.002, 0], [0., 1.002, 0]])  # 2 mm x 15 cm open strip
    strip = (positions, np.array([[0, 1, 2], [0, 2, 3]]))
    button = trimesh.creation.box(extents=(.02, .02, .02))
    button_positions = button.vertices+[.5, .1, 0]
    p, f, _ = combine(wall, strip, (button_positions, button.faces))
    drop, report = sliver_components(p, f)
    strip_start = len(wall[1])
    assert drop[strip_start:strip_start+2].all()
    assert not drop[:strip_start].any() and not drop[strip_start+2:].any()
    assert report["droppedComponents"] == 1


def test_smooth_openings_relaxes_jagged_hems_without_opening_seams():
    p, f, uv = tube(.1, 0, .3, rows=6)
    rows, columns = 7, 49
    jagged = p.copy()
    grid = np.arange(len(p)).reshape(rows, columns)
    for c in range(0, columns-1):
        jagged[grid[0, c], 1] -= .004*(c % 2)          # zig-zag hem
    jagged[grid[0, -1]] = jagged[grid[0, 0]]            # seam copies stay identical
    smoothed, report = smooth_openings(jagged, f, iterations=2, strength=.5)
    assert report["smoothedLoops"] == 2
    hem = grid[0, :-1]
    assert np.ptp(smoothed[hem, 1]) < .5*np.ptp(jagged[hem, 1])
    assert np.allclose(smoothed[grid[0, -1]], smoothed[grid[0, 0]])
    interior = grid[2:5].reshape(-1)
    assert np.allclose(smoothed[interior], jagged[interior])
    assert mesh_summary(smoothed, f)["boundaryLoops"] == 2


def test_outer_shell_keeps_one_clean_loop_per_opening():
    body = trimesh.creation.cylinder(radius=.09, height=.9, sections=48)
    body.apply_translation([0, .15, 0])
    wall = tube(.1, 0, .3)
    lining = tube(.0995, 0, .05, rows=2, flip=True)   # double layer at the lower hem
    parts = [Part(wall[0], wall[1], wall[2], material()), Part(lining[0], lining[1], lining[2], material())]
    result, report = outer_shell(parts, body.vertices, body.faces)
    kept = np.concatenate([p.positions[p.faces] for p in result])
    assert kept.shape[0] == len(wall[1])
    final = mesh_summary(*combine(*[(p.positions, p.faces) for p in result])[:2])
    assert final["boundaryLoops"] == 2 and final["nonManifoldEdges"] == 0
    assert report["topologyAfterCleanup"]["boundaryLoops"] == 2


def test_retexture_keeps_geometry_and_copies_the_donor_material(tmp_path):
    import io
    import pygltflib as gl
    from PIL import Image
    from garmentimport.mesh_io import write_glb
    from garmentimport.retexture import transplant_material
    p, f, uv = tube(.1, 0, .3, around=12, rows=3)
    own = {"base": np.full((16, 16, 4), 200, np.uint8), "mr": np.full((16, 16, 4), 255, np.uint8)}
    donor_channels = {"base": np.full((32, 32, 4), 90, np.uint8), "mr": np.full((16, 16, 4), 255, np.uint8),
                      "normal": np.full((8, 8, 4), 128, np.uint8)}
    write_glb(tmp_path/"new.glb", p, f, uv, own, "OPAQUE")
    write_glb(tmp_path/"donor.glb", p*.5, f, uv, donor_channels, "OPAQUE")
    transplant_material(tmp_path/"new.glb", tmp_path/"donor.glb", tmp_path/"out.glb")
    out = gl.GLTF2().load_binary(str(tmp_path/"out.glb"))
    blob = out.binary_blob()
    sizes = []
    for image in out.images:
        view = out.bufferViews[image.bufferView]
        sizes.append(Image.open(io.BytesIO(blob[view.byteOffset:view.byteOffset+view.byteLength])).size)
    assert sizes.count((16, 16)) == 1 and (32, 32) in sizes and (8, 8) in sizes  # donor images only (own base is gone)
    assert out.materials[0].normalTexture is not None
    primitive = out.meshes[0].primitives[0]
    count = out.accessors[primitive.attributes.POSITION].count
    assert count == len(p) and out.accessors[primitive.indices].count == f.size
    view = out.bufferViews[out.accessors[primitive.attributes.POSITION].bufferView]
    positions = np.frombuffer(blob, dtype="<f4", count=count*3, offset=view.byteOffset).reshape(-1, 3)
    assert np.allclose(positions, p, atol=1e-6)  # the new geometry, not the donor's
