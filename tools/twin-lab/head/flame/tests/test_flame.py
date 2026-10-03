import json

import numpy as np
import pytest
import trimesh
from flamehead.assets import region_mask
from flamehead.camera import Camera, load_cameras
from flamehead.geometry import (
    blend_with_foldover_control,
    boundary_loops,
    bridge_rings,
    edge_table,
    replace_face,
    similarity,
    symmetry_map,
    transform,
)
from flamehead.glb import restore_extras, save
from flamehead.pipeline import split_uv
from flamehead.texture import bake, pack_atlases, uv_atlas, visibility
from scipy.spatial.transform import Rotation
from twinrefine.scan import Scan, save_scan


def camera():
    return Camera(
        np.array([[100, 0, 64], [0, 100, 64], [0, 0, 1.0]]),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.5], [0, 0, 0, 1.0]]),
        np.array([30, 286, -10, 246.0]),
        (256, 320),
        (128, 128),
    )


def test_camera_export_conventions_roundtrip(tmp_path):
    c = camera()
    c.extrinsic[:3, :3] = Rotation.from_euler("xyz", [17, -35, 9], degrees=True).as_matrix()
    c.extrinsic[:3, 3] = [0.01, -0.02, -0.5]
    data = {
        "schema": "dt-flame-head-cameras/1",
        "projection": "q = worldToCamera @ [x,y,z,1]; depth=-q.z; u=fx*q.x/depth+cx; v=cy-fy*q.y/depth",
        "views": {
            n: {
                "intrinsics": c.intrinsic.tolist(),
                "worldToCamera": c.extrinsic.tolist(),
                "cropBoundsYminYmaxXminXmax": c.crop.tolist(),
                "originalSizeWH": list(c.original_size),
                "imageSizeWH": list(c.size),
            }
            for n in ("front", "right")
        },
    }
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps(data))
    loaded = load_cameras(path)["right"]
    points = np.random.default_rng(10).uniform(-0.05, 0.05, (50, 3))
    projected = loaded.project(points)
    np.testing.assert_allclose(loaded.unproject(projected[:, :2], projected[:, 2]), points, atol=1e-12)
    np.testing.assert_allclose(loaded.from_original(loaded.to_original(projected[:, :2])), projected[:, :2], atol=1e-12)
    np.testing.assert_allclose(loaded.to_original(np.array([[0.5, 0.5]])), [[-9, 31]])
    # OpenGL +Y is upwards and near points have positive -Z depth.
    basic = camera().project(np.array([[0, 0.01, 0]]))[0]
    assert basic[1] < 64 and basic[2] == 0.5
    data["projection"] = "unsupported"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="projection"):
        load_cameras(path)


def test_symmetry_and_similarity():
    m = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
    ids, report = symmetry_map(m.vertices)
    np.testing.assert_array_equal(ids[ids], np.arange(len(ids)))
    mirrored = m.vertices.copy()
    mirrored[:, 0] *= -1
    np.testing.assert_allclose(m.vertices[ids], mirrored, atol=1e-12)
    assert report["max_distance_mm"] < 1e-6
    rotation = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_matrix()
    target = transform(m.vertices, 1.12, rotation, np.array([0, 1.6, 0.03]))
    s, r, t, hit = similarity(m.vertices, target)
    np.testing.assert_allclose(transform(m.vertices, s, r, t), target, atol=1e-12)
    assert not hit
    assert similarity(m.vertices, target * 2)[3]


@pytest.mark.parametrize("use_xatlas", [False, True])
def test_atlas_fill_synthetic_photos(use_xatlas):
    if use_xatlas:
        pytest.importorskip("xatlas")
    m = trimesh.creation.icosphere(subdivisions=1, radius=0.08)
    v, f = np.asarray(m.vertices), np.asarray(m.faces)
    mapping, af, uv, method = uv_atlas(v, f, 256, use_xatlas)
    np.testing.assert_array_equal(mapping[af], f)
    assert np.all((uv >= 0) & (uv <= 1))
    c = camera()
    # Match the synthetic photo's declared dimensions and crop.
    c.crop, c.original_size = np.array([0, 128, 0, 128.0]), (128, 128)
    photos = {n: np.full((128, 128, 3), [160, 110, 90], np.uint8) for n in ("front", "right")}
    sym, _ = symmetry_map(v)
    tex, report = bake(
        v,
        f,
        mapping,
        af,
        uv,
        {n: v for n in photos},
        {n: c for n in photos},
        photos,
        sym,
        {"left_eyeball": np.array([], int), "right_eyeball": np.array([], int)},
        256,
    )
    assert report["texture_fill_ratio"] == 1 and report["photo_observed_ratio"] > 0.1
    assert tex.min() > 0
    np.testing.assert_allclose(np.median(tex.reshape(-1, 3), axis=0), [160, 110, 90], atol=2)
    assert method == ("xatlas" if use_xatlas else "triangle-grid")


def test_visibility_is_perspective_correct():
    c = camera()
    # All vertices have positive depth; one slanted triangle covers the center.
    v = np.array([[-0.1, -0.1, 0.1], [0.1, -0.1, -0.1], [0, 0.1, 0]])
    f = np.array([[0, 1, 2]])
    d, edge, scale = visibility(v, f, c, resolution=128)
    p = c.project(v)
    # Compute analytic screen barycentrics at the chosen pixel center.
    pixel = np.array([64.5, 64.5])
    matrix = np.vstack((p[:, :2].T, np.ones(3)))
    bary = np.linalg.solve(matrix, np.r_[pixel, 1])
    expected = 1 / np.sum(bary / p[:, 2])
    assert d[64, 64] == pytest.approx(expected, abs=1e-6)
    assert edge[64, 64] > 0


def test_stitch_manifold_preserves_scan_outside_patch():
    scan = trimesh.creation.icosphere(subdivisions=3, radius=0.11)
    flame = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
    v, f, removed, caps, ring, fr, report = replace_face(
        np.asarray(scan.vertices),
        np.asarray(scan.faces),
        np.asarray(flame.vertices),
        np.asarray(flame.faces),
        flame.vertices[:, 2] > 0.015,
    )
    np.testing.assert_array_equal(v[: len(scan.vertices)], scan.vertices)
    np.testing.assert_array_equal(f[: (~removed).sum()], scan.faces[~removed])
    _, count = edge_table(f)
    assert np.all(count == 2) and boundary_loops(f) == []
    mesh = trimesh.Trimesh(v, f, process=False)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert report["stitch_gap_max_mm"] == 0 and report["removed_triangles"] > 0


def test_ring_zipper_unequal_counts():
    angles = np.arange(7) * 2 * np.pi / 7
    a = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(7)))
    angles = np.arange(13) * 2 * np.pi / 13
    b = np.column_stack((0.8 * np.cos(angles), 0.8 * np.sin(angles), np.full(13, 0.1)))
    v = np.vstack((a, b))
    faces = bridge_rings(v, np.arange(7), np.arange(7, 20)[::-1])
    assert len(faces) == 20
    _, count = edge_table(faces)
    assert (count == 1).sum() == 20 and (count > 2).sum() == 0


def test_blend_relaxes_thin_fold_and_keeps_distant_geometry():
    vertices = np.array([[0, 0, 0], [0.001, 0, 0], [0, 0.001, 0], [0.1, 0.1, 0.1]])
    patch = np.array([[0, 1, 2]])
    field = np.zeros_like(vertices)
    field[2, 1] = -0.0035
    result, strength = blend_with_foldover_control(vertices, patch, field)
    assert result[2, 1] > 0 and strength.min() < 1
    np.testing.assert_array_equal(result[3], vertices[3])
    assert strength[3] == 1


def test_protected_mask_regions_and_ears_option():
    masks = {
        k: np.array([], int)
        for k in (
            "face",
            "nose",
            "lips",
            "eye_region",
            "left_eye_region",
            "right_eye_region",
            "left_eyeball",
            "right_eyeball",
            "scalp",
            "neck",
            "boundary",
            "left_ear",
            "right_ear",
        )
    }
    masks.update(
        face=np.arange(8), scalp=np.array([0]), neck=np.array([1]), boundary=np.array([2]), left_ear=np.array([3])
    )
    assert region_mask(masks, 10, False).tolist() == [False] * 4 + [True] * 4 + [False] * 2
    assert region_mask(masks, 10, True)[3]


def test_atlas_pack_and_uv_split():
    scan = np.full((32, 32, 3), [70, 80, 90], np.uint8)
    flame = np.full((16, 16, 3), [130, 140, 150], np.uint8)
    atlas, a, b = pack_atlases(scan, flame, np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]]))
    np.testing.assert_array_equal(atlas[int(a[0, 1] * len(atlas)), int(a[0, 0] * len(atlas))], scan[0, 0])
    np.testing.assert_array_equal(atlas[int(b[0, 1] * len(atlas)), int(b[0, 0] * len(atlas))], flame[0, 0])
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    uv = np.array([[[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 1], [0, 1]]], float)
    out, f, t = split_uv(vertices, faces, uv)
    assert len(out) == 5
    np.testing.assert_array_equal(out[f], vertices[faces])


def test_extras_preserved_glb_roundtrip(tmp_path):
    import struct

    from glbio import _split

    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.1)
    scan = Scan(
        mesh.vertices,
        mesh.faces,
        np.full((len(mesh.vertices), 2), 0.5, np.float32),
        np.full((16, 16, 3), 130, np.uint8),
        "image/png",
    )
    source = tmp_path / "scan.glb"
    save_scan(source, scan)
    js, binary = _split(source.read_bytes())
    old = {
        "asset": {"extras": {"dtRefine": True}},
        "extras": {"root": 4},
        "meshes": [{"extras": {"mesh": 1}, "primitives": [{"extras": {"prim": 2}}]}],
        "nodes": [{"extras": {"node": 3}}],
        "materials": [{"extras": {"material": "synthetic"}}],
    }
    restore_extras(old, js, {"version": 1})
    assert js["extras"] == old["extras"] and js["meshes"][0]["primitives"][0]["extras"] == {"prim": 2}
    assert js["asset"]["extras"]["dtRefine"]
    encoded = json.dumps(js).encode()
    encoded += b" " * (-len(encoded) % 4)
    source.write_bytes(
        struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(binary))
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )
    # Serialize using the wrapper, then make that output the input to verify preservation.
    first = tmp_path / "first.glb"
    save(first, scan, source, {"version": 1, "synthetic": True})
    final = tmp_path / "final.glb"
    save(final, scan, first, {"version": 2})
    checked, _ = _split(final.read_bytes())
    assert checked["asset"]["extras"]["dtFlameHead"]["version"] == 2
    assert checked["asset"]["extras"]["dtRefine"]
    assert checked["extras"] == {"root": 4}
    assert checked["nodes"][0]["extras"] == {"node": 3}
    assert checked["meshes"][0]["extras"] == {"mesh": 1}
    assert checked["meshes"][0]["primitives"][0]["extras"] == {"prim": 2}
    assert checked["materials"][0]["extras"] == {"material": "synthetic"}
    assert len(checked["meshes"]) == len(checked["materials"]) == 1


def test_synthetic_pipeline_single_material_and_previews(tmp_path, monkeypatch):
    import pickle
    import zipfile

    from flamehead import pipeline
    from glbio import _split
    from PIL import Image
    from twinrefine.scan import load_scan, weld_ids

    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    fit, photos, assets = [tmp_path / name for name in ("fit", "photos", "assets")]
    for folder in (fit / "fitted_views", photos, assets):
        folder.mkdir(parents=True)
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    mesh.export(fit / "head_neutral.obj")
    for n in ("front", "right"):
        mesh.export(fit / "fitted_views" / f"{n}.ply")
        Image.fromarray(np.full((128, 128, 3), [165, 115, 90], np.uint8)).save(photos / f"{n}.jpg")
    c = camera()
    c.crop, c.original_size = np.array([0, 128, 0, 128.0]), (128, 128)
    camera_data = {
        "schema": "dt-flame-head-cameras/1",
        "projection": "q = worldToCamera @ [x,y,z,1]; depth=-q.z; u=fx*q.x/depth+cx; v=cy-fy*q.y/depth",
        "views": {
            n: {
                "intrinsics": c.intrinsic.tolist(),
                "worldToCamera": c.extrinsic.tolist(),
                "cropBoundsYminYmaxXminXmax": c.crop.tolist(),
                "originalSizeWH": list(c.original_size),
                "imageSizeWH": list(c.size),
            }
            for n in ("front", "right")
        },
    }
    (fit / "cameras.json").write_text(json.dumps(camera_data))
    masks = {
        k: np.array([], int)
        for k in (
            "face",
            "nose",
            "lips",
            "eye_region",
            "left_eye_region",
            "right_eye_region",
            "left_eyeball",
            "right_eyeball",
            "scalp",
            "neck",
            "boundary",
            "left_ear",
            "right_ear",
        )
    }
    masks["face"] = np.flatnonzero(vertices[:, 2] > 0.015)
    with zipfile.ZipFile(assets / "FLAME_masks.zip", "w") as z:
        z.writestr("FLAME_masks.pkl", pickle.dumps(masks, protocol=4))
    ids = np.array([33, 133, 263, 362, 159, 386, 55, 285, 105, 334, 1, 168, 129, 358, 61, 291, 13, 14, 152])
    fi = np.flatnonzero((vertices[faces, 2] > 0.02).all(1))[: len(ids)]
    bary = np.full((len(ids), 3), 1 / 3)
    np.savez(assets / "mediapipe_landmark_embedding.npz", lmk_face_idx=fi, lmk_b_coords=bary, landmark_indices=ids)
    lm = np.zeros((468, 3))
    lm[ids] = vertices[faces[fi]].mean(1)
    monkeypatch.setattr(pipeline, "detect_twin_landmarks", lambda *args: lm)
    scan_mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
    # Give the synthetic scan a stable atlas; it has no owner's pixels.
    scan = Scan(
        scan_mesh.vertices,
        scan_mesh.faces,
        np.full((len(scan_mesh.vertices), 2), 0.5, np.float32),
        np.full((256, 256, 3), [145, 110, 90], np.uint8),
        "image/png",
    )
    source = tmp_path / "input.glb"
    save_scan(source, scan)
    out = tmp_path / "user-data/head/head.glb"
    report = pipeline.run(source, fit, photos, assets, out, texture_size=256)
    assert report["texture"]["texture_fill_ratio"] == 1 and report["alignment"]["scale"] == pytest.approx(1, abs=0.02)
    assert report["stitch"]["nonmanifold_edges"] == 0
    checked = load_scan(out)
    inv, first = weld_ids(checked.verts)
    welded = trimesh.Trimesh(checked.verts[first], inv[checked.faces], process=False)
    assert welded.is_watertight and welded.is_winding_consistent
    js, _ = _split(out.read_bytes())
    assert len(js["meshes"]) == len(js["materials"]) == 1
    assert "dtFlameHead" in js["asset"]["extras"]
    assert len(report["previews"]["views"]) == 4
    assert (out.parent / "previews/contact.png").is_file()
