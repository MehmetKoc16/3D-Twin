"""Synthetic tests for the automated shoe alignment sidecar generator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from garmentimport.body import align
from garmentimport.mesh_io import Part, clean, write_glb
from garmentimport.shoe_align import (
    align_shoe,
    compute_shoe_alignment,
    detect_length_and_toe,
    detect_side,
    detect_up_and_sole,
    main,
    parser,
)

PIPELINE = Path(__file__).resolve().parents[1]
BODY_DIR = PIPELINE.parents[1] / "apps/web/public/assets/body"


def make_material():
    return {
        "base": None,
        "mr": None,
        "normal": None,
        "occlusion": None,
        "emissive": None,
        "factor": [0.8, 0.8, 0.8, 1.0],
        "roughness": 0.8,
        "metallic": 0.1,
        "emissiveFactor": [0, 0, 0],
        "alphaMode": "OPAQUE",
        "alphaCutoff": 0.5,
        "normalScale": 1.0,
        "occlusionStrength": 1.0,
    }


def make_synthetic_shoe_mesh(
    side: str = "left",
    length: float = 0.26,
    width: float = 0.09,
    heel_height: float = 0.12,
    toe_height: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic shoe mesh with flat sole, high heel collar, and low toe box."""
    nx, nz = 7, 15
    z_coords = np.linspace(-0.05, -0.05 + length, nz)
    half_w = width / 2.0

    sole_verts = []
    for j, z in enumerate(z_coords):
        z_norm = (z - (-0.05)) / length
        # Left shoe has medial arch indentation on negative X; right shoe on positive X.
        if 0.25 <= z_norm <= 0.55:
            if side == "left":
                min_x = -half_w * 0.45
                max_x = half_w
            else:
                min_x = -half_w
                max_x = half_w * 0.45
        else:
            min_x = -half_w
            max_x = half_w

        # Toe apex offset: big toe is medial
        if z_norm >= 0.90:
            toe_shift = -0.015 if side == "left" else 0.015
            min_x += toe_shift
            max_x += toe_shift

        xs = np.linspace(min_x, max_x, nx)
        for x in xs:
            sole_verts.append([x, 0.0, z])

    sole_verts = np.array(sole_verts, float)

    # Sole grid faces
    sole_faces = []
    for j in range(nz - 1):
        for i in range(nx - 1):
            v00 = j * nx + i
            v10 = j * nx + (i + 1)
            v01 = (j + 1) * nx + i
            v11 = (j + 1) * nx + (i + 1)
            sole_faces.append([v00, v10, v01])
            sole_faces.append([v10, v11, v01])
    sole_faces = np.array(sole_faces, int)

    # Perimeter indices of the sole grid (ordered loop)
    perimeter: list[int] = []
    for i in range(nx):
        perimeter.append(i)
    for j in range(1, nz):
        perimeter.append(j * nx + (nx - 1))
    for i in range(nx - 2, -1, -1):
        perimeter.append((nz - 1) * nx + i)
    for j in range(nz - 2, 0, -1):
        perimeter.append(j * nx)

    # Upper rim vertices
    upper_verts = []
    offset = len(sole_verts)
    for idx in perimeter:
        sv = sole_verts[idx]
        z_norm = (sv[2] - (-0.05)) / length
        if z_norm <= 0.30:
            h = heel_height
        elif z_norm >= 0.70:
            h = toe_height
        else:
            t = (z_norm - 0.30) / 0.40
            h = heel_height * (1.0 - t) + toe_height * t
        # Inward upper taper
        upper_verts.append([sv[0] * 0.85, h, sv[2]])
    upper_verts = np.array(upper_verts, float)

    # Wall faces connecting sole perimeter to upper rim
    wall_faces = []
    n_perim = len(perimeter)
    for k in range(n_perim):
        k_next = (k + 1) % n_perim
        s0 = perimeter[k]
        s1 = perimeter[k_next]
        u0 = offset + k
        u1 = offset + k_next
        wall_faces.append([s0, s1, u0])
        wall_faces.append([s1, u1, u0])
    wall_faces = np.array(wall_faces, int)

    # Collar cap faces (fan triangulation of upper rim)
    cap_faces = []
    for k in range(1, n_perim - 1):
        cap_faces.append([offset, offset + k + 1, offset + k])
    cap_faces = np.array(cap_faces, int)

    all_positions = np.vstack([sole_verts, upper_verts])
    all_faces = np.vstack([sole_faces, wall_faces, cap_faces])
    uvs = np.zeros((len(all_positions), 2))

    return all_positions, all_faces, uvs


def create_shoe_glb(
    path: Path,
    side: str = "left",
    transform: np.ndarray | None = None,
) -> Path:
    """Helper to write a synthetic shoe GLB with optional rotation/scale/translation."""
    pos, faces, uvs = make_synthetic_shoe_mesh(side=side)
    if transform is not None:
        pos = pos @ transform[:3, :3].T + transform[:3, 3]

    part = clean(Part(pos, faces, uvs, make_material()))
    channels = {
        "base": np.full((16, 16, 4), 200, dtype=np.uint8),
        "mr": np.full((16, 16, 4), [0, 200, 20, 255], dtype=np.uint8),
    }
    write_glb(path, part.positions, part.faces, part.uv, channels, "OPAQUE")
    return path


def test_detect_up_and_sole_finds_sole_plane():
    pos, _, _ = make_synthetic_shoe_mesh(side="left")
    up, info = detect_up_and_sole(pos)
    # Sole is at Y=0, normal pointing UP is +Y
    assert np.allclose(up, [0.0, 1.0, 0.0], atol=0.1)
    assert info["bestScore"] > 0


def test_detect_length_and_toe_finds_correct_toe():
    pos, _, _ = make_synthetic_shoe_mesh(side="left")
    up, _ = detect_up_and_sole(pos)
    fwd, conf, flipped, info = detect_length_and_toe(pos, up, toe_flip=False)
    # Forward toe direction is +Z
    assert np.allclose(fwd, [0.0, 0.0, 1.0], atol=0.1)
    assert conf > 0.5
    assert not flipped


def test_detect_side_classifies_left_and_right():
    pos_l, _, _ = make_synthetic_shoe_mesh(side="left")
    up_l, _ = detect_up_and_sole(pos_l)
    fwd_l, _, _, _ = detect_length_and_toe(pos_l, up_l)
    u_lat_l, detected_l, conf_l, _ = detect_side(pos_l, up_l, fwd_l, requested_side="left")
    assert detected_l == "left"
    assert np.allclose(u_lat_l, [1.0, 0.0, 0.0], atol=0.1)

    pos_r, _, _ = make_synthetic_shoe_mesh(side="right")
    up_r, _ = detect_up_and_sole(pos_r)
    fwd_r, _, _, _ = detect_length_and_toe(pos_r, up_r)
    # detect_side raises error when right shoe is detected or requested
    with pytest.raises(ValueError, match="(?i)right shoe"):
        detect_side(pos_r, up_r, fwd_r, requested_side="left")


def test_canonical_shoe_alignment(tmp_path):
    glb = create_shoe_glb(tmp_path / "canonical.glb", side="left")
    sidecar_path = tmp_path / "alignment.json"

    sidecar, report = align_shoe(glb, out_path=sidecar_path)
    assert sidecar_path.exists()
    assert sidecar["pose"] == "a"

    matrix = np.array(sidecar["matrix"], float)
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix[3], [0, 0, 0, 1])

    det = np.linalg.det(matrix[:3, :3])
    assert det > 0, "Matrix determinant must be strictly positive (orientation-preserving)"

    # Check transformed points
    pos, _, _ = make_synthetic_shoe_mesh(side="left")
    aligned = pos @ matrix[:3, :3].T + matrix[:3, 3]

    # 1. Sole grounded on y = 0
    assert aligned[:, 1].min() == pytest.approx(0.0, abs=0.005)

    # 2. Entire shoe is on the left foot (+X > 0)
    assert aligned[:, 0].min() > 0.05
    assert aligned[:, 0].max() < 0.30

    # 3. Length along Z axis
    assert np.ptp(aligned[:, 2]) == pytest.approx(report["targetShoeLengthM"], abs=0.01)

    # 4. Toe is pointing +Z (front points have larger Z than heel points)
    heel_pts = aligned[pos[:, 2] < 0.0]
    toe_pts = aligned[pos[:, 2] > 0.15]
    assert toe_pts[:, 2].mean() > heel_pts[:, 2].mean()


def test_centimeter_unit_inference(tmp_path):
    # Centimeter units: 100x scale
    trans = np.eye(4)
    trans[:3, :3] *= 100.0
    glb = create_shoe_glb(tmp_path / "cm_shoe.glb", side="left", transform=trans)

    sidecar, report = align_shoe(glb)
    matrix = np.array(sidecar["matrix"], float)

    # Scale should be roughly 1/100 = 0.01
    assert report["scale"] == pytest.approx(0.01, rel=0.15)
    det = np.linalg.det(matrix[:3, :3])
    assert det > 0

    pos, _, _ = make_synthetic_shoe_mesh(side="left")
    scaled_pos = pos * 100.0
    aligned = scaled_pos @ matrix[:3, :3].T + matrix[:3, 3]

    assert aligned[:, 1].min() == pytest.approx(0.0, abs=0.01)
    assert aligned[:, 0].min() > 0.05
    assert np.ptp(aligned[:, 2]) == pytest.approx(report["targetShoeLengthM"], abs=0.01)


@pytest.mark.parametrize(
    "rot_angles,translation",
    [
        ([0, 90, 0], [5.0, 2.0, -3.0]),       # 90 deg around Y (length along X)
        ([90, 0, 0], [-10.0, 15.0, 25.0]),   # 90 deg around X (length along Y)
        ([0, 0, 180], [1.0, -4.0, 0.5]),      # upside down
        ([30, -60, 45], [12.0, 8.0, -15.0]),  # arbitrary 3D rotation
    ],
)
def test_rotated_and_translated_shoe(tmp_path, rot_angles, translation):
    rot = Rotation.from_euler("xyz", rot_angles, degrees=True).as_matrix()
    trans = np.eye(4)
    trans[:3, :3] = rot * 1.5  # also scaled by 1.5
    trans[:3, 3] = translation

    glb = create_shoe_glb(tmp_path / "transformed.glb", side="left", transform=trans)
    sidecar, report = align_shoe(glb, up="auto")
    matrix = np.array(sidecar["matrix"], float)

    assert np.linalg.det(matrix[:3, :3]) > 0

    pos, _, _ = make_synthetic_shoe_mesh(side="left")
    source_pts = pos @ trans[:3, :3].T + trans[:3, 3]
    aligned = source_pts @ matrix[:3, :3].T + matrix[:3, 3]

    assert aligned[:, 1].min() == pytest.approx(0.0, abs=0.02)
    assert aligned[:, 0].min() > 0.0
    assert np.ptp(aligned[:, 2]) == pytest.approx(report["targetShoeLengthM"], abs=0.02)

    # Verify toe is pointing +Z
    heel_pts = aligned[pos[:, 2] < 0.0]
    toe_pts = aligned[pos[:, 2] > 0.15]
    assert toe_pts[:, 2].mean() > heel_pts[:, 2].mean()


def test_toe_flip_reverses_burun(tmp_path):
    glb = create_shoe_glb(tmp_path / "shoe.glb", side="left")

    sidecar_normal, report_normal = align_shoe(glb, toe_flip=False)
    sidecar_flipped, report_flipped = align_shoe(glb, toe_flip=True)

    assert not report_normal["toeFlipped"]
    assert report_flipped["toeFlipped"]

    m_norm = np.array(sidecar_normal["matrix"], float)
    m_flip = np.array(sidecar_flipped["matrix"], float)

    assert np.linalg.det(m_norm[:3, :3]) > 0
    assert np.linalg.det(m_flip[:3, :3]) > 0

    pos, _, _ = make_synthetic_shoe_mesh(side="left")
    aligned_norm = pos @ m_norm[:3, :3].T + m_norm[:3, 3]
    aligned_flip = pos @ m_flip[:3, :3].T + m_flip[:3, 3]

    heel_idx = np.flatnonzero(pos[:, 2] < 0.0)
    toe_idx = np.flatnonzero(pos[:, 2] > 0.15)

    # In normal orientation, toe has greater Z than heel
    assert aligned_norm[toe_idx, 2].mean() > aligned_norm[heel_idx, 2].mean()
    # With --toe-flip, toe has smaller Z than heel (flipped)
    assert aligned_flip[toe_idx, 2].mean() < aligned_flip[heel_idx, 2].mean()


def test_right_shoe_rejected(tmp_path):
    glb_right = create_shoe_glb(tmp_path / "shoe_r.glb", side="right")

    with pytest.raises(ValueError, match="(?i)right shoe"):
        align_shoe(glb_right, side="left")

    with pytest.raises(ValueError, match="(?i)right shoe"):
        align_shoe(glb_right, side="right")


def test_cli_produces_valid_sidecar_and_align_loads_it(tmp_path):
    glb = create_shoe_glb(tmp_path / "cli_shoe.glb", side="left")
    out_json = tmp_path / "alignment.json"
    rep_json = tmp_path / "report.json"

    ret = main([
        str(glb),
        "--out",
        str(out_json),
        "--length-margin-cm",
        "1.5",
        "--report",
        str(rep_json),
    ])
    assert ret == 0
    assert out_json.exists()
    assert rep_json.exists()

    sidecar = json.loads(out_json.read_text(encoding="utf-8"))
    assert sidecar["pose"] == "a"
    assert "matrix" in sidecar

    # Verify that the sidecar matrix applies cleanly
    pos, _, _ = make_synthetic_shoe_mesh(side="left")
    matrix = np.array(sidecar["matrix"], float)
    aligned = pos @ matrix[:3, :3].T + matrix[:3, 3]

    assert aligned[:, 1].min() == pytest.approx(0.0, abs=0.01)
    assert aligned[:, 0].min() > 0.0

    # If full body assets exist, verify garmentimport.body.align executes without error
    if (BODY_DIR / "base.glb").exists():
        from garmentimport.body import load_body

        body = load_body(BODY_DIR)
        aligned_body, report = align(pos, body, "shoes", "a", sidecar=sidecar)
        assert report["pose"] == "a"
        assert report["scale"] > 0
        assert np.allclose(aligned, aligned_body, atol=1e-5)


def test_obj_source_support(tmp_path):
    pos, faces, _ = make_synthetic_shoe_mesh(side="left")
    obj_path = tmp_path / "shoe.obj"

    lines = [f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n" for p in pos]
    lines.extend([f"vt 0.5 0.5\n" for _ in pos])
    for face in faces:
        v1, v2, v3 = face[0] + 1, face[1] + 1, face[2] + 1
        lines.append(f"f {v1}/{v1} {v2}/{v2} {v3}/{v3}\n")
    obj_path.write_text("".join(lines), encoding="utf-8")

    out_json = tmp_path / "obj_alignment.json"
    sidecar, report = align_shoe(obj_path, out_path=out_json)

    assert out_json.exists()
    assert np.linalg.det(np.array(sidecar["matrix"])[:3, :3]) > 0
    assert report["detectedSide"] == "left"


def test_node_filtering_and_exclude_material(tmp_path):
    glb = create_shoe_glb(tmp_path / "single_shoe.glb", side="left")
    import pygltflib as gl

    g = gl.GLTF2().load(str(glb))
    # Add a second node with mesh 0 named "Shoe_R_node" translated away
    g.nodes.append(gl.Node(mesh=0, name="Shoe_R_node", translation=[5.0, 0.0, 0.0]))
    g.scenes[0].nodes.append(len(g.nodes) - 1)
    g.nodes[0].name = "Shoe_L_node"
    multi_path = tmp_path / "pair.glb"
    g.save_binary(str(multi_path))

    # Without node filtering, both shoes combined would distort the bounding box / extent
    # With node filtering, only Shoe_L_node is loaded and aligned
    sidecar, report = align_shoe(multi_path, node=["Shoe_L_node"])
    assert report["detectedSide"] == "left"
    assert np.linalg.det(np.array(sidecar["matrix"])[:3, :3]) > 0


def test_override_up_and_toe_alignment(tmp_path):
    """Verify that --up +y --toe -x overrides produce exact alignment for a shoe with length along X."""
    # Rotate canonical shoe 90 deg around Y:
    # +Z (toe) -> -X
    # +Y (up)  -> +Y
    # +X (lat) -> +Z
    rot_90_y = np.array([
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    trans = np.eye(4)
    trans[:3, :3] = rot_90_y

    glb = create_shoe_glb(tmp_path / "x_shoe.glb", side="left", transform=trans)
    out_json = tmp_path / "alignment.json"

    # GLB defaults to up="+y", pass toe="-x"
    sidecar, report = align_shoe(glb, out_path=out_json, up="+y", toe="-x")

    assert out_json.exists()
    report_json = tmp_path / "alignment.report.json"
    assert report_json.exists()

    # Exact axes check (directly from override, not snapped or perturbed)
    assert np.allclose(report["axes"]["up"], [0.0, 1.0, 0.0])
    assert np.allclose(report["axes"]["forward"], [-1.0, 0.0, 0.0])
    assert np.allclose(report["axes"]["lateral"], [0.0, 0.0, 1.0])

    # Check source extents: length along X (-x), height along Y (+y), width along Z (+z)
    assert report["sourceExtent"]["length"] == pytest.approx(0.26, abs=0.001)
    assert report["sourceExtent"]["height"] == pytest.approx(0.12, abs=0.001)
    # The medial toe apex extends 15 mm past the nominal 90 mm sole width.
    assert report["sourceExtent"]["width"] == pytest.approx(0.105, abs=0.001)

    # Scale check: targetShoeLengthM / source_shoe_len
    expected_scale = report["targetShoeLengthM"] / 0.26
    assert report["scale"] == pytest.approx(expected_scale, rel=1e-5)

    # Rotation matrix check: maps [-1, 0, 0] to +Z, [0, 1, 0] to +Y, [0, 0, 1] to +X
    matrix = np.array(sidecar["matrix"], float)
    rot_extracted = matrix[:3, :3] / report["scale"]
    expected_rot = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    assert np.allclose(rot_extracted, expected_rot, atol=1e-6)

    # Check transformed points
    pos_orig, _, _ = make_synthetic_shoe_mesh(side="left")
    pos_x = pos_orig @ rot_90_y.T

    aligned = pos_x @ matrix[:3, :3].T + matrix[:3, 3]

    # Sole grounded at y = 0
    assert aligned[:, 1].min() == pytest.approx(0.0, abs=1e-4)

    # Shoe is on the left foot (+X > 0)
    assert aligned[:, 0].min() > 0.05
    assert aligned[:, 0].max() < 0.30

    # Toe (which was at negative X in pos_x) must point at +Z in aligned
    toe_mask = pos_x[:, 0] < -0.15
    heel_mask = pos_x[:, 0] > 0.0
    assert aligned[toe_mask, 2].mean() > aligned[heel_mask, 2].mean()

    # Length along Z matches target
    assert np.ptp(aligned[:, 2]) == pytest.approx(report["targetShoeLengthM"], abs=1e-4)


def test_override_toe_flip_with_explicit_toe(tmp_path):
    """Verify that --toe-flip inverts toe direction even when --toe override is specified."""
    rot_90_y = np.array([
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    trans = np.eye(4)
    trans[:3, :3] = rot_90_y

    glb = create_shoe_glb(tmp_path / "x_shoe_flip.glb", side="left", transform=trans)
    sidecar, report = align_shoe(glb, up="+y", toe="-x", toe_flip=True)

    assert report["toeFlipped"]
    assert np.allclose(report["axes"]["forward"], [1.0, 0.0, 0.0])

    pos_orig, _, _ = make_synthetic_shoe_mesh(side="left")
    pos_x = pos_orig @ rot_90_y.T

    matrix = np.array(sidecar["matrix"], float)
    aligned = pos_x @ matrix[:3, :3].T + matrix[:3, 3]

    toe_mask = pos_x[:, 0] < -0.15
    heel_mask = pos_x[:, 0] > 0.0
    # Toe is flipped to -Z, so its mean Z is lower than heel
    assert aligned[toe_mask, 2].mean() < aligned[heel_mask, 2].mean()


def test_cli_up_and_toe_options(tmp_path):
    """Test CLI options for --up and --toe and verification of alignment.report.json writing."""
    rot_90_y = np.array([
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    trans = np.eye(4)
    trans[:3, :3] = rot_90_y
    glb = create_shoe_glb(tmp_path / "cli_shoe_x.glb", side="left", transform=trans)

    out_json = tmp_path / "alignment.json"
    ret = main([
        str(glb),
        "--out",
        str(out_json),
        "--up",
        "+y",
        "--toe",
        "-x",
    ])
    assert ret == 0
    assert out_json.exists()
    assert (tmp_path / "alignment.report.json").exists()

    sidecar = json.loads(out_json.read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "alignment.report.json").read_text(encoding="utf-8"))

    assert np.allclose(report["axes"]["up"], [0.0, 1.0, 0.0])
    assert np.allclose(report["axes"]["forward"], [-1.0, 0.0, 0.0])
    assert np.linalg.det(np.array(sidecar["matrix"])[:3, :3]) > 0




def synthetic_target_body():
    """A tiny weighted neutral foot; no real person's geometry or measurements."""
    from types import SimpleNamespace

    positions = np.array([[x, y, z] for x in (.14, .23) for y in (0., .07)
                          for z in (-.03, .2038)])
    return SimpleNamespace(positions=positions, render_count=len(positions),
                           names=["foot_l"], joints=np.zeros((len(positions), 1), int),
                           weights=np.ones((len(positions), 1)), measures={})


def test_complete_matrix_for_y_up_negative_x_toe():
    points, _, _ = make_synthetic_shoe_mesh()
    rotation = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    matrix, report = compute_shoe_alignment(points @ rotation.T, body=synthetic_target_body(),
                                           up="+y", toe="-x")
    scale = (.2338 + .012 + .008) / .26
    expected = np.array([[0., 0., scale, .185 + .0075 * scale],
                         [0., scale, 0., 0.],
                         [-scale, 0., 0., -.035 + .05 * scale],
                         [0., 0., 0., 1.]])
    np.testing.assert_allclose(matrix, expected, atol=1e-12)
    assert report["yawCorrectionDeg"] == 0
    assert report["targetFoot"]["length"] == pytest.approx(.2338)


@pytest.mark.parametrize("yaw", [-20., -2., 2., 20., 40.])
def test_explicit_toe_corrects_diagonal_shoe_with_bounded_yaw(yaw):
    points, _, _ = make_synthetic_shoe_mesh()
    rotation = Rotation.from_euler("y", yaw - 90, degrees=True).as_matrix()
    source = points @ rotation.T * 100 + [25., 10., -30.]
    matrix, report = compute_shoe_alignment(source, body=synthetic_target_body(),
                                           up="+y", toe="-x")
    assert report["yawCorrectionDeg"] == pytest.approx(min(yaw, 30.), abs=1e-6)
    assert np.linalg.det(matrix[:3, :3]) > 0
    aligned = source @ matrix[:3, :3].T + matrix[:3, 3]
    assert aligned[:, 1].min() == pytest.approx(0., abs=1e-12)
    assert np.ptp(aligned[:, 2]) == pytest.approx(report["targetShoeLengthM"])
    if abs(yaw) <= 30:
        recovered = matrix[:3, :3] / report["scale"] @ rotation
        np.testing.assert_allclose(recovered, np.eye(3), atol=1e-12)
        assert report["sourceExtent"]["length"] == pytest.approx(26.)
    else:
        assert any("limited to 30" in warning for warning in report["warnings"])


def test_x_length_glb_default_up_does_not_become_length_axis(tmp_path):
    rotation = np.eye(4)
    rotation[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    path = create_shoe_glb(tmp_path / "y_up_x_length.glb", transform=rotation)
    _, report = align_shoe(path, body=synthetic_target_body())
    np.testing.assert_allclose(report["axes"]["up"], [0., 1., 0.])
    assert report["sourceExtent"]["length"] > .25
    assert report["targetShoeLengthM"] == pytest.approx(.2538)
    assert report["scale"] < 1.05


def test_missing_target_assets_fail_instead_of_guessing(tmp_path):
    points, _, _ = make_synthetic_shoe_mesh()
    with pytest.raises(FileNotFoundError):
        compute_shoe_alignment(points, body_path=tmp_path)


@pytest.mark.parametrize("margin", [-1., float("nan"), float("inf")])
def test_invalid_margins_rejected(margin):
    points, _, _ = make_synthetic_shoe_mesh()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        compute_shoe_alignment(points, body=synthetic_target_body(), length_margin_cm=margin)


def test_parallel_axes_and_flat_geometry_rejected():
    points, _, _ = make_synthetic_shoe_mesh()
    with pytest.raises(ValueError, match="perpendicular"):
        compute_shoe_alignment(points, body=synthetic_target_body(), up="+y", toe="-y")
    points[:, 1] = 0
    with pytest.raises(ValueError, match="nonzero"):
        compute_shoe_alignment(points, body=synthetic_target_body())


@pytest.mark.parametrize("axis", ["-x", "-y", "-z", "+x", "+y", "+z", "auto"])
def test_parser_accepts_separate_signed_axis_values(axis):
    args = parser().parse_args(["shoe.glb", "--up", axis, "--toe", axis,
                                "--node", "left", "--node", "sole",
                                "--exclude-material", "logo", "--exclude-material", "floor"])
    assert args.up == axis and args.toe == axis
    assert args.node == ["left", "sole"]
    assert args.exclude_material == ["logo", "floor"]


def test_material_exclusion_is_applied(tmp_path):
    path = create_shoe_glb(tmp_path / "excluded.glb")
    import pygltflib as gl

    g = gl.GLTF2().load(str(path))
    g.materials[0].name = "excluded-material"
    g.save_binary(str(path))
    with pytest.raises(ValueError):
        align_shoe(path, exclude_material=["excluded-material"], body=synthetic_target_body())
