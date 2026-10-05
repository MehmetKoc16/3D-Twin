"""Synthetic geometry only: no user meshes, landmarks, textures or photos."""

import importlib

import numpy as np
import pytest
import trimesh

from hybridbody.glasses_hy3d import TwinHead, fit_accessory, measured_curves, segment_glasses

# Import the standalone generator after glasses_hy3d registers its stage folder.
accessory = importlib.import_module("accessory_glb")
generator = importlib.import_module("make_glasses")
accessor_array = accessory.accessor_array
read_glb_bytes = accessory.read_glb_bytes
validate_glasses = accessory.validate_glasses
encode_glasses = generator.encode_glasses


def synthetic_head():
    mesh = trimesh.creation.icosphere(subdivisions=3)
    points = mesh.vertices * [0.078, 0.12, 0.085] + [0, 0.10, 0]
    return TwinHead(
        points,
        mesh.faces,
        np.zeros((len(points), 2)),
        np.array([[-0.032, 0.105, 0.072], [0.032, 0.105, 0.072]]),
        np.array([0, 0.102, 0.10]),
        np.array([[-0.080, 0.110, 0], [0.080, 0.110, 0]]),
        np.empty((0, 3)),
        np.empty(0, int),
        np.eye(4),
        {},
    )


def test_symmetric_fit_eye_centres_clearance_and_contract():
    head = synthetic_head()
    params, geometry, _, arms, metrics = fit_accessory(head, np.array([0.024, 0.023]))
    assert metrics["triangles"] < 6000
    assert max(metrics["eye_xy_error_mm"]) < 1e-7
    assert metrics["min_surface_clearance_mm"] >= 1
    assert metrics["bridge_clearance_mm"] >= 1
    assert np.allclose(arms[:28] * [-1, 1, 1], arms[28:])
    # Verify triangle topology by welding cap copies, not array index seams.
    mesh = trimesh.Trimesh(geometry[0], geometry[2], process=True)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert np.min(mesh.area_faces) > 1e-12
    raw = encode_glasses(params, np.zeros(3), "synthetic measured fit", geometry=geometry, placement=metrics)
    assert validate_glasses(raw) == params
    doc, blob = read_glb_bytes(raw)
    attrs = doc["meshes"][0]["primitives"][0]["attributes"]
    assert np.all(accessor_array(doc, blob, attrs["WEIGHTS_0"]) == [1, 0, 0, 0])
    assert doc["materials"][0]["pbrMetallicRoughness"]["metallicFactor"] == 1
    assert not doc.get("images")


def test_segmentation_excludes_blue_hair_and_flags_skin_filled_lenses():
    eyes = np.array([[-0.032, 0.1, 0.08], [0.032, 0.1, 0.08]])
    rims = np.array([[-0.032, 0.09, 0.024, 0.024], [0.032, 0.09, 0.024, 0.024]])
    p = np.array([[-0.056, 0.09, 0.115], [-0.032, 0.09, 0.115], [0.056, 0.09, 0.115], [0.056, 0.09, 0.115]])
    n = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0]])
    colours = np.array([[185, 180, 175], [180, 145, 125], [30, 50, 190], [20, 20, 20]])
    frame, lens, report = segment_glasses(p, n, colours, eyes, rims)
    assert frame.tolist() == [True, False, False, False]
    assert lens.tolist() == [False, True, False, False]
    assert report["skin_coloured_lens_fraction"] == 1


def test_measured_mask_keeps_original_vertical_drop():
    eyes = np.array([[-0.032, 0.10, 0.08], [0.032, 0.10, 0.08]])
    rims = np.array([[-0.032, 0.082, 0.024, 0.024], [0.032, 0.082, 0.024, 0.024]])
    curves = measured_curves(rims, eyes, 0.12)
    assert np.isfinite(curves).all()
    assert np.allclose(curves[:360, 1].mean(), 0.082)
    assert curves[:, 1].min() < eyes[:, 1].min() - 0.04


@pytest.mark.parametrize("radii", [[np.nan, 0.02], [0.05, 0.02], [0.02, 0.001]])
def test_invalid_radii_fail_closed(radii):
    with pytest.raises(ValueError, match="radii"):
        fit_accessory(synthetic_head(), np.array(radii))
