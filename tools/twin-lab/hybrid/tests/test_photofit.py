"""Synthetic cameras/photos only; never load private data or licensed FLAME files."""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest
from hybridbody.photofit import (
    camera_from_cv,
    fit_camera,
    mirror_correspondences,
    mirrored_camera,
    resolve_photos_set,
    view_preferences,
)


@pytest.mark.parametrize("yaw", [0, 65, -55])
def test_pnp_recovers_known_perspective_with_outliers(yaw):
    rng = np.random.default_rng(8)
    points = rng.uniform([-0.065, -0.080, 0.02], [0.065, 0.080, 0.1], (105, 3))
    # A convex face-like surface, with a nose protrusion and enough depth for focal recovery.
    points[:, 2] = 0.025 + 0.06 * np.exp(-((points[:, 0] / 0.04) ** 2 + (points[:, 1] / 0.06) ** 2))
    rotation = np.diag([1.0, -1.0, -1.0]) @ cv2.Rodrigues(np.array([0.0, np.radians(yaw), 0.0]))[0]
    expected = camera_from_cv(np.r_[cv2.Rodrigues(rotation)[0].ravel(), [0.01, 0.005, 0.5], np.log(1100)], (1200, 1600))
    clean = expected.project(points)[:, :2]
    pixels = clean + rng.normal(0, 0.3, clean.shape)
    pixels[:10] += rng.uniform(40, 90, (10, 2))
    camera, report, inliers = fit_camera(points, pixels, (1200, 1600))
    error = np.linalg.norm(camera.project(points)[10:, :2] - clean[10:], axis=1)
    assert np.sqrt(np.mean(error**2)) < 0.5
    assert abs(camera.intrinsic[0, 0] / 1100 - 1) < 0.05
    assert report["rms_px"] < 1
    assert report["all_rms_px"] > 15
    assert inliers.sum() >= 90
    assert not inliers[:10].any()
    lifted = camera.unproject(camera.project(points)[:, :2], camera.project(points)[:, 2])
    assert np.allclose(lifted, points)


def test_anatomical_weights_favour_front_and_reject_tilted_right_chin():
    points = np.array([[0.0, 0.05, 0.08], [0.08, 0.05, 0.0], [0.0, -0.01, 0.04], [0.06, -0.02, 0.0]])
    weight = view_preferences(points, chin_y=0.0)
    assert weight["front"][0] / weight["left"][0] > 70
    assert weight["left"][1] == weight["right"][1] == weight["front"][1] == 1
    assert weight["front"][2] > weight["front"][0]
    assert weight["right"][2] < weight["left"][2] / 10
    assert weight["right"][3] < 0.03
    assert set(weight) == {"front", "left", "right"}


def test_auto_defaults_and_incomplete_set_fails(tmp_path):
    old = tmp_path / "colab_upload"
    old.mkdir()
    assert resolve_photos_set("auto", old) == ("glasses", old)
    (tmp_path / "nog_front.jpeg").touch()
    with pytest.raises(ValueError, match="Incomplete"):
        resolve_photos_set("auto", old)
    for view in ("left", "right"):
        (tmp_path / f"nog_{view}.jpeg").touch()
    assert resolve_photos_set("auto", old) == ("noglasses", tmp_path)
    assert resolve_photos_set("glasses", old) == ("glasses", old)


def test_source_unmirror_swaps_bilateral_correspondences_and_keeps_proper_camera():
    rng = np.random.default_rng(12)
    side = rng.uniform([0.01, -0.08, 0.02], [0.08, 0.08, 0.08], (52, 3))
    points = np.vstack((side, side * [-1, 1, 1], [[0.0, 0.0, 0.08]]))
    permutation = mirror_correspondences(points)
    assert np.array_equal(permutation[:52], np.arange(52, 104))
    rotation = np.diag([1.0, -1.0, -1.0]) @ cv2.Rodrigues(np.array([0.0, 0.7, 0.0]))[0]
    camera = camera_from_cv(np.r_[cv2.Rodrigues(rotation)[0].ravel(), [0.02, 0.01, 0.6], np.log(1100)], (1200, 1600))
    mirrored = mirrored_camera(camera)
    expected = camera.project(points[permutation])
    expected[:, 0] = 1200 - expected[:, 0]
    assert np.allclose(mirrored.project(points), expected)
    assert np.linalg.det(mirrored.extrinsic[:3, :3]) == pytest.approx(1.0)


def test_outline_refinement_preserves_a_known_synthetic_camera():
    rng = np.random.default_rng(10)
    points = rng.uniform([-0.07, -0.08, 0.02], [0.07, 0.08, 0.1], (105, 3))
    rotation = np.diag([1.0, -1.0, -1.0]) @ cv2.Rodrigues(np.array([0.0, 0.8, 0.0]))[0]
    camera = camera_from_cv(np.r_[cv2.Rodrigues(rotation)[0].ravel(), [0.0, 0.0, 0.5], np.log(1100)], (1200, 1600))
    theta = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    outline = np.c_[0.07 * np.cos(theta), 0.08 * np.sin(theta), np.full(len(theta), 0.04)]
    contour = camera.project(outline[::15])[:, :2]
    fitted, report, _ = fit_camera(
        points, camera.project(points)[:, :2], (1200, 1600), outline_points=outline, outline_pixels=contour
    )
    assert np.max(np.linalg.norm(fitted.project(points)[:, :2] - camera.project(points)[:, :2], axis=1)) < 0.01
    assert report["outline_refinement"]["rms_after_px"] < 0.01


def test_three_view_bake_never_adds_a_mirror_and_masks_photo_ears_off_face(monkeypatch):
    from flamehead import texture as baker
    from flamehead.camera import Camera

    neutral = np.array([[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [-0.1, 0.1, 0.0]])
    faces = np.array([[0, 1, 2]])
    fid = np.zeros((32, 32), int)
    y, x = np.nonzero(fid >= 0)
    tri = np.tile(faces, (len(y), 1))
    bary = np.full((len(y), 3), 1 / 3)
    names = ("front", "left", "right")
    camera = Camera(np.eye(3), np.eye(4), np.array([0, 32, 0, 32]), (32, 32), (32, 32))
    photos = {
        name: np.full((32, 32, 3), colour, np.uint8)
        for name, colour in zip(names, ([160, 110, 90], [100, 160, 110], [110, 100, 160]), strict=True)
    }
    monkeypatch.setattr(baker, "visibility", lambda *args: (None, None, None))
    monkeypatch.setattr(
        baker,
        "project_samples",
        lambda points, normals, photo, *args: (
            np.tile(photo[0, 0], (len(points), 1)),
            np.ones(len(points), np.float32),
        ),
    )
    monkeypatch.setattr(camera, "project", lambda points: np.tile([10.0, 10.0, 1.0], (len(points), 1)))
    masks = {"face": np.arange(3), "left_eyeball": np.array([], int), "right_eyeball": np.array([], int)}
    view_masks = {name: np.ones((32, 32), np.float32) for name in names}
    args = (
        neutral,
        faces,
        fid,
        y,
        x,
        tri,
        bary,
        dict.fromkeys(names, neutral),
        dict.fromkeys(names, camera),
        photos,
        np.arange(3),
        masks,
    )
    _, direct = baker.bake_texels(*args, view_names=names, view_face_masks=view_masks)
    assert set(direct["view_weight_share"]) == set(names)
    assert direct["view_weight_share"]["left"] > 0.1
    view_masks["left"][:] = 0  # source pixels belong to the photographed ear/background, not the face
    _, masked = baker.bake_texels(*args, view_names=names, view_face_masks=view_masks)
    assert masked["view_weight_share"]["left"] == 0
    # A template forehead labelled as FLAME scalp still obeys the source oval.
    masks["face"] = np.array([], int)
    masks["photo_face_region"] = np.arange(3)
    _, scalp = baker.bake_texels(*args, view_names=names, view_face_masks=view_masks)
    assert scalp["view_weight_share"]["left"] == 0
    # Different white balance may need more than ten Lab b units of correction.
    from flamehead.colour import from_lab

    view_masks["left"][:] = 1
    for name, lab in zip(names, ([60., 9., 8.], [60., 9., 25.], [60., 9., 8.]), strict=True):
        photos[name][:] = np.rint(from_lab(np.array([lab], np.float32))[0] * 255).astype(np.uint8)
    _, harmonized = baker.bake_texels(*args, view_names=names, view_face_masks=view_masks, harmonize_chroma=True)
    assert harmonized["gains"]["left"]["chroma_offset_lab"][1] < -15
    assert harmonized["gains"]["left"]["overlap_colour"]["after_mean_delta_e76"] < 1
    photos["left"][:] = np.rint(from_lab(np.array([[25., 2., 3.]], np.float32))[0] * 255).astype(np.uint8)
    masks["eye_region"] = np.array([0, 1])  # synthetic texels lie well above this eye height
    _, clean_forehead = baker.bake_texels(*args, view_names=names, view_face_masks=view_masks, reject_forehead_hair=True)
    assert clean_forehead["forehead_photo_hair_rejected_texels"]["left"] == len(y)
    assert clean_forehead["view_weight_share"]["left"] == 0
    masks["eye_region"] = np.array([1, 2])  # eye height above the samples: brows/beard are protected
    _, protected = baker.bake_texels(*args, view_names=names, view_face_masks=view_masks, reject_forehead_hair=True)
    assert protected["forehead_photo_hair_rejected_texels"]["left"] == 0
    assert protected["view_weight_share"]["left"] > 0.1


def test_cli_passes_auto_and_explicit_photo_set(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / "hybrid.py"
    spec = importlib.util.spec_from_file_location("hybrid_cli_photos_test", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls = []
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(cli, "verify_report", lambda report: [])
    base = ["--bodyfix", str(tmp_path), "--out", str(tmp_path / "hybrid.glb")]
    assert cli.main(base) == 0
    assert calls[-1]["photos_set"] == "auto"
    assert cli.main([*base, "--photos-set", "noglasses"]) == 0
    assert calls[-1]["photos_set"] == "noglasses"
