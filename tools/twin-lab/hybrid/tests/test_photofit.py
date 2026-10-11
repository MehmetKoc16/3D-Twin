"""Synthetic cameras/photos only; never load private data or licensed FLAME files."""

import importlib.util
import io
from pathlib import Path

import cv2
import numpy as np
import pytest

from hybridbody.photofit import (
    CameraPrior,
    camera_from_cv,
    camera_pose,
    ear_edge_guides,
    fit_camera,
    focal_prior,
    mirror_correspondences,
    mirrored_camera,
    photo_metadata,
    resolve_photos_set,
    same_lens,
    view_preferences,
    visible_landmarks,
)


def test_exif_focal_prior_uses_diagonal_and_does_not_group_unknown_or_zoomed_lenses():
    from PIL import Image

    image = Image.new("RGB", (600, 800))
    image.getexif()[271] = "Synthetic camera maker"
    image.getexif()[272] = "Synthetic phone"
    image.getexif()[34665] = {37386: 3.5, 41989: 24, 42036: "Synthetic selfie lens", 41988: 1}
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", exif=image.getexif())
    encoded.seek(0)
    with Image.open(encoded) as decoded:
        metadata = photo_metadata(decoded)
    assert metadata["focal_mm"] == 3.5
    assert metadata["focal_35mm"] == 24
    assert focal_prior(metadata, (600, 800)).focal_px == pytest.approx(24 * 1000 / np.hypot(36, 24))
    assert same_lens(metadata, metadata)
    assert not same_lens(metadata, {**metadata, "digital_zoom": 2})
    assert not same_lens(metadata, {**metadata, "lens_model": "Synthetic rear lens"})
    unknown = {key: None if isinstance(value, (int, float)) else "" for key, value in metadata.items()}
    assert not same_lens(unknown, unknown)
    assert focal_prior(unknown, (800, 600)).focal_px == focal_prior(unknown, (600, 800)).focal_px
    physical_only = {**metadata, "focal_35mm": None}
    assert focal_prior(physical_only, (600, 800)).focal_log_sigma == 0.30


def test_z_test_removes_occluded_front_facing_landmarks():
    from flamehead.camera import Camera

    # Both planes face the camera. A normal-only test retains the hidden plane.
    vertices = np.array([[-.1, -.1, .1], [.1, -.1, .1], [.1, .1, .1], [-.1, .1, .1]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    points = np.array([[.01, .02, .1], [.01, .02, .05]])
    normals = np.tile([0., 0., 1.], (2, 1))
    extrinsic = np.eye(4)
    extrinsic[2, 3] = -.5
    camera = Camera(np.array([[200, 0, 128], [0, 200, 128], [0, 0, 1]]), extrinsic,
                    np.array([0, 256, 0, 256]), (256, 256), (256, 256))
    assert visible_landmarks(camera, points, normals).tolist() == [True, True]
    assert visible_landmarks(camera, points, normals, vertices, faces).tolist() == [True, False]


def test_ear_contour_uses_only_local_synthetic_edge_evidence():
    import trimesh
    from flamehead.camera import Camera

    mesh = trimesh.creation.icosphere(subdivisions=2, radius=.015)
    mesh.vertices += [.085, 0, .03]
    extrinsic = np.eye(4)
    extrinsic[2, 3] = -.5
    camera = Camera(np.array([[500, 0, 128], [0, 500, 128], [0, 0, 1]]), extrinsic,
                    np.array([0, 256, 0, 256]), (256, 256), (256, 256))
    rgb = np.full((256, 256, 3), 180, np.uint8)
    ids = np.arange(len(mesh.vertices))
    points, _, report = ear_edge_guides(mesh.vertices, mesh.faces, ids, camera, rgb)
    assert points is None and report["samples"] == 0
    hull = cv2.convexHull(np.rint(camera.project(mesh.vertices)[:, :2]).astype(np.int32))
    cv2.fillConvexPoly(rgb, hull, (70, 70, 70))
    points, pixels, report = ear_edge_guides(mesh.vertices, mesh.faces, ids, camera, rgb)
    assert report["samples"] >= 6
    assert report["semantic_ear_detection"] is False
    assert np.linalg.norm(camera.project(points)[:, :2] - pixels, axis=1).max() <= 12


@pytest.mark.parametrize("yaw", [-78, 78])
def test_near_profile_with_hallucinated_hidden_features_recovers_known_camera(yaw):
    import trimesh
    from scipy.spatial.transform import Rotation

    mesh = trimesh.creation.icosphere(subdivisions=3, radius=.085)
    mesh.vertices[:, 1] *= 1.15
    rng = np.random.default_rng(817)
    ids = rng.choice(np.flatnonzero(mesh.vertices[:, 2] > .005), 105, replace=False)
    points = mesh.vertices[ids]
    normals = mesh.vertex_normals[ids]
    rotation = np.diag([1., -1., -1.]) @ Rotation.from_euler("xyz", [3, -yaw, 8], degrees=True).as_matrix()
    expected = camera_from_cv(np.r_[cv2.Rodrigues(rotation)[0].ravel(), [0., 0., .55], np.log(1100)], (1200, 1600))
    visible = visible_landmarks(expected, points, normals, mesh.vertices, mesh.faces)
    clean = expected.project(points)[:, :2]
    pixels = clean + rng.normal(0, .3, clean.shape)
    pixels[~visible] += rng.uniform(30, 65, ((~visible).sum(), 2))
    weak_perspective_seed = camera_from_cv(
        np.r_[cv2.Rodrigues(rotation)[0].ravel(), [0., 0., .55 * 8000 / 1100], np.log(8000)], (1200, 1600),
    )
    fitted, report, inliers = fit_camera(
        points, pixels, (1200, 1600), normals=normals, mesh=mesh.vertices, faces=mesh.faces,
        prior=CameraPrior(1100, .025, .06, azimuth_hint_deg=yaw), initial_camera=weak_perspective_seed,
    )
    error = np.linalg.norm(fitted.project(points)[visible, :2] - clean[visible], axis=1)
    assert np.sqrt(np.mean(error ** 2)) < 1.5
    assert abs(fitted.intrinsic[0, 0] / 1100 - 1) < .04
    expected_pose = camera_pose(expected, points.mean(0))
    fitted_pose = camera_pose(fitted, points.mean(0))
    assert np.max(np.abs(np.asarray(expected_pose) - fitted_pose)) < 3
    assert inliers.sum() >= visible.sum() - 5
    assert not inliers[~visible].any()
    assert report["all_rms_px"] > report["visible_rms_px"] * 5
    assert report["visibility_method"] == "normal facing + iterative mesh z-test"


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
        points, camera.project(points)[:, :2], (1200, 1600), outline_points=outline, outline_pixels=contour,
        ear_points=outline[10:30:3], ear_pixels=camera.project(outline[10:30:3])[:, :2],
    )
    assert np.max(np.linalg.norm(fitted.project(points)[:, :2] - camera.project(points)[:, :2], axis=1)) < 0.01
    assert report["outline_refinement"]["rms_after_px"] < 0.01
    assert report["ear_refinement"]["rms_after_px"] < 0.01


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


def _synthetic_head(seed=5):
    """Ellipsoid 'head' with 105 embedding points on its front, plus masks shaped like the FLAME ones."""
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=3, radius=.085)
    mesh.vertices[:, 1] *= 1.15
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.flatnonzero(mesh.vertices[:, 2] > .005), 105, replace=False)
    front = np.flatnonzero(mesh.vertices[:, 2] > .0)
    masks = {key: np.array([], int) for key in ("forehead", "eye_region", "nose", "lips")}
    masks["face"] = front
    masks["left_ear"] = np.flatnonzero((mesh.vertices[:, 0] > .07) & (np.abs(mesh.vertices[:, 2]) < .03))
    masks["right_ear"] = np.flatnonzero((mesh.vertices[:, 0] < -.07) & (np.abs(mesh.vertices[:, 2]) < .03))
    return mesh, ids, masks


def _synthetic_camera(yaw, pitch, roll, size=(1200, 1600), focal=1100):
    from scipy.spatial.transform import Rotation

    rotation = np.diag([1., -1., -1.]) @ Rotation.from_euler("xyz", [pitch, -yaw, roll], degrees=True).as_matrix()
    return camera_from_cv(np.r_[cv2.Rodrigues(rotation)[0].ravel(), [0., 0., .55], np.log(focal)], size)


def test_fit_photo_set_profiles_use_loose_roll_tight_elevation_and_skip_oval_guides(tmp_path, monkeypatch):
    """A profile with a 35 degree head nod (camera roll) and a level camera: the nod must survive the priors."""
    from types import SimpleNamespace

    from PIL import Image
    from twinrefine.landmarks import FaceLandmarks

    import hybridbody
    from hybridbody import photofit

    mesh, ids, masks = _synthetic_head()
    flame = SimpleNamespace(neutral=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces),
                            landmark_points=np.asarray(mesh.vertices[ids]), landmark_ids=np.arange(105) * 4 + 3, masks=masks)
    truth = {"front": _synthetic_camera(0, 2, 0), "left": _synthetic_camera(72, 0, 35),
             "right": _synthetic_camera(-72, 0, -35)}
    size = (1200, 1600)
    rng = np.random.default_rng(3)
    folder = tmp_path / "head"
    folder.mkdir()
    normals = mesh.vertex_normals[ids]
    detections = {}
    oval_points = rng.uniform([100, 100], [1100, 1500], (478, 2))
    angle = np.linspace(0, 2 * np.pi, len(photofit.FACE_OVAL), endpoint=False)
    oval_points[photofit.FACE_OVAL] = np.c_[600 + 400 * np.cos(angle), 800 + 600 * np.sin(angle)]
    for index, (name, camera) in enumerate(truth.items()):
        Image.new("RGB", size, (140, 120, 110)).save(folder / f"nog_{name}.jpeg")
        xy = oval_points.copy()  # the same detected oval in every photo
        projected = camera.project(flame.landmark_points)[:, :2] + rng.normal(0, .3, (105, 2))
        hidden = ~visible_landmarks(camera, flame.landmark_points, normals, mesh.vertices, mesh.faces)
        projected[hidden] += rng.uniform(30, 65, (hidden.sum(), 2))
        xy[flame.landmark_ids] = projected
        detections[index] = FaceLandmarks(xy, np.zeros(478), (0, 0, *size))
    calls = iter(range(3))
    monkeypatch.setattr(photofit, "detect_photo", lambda rgb: (detections[next(calls)], {"method": "synthetic"}))
    monkeypatch.setattr(hybridbody, "REPO", tmp_path)
    audit = tmp_path / "user-data" / "photo_fits"
    cameras, photos, report, face_masks = photofit.fit_photo_set(flame, folder, audit)
    for name in ("left", "right"):
        expected = np.asarray(camera_pose(truth[name], flame.landmark_points.mean(0)))
        fitted = np.asarray(camera_pose(cameras[name], flame.landmark_points.mean(0)))
        assert np.max(np.abs(expected - fitted)) < 4, (name, expected, fitted)
        assert report[name]["outline_refinement"] is None and report[name]["outline_guides_enabled"] is False
        assert report[name]["priors"]["elevation_sigma_deg"] == photofit.PROFILE_ELEVATION_SIGMA_DEG
        assert report[name]["priors"]["roll_sigma_deg"] == photofit.PROFILE_ROLL_SIGMA_DEG
        assert report[name]["neck_pitch_correction"]["applied_deg"] == 0
        assert abs(report[name]["neck_pitch_correction"]["profile_head_pitch_equivalent_deg"] - fitted[2]) < 1e-6
        assert "large roll" not in " ".join(report[name]["quality_flags"])
        registration = report[name]["landmark_registration_vs_front"]["after"]
        assert registration["median_mm"] < 3
    assert report["front"]["priors"]["azimuth_hint_deg"] is None
    assert (audit / "camera_fits.json").is_file() and set(face_masks) == {"front", "left", "right"}
    # Side photos weight the oval boundary band down more than the front (hair strands sit there in a nodded profile).
    soft = {name: ((0 < mask) & (mask < 1)).sum() for name, mask in face_masks.items()}
    assert soft["left"] > 3 * soft["front"] and soft["right"] > 3 * soft["front"]


def test_landmark_registration_measures_cross_view_displacement_in_millimetres():
    import trimesh

    from hybridbody.photofit import landmark_registration, surface_hits

    mesh = trimesh.creation.icosphere(subdivisions=3, radius=.085)
    rng = np.random.default_rng(11)
    ids = rng.choice(np.flatnonzero(mesh.vertices[:, 2] > .02), 60, replace=False)
    points, normals = mesh.vertices[ids], mesh.vertex_normals[ids]
    front, side = _synthetic_camera(0, 0, 0), _synthetic_camera(35, 0, 0)
    hits = surface_hits(mesh.vertices, mesh.faces, front, front.project(points)[:, :2])
    assert np.allclose(hits, points, atol=1e-4)
    assert np.isnan(surface_hits(mesh.vertices, mesh.faces, front, np.array([[5., 5.]]))).all()
    args = (mesh.vertices, mesh.faces, points, normals, front, front.project(points)[:, :2], side)
    perfect = landmark_registration(*args, side.project(points)[:, :2])
    assert perfect["landmarks_seen_by_both"] > 20 and perfect["median_mm"] < .1
    shifted = side.project(points)[:, :2] + [20., 0.]
    report = landmark_registration(*args, shifted)
    # 20 px at ~0.55 m with a 1100 px focal is ~10 mm across the image plane, a bit more on the oblique surface.
    assert 8 < report["median_mm"] < 25
    subset = np.arange(len(points)) < 10
    assert landmark_registration(*args, shifted, only=subset)["landmarks_seen_by_both"] <= 10
    assert landmark_registration(*args, shifted, only=np.zeros(len(points), bool))["median_mm"] is None


def test_ear_edge_guides_alone_refine_the_camera_without_oval_guides():
    rng = np.random.default_rng(10)
    points = rng.uniform([-0.07, -0.08, 0.02], [0.07, 0.08, 0.1], (105, 3))
    camera = _synthetic_camera(40, 0, 0)
    theta = np.linspace(0, np.pi, 40)
    ear = np.c_[0.075 * np.ones(40), 0.02 * np.cos(theta), 0.05 + 0.02 * np.sin(theta)]
    off_guides = camera.project(ear)[:, :2] + [12., -9.]
    fitted, report, _ = fit_camera(points, camera.project(points)[:, :2], (1200, 1600), ear_points=ear, ear_pixels=off_guides)
    assert report["outline_refinement"] is None
    assert report["ear_refinement"]["samples"] == 40
    assert report["ear_refinement"]["rms_after_px"] < report["ear_refinement"]["rms_before_px"]
    # Landmarks stay the anchor: an inconsistent 15 px ear guide moves the landmarks by less than the guide offset.
    assert np.mean(np.linalg.norm(fitted.project(points)[:, :2] - camera.project(points)[:, :2], axis=1)) < 8
