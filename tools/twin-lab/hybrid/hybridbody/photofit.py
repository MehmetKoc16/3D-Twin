"""Local glasses-free photo cameras on the existing private FLAME identity.

The user supplies the licensed FLAME MediaPipe embedding in user-data/flame
(obtain it from flame.is.tue.mpg.de). No model data is bundled here. Detection
never draws, exports or displays a photo/crop; private numeric audits stay in
user-data. Camera axes are converted explicitly from OpenCV to FLAME.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from flamehead.camera import Camera
from PIL import Image, ImageOps
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial.transform import Rotation
from twinrefine.landmarks import FaceLandmarks, _run_detector, detect_landmarks

from .register import smoothstep

VIEWS = ("front", "left", "right")
CV_TO_FLAME = np.diag([1.0, -1.0, -1.0])
# Profile pose priors. A phone held at head height has a small camera elevation. In a profile, rotation about the
# optical axis (camera roll) is the same motion as the head nodding about the ear-to-ear axis, so roll must stay
# loose (the head may be tilted) while elevation is the quantity to pin. On the user's set a looser elevation prior
# (8 or 15 deg) left -14..-25 / +8..+20 deg elevations and a lower ear-region skin fraction (README, "Glasses-free
# profile camera audit").
PROFILE_ELEVATION_SIGMA_DEG = 4.0
PROFILE_ROLL_SIGMA_DEG = 40.0
# Photo face-oval weight of the side photos: zero within MARGIN of the detected oval, then a RAMP (pixels for a 2048 px
# long edge; the front keeps its 6 px ramp). Hair strands at the oval boundary of a nodded profile otherwise land on the
# forehead/temple as a grey smudge; with 12/24 the smudge fades while the side-only high-pass detail drops by about 8 %.
PROFILE_MASK_MARGIN_PX = 12.0
PROFILE_MASK_RAMP_PX = 24.0
FACE_OVAL = np.array(
    [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ]
)


@dataclass(frozen=True)
class CameraPrior:
    """Pixel focal prior and physical pose priors, independent of Euler conventions."""

    focal_px: float
    focal_log_sigma: float = 0.25
    focal_log_limit: float = 0.45
    elevation_deg: float = 0.0
    elevation_sigma_deg: float = 8.0
    roll_sigma_deg: float = 12.0
    azimuth_hint_deg: float | None = None

    def __post_init__(self):
        positive = (self.focal_px, self.focal_log_sigma, self.focal_log_limit,
                    self.elevation_sigma_deg, self.roll_sigma_deg)
        if not np.isfinite(positive).all() or min(positive) <= 0 or not np.isfinite(self.elevation_deg):
            raise ValueError("Camera prior scales must be finite and positive")
        if self.azimuth_hint_deg is not None and not np.isfinite(self.azimuth_hint_deg):
            raise ValueError("Invalid azimuth hint")


def photo_metadata(image) -> dict:
    """Read only camera metadata; physical focal length alone cannot give pixel focal."""
    exif = image.getexif()
    tags = dict(exif)
    try:
        tags.update(exif.get_ifd(34665))
    except (ValueError, TypeError, OSError, SyntaxError):
        # Damaged/missing nested metadata must not prevent a phone-FOV fallback.
        pass

    def positive(tag):
        try:
            value = float(tags.get(tag, 0))
            return value if np.isfinite(value) and value > 0 else None
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    return {
        "make": str(tags.get(271, "")).strip(),
        "model": str(tags.get(272, "")).strip(),
        "lens_model": str(tags.get(42036, "")).strip(),
        "focal_mm": positive(37386),
        "focal_35mm": positive(41989),
        "digital_zoom": positive(41988),
    }


def focal_prior(metadata, size):
    """35 mm equivalence uses image diagonal; fallback is an explicit 24 mm phone prior."""
    equivalent = metadata["focal_35mm"] or 24.0
    focal = equivalent * np.hypot(*size) / np.hypot(36, 24)
    measured = metadata["focal_35mm"] is not None
    return CameraPrior(focal, 0.12 if measured else 0.30, 0.25 if measured else 0.55)


def same_lens(a, b):
    """Do not equate absent EXIF, different phone lenses, or different digital zooms."""
    if not a["model"] or (a["make"], a["model"]) != (b["make"], b["model"]):
        return False
    if a["lens_model"] != b["lens_model"]:
        return False
    for key in ("focal_mm", "focal_35mm", "digital_zoom"):
        if (a[key] is None) != (b[key] is None):
            return False
        if a[key] is not None and not np.isclose(a[key], b[key], rtol=0.02):
            return False
    return bool(a["lens_model"] or a["focal_mm"] or a["focal_35mm"])


def camera_pose(camera, centre):
    """Azimuth/elevation around the embedded face; roll is projected world-up tilt."""
    toward = camera.center - centre
    yaw = np.degrees(np.arctan2(toward[0], toward[2]))
    elevation = np.degrees(np.arctan2(toward[1], np.hypot(toward[0], toward[2])))
    rotation = camera.extrinsic[:3, :3]
    roll = np.degrees(np.arctan2(rotation[0, 1], rotation[1, 1]))
    return float(yaw), float(elevation), float(roll)


def visible_landmarks(camera, points, normals=None, mesh=None, faces=None):
    """Facing plus perspective mesh z-test, refreshed outside each continuous solve."""
    projection = camera.project(points)
    visible = projection[:, 2] > 0
    if normals is not None:
        toward = camera.center - points
        toward /= np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-12)
        visible &= np.einsum("ij,ij->i", normals, toward) > 0.05
    if mesh is not None:
        from flamehead.texture import visibility

        depth, _, scale = visibility(mesh, faces, camera)
        xy = np.floor(projection[:, :2] * scale).astype(int)
        inside = ((xy >= 0) & (xy < [depth.shape[1], depth.shape[0]])).all(1)
        xy = np.clip(xy, 0, [depth.shape[1] - 1, depth.shape[0] - 1])
        surface = depth[xy[:, 1], xy[:, 0]]
        # Embedding is barycentric on the mesh; allow raster sampling error,
        # but not far-side landmarks behind the nose/cheek (typically >10 mm).
        visible &= inside & np.isfinite(surface) & (np.abs(projection[:, 2] - surface) < 0.0025)
    return visible


def surface_hits(mesh, faces, camera, pixels):
    """First mesh intersection of the rays through ``pixels`` (Moller-Trumbore); NaN rows where a ray misses."""
    triangle = mesh[faces]
    edge1, edge2 = triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
    origin = camera.center
    towards = camera.unproject(np.asarray(pixels, float), np.ones(len(pixels))) - origin
    towards /= np.linalg.norm(towards, axis=1, keepdims=True)
    hits = np.full((len(pixels), 3), np.nan)
    for i, ray in enumerate(towards):
        h = np.cross(ray, edge2)
        determinant = np.einsum("ij,ij->i", edge1, h)
        parallel = np.abs(determinant) < 1e-14
        inverse = 1 / np.where(parallel, 1, determinant)
        s = origin - triangle[:, 0]
        u = inverse * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, edge1)
        v = inverse * (q @ ray)
        t = inverse * np.einsum("ij,ij->i", edge2, q)
        # A ray through a shared vertex/edge must still hit: tolerate rounding at the triangle borders.
        inside = ~parallel & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 0)
        if inside.any():
            hits[i] = origin + ray * t[inside].min()
    return hits


def seen_by_both(mesh, faces, points, normals, front, other):
    return visible_landmarks(front, points, normals, mesh, faces) & visible_landmarks(other, points, normals, mesh, faces)


def landmark_registration(mesh, faces, points, normals, front, front_pixels, other, other_pixels, only=None):
    """Texture registration of two photo cameras at the facial landmarks both can see, in millimetres.

    Each detected landmark is cast through its own camera onto the identity mesh. If both cameras and detections were
    perfect the two hits would coincide; their distance is how far a feature is displaced between the front bake and
    the side bake (a ghost), without any texture. Detector noise adds to it, so compare cameras on identical photos
    and, via ``only``, on the same landmark subset.
    """
    seen = seen_by_both(mesh, faces, points, normals, front, other)
    if only is not None:
        seen &= only
    report = {"landmarks_seen_by_both": int(seen.sum())}
    if seen.sum() < 4:
        return {**report, "median_mm": None}
    a = surface_hits(mesh, faces, front, front_pixels[seen])
    b = surface_hits(mesh, faces, other, other_pixels[seen])
    distance = np.linalg.norm(a - b, axis=1) * 1000
    distance = distance[np.isfinite(distance)]
    if len(distance) < 4:
        return {**report, "median_mm": None}
    return {**report, "ray_pairs": int(len(distance)), "median_mm": float(np.median(distance)),
            "mean_mm": float(distance.mean()), "p90_mm": float(np.percentile(distance, 90))}


def robust_residual(values, scale):
    """Soft-L1 data residuals with linear least_squares: priors remain quadratic."""
    values = np.asarray(values)
    ratio = values / scale
    return values * np.sqrt(2 / (np.sqrt(1 + ratio * ratio) + 1))


def resolve_photos_set(choice: str, photos: Path) -> tuple[str, Path]:
    """The launcher passes colab_upload; its parent contains nog_*.jpeg."""
    if choice not in ("auto", "glasses", "noglasses"):
        raise ValueError("Unknown photo set")
    photos = Path(photos)
    nog = photos.parent if photos.name == "colab_upload" else photos
    selected = "noglasses" if choice == "auto" and (nog / "nog_front.jpeg").is_file() else choice
    if selected == "auto":
        selected = "glasses"
    if selected == "noglasses":
        missing = [f"nog_{view}.jpeg" for view in VIEWS if not (nog / f"nog_{view}.jpeg").is_file()]
        if missing:
            raise ValueError("Incomplete glasses-free set: " + ", ".join(missing))
        return selected, nog
    return selected, photos


def camera_from_cv(parameters: np.ndarray, size: tuple[int, int]) -> Camera:
    rotation = cv2.Rodrigues(parameters[:3])[0]
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = CV_TO_FLAME @ rotation
    extrinsic[:3, 3] = CV_TO_FLAME @ parameters[3:6]
    w, h = size
    focal = np.exp(parameters[6])
    intrinsic = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
    return Camera(intrinsic, extrinsic, np.array([0, h, 0, w]), size, size)


def fit_camera(
    points, pixels, size, *, normals=None, initial_camera=None, outline_points=None, outline_pixels=None,
    prior=None, mesh=None, faces=None, ear_points=None, ear_pixels=None,
) -> tuple[Camera, dict, np.ndarray]:
    """Focal grid + RANSAC PnP, then robust joint rotation/translation/focal refinement.

    Hidden/back-facing embedded points are excluded after the initial pose. RMS
    in mm is the image-plane reprojection displacement at the fitted depth,
    not a measured 3D geometry error. Both all-point and inlier RMS are reported.
    """
    points, pixels = np.asarray(points, float), np.asarray(pixels, float)
    if points.shape != (len(pixels), 3) or pixels.shape[1:] != (2,) or len(points) < 12:
        raise ValueError("At least twelve 3D/2D landmarks required")
    if not np.isfinite(points).all() or not np.isfinite(pixels).all() or min(size) <= 0:
        raise ValueError("Invalid camera inputs")
    if (mesh is None) != (faces is None):
        raise ValueError("Visibility requires both mesh vertices and faces")
    if normals is not None and (np.shape(normals) != points.shape or not np.isfinite(normals).all()):
        raise ValueError("Invalid landmark normals")
    if (ear_points is None) != (ear_pixels is None):
        raise ValueError("Ear contour requires both 3D and image guides")
    image_scale = max(size)
    face_span = np.linalg.norm(np.ptp(pixels, axis=0))
    threshold = max(3.0, face_span * 0.025)
    if prior is not None:
        # A profile foreshortens face span, not detector pixel uncertainty.
        # Keep the resolution-dependent noise floor identical across the set.
        threshold = max(threshold, image_scale * 0.008)
    lower = np.r_[np.full(6, -np.inf), np.log(image_scale * 0.35)]
    upper = np.r_[np.full(6, np.inf), np.log(image_scale * 4.0)]
    if prior is not None:
        lower[6] = np.log(prior.focal_px) - prior.focal_log_limit
        upper[6] = np.log(prior.focal_px) + prior.focal_log_limit

    def pose_prior(parameters):
        if prior is None:
            return np.empty(0)
        cam = camera_from_cv(parameters, size)
        yaw, elevation, roll = camera_pose(cam, points.mean(0))
        weight = threshold
        residual = [
            (parameters[6] - np.log(prior.focal_px)) / prior.focal_log_sigma,
            (elevation - prior.elevation_deg) / prior.elevation_sigma_deg,
            roll / prior.roll_sigma_deg,
        ]
        if prior.azimuth_hint_deg is not None:
            residual.append(max(0, -np.sign(prior.azimuth_hint_deg) * yaw) / 10)
        return weight * np.asarray(residual)

    def candidate_score(parameters, error):
        # Pose priors participate in seed selection as well as refinement.
        if prior is not None and prior.azimuth_hint_deg is not None:
            visible = visible_landmarks(camera_from_cv(parameters, size), points, normals, mesh, faces)
            if visible.sum() < 12:
                return np.inf
            error = error[visible]
        return np.minimum(error, threshold).mean() + np.linalg.norm(pose_prior(parameters)) / 12

    candidates = []
    if initial_camera is not None:
        cv_rotation = CV_TO_FLAME @ initial_camera.extrinsic[:3, :3]
        cv_translation = CV_TO_FLAME @ initial_camera.extrinsic[:3, 3]
        p = np.r_[cv2.Rodrigues(cv_rotation)[0].ravel(), cv_translation, np.log(initial_camera.intrinsic[0, 0])]
        # Preserve approximate magnification when an old focal lies outside the new prior.
        focal_log = float(np.clip(p[6], lower[6] + 1e-8, upper[6] - 1e-8))
        p[3:6] *= np.exp(focal_log - p[6])
        p[6] = focal_log
        initial_camera = camera_from_cv(p, size)
        error = np.linalg.norm(initial_camera.project(points)[:, :2] - pixels, axis=1)
        candidates.append((candidate_score(p, error), p))
    cv2.setRNGSeed(1729)
    focals = image_scale * np.array([0.45, 0.65, 0.9, 1.2, 1.7, 2.4, 3.2])
    if prior is not None:
        focals = prior.focal_px * np.exp(np.linspace(-prior.focal_log_limit * 0.8, prior.focal_log_limit * 0.8, 5))
    for focal in focals:
        k = np.array([[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1]])
        ok, r, t, inliers = cv2.solvePnPRansac(
            points,
            pixels,
            k,
            None,
            iterationsCount=500,
            reprojectionError=threshold,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < 12:
            continue
        p = np.r_[r.ravel(), t.ravel(), np.log(focal)]
        cam = camera_from_cv(p, size)
        projected = cam.project(points)
        error = np.linalg.norm(projected[:, :2] - pixels, axis=1)
        if (projected[:, 2] > 0).all():
            candidates.append((candidate_score(p, error), p))
    if prior is not None and prior.azimuth_hint_deg is not None:
        # Profile MediaPipe far-side features are hallucinations. Generate seeds
        # from SUBJECT-side pose hypotheses and solve PnP only on their z-visible
        # subset, instead of letting the hidden eye win the initial RANSAC vote.
        centre = points.mean(0)
        for yaw in (prior.azimuth_hint_deg - 15, prior.azimuth_hint_deg, prior.azimuth_hint_deg + 15):
            angle = np.radians(yaw)
            back = np.array([np.sin(angle), 0, np.cos(angle)])
            right = np.cross([0, 1, 0], back)
            rotation = np.vstack((right, np.cross(back, right), back))
            distance = prior.focal_px * np.ptp(points[:, 1]) / max(np.ptp(pixels[:, 1]), 1)
            translation = -rotation @ (centre + back * distance)
            target = np.median(pixels, axis=0) - np.array(size) / 2
            translation[:2] += target * [1, -1] * distance / prior.focal_px
            seed = np.r_[cv2.Rodrigues(CV_TO_FLAME @ rotation)[0].ravel(),
                         CV_TO_FLAME @ translation, np.log(prior.focal_px)]
            visible = visible_landmarks(camera_from_cv(seed, size), points, normals, mesh, faces)
            if visible.sum() < 12:
                continue
            for focal in focals:
                k = np.array([[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1]])
                ok, r, t, inliers = cv2.solvePnPRansac(
                    points[visible], pixels[visible], k, None, iterationsCount=500,
                    reprojectionError=threshold, confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
                )
                if ok and inliers is not None and len(inliers) >= 12:
                    p = np.r_[r.ravel(), t.ravel(), np.log(focal)]
                    cam = camera_from_cv(p, size)
                    projected = cam.project(points)
                    if (projected[:, 2] > 0).all():
                        error = np.linalg.norm(projected[:, :2] - pixels, axis=1)
                        candidates.append((candidate_score(p, error), p))
    candidates = [candidate for candidate in candidates if np.isfinite(candidate[0])]
    if not candidates:
        raise ValueError("No positive-depth robust PnP solution")
    p = min(candidates, key=lambda item: item[0])[1]
    selected = np.ones(len(points), bool)
    visibility_history = []
    for _ in range(6):
        cam = camera_from_cv(p, size)
        selected = visible_landmarks(cam, points, normals, mesh, faces)
        visibility_history.append(int(selected.sum()))
        error = np.linalg.norm(cam.project(points)[:, :2] - pixels, axis=1)
        selected &= error < max(threshold * 2, np.median(error[selected]) * 3) if selected.any() else False
        if selected.sum() < 12:
            raise ValueError("Too few visible landmarks for reliable profile fit")

        def residual(parameters, subset=selected):
            data = (camera_from_cv(parameters, size).project(points[subset])[:, :2] - pixels[subset]).ravel()
            return np.r_[robust_residual(data, threshold / 3), pose_prior(parameters)]

        result = least_squares(
            residual, p, bounds=(lower, upper), x_scale="jac", max_nfev=400
        )
        p = result.x
    outline_report = None
    ear_before = None
    have_outline = outline_points is not None and outline_pixels is not None
    if have_outline or ear_points is not None:
        from scipy.spatial import cKDTree

        seed = p.copy()

        def outline_error(camera):
            keep = visible_landmarks(camera, outline_points, mesh=mesh, faces=faces)
            if keep.sum() < 3:
                raise ValueError("Too few visible outline samples")
            return cKDTree(camera.project(outline_points[keep])[:, :2]).query(outline_pixels)[0]

        before = outline_error(camera_from_cv(p, size)) if have_outline else None
        if ear_points is not None:
            ear_before = np.linalg.norm(camera_from_cv(p, size).project(ear_points)[:, :2] - ear_pixels, axis=1)
        # Guide ICP refreshes 3D correspondence each iteration. Keep the robust
        # interior landmarks as anchors and bound pose drift from PnP.
        low = np.maximum(lower, seed - np.r_[np.full(3, 0.4), np.full(3, 0.10), 0.4])
        high = np.minimum(upper, seed + np.r_[np.full(3, 0.4), np.full(3, 0.10), 0.4])
        for _ in range(6):
            cam = camera_from_cv(p, size)
            selected = visible_landmarks(cam, points, normals, mesh, faces)
            errors = np.linalg.norm(cam.project(points)[:, :2] - pixels, axis=1)
            selected &= errors < max(threshold * 2, np.median(errors[selected]) * 3) if selected.any() else False
            if selected.sum() < 12:
                raise ValueError("Too few visible guide anchors")
            anchors = None
            if have_outline:
                boundary_visible = visible_landmarks(cam, outline_points, mesh=mesh, faces=faces)
                boundary_points = outline_points[boundary_visible]
                if len(boundary_points) < 3:
                    raise ValueError("Too few visible outline samples")
                projected = cam.project(boundary_points)[:, :2]
                anchors = boundary_points[cKDTree(projected).query(outline_pixels)[1]]
            ear_visible = (visible_landmarks(cam, ear_points, mesh=mesh, faces=faces)
                           if ear_points is not None else None)

            def guided_residual(parameters, subset=selected, anchor_points=anchors, ear_subset=ear_visible):
                cam = camera_from_cv(parameters, size)
                data = (cam.project(points[subset])[:, :2] - pixels[subset]).ravel()
                if anchor_points is not None:
                    boundary = cam.project(anchor_points)[:, :2] - outline_pixels
                    data = np.r_[data, boundary.ravel() * (3 if prior is not None else 2)]
                if ear_points is not None and len(ear_points):
                    ear = cam.project(ear_points[ear_subset])[:, :2] - ear_pixels[ear_subset]
                    data = np.r_[data, ear.ravel() * 2]
                return np.r_[robust_residual(data, threshold / 2), pose_prior(parameters)]

            p = least_squares(
                guided_residual,
                p,
                bounds=(low, high),
                x_scale="jac",
                max_nfev=200,
            ).x
        if have_outline:
            after = outline_error(camera_from_cv(p, size))
            outline_report = {
                "method": "visible FLAME cheek/jaw boundary to MediaPipe outline, robust ICP with pose/focal priors",
                "samples": len(outline_pixels),
                "rms_before_px": float(np.sqrt(np.mean(before**2))),
                "rms_after_px": float(np.sqrt(np.mean(after**2))),
            }
    cam = camera_from_cv(p, size)
    projection = cam.project(points)
    if (projection[:, 2] <= 0).any():
        raise ValueError("Refinement placed landmarks behind the camera")
    error = np.linalg.norm(projection[:, :2] - pixels, axis=1)
    visible = visible_landmarks(cam, points, normals, mesh, faces)
    inliers = visible & (error <= threshold)
    if inliers.sum() < 12:
        raise ValueError(f"Too few robust camera inliers: {inliers.sum()}/{visible.sum()} visible; "
                         f"pose={camera_pose(cam, points.mean(0))}; visible RMS={np.sqrt(np.mean(error[visible] ** 2)):.2f}px")
    mm = error * projection[:, 2] / cam.intrinsic[0, 0] * 1000

    def rms(values):
        return float(np.sqrt(np.mean(values**2)))

    pose = Rotation.from_matrix(cam.extrinsic[:3, :3]).as_euler("xyz", degrees=True)
    yaw, elevation, roll = camera_pose(cam, points.mean(0))
    report = {
        "landmarks": len(points),
        "visible_landmarks": int(visible.sum()),
        "inliers": int(inliers.sum()),
        "optimisation_anchors": int(selected.sum()),
        "rms_px": rms(error[inliers]),
        "rms_mm": rms(mm[inliers]),
        "visible_rms_px": rms(error[visible]),
        "all_rms_px": rms(error),
        "all_rms_mm": rms(mm),
        "p95_inlier_px": float(np.percentile(error[inliers], 95)),
        "ransac_threshold_px": threshold,
        "pose_xyz_deg": pose.tolist(),
        "pose_convention": "FLAME world-to-camera XYZ Euler; neutral front is 0,0,0",
        "mm_convention": "image-plane displacement at fitted depth; identity mesh in metres",
        "focal_px": float(cam.intrinsic[0, 0]),
        "original_size_wh": list(size),
        "camera_azimuth_deg": float(yaw),
        "camera_elevation_deg": float(elevation),
        "camera_roll_deg": roll,
        "visibility_method": "normal facing + iterative mesh z-test" if mesh is not None else "normal facing",
        "visibility_iterations": visibility_history,
        "priors": vars(prior) if prior is not None else None,
        "focal_near_bound": bool(p[6] < lower[6] + 0.01 or p[6] > upper[6] - 0.01),
        "intrinsics": cam.intrinsic.tolist(),
        "world_to_camera": cam.extrinsic.tolist(),
        "expression": "existing neutral identity, no expression or jaw optimisation",
        "outline_refinement": outline_report,
        "ear_refinement": {
            "samples": len(ear_points),
            "rms_before_px": rms(ear_before),
            "rms_after_px": rms(np.linalg.norm(cam.project(ear_points)[:, :2] - ear_pixels, axis=1)),
        } if ear_points is not None else None,
    }
    return cam, report, inliers


def detect_photo(rgb):
    """Retry difficult profiles with crops/rotations in memory; invert every transform."""
    detected = detect_landmarks(rgb, full_frame=True, target=1280)
    if detected is not None:
        return detected, {"method": "full frame"}
    h, w = rgb.shape[:2]
    for confidence in (0.5, 0.2):
        for height in (1.0, 0.85, 0.65):
            ch = int(h * height)
            for top in sorted({0, int((h - ch) / 2), h - ch}):
                crop = rgb[top : top + ch]
                for angle in (-90, 90, 180, 0, -20, 20, -40, 40):
                    detected = _detect_transformed(crop, top, angle, confidence, w, ch)
                    if detected is not None:
                        return detected, {
                            "method": "in-memory crop/rotation",
                            "crop_xyxy": [0, top, w, top + ch],
                            "rotation_deg": angle,
                            "confidence_threshold": confidence,
                        }
    return None, {"method": "failed"}


def _detect_transformed(crop, top, angle, confidence, w, ch):
    """One in-memory transformed detection with exact inverse pixel mapping."""
    scale = 1024 / max(ch, w)
    affine = cv2.getRotationMatrix2D((w / 2, ch / 2), angle, scale)
    affine[:, 2] += np.array([512 - w / 2, 512 - ch / 2])
    sub = cv2.warpAffine(crop, affine, (1024, 1024), borderMode=cv2.BORDER_REPLICATE)
    result = _run_detector(sub, confidence)
    if not result.face_landmarks:
        return None
    pts = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]])
    xy = np.c_[pts[:, :2] * 1024, np.ones(len(pts))] @ cv2.invertAffineTransform(affine).T
    xy[:, 1] += top
    return FaceLandmarks(xy, pts[:, 2] * 1024 / scale, (0, top, w, top + ch))


def mirror_correspondences(points):
    """Find semantic bilateral pairs in the local embedding; require an involution."""
    cost = np.linalg.norm((points * [-1, 1, 1])[:, None] - points[None, :], axis=2)
    _, permutation = linear_sum_assignment(cost)
    if (
        not np.array_equal(permutation[permutation], np.arange(len(points)))
        or cost[np.arange(len(points)), permutation].max() > 0.005
    ):
        raise ValueError("Embedding does not provide reliable bilateral landmark pairs")
    return permutation


def mirrored_camera(camera):
    """Reflect image and world X together: a proper rotation, never a mirrored mesh."""
    reflection = np.diag([-1.0, 1.0, 1.0, 1.0])
    return Camera(
        camera.intrinsic.copy(),
        reflection @ camera.extrinsic @ reflection,
        camera.crop.copy(),
        camera.original_size,
        camera.size,
    )


def ear_edge_guides(mesh, faces, ear_ids, camera, rgb):
    """Local edge evidence near the visible ear silhouette, never a semantic ear detector.

    The search is bounded to 12 source pixels around a posed ear contour. Hair
    and background can still produce edges: these guides are secondary, robust
    constraints, and their provenance/support are reported explicitly.
    """
    from scipy.spatial import cKDTree

    member = np.zeros(len(mesh), bool)
    member[ear_ids] = True
    triangles = mesh[faces]
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    facing = np.einsum("ij,ij->i", normal, camera.center - triangles.mean(1)) > 0
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    edges, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    front_count = np.bincount(inverse, weights=np.tile(facing, 3))
    contour = member[edges].all(1) & (front_count > 0) & ((front_count < counts) | (counts == 1))
    edge = edges[contour]
    t = np.linspace(0, 1, 5)[None, :, None]
    points = (mesh[edge[:, 0], None] * (1 - t) + mesh[edge[:, 1], None] * t).reshape(-1, 3)
    points = points[visible_landmarks(camera, points, mesh=mesh, faces=faces)]
    # Bound the number of correlated samples so a dense ear never dominates PnP.
    points = points[::max(1, int(np.ceil(len(points) / 40)))]
    if len(points) < 6:
        return None, None, {"samples": 0, "reason": "insufficient visible ear contour"}
    scale = min(1, 1024 / max(rgb.shape[:2]))
    grey = cv2.cvtColor(cv2.resize(rgb, None, fx=scale, fy=scale), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(grey, (5, 5), 1), 60, 120)
    y, x = np.nonzero(edges)
    if len(x) == 0:
        return None, None, {"samples": 0, "reason": "no local image edges"}
    observed = (np.c_[x, y] + 0.5) / scale
    projected = camera.project(points)[:, :2]
    distance, nearest = cKDTree(observed).query(projected)
    keep = distance <= 12
    report = {"method": "Canny edges within 12 px of z-visible FLAME ear silhouette",
              "semantic_ear_detection": False, "samples": int(keep.sum()), "weight": 2.0}
    if keep.sum() < 6:
        report["reason"] = "insufficient local edge support"
        return None, None, report
    return points[keep], observed[nearest[keep]], report


def preserve_bake_baseline(audit_dir):
    """Keep the original numeric bake metrics once, only if its cameras match the baseline."""
    audit_dir = Path(audit_dir)
    target = audit_dir / "bake_baseline.json"
    source = audit_dir.parent / "hybrid_report.json"
    baseline_path = audit_dir / "camera_baseline.json"
    if target.is_file() or not source.is_file() or not baseline_path.is_file():
        return
    original = json.loads(source.read_text(encoding="utf8"))["texture"]
    baseline = json.loads(baseline_path.read_text(encoding="utf8"))
    if all(np.allclose(original["camera_fits"][name][key], baseline[name][key])
           for name in VIEWS for key in ("intrinsics", "world_to_camera")):
        target.write_text(json.dumps({"gains": original["gains"]}, indent=2) + "\n", encoding="utf8")


def fit_photo_set(flame, folder: Path, audit_dir: Path, *, use_outline_guides=False, use_ear_edges=True):
    """Run 478-point detection and evaluate the official 105-point embedding.

    Profile cameras use the 105 landmarks, the focal/elevation/roll priors, the mesh z-visibility test and local
    ear-edge guides. The MediaPipe face-oval guides are opt-in: on the user's set they pulled the profile pose and
    lowered the skin fraction of the samples baked onto the ear (README, "Profile camera audit").
    """
    import trimesh
    from scipy.spatial import cKDTree

    audit_dir = Path(audit_dir)
    from . import REPO

    if not audit_dir.resolve().is_relative_to((REPO / "user-data").resolve()):
        raise ValueError("Camera audits must stay in user-data/")
    audit_dir.mkdir(parents=True, exist_ok=True)
    vertex_normal = trimesh.Trimesh(flame.neutral, flame.faces, process=False).vertex_normals
    normals = vertex_normal[cKDTree(flame.neutral).query(flame.landmark_points)[1]]
    photos, cameras, report, detections, metadata = {}, {}, {}, {}, {}
    previous_path = audit_dir / "camera_fits.json"
    baseline_path = audit_dir / "camera_baseline.json"
    if previous_path.is_file() and not baseline_path.is_file():
        previous = json.loads(previous_path.read_text(encoding="utf8"))
        if all(previous.get(name, {}).get("priors") is None for name in VIEWS):
            baseline_path.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf8")
    preserve_bake_baseline(audit_dir)
    for name in VIEWS:
        with Image.open(folder / f"nog_{name}.jpeg") as im:
            metadata[name] = photo_metadata(im)
            photos[name] = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))

    def fit_view(name):
        detection, detection_report = detect_photo(photos[name])
        if detection is None or len(detection.xy) != 478:
            raise ValueError(f"{name}: MediaPipe did not return 478 landmarks")
        pixels = detection.xy[flame.landmark_ids]
        detections[name] = (detection, detection_report)
        h, w = photos[name].shape[:2]
        options = ({"prior": focal_prior(metadata[name], (w, h)), "mesh": flame.neutral, "faces": flame.faces}
                   if name == "front" else {})
        cameras[name], report[name], inliers = fit_camera(
            flame.landmark_points, pixels, (w, h), normals=normals, **options,
        )
        report[name]["detection"] = detection_report
        np.savez_compressed(
            audit_dir / f"{name}_landmarks.npz",
            xy=detection.xy,
            z=detection.z,
            embedding_ids=flame.landmark_ids,
            inliers=inliers,
            reprojected=cameras[name].project(flame.landmark_points),
        )

    for name in VIEWS:
        fit_view(name)
    # Selfie apps may encode mirrored pixels without EXIF metadata. Correct only
    # sides that contradict the user's SUBJECT-side labels. Front is corrected
    # together with the set only when both side cameras imply a shared mirror.
    mirrored_views = set()
    if report["left"]["camera_azimuth_deg"] < -10:
        mirrored_views.add("left")
    if report["right"]["camera_azimuth_deg"] > 10:
        mirrored_views.add("right")
    if len(mirrored_views) == 2:
        mirrored_views.add("front")
    if mirrored_views:
        permutation = mirror_correspondences(flame.landmark_points)
        for name in VIEWS:
            if name not in mirrored_views:
                continue
            initial_azimuth = report[name]["camera_azimuth_deg"]
            photos[name] = np.ascontiguousarray(photos[name][:, ::-1])
            detection, detection_report = detections[name]
            h, w = photos[name].shape[:2]
            pixels = detection.xy[flame.landmark_ids[permutation]].copy()
            pixels[:, 0] = w - pixels[:, 0]
            cameras[name], report[name], inliers = fit_camera(
                flame.landmark_points,
                pixels,
                (w, h),
                normals=normals,
                initial_camera=mirrored_camera(cameras[name]),
                **({"prior": focal_prior(metadata[name], (w, h)), "mesh": flame.neutral, "faces": flame.faces}
                   if name == "front" else {}),
            )
            report[name]["detection"] = detection_report
            np.savez_compressed(
                audit_dir / f"{name}_landmarks.npz",
                xy=detection.xy,
                z=detection.z,
                embedding_ids=flame.landmark_ids,
                fitted_embedding_xy=pixels,
                inliers=inliers,
                bilateral_permutation=permutation,
                xy_space="decoded source before horizontal unmirror",
                reprojected=cameras[name].project(flame.landmark_points),
            )
            report[name]["source_orientation"] = {
                "horizontal_unmirror": True,
                "initial_camera_azimuth_deg": initial_azimuth,
                "reason": "camera sign contradicted the user-defined subject side; front follows only a shared mirror",
            }
    for name in VIEWS:
        report[name].setdefault("source_orientation", {"horizontal_unmirror": False})
    baseline = json.loads(baseline_path.read_text(encoding="utf8")) if baseline_path.is_file() else {}
    # The 105-point embedding has no jaw/cheek oval: profiles additionally need
    # this face-outline constraint, rather than trusting hallucinated far-side eyes.
    region = np.zeros(len(flame.neutral), bool)
    for key in ("face", "forehead", "eye_region", "nose", "lips"):
        region[flame.masks[key]] = True
    region_faces = flame.faces[region[flame.faces].all(1)]
    edges = np.sort(np.concatenate([region_faces[:, [0, 1]], region_faces[:, [1, 2]], region_faces[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    edge = edges[counts == 1]
    t = np.linspace(0, 1, 12)[None, :, None]
    outline = (flame.neutral[edge[:, 0], None] * (1 - t) + flame.neutral[edge[:, 1], None] * t).reshape(-1, 3)
    face_masks = {}
    for name in VIEWS:
        detection, _ = detections[name]
        h, w = photos[name].shape[:2]
        oval = detection.xy[FACE_OVAL].copy()
        if name in mirrored_views:
            oval[:, 0] = w - oval[:, 0]
            oval = oval[np.r_[0, np.arange(35, 0, -1)]]
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.rint(oval).astype(np.int32)], 1)
        margin, ramp = (0.0, 6.0) if name == "front" else (PROFILE_MASK_MARGIN_PX, PROFILE_MASK_RAMP_PX)
        scale = max(h, w) / 2048  # pixel constants are defined for a 2048 px long edge
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        face_masks[name] = np.clip((distance - margin * scale) / (ramp * scale), 0, 1).astype(np.float32)
        if name != "front":
            with np.load(audit_dir / f"{name}_landmarks.npz") as data:
                pixels = data["fitted_embedding_xy"] if name in mirrored_views else data["xy"][flame.landmark_ids]
            guides = oval[6:19] if report[name]["camera_azimuth_deg"] > 0 else oval[18:31]
            old = report[name]
            shared = same_lens(metadata["front"], metadata[name])
            no_exif = not any(metadata[v]["model"] or metadata[v]["focal_35mm"] or metadata[v]["focal_mm"] for v in VIEWS)
            pose_priors = {
                "elevation_sigma_deg": PROFILE_ELEVATION_SIGMA_DEG,
                "roll_sigma_deg": PROFILE_ROLL_SIGMA_DEG,
                "azimuth_hint_deg": 65 if name == "left" else -65,
            }
            if shared or no_exif:
                normalized = cameras["front"].intrinsic[0, 0] / np.hypot(*cameras["front"].size)
                prior = CameraPrior(normalized * np.hypot(w, h), 0.025 if shared else 0.15,
                                    0.06 if shared else 0.30, **pose_priors)
                constraint_source = "EXIF same lens; front-calibrated diagonal focal" if shared else (
                    "EXIF absent; assumed common selfie lens, phone FOV prior calibrated by front (not verified)"
                )
            else:
                prior = replace(focal_prior(metadata[name], (w, h)), **pose_priors)
                constraint_source = "independent EXIF/FOV prior; same lens not established"
            options = {"prior": prior, "mesh": flame.neutral, "faces": flame.faces}
            if use_outline_guides:
                options.update(outline_points=outline, outline_pixels=guides)
            cameras[name], report[name], inliers = fit_camera(
                flame.landmark_points, pixels, (w, h), normals=normals, initial_camera=cameras[name], **options,
            )
            ear_report = {"enabled": use_ear_edges}
            if use_ear_edges:
                ear_points, ear_pixels, ear_report = ear_edge_guides(
                    flame.neutral, flame.faces, flame.masks[f"{name}_ear"], cameras[name], photos[name],
                )
                if ear_points is not None:
                    cameras[name], report[name], inliers = fit_camera(
                        flame.landmark_points, pixels, (w, h), normals=normals, initial_camera=cameras[name],
                        ear_points=ear_points, ear_pixels=ear_pixels, **options,
                    )
                    ear_report.update(report[name]["ear_refinement"] or {})
            report[name]["ear_refinement"] = ear_report
            report[name]["outline_guides_enabled"] = use_outline_guides
            report[name]["focal_constraint_source"] = constraint_source
            report[name]["source_orientation"] = old["source_orientation"]
            report[name]["detection"] = old["detection"]
            with np.load(audit_dir / f"{name}_landmarks.npz") as data:
                audit = dict(data)
            audit.update(inliers=inliers, reprojected=cameras[name].project(flame.landmark_points), outline_xy=guides)
            np.savez_compressed(audit_dir / f"{name}_landmarks.npz", **audit)
        report[name]["exif_camera"] = metadata[name]
        profile = name != "front"
        report[name]["neck_pitch_correction"] = {
            "applied_deg": 0.0,
            "profile_head_pitch_equivalent_deg": report[name]["camera_roll_deg"] if profile else None,
            "reason": (
                "In a profile, camera roll about the optical axis and head pitch about the ear-to-ear axis are the same "
                "rotation of the face relative to the image, so the fitted roll (prior sigma "
                f"{PROFILE_ROLL_SIGMA_DEG:g} deg, not clamped) already poses the head; a separate neck pitch would be "
                "indistinguishable from it without independent neck/body observations"
            ) if profile else "frontal view: no pitch correction applied",
        }
        if name in baseline:
            before = baseline[name]
            before_camera = Camera(np.asarray(before["intrinsics"]), np.asarray(before["world_to_camera"]),
                                   np.array([0, h, 0, w]), (w, h), (w, h))
            before_visible = visible_landmarks(before_camera, flame.landmark_points, normals, flame.neutral, flame.faces)
            after_visible = visible_landmarks(cameras[name], flame.landmark_points, normals, flame.neutral, flame.faces)
            with np.load(audit_dir / f"{name}_landmarks.npz") as data:
                pixels = data["fitted_embedding_xy"] if name in mirrored_views else data["xy"][flame.landmark_ids]
            common = before_visible & after_visible
            before_error = np.linalg.norm(before_camera.project(flame.landmark_points)[:, :2] - pixels, axis=1)
            after_error = np.linalg.norm(cameras[name].project(flame.landmark_points)[:, :2] - pixels, axis=1)
            report[name]["before"] = {key: before[key] for key in (
                "inliers", "rms_px", "all_rms_px", "focal_px", "camera_azimuth_deg", "camera_elevation_deg",
                "ransac_threshold_px", "focal_near_bound",
            )}
            report[name]["before"]["camera_roll_deg"] = camera_pose(before_camera, flame.landmark_points.mean(0))[2]
            report[name]["common_visible_comparison"] = {
                "landmarks": int(common.sum()),
                "before_rms_px": float(np.sqrt(np.mean(before_error[common] ** 2))) if common.any() else None,
                "after_rms_px": float(np.sqrt(np.mean(after_error[common] ** 2))) if common.any() else None,
                "threshold_px": report[name]["ransac_threshold_px"],
                "before_inliers": int((common & (before_error <= report[name]["ransac_threshold_px"])).sum()),
                "after_inliers": int((common & (after_error <= report[name]["ransac_threshold_px"])).sum()),
            }
        report[name]["quality_flags"] = [
            flag for condition, flag in (
                (abs(report[name]["camera_elevation_deg"]) > 15, "elevation exceeds handheld prior"),
                # In a profile a large roll is the head nod, not an error; only a frontal roll is suspicious.
                (not profile and abs(report[name]["camera_roll_deg"]) > 20, "large roll: head tilt or detector ambiguity"),
                (report[name]["inliers"] < 30, "sparse profile landmark agreement"),
            ) if condition
        ]
        if name == "front":
            report[name]["focal_constraint_source"] = "EXIF 35 mm-equivalent" if metadata[name]["focal_35mm"] else "24 mm-equivalent phone FOV prior"
        print(f"[hybrid] {name} camera: {json.dumps(report[name])}", flush=True)
    registration_pixels = {}
    for name in VIEWS:
        with np.load(audit_dir / f"{name}_landmarks.npz") as data:
            registration_pixels[name] = data["fitted_embedding_xy"] if name in mirrored_views else data["xy"][flame.landmark_ids]
    mesh_args = (flame.neutral, flame.faces, flame.landmark_points, normals)
    for name in ("left", "right"):
        registration = {"after": landmark_registration(
            *mesh_args, cameras["front"], registration_pixels["front"], cameras[name], registration_pixels[name],
        )}
        if "front" in baseline and name in baseline:
            old_cameras = {}
            for view in ("front", name):
                w, h = report[view]["original_size_wh"]
                old_cameras[view] = Camera(np.asarray(baseline[view]["intrinsics"]), np.asarray(baseline[view]["world_to_camera"]),
                                           np.array([0, h, 0, w]), (w, h), (w, h))
            registration["before"] = landmark_registration(
                *mesh_args, old_cameras["front"], registration_pixels["front"], old_cameras[name], registration_pixels[name],
            )
            common = (seen_by_both(*mesh_args, cameras["front"], cameras[name])
                      & seen_by_both(*mesh_args, old_cameras["front"], old_cameras[name]))
            registration["common_landmarks"] = {
                "landmarks": int(common.sum()),
                "after": landmark_registration(
                    *mesh_args, cameras["front"], registration_pixels["front"], cameras[name], registration_pixels[name], only=common,
                ),
                "before": landmark_registration(
                    *mesh_args, old_cameras["front"], registration_pixels["front"], old_cameras[name], registration_pixels[name],
                    only=common,
                ),
            }
        report[name]["landmark_registration_vs_front"] = registration
        print(f"[hybrid] {name} landmark registration vs front (mm): {json.dumps(registration)}", flush=True)
    (audit_dir / "camera_fits.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    return cameras, photos, report, face_masks


def view_preferences(points, chin_y, centre_x=0.0):
    """Smooth anatomical priors; visibility still decides whether any photo can contribute."""
    centre = 1 - smoothstep((np.abs(points[:, 0] - centre_x) - 0.018) / 0.04)
    lower = 1 - smoothstep((points[:, 1] - chin_y - 0.005) / 0.030)
    return {
        "front": 1 + 11 * centre + 19 * lower,
        "left": (1 - 0.85 * centre) * (1 - 0.75 * lower),
        "right": (1 - 0.85 * centre) * (1 - 0.98 * lower),
    }
