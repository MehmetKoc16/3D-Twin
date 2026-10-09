"""Local glasses-free photo cameras on the existing private FLAME identity.

The user supplies the licensed FLAME MediaPipe embedding in user-data/flame
(obtain it from flame.is.tue.mpg.de). No model data is bundled here. Detection
never draws, exports or displays a photo/crop; private numeric audits stay in
user-data. Camera axes are converted explicitly from OpenCV to FLAME.
"""

from __future__ import annotations

import json
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
    points, pixels, size, *, normals=None, initial_camera=None, outline_points=None, outline_pixels=None
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
    image_scale = max(size)
    face_span = np.linalg.norm(np.ptp(pixels, axis=0))
    threshold = max(3.0, face_span * 0.025)
    candidates = []
    if initial_camera is not None:
        cv_rotation = CV_TO_FLAME @ initial_camera.extrinsic[:3, :3]
        cv_translation = CV_TO_FLAME @ initial_camera.extrinsic[:3, 3]
        p = np.r_[cv2.Rodrigues(cv_rotation)[0].ravel(), cv_translation, np.log(initial_camera.intrinsic[0, 0])]
        error = np.linalg.norm(initial_camera.project(points)[:, :2] - pixels, axis=1)
        candidates.append((np.minimum(error, threshold).mean(), p))
    cv2.setRNGSeed(1729)
    for ratio in (0.45, 0.65, 0.9, 1.2, 1.7, 2.4, 3.2):
        focal = image_scale * ratio
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
            candidates.append((np.minimum(error, threshold).mean(), p))
    if not candidates:
        raise ValueError("No positive-depth robust PnP solution")
    p = min(candidates, key=lambda item: item[0])[1]
    selected = np.ones(len(points), bool)
    lower = np.r_[np.full(6, -np.inf), np.log(image_scale * 0.35)]
    upper = np.r_[np.full(6, np.inf), np.log(image_scale * 4.0)]
    for _ in range(4):
        cam = camera_from_cv(p, size)
        if normals is not None:
            toward = cam.center - points
            toward /= np.linalg.norm(toward, axis=1, keepdims=True)
            selected = np.einsum("ij,ij->i", normals, toward) > 0.08
        error = np.linalg.norm(cam.project(points)[:, :2] - pixels, axis=1)
        selected &= error < max(threshold * 2, np.median(error[selected]) * 3) if selected.any() else False
        if selected.sum() < 12:
            raise ValueError("Too few visible landmarks for reliable profile fit")

        def residual(parameters, subset=selected):
            return (camera_from_cv(parameters, size).project(points[subset])[:, :2] - pixels[subset]).ravel()

        result = least_squares(
            residual, p, bounds=(lower, upper), loss="soft_l1", f_scale=threshold / 3, x_scale="jac", max_nfev=400
        )
        p = result.x
    outline_report = None
    if outline_points is not None and outline_pixels is not None:
        from scipy.spatial import cKDTree

        seed = p.copy()
        before = cKDTree(camera_from_cv(p, size).project(outline_points)[:, :2]).query(outline_pixels)[0]
        selected &= np.linalg.norm(camera_from_cv(p, size).project(points)[:, :2] - pixels, axis=1) <= threshold
        # Face-outline ICP refreshes 3D correspondence each iteration. Keep the
        # robust interior landmarks as anchors and bound pose drift from PnP.
        low = np.maximum(lower, seed - np.r_[np.full(3, 0.25), np.full(3, 0.06), 0.4])
        high = np.minimum(upper, seed + np.r_[np.full(3, 0.25), np.full(3, 0.06), 0.4])
        for _ in range(6):
            projected = camera_from_cv(p, size).project(outline_points)[:, :2]
            nearest = cKDTree(projected).query(outline_pixels)[1]
            anchors = outline_points[nearest]

            def outline_residual(parameters, subset=selected, anchor_points=anchors):
                cam = camera_from_cv(parameters, size)
                interior = cam.project(points[subset])[:, :2] - pixels[subset]
                boundary = cam.project(anchor_points)[:, :2] - outline_pixels
                return np.r_[interior.ravel(), boundary.ravel() * 2]

            p = least_squares(
                outline_residual,
                p,
                bounds=(low, high),
                loss="soft_l1",
                f_scale=threshold / 2,
                x_scale="jac",
                max_nfev=200,
            ).x
        after = cKDTree(camera_from_cv(p, size).project(outline_points)[:, :2]).query(outline_pixels)[0]
        outline_report = {
            "method": "MediaPipe observed cheek/jaw outline to FLAME face boundary, bounded robust ICP",
            "samples": len(outline_pixels),
            "rms_before_px": float(np.sqrt(np.mean(before**2))),
            "rms_after_px": float(np.sqrt(np.mean(after**2))),
        }
    cam = camera_from_cv(p, size)
    projection = cam.project(points)
    if (projection[:, 2] <= 0).any():
        raise ValueError("Refinement placed landmarks behind the camera")
    error = np.linalg.norm(projection[:, :2] - pixels, axis=1)
    visible = np.ones(len(points), bool)
    if normals is not None:
        direction = cam.center - points
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        visible = np.einsum("ij,ij->i", normals, direction) > 0.08
    inliers = visible & (error <= threshold)
    if inliers.sum() < 12:
        raise ValueError("Too few robust camera inliers")
    mm = error * projection[:, 2] / cam.intrinsic[0, 0] * 1000

    def rms(values):
        return float(np.sqrt(np.mean(values**2)))

    pose = Rotation.from_matrix(cam.extrinsic[:3, :3]).as_euler("xyz", degrees=True)
    toward = cam.center - points.mean(0)
    yaw = np.degrees(np.arctan2(toward[0], toward[2]))
    elevation = np.degrees(np.arctan2(toward[1], np.hypot(toward[0], toward[2])))
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
        "focal_near_bound": bool(p[6] < lower[6] + 0.01 or p[6] > upper[6] - 0.01),
        "intrinsics": cam.intrinsic.tolist(),
        "world_to_camera": cam.extrinsic.tolist(),
        "expression": "existing neutral identity, no expression or jaw optimisation",
        "outline_refinement": outline_report,
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


def fit_photo_set(flame, folder: Path, audit_dir: Path):
    """Run 478-point detection and evaluate the official 105-point embedding."""
    import trimesh
    from scipy.spatial import cKDTree

    audit_dir = Path(audit_dir)
    from . import REPO

    if not audit_dir.resolve().is_relative_to((REPO / "user-data").resolve()):
        raise ValueError("Camera audits must stay in user-data/")
    audit_dir.mkdir(parents=True, exist_ok=True)
    vertex_normal = trimesh.Trimesh(flame.neutral, flame.faces, process=False).vertex_normals
    normals = vertex_normal[cKDTree(flame.neutral).query(flame.landmark_points)[1]]
    photos, cameras, report, detections = {}, {}, {}, {}
    for name in VIEWS:
        with Image.open(folder / f"nog_{name}.jpeg") as im:
            photos[name] = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))

    def fit_view(name):
        detection, detection_report = detect_photo(photos[name])
        if detection is None or len(detection.xy) != 478:
            raise ValueError(f"{name}: MediaPipe did not return 478 landmarks")
        pixels = detection.xy[flame.landmark_ids]
        detections[name] = (detection, detection_report)
        h, w = photos[name].shape[:2]
        cameras[name], report[name], inliers = fit_camera(flame.landmark_points, pixels, (w, h), normals=normals)
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
        face_masks[name] = np.minimum(cv2.distanceTransform(mask, cv2.DIST_L2, 3) / 6, 1).astype(np.float32)
        if name != "front" and report[name]["inliers"] < 60:
            with np.load(audit_dir / f"{name}_landmarks.npz") as data:
                pixels = data["fitted_embedding_xy"] if name in mirrored_views else data["xy"][flame.landmark_ids]
            guides = oval[6:19] if report[name]["camera_azimuth_deg"] > 0 else oval[18:31]
            old = report[name]
            cameras[name], report[name], inliers = fit_camera(
                flame.landmark_points,
                pixels,
                (w, h),
                normals=normals,
                initial_camera=cameras[name],
                outline_points=outline,
                outline_pixels=guides,
            )
            report[name]["source_orientation"] = old["source_orientation"]
            report[name]["detection"] = old["detection"]
            with np.load(audit_dir / f"{name}_landmarks.npz") as data:
                audit = dict(data)
            audit.update(inliers=inliers, reprojected=cameras[name].project(flame.landmark_points), outline_xy=guides)
            np.savez_compressed(audit_dir / f"{name}_landmarks.npz", **audit)
        print(f"[hybrid] {name} camera: {json.dumps(report[name])}", flush=True)
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
