"""Estimate round or rounded-polygon eyeglasses using assumed interpupillary distance."""
from pathlib import Path
import argparse
import math

import cv2
import numpy as np

from core import (ROOT, Landmarker, Detection, EYES, detect_glasses, private_path,
                  pupil_centers, read_image, write_json)
from models import AccessorySegmenter


def estimate(image: np.ndarray, landmarks: np.ndarray, detection: Detection,
             ipd_mm: float = 63.0, lens_shape: str = "round") -> dict:
    if not math.isfinite(ipd_mm) or not 40 <= ipd_mm <= 85:
        raise ValueError("IPD must be finite and between 40 and 85 mm")
    if len(detection.rings) != 2:
        raise ValueError("Two supported rims are required; refusing invented dimensions")
    if lens_shape not in ("round","rounded-polygon"):
        raise ValueError("Unsupported lens shape")
    pupils = pupil_centers(landmarks)
    distance = float(np.linalg.norm(pupils[1] - pupils[0]))
    if distance <= 0:
        raise ValueError("Invalid pupil distance")
    scale = ipd_mm / 1000 / distance
    left, right = sorted(detection.rings, key=lambda ring: ring["center"][0])
    outer_radius = float(np.mean([r["radius_x"] for r in (left, right)]))
    bridge = (right["center"][0] - right["radius_x"]
              - left["center"][0] - left["radius_x"])
    if bridge <= 0:
        raise ValueError("Rim fits overlap; bridge estimate is unreliable")
    # The local maxima of the distance transform estimate half the thin line width.
    dt = cv2.distanceTransform(detection.core_mask, cv2.DIST_L2, 5)
    peaks = (dt > 0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)))
    thickness_px = float(2 * np.median(dt[peaks])) if np.any(peaks) else 1.0
    pixels = image[detection.mask > 0]
    if not len(pixels):
        raise ValueError("Empty frame mask")
    rgb = np.rint(np.median(pixels[:, ::-1], axis=0)).astype(int).tolist()
    vertical = (float(np.mean([r["center"][1] for r in (left, right)]))
                - float(pupils[:, 1].mean())) * scale
    width = right["center"][0] + right["radius_x"] - left["center"][0] + left["radius_x"]
    # Eye corners define the eye-line tilt; dimensions are projected, not 3D scans.
    counts,roundness = [],[]
    for rim in (left,right):
        if "contour" in rim:
            hull = cv2.convexHull(np.array(rim["contour"],dtype=np.float32))
            perimeter = cv2.arcLength(hull,True)
            if perimeter>0:
                counts.append(len(cv2.approxPolyDP(hull,.015*perimeter,True)))
                roundness.append(min(1.0,4*math.pi*cv2.contourArea(hull)/perimeter**2))
    return {
        "version": 2, "units": "meters", "lens_shape": lens_shape,
        "lens_shape_source": "explicit shape hint; contour is not a reliable shape classifier",
        "lens_corner_count_approx": int(round(float(np.median(counts)))) if counts and lens_shape!="round" else 0 if lens_shape=="round" else None,
        "lens_roundness": round(float(np.mean(roundness)),4) if roundness else None,
        "shape_descriptor_method": "convex contour hull; 1.5%-perimeter polygon simplification; roundness 4*pi*area/perimeter^2",
        "lens_outer_radius_m": round(outer_radius * scale, 6),
        "lens_outer_radii_m": [round(r["radius_x"] * scale, 6) for r in (left, right)],
        "bridge_width_m": round(bridge * scale, 6),
        "frame_width_at_temples_m": round(width * scale, 6),
        "frame_thickness_m": round(thickness_px * scale, 6),
        "frame_colour_rgb": rgb,
        "frame_colour_hex": "#" + "".join(f"{c:02x}" for c in rgb),
        "temple_arm_length_m": 0.14,
        "temple_arm_method": "default_no_reliable_profile_arm",
        "lens_vertical_offset_m": round(vertical, 6),
        "vertical_offset_convention": "positive downward from mean pupil center",
        "assumed_ipd_m": ipd_mm / 1000,
        "method": "landmark IPD scale; flexible ridge-supported rim contours; accessory segmentation; median masked RGB; distance-transform line width",
        "diagnostics": {"pupil_distance_px": round(distance, 3),
                        "meters_per_pixel": scale,
                        "rim_coverage": [r["coverage"] for r in (left, right)]},
        "uncertainty": {
            "scale": "Every length scales linearly with assumed IPD; no physical calibration.",
            "dimensions": "Heuristic +/-15% for radius/width/bridge; no statistical confidence interval. Pose and refraction add bias.",
            "frame_thickness": "Heuristic +/-50%; resolution, ridge response and metal highlights bias width.",
            "colour": "Median final-mask pixels includes skin at dilation borders, reflections and shadows; not intrinsic metal colour.",
            "temple_arm": "Default 0.14 m if no reliable projected segment; heuristic +/-0.03 m. Profiles do not recover hidden length.",
            "vertical_offset": "Heuristic +/-0.005 m; sensitive to rim fit and pupil localization."},
        "review_required": True}


def profile_arm(image: np.ndarray, landmarks: np.ndarray, detection: Detection,
                front_eye_width_px: float, front_scale: float) -> float | None:
    """A visible straight segment plus 25 mm for the curved/occluded ear end."""
    widths = [float(np.linalg.norm(landmarks[e[0]] - landmarks[e[3]])) for e in EYES]
    eye_width = max(widths)
    if eye_width < 5:
        return None
    scale = front_scale * front_eye_width_px / eye_width
    pupils = pupil_centers(landmarks)
    y = float(pupils[:, 1].mean())
    mask = detection.core_mask.copy()
    yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]]
    visible_eye = pupils[int(np.argmax(widths))]
    mask[(np.abs(yy - y) > eye_width * 0.5) | (np.abs(xx - visible_eye[0]) < eye_width)] = 0
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 20,
                            minLineLength=max(10, round(0.045 / scale)), maxLineGap=round(0.006 / scale))
    spans = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(y2 - y1) > abs(x2 - x1) * 0.5:
                continue
            length = float(np.hypot(x2 - x1, y2 - y1)) * scale
            if 0.045 <= length <= 0.16:
                spans.append(length)
    return min(0.18, max(spans) + 0.025) if spans else None


def main() -> None:
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=Path("user-data/twin/head"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--ipd-mm", type=float, default=63.0)
    parser.add_argument("--lens-shape",choices=("round","rounded-polygon"),default="rounded-polygon")
    parser.add_argument("--model", type=Path, default=ROOT / "apps/web/public/models/face_landmarker.task")
    args = parser.parse_args()
    source = private_path(args.in_dir)
    target = private_path(args.out or source / "glasses.json")
    if target.suffix.lower() != ".json":
        parser.error("Measurement output must be JSON")
    detector = Landmarker(args.model)
    segmenter = AccessorySegmenter()
    try:
        image = read_image(source / "front.jpg")
        landmarks = detector.detect(image)
        if landmarks is None:
            raise ValueError("Front face not detected; no metric measurements written")
        accessory,hair = segmenter.segment(image,landmarks)
        detection = detect_glasses(image, landmarks,accessory_probability=accessory,hair_mask=hair)
        result = estimate(image, landmarks, detection, args.ipd_mm,args.lens_shape)
        result["uncertainty"]["shape"] = "Approximate corners depend on smoothing and contour coverage; rounded-polygon radius is an ellipse-envelope proxy, not a circular lens."
        front_eye_width = max(float(np.linalg.norm(landmarks[e[0]] - landmarks[e[3]])) for e in EYES)
        front_height = float(np.linalg.norm(landmarks[10]-landmarks[152]))
        front_ipd = detection.pupil_distance_px
        arms = []
        for name in ("profile_nose_left", "profile_nose_right"):
            path = source / f"{name}.jpg"
            if not path.exists():
                continue
            profile = read_image(path)
            points = detector.detect(profile)
            if points is not None:
                detected = detect_glasses(profile, points, name,
                                           front_ipd/front_height, front_eye_width/front_ipd,
                                           *segmenter.segment(profile,points))
                arm = profile_arm(profile, points, detected, front_eye_width,
                                  result["diagnostics"]["meters_per_pixel"])
                if arm is not None:
                    arms.append(arm)
        if arms:
            result["temple_arm_length_m"] = round(float(np.median(arms)), 6)
            result["temple_arm_method"] = "profile_projected_segment_plus_0.025m_occluded_end"
        result["diagnostics"]["profiles_with_arm_estimate"] = len(arms)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, result)
        print("Measurements written locally; no image contents displayed.")
    finally:
        detector.close()
        segmenter.close()


if __name__ == "__main__":
    main()
