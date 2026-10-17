"""Automated alignment sidecar generator for a single left shoe to MakeHuman neutral frame."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .mesh_io import clean, load_parts

PIPELINE = Path(__file__).resolve().parents[1]
ROOT = PIPELINE.parents[1]
DEFAULT_BODY = ROOT / "apps/web/public/assets/body"


def extract_target_foot(body: Any = None, body_path: Path | None = None) -> dict[str, float]:
    """Extract left foot bounds and landmarks from the MakeHuman runtime neutral body."""
    if body is None:
        target_dir = (body_path or DEFAULT_BODY).resolve()
        from .body import load_body

        # Never silently substitute guessed anatomy when assets are missing or corrupt.
        body = load_body(target_dir)

    foot_length = None
    if body is not None and hasattr(body, "measures") and isinstance(body.measures, dict):
        for m in body.measures.get("measures", []):
            if m.get("id") == "footLength" and "verts" in m:
                v1, v2 = m["verts"]
                foot_length = abs(float(body.positions[v1, 2] - body.positions[v2, 2]))
                break

    indices = [i for i, name in enumerate(body.names) if name in ("foot_l", "ball_l")]
    in_foot = np.isin(body.joints, indices) & (body.weights > 0.05)
    render = body.positions[:body.render_count]
    left_mask = in_foot.any(axis=1) & (render[:, 0] > 0)
    foot_pts = render[left_mask]
    if len(foot_pts) > 0 and np.isfinite(foot_pts).all():
        heel_z = float(foot_pts[:, 2].min())
        toe_z = float(foot_pts[:, 2].max())
        ground_y = float(foot_pts[:, 1].min())
        center_x = float((foot_pts[:, 0].min() + foot_pts[:, 0].max()) / 2.0)
        width_x = float(foot_pts[:, 0].max() - foot_pts[:, 0].min())
    else:
        raise ValueError("Target body has no weighted left foot vertices")

    if foot_length is None or not (0.10 <= foot_length <= 0.40):
        foot_length = float(toe_z - heel_z)
    if not (0.10 <= foot_length <= 0.40) or width_x <= 0:
        raise ValueError("Target body has invalid foot dimensions in meters")

    return {
        "foot_length": float(foot_length),
        "heel_z": float(heel_z),
        "toe_z": float(toe_z),
        "ground_y": float(ground_y),
        "center_x": float(center_x),
        "width_x": float(width_x),
    }


AXIS_CHOICES = ("auto", "+x", "-x", "+y", "-y", "+z", "-z")
AXIS_MAP: dict[str, np.ndarray] = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "x": np.array([1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def parse_axis(val: str | None) -> np.ndarray | None:
    if val is None or val.strip().lower() == "auto":
        return None
    k = val.strip().lower()
    if k not in AXIS_MAP:
        raise ValueError(f"Invalid axis '{val}'. Must be one of {', '.join(AXIS_CHOICES)}")
    return AXIS_MAP[k].copy()


def detect_up_and_sole(points: np.ndarray) -> tuple[np.ndarray, dict]:
    """Find a broad supporting sole plane, including when PCA mixes height and length."""
    from scipy.spatial import ConvexHull, QhullError

    points = np.unique(np.asarray(points, float), axis=0)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError("Shoe point cloud must contain at least four finite 3D vertices")
    centered = points - points.mean(axis=0)
    extent = float(np.linalg.norm(np.ptp(centered, axis=0)))
    if extent <= 0 or np.linalg.matrix_rank(centered) < 3:
        raise ValueError("Shoe point cloud must have nonzero length, width and height")
    normalized = centered / extent
    # Bounded, deterministic sampling keeps dense scans practical. Hull facets supply
    # genuine supporting plane normals; PCA alone tilts with the high heel collar.
    sample = normalized[::max(1, (len(points) + 2047) // 2048)]
    try:
        normals = -ConvexHull(sample).equations[:, :3]
    except QhullError as exc:
        raise ValueError("Cannot determine a supporting shoe sole plane") from exc
    _, eigvecs = np.linalg.eigh(normalized.T @ normalized)
    normals = np.vstack([normals, eigvecs.T, -eigvecs.T])
    normals = np.unique(np.round(normals, 7), axis=0)
    diagnostics = []
    best_score, best_candidate = -1.0, None
    for normal in normals:
        d = normal / np.linalg.norm(normal)
        h = normalized @ d
        span = float(np.ptp(h))
        if span < 1e-10:
            continue
        mask = h <= h.min() + .025 * span
        slab = normalized[mask]
        if len(slab) < 4:
            continue
        horizontal = normalized - h[:, None] * d
        _, basis = np.linalg.eigh(horizontal.T @ horizontal)
        plane = basis[:, -2:]
        coverage = float(np.prod(np.ptp(slab @ plane, axis=0) /
                                 np.maximum(np.ptp(normalized @ plane, axis=0), 1e-12)))
        thickness = float(np.std(h[mask])) / span
        density = float(mask.mean())
        score = density * coverage / (1 + 40 * thickness)
        diagnostics.append({"dir": d.tolist(), "score": score,
                            "density": density, "areaCoverage": coverage})
        if score > best_score:
            best_score, best_candidate = score, d
    if best_candidate is None:
        raise ValueError("Cannot find a broad supporting shoe sole plane; specify --up")
    return best_candidate, {"bestScore": best_score, "candidates": diagnostics,
                            "confidence": float(np.clip(best_score, 0, 1))}


def detect_length_and_toe(
    points: np.ndarray, up_vector: np.ndarray, toe_flip: bool = False
) -> tuple[np.ndarray, float, bool, dict]:
    """Find the horizontal length axis and identify the forward toe direction."""
    # UV seam copies must not change the orientation decision.
    points = np.unique(points, axis=0)
    # Project points onto horizontal plane orthogonal to up_vector
    p_horiz = points - (points @ up_vector)[:, None] * up_vector
    centroid = p_horiz.mean(axis=0)
    centered = p_horiz - centroid
    cov = centered.T @ centered / len(points)
    _, eigvecs = np.linalg.eigh(cov)

    v_len = eigvecs[:, 2]
    # Orthogonalize against up_vector
    v_len = v_len - (v_len @ up_vector) * up_vector
    v_len = v_len / np.linalg.norm(v_len)

    # Distinguish heel from toe along v_len
    z = (points - points.mean(axis=0)) @ v_len
    z_min, z_max = float(z.min()), float(z.max())
    length_span = max(z_max - z_min, 1e-6)

    h = points @ up_vector - (points @ up_vector).min()
    h_max = max(float(h.max()), 1e-6)

    # Ends A (lower z) and B (higher z)
    end_a_mask = z <= (z_min + 0.30 * length_span)
    end_b_mask = z >= (z_max - 0.30 * length_span)

    h_max_a = float(h[end_a_mask].max()) if end_a_mask.any() else 0.0
    h_max_b = float(h[end_b_mask].max()) if end_b_mask.any() else 0.0

    # Top collar vertices (highest 15% of shoe height)
    collar_mask = h >= (0.85 * h_max)
    if collar_mask.any():
        z_collar_mean = float(z[collar_mask].mean())
        collar_pos = (z_collar_mean - (z_min + z_max) / 2.0) / length_span
    else:
        collar_pos = 0.0

    # Evidence that End B is HEEL vs End A is HEEL
    evidence_b_is_heel = ((h_max_b - h_max_a) / h_max) + 2.0 * collar_pos

    if evidence_b_is_heel > 0:
        # End B is heel, End A is toe -> toe direction is -v_len
        fwd_vector = -v_len
    else:
        # End A is heel, End B is toe -> toe direction is +v_len
        fwd_vector = v_len

    confidence = min(1.0, max(0.1, abs(evidence_b_is_heel) / 1.5))

    if toe_flip:
        fwd_vector = -fwd_vector

    return fwd_vector, float(confidence), toe_flip, {
        "evidenceBIsHeel": float(evidence_b_is_heel),
        "hMaxA": float(h_max_a),
        "hMaxB": float(h_max_b),
        "collarPos": float(collar_pos),
    }


def detect_side(
    points: np.ndarray,
    up_vector: np.ndarray,
    fwd_vector: np.ndarray,
    requested_side: str = "left",
    allow_right: bool = False,
) -> tuple[np.ndarray, str, float, dict]:
    """Determine lateral vector (u_lat = u_up x u_fwd) and classify Left vs Right shoe."""
    u_lat = np.cross(up_vector, fwd_vector)
    u_lat = u_lat / np.linalg.norm(u_lat)

    # Local coordinates: x along u_lat, z along fwd_vector (0=heel, 1=toe)
    pts_centered = points - points.mean(axis=0)
    x = pts_centered @ u_lat
    z = pts_centered @ fwd_vector
    z_min, z_max = float(z.min()), float(z.max())
    z_len = max(z_max - z_min, 1e-6)
    z_norm = (z - z_min) / z_len

    # In a LEFT shoe, medial (arch) is on -u_lat, lateral (outer) is on +u_lat.
    # In a RIGHT shoe, medial (arch) is on +u_lat, lateral (outer) is on -u_lat.
    arch_mask = (z_norm >= 0.25) & (z_norm <= 0.55)
    ball_mask = (z_norm >= 0.60) & (z_norm <= 0.75)

    x_span = max(float(np.ptp(x)), 1e-6)
    score_left = 0.0

    if arch_mask.any() and ball_mask.any():
        x_neg_arch = abs(float(x[arch_mask].min()))
        x_pos_arch = abs(float(x[arch_mask].max()))
        x_neg_ball = abs(float(x[ball_mask].min()))
        x_pos_ball = abs(float(x[ball_mask].max()))

        indent_neg = x_neg_ball - x_neg_arch
        indent_pos = x_pos_ball - x_pos_arch
        score_left += (indent_neg - indent_pos) / x_span

    # Toe box apex offset: big toe is medial
    toe_mask = z_norm >= 0.90
    if toe_mask.any():
        x_toe_apex = float(x[toe_mask].mean())
        # For a left shoe, big toe is at negative x (towards midline)
        score_left += -1.5 * (x_toe_apex / x_span)

    if score_left > 0.08:
        detected = "left"
        confidence = min(1.0, 0.5 + score_left * 0.8)
    elif score_left < -0.08:
        detected = "right"
        confidence = min(1.0, 0.5 + abs(score_left) * 0.8)
    else:
        detected = requested_side
        confidence = 0.40

    if requested_side == "right":
        raise ValueError(
            "Right shoe specified (--side right). The garment import pipeline requires "
            "an authored single left shoe; --mirror-pair generates the right shoe automatically."
        )

    if detected == "right" and confidence >= 0.65 and requested_side == "left" and not allow_right:
        raise ValueError(
            f"Detected right shoe (confidence {confidence:.2f}). "
            "A single left shoe is required for import and --mirror-pair. "
            "If this model is indeed a right shoe, author or extract the left shoe instead."
        )

    return u_lat, detected, float(confidence), {
        "scoreLeft": float(score_left),
        "detected": detected,
        "confidence": float(confidence),
    }


def refine_horizontal_yaw(points: np.ndarray, up: np.ndarray, forward: np.ndarray, pca_yaw: float) -> float:
    """Remove small PCA skew from an asymmetric toe using nearby footprint support edges."""
    from scipy.spatial import ConvexHull

    lateral = np.cross(up, forward)
    coordinates = (points - points.mean(axis=0)) @ np.column_stack([lateral, forward])
    hull = coordinates[ConvexHull(coordinates).vertices]
    edges = np.roll(hull, -1, axis=0) - hull
    angles = (np.degrees(np.arctan2(edges[:, 0], edges[:, 1])) + 90) % 180 - 90
    # Only refine near the PCA axis: a footprint edge must not select another axis.
    candidates = [pca_yaw] + [float(a) for a in angles if abs(a - pca_yaw) <= 5.0]
    areas = []
    for angle in candidates:
        radians = np.radians(angle)
        frame = np.array([[np.cos(radians), np.sin(radians)],
                          [-np.sin(radians), np.cos(radians)]])
        areas.append(float(np.prod(np.ptp(coordinates @ frame, axis=0))))
    return candidates[int(np.argmin(areas))]


def compute_shoe_alignment(
    points: np.ndarray,
    body: Any = None,
    body_path: Path | None = None,
    side: str = "left",
    up: str = "auto",
    toe: str = "auto",
    toe_flip: bool = False,
    length_margin_cm: float = 1.2,
    wall_allowance_cm: float = 0.8,
) -> tuple[np.ndarray, dict]:
    """Compute the orientation-preserving row-major affine 4x4 matrix for the left shoe."""
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or len(points) < 4:
        raise ValueError("Invalid shoe point coordinates")
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 3:
        raise ValueError("Shoe must have nonzero length, width and height")
    if side not in ("left", "right"):
        raise ValueError("Side must be left or right")
    if not all(np.isfinite(v) and v >= 0 for v in (length_margin_cm, wall_allowance_cm)):
        raise ValueError("Length margin and wall allowance must be finite and nonnegative")

    target = extract_target_foot(body=body, body_path=body_path)
    warnings: list[str] = []

    # 1. Up and sole plane detection or override
    u_up_explicit = parse_axis(up)
    if u_up_explicit is not None:
        u_up = u_up_explicit / np.linalg.norm(u_up_explicit)
        up_info = {"override": up, "vector": u_up.tolist()}
    else:
        u_up, up_info = detect_up_and_sole(points)
        if up_info["confidence"] < 0.2:
            warnings.append("Low confidence in automatic sole plane; inspect alignment or specify --up.")

    # 2. Length axis and toe direction or override
    u_toe_explicit = parse_axis(toe)
    yaw_deg = 0.0
    if u_toe_explicit is not None:
        if abs(float(np.dot(u_toe_explicit, u_up))) > 1e-6:
            raise ValueError(f"Toe axis ({toe}) must be perpendicular to Up axis ({up})")
        u_fwd = u_toe_explicit - np.dot(u_toe_explicit, u_up) * u_up
        u_fwd = u_fwd / np.linalg.norm(u_fwd)
        pca_fwd, _, _, _ = detect_length_and_toe(points, u_up)
        if pca_fwd @ u_fwd < 0:
            pca_fwd = -pca_fwd
        angle = float(np.degrees(np.arctan2(np.dot(u_up, np.cross(u_fwd, pca_fwd)),
                                            np.dot(u_fwd, pca_fwd))))
        refined_angle = refine_horizontal_yaw(points, u_up, u_fwd, angle)
        yaw_deg = float(np.clip(refined_angle, -30.0, 30.0))
        radians = np.radians(yaw_deg)
        u_fwd = u_fwd * np.cos(radians) + np.cross(u_up, u_fwd) * np.sin(radians)
        if abs(refined_angle) > 30.0:
            warnings.append("Horizontal PCA yaw exceeds 30 degrees; correction limited to 30 degrees.")
        if toe_flip:
            u_fwd = -u_fwd
        toe_conf = 1.0
        flipped = bool(toe_flip)
        toe_info = {"override": toe, "vector": u_fwd.tolist(), "pcaYawDeg": angle,
                    "footprintYawDeg": refined_angle, "yawLimitDeg": 30.0}
    else:
        u_fwd, toe_conf, flipped, toe_info = detect_length_and_toe(
            points, u_up, toe_flip=toe_flip
        )
        if toe_conf < 0.25:
            warnings.append(
                f"Low confidence ({toe_conf:.2f}) in automatic toe direction. Verify orientation or use --toe-flip."
            )
        if flipped:
            warnings.append("Toe direction manually inverted via --toe-flip.")

    # 3. Lateral axis and Left/Right detection
    allow_right = bool(toe_flip or (toe and toe != "auto"))
    u_lat, detected_side, side_conf, side_info = detect_side(
        points, u_up, u_fwd, requested_side=side, allow_right=allow_right
    )
    if detected_side != "left":
        warnings.append(f"Shoe side classified as '{detected_side}' (confidence {side_conf:.2f}).")
    elif side_conf < 0.65:
        warnings.append("Shoe side is ambiguous; verify that the source is a single left shoe.")

    # 4. Scale calculation
    source_fwd = points @ u_fwd
    source_shoe_len = float(np.ptp(source_fwd))
    if source_shoe_len < 1e-6:
        raise ValueError("Source shoe length is zero")

    inner_margin_m = max(0.0, length_margin_cm / 100.0)
    wall_m = max(0.0, wall_allowance_cm / 100.0)
    target_shoe_len = target["foot_length"] + inner_margin_m + wall_m
    scale = target_shoe_len / source_shoe_len

    # 5. Rotation matrix: [u_lat, u_up, u_fwd] -> [X, Y, Z]
    # R maps u_lat to [1,0,0], u_up to [0,1,0], u_fwd to [0,0,1]
    rot = np.vstack([u_lat, u_up, u_fwd])
    if np.linalg.det(rot) <= 0:
        raise ValueError("Basis vectors do not form a right-handed orientation-preserving frame")

    scaled_rot = scale * rot
    p_rot = points @ scaled_rot.T

    # 6. Translations
    # a. Ground sole on y = 0
    t_y = -float(p_rot[:, 1].min())

    # b. Align heel 5 mm behind target foot heel
    heel_offset_m = 0.005
    t_z = (target["heel_z"] - heel_offset_m) - float(p_rot[:, 2].min())

    # c. Center on target foot center line in X
    shoe_center_x = float((p_rot[:, 0].min() + p_rot[:, 0].max()) / 2.0)
    t_x = target["center_x"] - shoe_center_x

    # 7. Affine 4x4 row-major matrix
    matrix = np.eye(4)
    matrix[:3, :3] = scaled_rot
    matrix[:3, 3] = [t_x, t_y, t_z]

    source_up = points @ u_up
    source_lat = points @ u_lat

    report = {
        "pose": "a",
        "matrix": matrix.tolist(),
        "axes": {
            "up": u_up.tolist(),
            "forward": u_fwd.tolist(),
            "lateral": u_lat.tolist(),
        },
        "scale": float(scale),
        "yawCorrectionDeg": yaw_deg,
        "upConfidence": float(up_info.get("confidence", 1.0)),
        "sourceExtent": {
            "length": float(source_shoe_len),
            "height": float(np.ptp(source_up)),
            "width": float(np.ptp(source_lat)),
        },
        "targetFoot": {
            "length": float(target["foot_length"]),
            "heelZ": float(target["heel_z"]),
            "toeZ": float(target["toe_z"]),
            "groundY": float(target["ground_y"]),
            "centerX": float(target["center_x"]),
        },
        "targetShoeLengthM": float(target_shoe_len),
        "toeConfidence": float(toe_conf),
        "toeFlipped": bool(flipped),
        "detectedSide": detected_side,
        "sideConfidence": float(side_conf),
        "warnings": warnings,
        "diagnostics": {
            "up": up_info,
            "toe": toe_info,
            "side": side_info,
        },
    }

    return matrix, report


def align_shoe(
    source: Path | str,
    out_path: Path | str | None = None,
    node: list[str] | None = None,
    exclude_material: list[str] | None = None,
    side: str = "left",
    up: str | None = None,
    toe: str = "auto",
    toe_flip: bool = False,
    length_margin_cm: float = 1.2,
    wall_allowance_cm: float = 0.8,
    body: Any = None,
    body_path: Path | None = None,
) -> tuple[dict, dict]:
    """Load a shoe mesh from source, compute left shoe alignment, and optionally write the sidecar."""
    path = Path(source).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    # Default: +y for glTF/GLB, auto for OBJ
    effective_up = up
    if effective_up is None:
        effective_up = "+y" if path.suffix.lower() in (".glb", ".gltf") else "auto"

    parts = load_parts(path, node_names=node, exclude_material_names=exclude_material)
    parts = [clean(p) for p in parts]
    if not parts or not any(len(p.positions) for p in parts):
        raise ValueError("No shoe geometry remains after node/material filtering")
    points = np.concatenate([p.positions for p in parts])

    matrix, report = compute_shoe_alignment(
        points=points,
        body=body,
        body_path=body_path,
        side=side,
        up=effective_up,
        toe=toe,
        toe_flip=toe_flip,
        length_margin_cm=length_margin_cm,
        wall_allowance_cm=wall_allowance_cm,
    )

    sidecar = {"pose": "a", "matrix": matrix.tolist()}

    if out_path is not None:
        target = Path(out_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        report_file = target.parent / f"{target.stem}.report.json"
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return sidecar, report


def _axis_arguments(argv: list[str] | None) -> list[str]:
    # argparse treats '-x' as another option, even when --toe expects a value.
    tokens = list(sys.argv[1:] if argv is None else argv)
    normalized = []
    i = 0
    while i < len(tokens):
        if tokens[i] in ("--up", "--toe") and i + 1 < len(tokens) and tokens[i + 1] in AXIS_CHOICES:
            normalized.append(f"{tokens[i]}={tokens[i + 1]}")
            i += 2
        else:
            normalized.append(tokens[i])
            i += 1
    return normalized


class AxisArgumentParser(argparse.ArgumentParser):
    """Accept signed coordinate values both as '--toe -x' and '--toe=-x'."""

    def parse_known_args(self, args=None, namespace=None):
        return super().parse_known_args(_axis_arguments(args), namespace)


def parser() -> argparse.ArgumentParser:
    p = AxisArgumentParser(
        description="Compute an orientation-preserving 4x4 alignment sidecar for a single left shoe."
    )
    p.add_argument("source", type=Path, help="GLB/glTF or OBJ source file")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("alignment.json"),
        help="Path for the output alignment sidecar JSON (default: alignment.json)",
    )
    p.add_argument(
        "--node",
        action="append",
        help="GLB node name to include (repeatable; isolates one shoe from a scene/pair)",
    )
    p.add_argument(
        "--exclude-material",
        action="append",
        help="GLB material name to exclude (repeatable)",
    )
    p.add_argument(
        "--side",
        choices=("left", "right"),
        default="left",
        help="Shoe side (default: left; right will be rejected to protect --mirror-pair)",
    )
    p.add_argument(
        "--up",
        choices=AXIS_CHOICES,
        default=None,
        help="Up vector override in source coordinates (default: +y for GLB/glTF, auto for OBJ)",
    )
    p.add_argument(
        "--toe",
        choices=AXIS_CHOICES,
        default="auto",
        help="Toe direction override in source coordinates (default: auto)",
    )
    p.add_argument(
        "--toe-flip",
        action="store_true",
        help="Invert the automatic toe direction decision",
    )
    p.add_argument(
        "--length-margin-cm",
        type=float,
        default=1.2,
        help="Shoe inner length margin beyond foot length in cm (default: 1.2)",
    )
    p.add_argument(
        "--wall-allowance-cm",
        type=float,
        default=0.8,
        help="Shoe wall/sole thickness allowance in cm (default: 0.8)",
    )
    p.add_argument(
        "--body",
        type=Path,
        default=None,
        help="Path to MakeHuman runtime body assets (default: apps/web/public/assets/body)",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write full diagnostic report JSON",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        sidecar, report = align_shoe(
            source=args.source,
            out_path=args.out,
            node=args.node,
            exclude_material=args.exclude_material,
            side=args.side,
            up=args.up,
            toe=args.toe,
            toe_flip=args.toe_flip,
            length_margin_cm=args.length_margin_cm,
            wall_allowance_cm=args.wall_allowance_cm,
            body_path=args.body,
        )
        if args.report:
            rep_path = Path(args.report).resolve()
            rep_path.parent.mkdir(parents=True, exist_ok=True)
            rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"shoe_align: wrote {args.out} (scale={report['scale']:.4f}, "
            f"toe_confidence={report['toeConfidence']:.2f})"
        )
        return 0
    except Exception as exc:
        print(f"shoe_align error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
