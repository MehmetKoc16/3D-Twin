"""Local, conservative thin-frame detection. Never display input pixels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
VIEWS = ("front", "back", "profile_nose_right", "profile_nose_left")
EYES = ((33, 160, 158, 133, 153, 144), (362, 385, 387, 263, 373, 380))
BROWS = ((70, 63, 105, 66, 107, 55, 65, 52, 53, 46),
         (336, 296, 334, 293, 300, 276, 283, 282, 295, 285))


def private_path(path: Path) -> Path:
    """All CLI inputs/outputs must stay inside gitignored user-data."""
    resolved = path.resolve()
    if not resolved.is_relative_to((ROOT / "user-data").resolve()):
        raise ValueError("Personal inputs and outputs must stay under user-data/")
    return resolved


def read_image(path: Path) -> np.ndarray:
    path = private_path(path)
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path.name}")
    return image


def write_image(path: Path, image: np.ndarray, quality: int = 95) -> None:
    path = private_path(path)
    options = [cv2.IMWRITE_JPEG_QUALITY, quality] if path.suffix == ".jpg" else []
    success, encoded = cv2.imencode(path.suffix, image, options)
    if not success:
        raise ValueError(f"Cannot encode {path.name}")
    encoded.tofile(path)


def write_json(path: Path, value: dict) -> None:
    path = private_path(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class Landmarker:
    def __init__(self, model: Path):
        import mediapipe as mp
        self.mp = mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1, min_face_detection_confidence=0.15,
            min_face_presence_confidence=0.15, min_tracking_confidence=0.15)
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, image: np.ndarray) -> np.ndarray | None:
        h, w = image.shape[:2]
        # Crops and small roll changes can recover a near-profile detection.
        # Map every returned point back into the original, unmirrored pixel grid.
        crops = [(0, 0, w, h), (int(w*.08),int(h*.1),int(w*.92),int(h*.8)),
                 (int(w*.15),int(h*.2),int(w*.85),int(h*.7))]
        for x0, y0, x1, y1 in crops:
            rgb = cv2.cvtColor(image[y0:y1,x0:x1], cv2.COLOR_BGR2RGB)
            ch, cw = rgb.shape[:2]
            for roll in (0, -15, 15):
                matrix = cv2.getRotationMatrix2D((cw/2,ch/2),roll,1)
                rotated = cv2.warpAffine(rgb,matrix,(cw,ch),borderMode=cv2.BORDER_REFLECT)
                inverse = cv2.invertAffineTransform(matrix)
                for mirrored in (False, True):
                    data = np.ascontiguousarray(rotated[:,::-1] if mirrored else rotated)
                    result = self.detector.detect(self.mp.Image(
                        image_format=self.mp.ImageFormat.SRGB, data=data))
                    if result.face_landmarks:
                        points = np.array([((1-p.x if mirrored else p.x)*cw,p.y*ch,1)
                                           for p in result.face_landmarks[0]],dtype=np.float32)
                        return (points @ inverse.T + [x0,y0]).astype(np.float32)
        return None

    def close(self) -> None:
        self.detector.close()


@dataclass
class Detection:
    mask: np.ndarray
    core_mask: np.ndarray
    protected: np.ndarray
    rings: list[dict]
    pupil_distance_px: float
    status: str


def pupil_centers(landmarks: np.ndarray) -> np.ndarray:
    if len(landmarks) >= 478:
        return landmarks[[468, 473]]
    return np.array([landmarks[list(eye)].mean(axis=0) for eye in EYES])


def protection(shape: tuple, landmarks: np.ndarray, distance: float) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    for ids in EYES + BROWS:
        cv2.fillConvexPoly(mask, cv2.convexHull(landmarks[list(ids)].astype(np.int32)), 255)
    margin = max(1, round(distance * 0.015))
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                    (margin * 2 + 1, margin * 2 + 1)))


def ridge_response(gray: np.ndarray, distance: float) -> np.ndarray:
    response = np.zeros_like(gray)
    for fraction in (0.025, 0.045, 0.075):
        size = max(3, round(distance * fraction) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        response = np.maximum(response, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel))
        response = np.maximum(response, cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel))
    return response


def fit_ring(response: np.ndarray, protected: np.ndarray, pupil: np.ndarray,
             distance: float, horizontal_ratio: float = 1.0) -> dict | None:
    """Score full rim coverage, rather than selecting isolated eyelid arcs."""
    h, w = response.shape
    angles = np.linspace(0, 2 * math.pi, 180, endpoint=False)
    cosine, sine = np.cos(angles), np.sin(angles)
    best_score, best = 0.0, None
    for dx in np.linspace(-0.12, 0.12, 7):
        for dy in np.linspace(-0.08, 0.20, 8):
            cx, cy = pupil + [dx * distance, dy * distance]
            for radius in np.linspace(0.28, 0.49, 18) * distance:
                for aspect in (0.88, 1.0, 1.12):
                    rx, ry = radius * horizontal_ratio, radius * aspect
                    values = []
                    for offset in (-1.5, 0, 1.5):
                        x = np.clip(np.rint(cx + (rx + offset) * cosine).astype(int), 0, w - 1)
                        y = np.clip(np.rint(cy + (ry + offset) * sine).astype(int), 0, h - 1)
                        values.append(np.where(protected[y, x] == 0, response[y, x], 0))
                    signal = np.max(values, axis=0)
                    coverage = float(np.mean(signal >= 8))
                    sector = np.mean((signal >= 8).reshape(12, 15), axis=1)
                    score = float(np.mean(np.minimum(signal, 60))) * coverage
                    if coverage >= 0.48 and np.count_nonzero(sector > 0.3) >= 9 and score > best_score:
                        best_score = score
                        best = dict(center=[float(cx), float(cy)], radius_x=float(rx),
                                    radius_y=float(ry), coverage=coverage, score=score)
    if best is not None:
        # Remove the offset-search bias toward the inside edge of a thick rim.
        yy, xx = np.mgrid[:h, :w]
        cx, cy = best["center"]
        radial = np.sqrt(((xx-cx)/best["radius_x"])**2 + ((yy-cy)/best["radius_y"])**2)
        supported = ((np.abs(radial-1)*min(best["radius_x"],best["radius_y"]) < distance*0.06)
                     & (response >= 8) & (protected == 0))
        if np.count_nonzero(supported) > 30:
            samples = np.column_stack((xx[supported], yy[supported])).astype(np.float32)
            (ex,ey),(diameter_a,diameter_b),angle = cv2.fitEllipse(samples)
            radians = math.radians(angle)
            a,b = diameter_a/2,diameter_b/2
            rx = math.hypot(a*math.cos(radians),b*math.sin(radians))
            ry = math.hypot(a*math.sin(radians),b*math.cos(radians))
            if (np.hypot(ex-cx,ey-cy) < distance*.08
                    and .8 < rx/best["radius_x"] < 1.2
                    and .8 < ry/best["radius_y"] < 1.2):
                best["center"] = [float(ex),float(ey)]
                best["radius_x"],best["radius_y"] = rx,ry
    return best


def trace_rim(response: np.ndarray, protected: np.ndarray, ring: dict,
              accessory: np.ndarray) -> tuple[np.ndarray,float]:
    """Closed, smoothly varying ridge path; radii can form a rounded polygon."""
    angles = np.linspace(0,2*math.pi,240,endpoint=False)
    factors = np.linspace(.72,1.24,49)
    cx,cy = ring["center"]
    x = np.rint(cx+np.cos(angles)[:,None]*ring["radius_x"]*factors).astype(int)
    y = np.rint(cy+np.sin(angles)[:,None]*ring["radius_y"]*factors).astype(int)
    x = np.clip(x,0,response.shape[1]-1)
    y = np.clip(y,0,response.shape[0]-1)
    signal = response[y,x].astype(np.float32)
    cost = -np.minimum(signal,100)/100 - accessory[y,x]*.5 + (factors[None,:]-1)**2*3
    cost += (protected[y,x]>0)*5
    transition = (factors[:,None]-factors[None,:])**2*70
    start = int(np.argmin(cost[0]))
    previous = np.full(len(factors),1e6)
    previous[start] = cost[0,start]
    parents = np.zeros_like(cost,dtype=np.int32)
    for i in range(1,len(angles)):
        options = previous[:,None]+transition
        parents[i] = np.argmin(options,axis=0)
        previous = options[parents[i],np.arange(len(factors))]+cost[i]
    index = int(np.argmin(previous+transition[:,start]))
    choices = [index]
    for i in range(len(angles)-1,0,-1):
        index = int(parents[i,index])
        choices.append(index)
    choices = np.array(choices[::-1])
    coverage = float(np.mean(signal[np.arange(len(angles)),choices]>=8))
    contour = np.column_stack((x[np.arange(len(angles)),choices],y[np.arange(len(angles)),choices]))
    return contour.astype(np.int32),coverage


def detect_glasses(image: np.ndarray, landmarks: np.ndarray | None,
                   view: str = "front", profile_ipd_to_face_height: float = 0.35,
                   eye_width_to_ipd: float = 0.32,
                   accessory_probability: np.ndarray | None = None,
                   hair_mask: np.ndarray | None = None) -> Detection:
    empty = np.zeros(image.shape[:2], np.uint8)
    if landmarks is None or view == "back":
        return Detection(empty, empty.copy(), empty.copy(), [], 0.0,
                         "back_no_visible_face" if view == "back" else "no_face_skipped")
    # Work at bounded resolution; restore masks at original photo resolution.
    scale = min(1.0, 1024 / max(image.shape[:2]))
    work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    points = landmarks * scale
    pupils = pupil_centers(points)
    distance = float(np.linalg.norm(pupils[1] - pupils[0]))
    if distance < 12:
        return Detection(empty, empty.copy(), empty.copy(), [], distance / scale, "degenerate_landmarks")
    protect = protection(work.shape, points, distance)
    size = (work.shape[1],work.shape[0])
    accessory = (cv2.resize(accessory_probability,size,interpolation=cv2.INTER_LINEAR)
                 if accessory_probability is not None else np.zeros(work.shape[:2],np.float32))
    hair = (cv2.resize(hair_mask,size,interpolation=cv2.INTER_NEAREST)
            if hair_mask is not None else np.zeros(work.shape[:2],np.uint8))
    protect[hair>0] = 255
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    yy, xx = np.mgrid[:h, :w]
    # Profiles use visible eye width for scale; IPD is foreshortened there.
    is_profile = view.startswith("profile")
    widths = [np.linalg.norm(points[e[0]] - points[e[3]]) for e in EYES]
    face_height = float(np.linalg.norm(points[10] - points[152]))
    reference = max(distance, face_height * profile_ipd_to_face_height) if is_profile else distance
    response = ridge_response(gray, reference)
    rings = []
    band = np.zeros_like(gray)
    traced = np.zeros_like(gray)
    for i, pupil in enumerate(pupils):
        ratio = max(0.18, min(1.0, widths[i] / (reference * eye_width_to_ipd))) if is_profile else 1.0
        ring = fit_ring(response, protect, pupil, reference, ratio)
        if ring is None:
            continue
        rings.append(ring)
        cx, cy = ring["center"]
        rx, ry = ring["radius_x"], ring["radius_y"]
        contour,coverage = trace_rim(response,protect,ring,accessory)
        if coverage >= .4:
            ring["contour"] = contour.tolist()
            ring["path_coverage"] = coverage
            cv2.polylines(traced,[contour],True,255,max(1,round(reference*.008)))
            local = np.zeros_like(gray)
            cv2.polylines(local,[contour],True,255,max(5,round(reference*.06)))
            band |= local
    # Bridge / nose pads and temple arms: bounded zones, never entire eye region.
    order = np.argsort(pupils[:, 0])
    left, right = pupils[order]
    middle_y = float(pupils[:, 1].mean())
    radii_x = [reference*.35,reference*.35]
    if len(rings) == 2:
        radii_x = [r["radius_x"] for r in sorted(rings,key=lambda r:r["center"][0])]
    elif is_profile:
        radii_x = [max(4,widths[i]/eye_width_to_ipd*.35) for i in order]
    bridge = ((xx > left[0] + radii_x[0] * 0.85) & (xx < right[0] - radii_x[1] * 0.85)
              & (yy > middle_y - reference * 0.22) & (yy < middle_y + reference * 0.12))
    pads = np.zeros_like(bridge)
    for pad_x in (left[0]+radii_x[0]+reference*.06,
                  right[0]-radii_x[1]-reference*.06):
        # Include each pad's wire attachment at the inner rim. Skeletonizing
        # a merged pad/rim junction otherwise drops its observed outer edge.
        pads |= (((xx-pad_x)/(reference*.11))**2
                 +((yy-middle_y-reference*.17)/(reference*.16))**2)<1
    min_x, max_x = float(points[:, 0].min()), float(points[:, 0].max())
    temples = (((xx < left[0] - radii_x[0] * 0.8) & (xx > min_x - reference * 0.15))
               | ((xx > right[0] + radii_x[1] * 0.8) & (xx < max_x + reference * 0.15)))
    temples &= (yy > middle_y - reference * 0.12) & (yy < middle_y + reference * 0.12)
    # Accessories confirm ridge pixels; their coarse regions are never area-filled.
    near_eyes = ((xx>min_x-reference*.1)&(xx<max_x+reference*.1)
                &(yy>middle_y-reference*.65)&(yy<middle_y+reference*.65))
    confirmation = ((accessory>=.08)&near_eyes&(protect==0)).astype(np.uint8)*255
    nearby = (cv2.dilate(confirmation,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
              if accessory_probability is not None else np.ones_like(bridge))
    allowed = (band>0)|bridge|pads|temples
    ridge = ((response >= 8) & allowed & nearby & (protect == 0)).astype(np.uint8) * 255
    wire_ridge = ridge.copy()
    # Only short, confirmed rim gaps can be joined; no unsupported contour fill.
    traced[(~nearby)|(protect>0)] = 0
    supported_trace = (cv2.dilate(ridge,np.ones((3,3),np.uint8))>0)&(traced>0)
    ridge[supported_trace] = 255
    ridge[protect>0] = 0
    # Reject isolated texture specks, preserve thin connected curves and pads.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ridge)
    core = np.zeros_like(gray)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= max(4, reference * 0.045):
            core[labels == label] = 255
    core = cv2.morphologyEx(core,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    core[protect>0] = 0
    mask = core.copy()
    # Measurement width must not count coarse accessory/lens-interior regions.
    core &= wire_ridge
    mask[protect > 0] = 0
    size = (image.shape[1], image.shape[0])
    mask, core, protect = [cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
                           for m in (mask, core, protect)]
    pad_pixels = cv2.resize(pads.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0
    observed_pads = ((mask>0)&pad_pixels).astype(np.uint8)*255
    # Protect original-resolution eye/brow boundaries after upsampling too.
    protect = protection(image.shape, landmarks, distance / scale)
    if hair_mask is not None:
        protect[hair_mask>0] = 255
    # Broad ridge responses (e.g. lens shadow) must not become area masks.
    # Use nominal 5px centerline bands, retaining actual edges only where the
    # measured local ridge radius is small (including rasterized corners).
    centers = cv2.ximgproc.thinning(mask)
    radius = cv2.distanceTransform(mask,cv2.DIST_L2,5)
    thin_centers = ((centers>0)&(radius<=3.8)).astype(np.uint8)*255
    observed_thin = (mask>0)&(cv2.dilate(thin_centers,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))>0)
    mask = cv2.dilate(centers,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    fringe = cv2.dilate(observed_thin.astype(np.uint8)*255,np.ones((3,3),np.uint8))>0
    corner_cap = cv2.dilate(centers,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))>0
    mask[fringe&corner_cap] = 255
    # Compact nose pads are real hardware, not wire: retain their confirmed
    # ridge outline/interior instead of collapsing a pad to a centerline.
    mask |= cv2.dilate(observed_pads,np.ones((3,3),np.uint8))
    mask[protect > 0] = 0
    core[protect > 0] = 0
    for ring in rings:
        ring["center"] = [v / scale for v in ring["center"]]
        ring["radius_x"] /= scale
        ring["radius_y"] /= scale
        if "contour" in ring:
            ring["contour"] = (np.array(ring["contour"])/scale).tolist()
    return Detection(mask, core, protect, rings, distance / scale,
                     "landmark_ridge_accessory_contour" if accessory_probability is not None
                     else "landmark_ridge" if rings else "landmark_lines_no_round_rim")


def inpaint(image: np.ndarray, mask: np.ndarray, method: str = "telea") -> np.ndarray:
    if not np.any(mask):
        return image.copy()
    radius = max(3.0, min(12.0, max(image.shape[:2]) * 0.003))
    return cv2.inpaint(image, mask, radius,
                       cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS)
