"""Fit a tidy wire accessory to glasses measured on a PRIVATE Hunyuan3D bust.

Run with the refine stage venv. The FLAME fit is research/non-commercial: obtain
the licensed masks and embedding as documented by head/flame; all fits, busts,
measurements, textures and previews must remain under user-data/. No photos are
loaded here. The generated accessory uses the existing head/glasses GLB contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "hybridbody"

from . import BODY_ASSETS, LAB, REPO

sys.path.insert(0, str(LAB / "head/glasses"))
from make_glasses import (  # noqa: E402
    DEFAULTS,
    encode_glasses,
    node_worlds,
    nose_pad,
    read_glb,
    transform,
    tube,
)


@dataclass
class TwinHead:
    positions: np.ndarray
    faces: np.ndarray
    uv: np.ndarray
    eyes: np.ndarray
    nose: np.ndarray
    ears: np.ndarray
    landmarks: np.ndarray
    landmark_index: np.ndarray
    inverse_head: np.ndarray
    report: dict


def private_output(path: Path) -> None:
    if not path.resolve().is_relative_to(REPO / "user-data"):
        raise ValueError("Personal glasses outputs must stay under user-data/")


def load_twin_head(twin: Path, hybrid_dir: Path, fit: Path, assets: Path) -> TwinHead:
    """Reconstruct the exact saved template and FLAME similarity, without photos.

    Ear targets are the topmost deformed template vertices nearest the aligned
    FLAME ear masks. Eye centres use the same aligned eyeball means as pipeline.
    Nose bridge is MediaPipe 168 on the aligned FLAME embedding.
    """
    from flamehead.geometry import transform as flame_transform
    from mh import MHModel
    from scipy.spatial import cKDTree

    from .body import neck_definition
    from .facetex import build_head_mesh
    from .headfit import build_template, landmark_similarity, load_flame_fit, read_face_map

    report = json.loads((hybrid_dir / "hybrid_report.json").read_text(encoding="utf8"))
    model = MHModel()
    solution = report["bodyfixSolution"]
    original = model.shape(solution["fittedMacros"], solution["fittedModifiers"], ground=True)
    template = build_template(
        original[: model.nr], model.faces, read_face_map(BODY_ASSETS / "face-map.json"), neck_definition()["verts"]
    )
    flame = load_flame_fit(fit, assets)
    sim = landmark_similarity(flame, template, original)
    if abs(sim["scale"] - report["head"]["similarity_scale"]) > 1e-6:
        raise ValueError("Saved head and reconstructed FLAME similarity differ")
    aligned = flame_transform(flame.neutral, sim["scale"], sim["rotation"], sim["translation"])
    flame_lm = flame_transform(flame.landmark_points, sim["scale"], sim["rotation"], sim["translation"])
    nose_ids = np.flatnonzero(flame.landmark_ids == 168)
    if len(nose_ids) != 1:
        raise ValueError("FLAME embedding needs nose bridge landmark 168")
    eyes = np.array([aligned[flame.masks[name]].mean(0) for name in ("right_eyeball", "left_eyeball")])
    eyes = eyes[np.argsort(eyes[:, 0])]
    offsets = np.frombuffer(
        (hybrid_dir / "face_asset/face-offsets.bin").read_bytes(), dtype=[("id", "<u4"), ("delta", "<f4", (3,))]
    )
    final = original[: model.nr].copy()
    final[offsets["id"]] += offsets["delta"]
    head = build_head_mesh(template)
    points = final[head.ids]
    ears = []
    for name in ("right_ear", "left_ear"):
        distance = cKDTree(aligned[flame.masks[name]]).query(points)[0]
        region = points[distance < 0.005]
        if len(region) < 4:
            raise ValueError("Insufficient template ear geometry")
        top = region[region[:, 1] > np.percentile(region[:, 1], 95)]
        ears.append(top.mean(0))
    ears = np.array(ears)
    ears = ears[np.argsort(ears[:, 0])]
    document, _ = read_glb(twin)
    worlds = node_worlds(document)
    joint = [i for i, node in enumerate(document["nodes"]) if node.get("name") == "head"]
    if len(joint) != 1:
        raise ValueError("Twin needs one head bone")
    landmarks = np.asarray(template.landmark_bary @ final[template.first])
    return TwinHead(
        points,
        head.faces,
        model.uv[head.ids],
        eyes,
        flame_lm[nose_ids[0]],
        ears,
        landmarks,
        template.landmark_index,
        np.linalg.inv(worlds[joint[0]]),
        report,
    )


def aligned_bust(path: Path, head: TwinHead):
    """Reuse the hair job's read-only axis/scale/alignment implementation."""
    from .hairgen import HairStyle, HeadSurface, measure_head
    from .hy3d import align_bust, load_bust

    bust = load_bust(path)
    surface = HeadSurface(head.positions, head.faces)
    frame = measure_head(surface, float(head.eyes[:, 1].mean()), HairStyle())
    alignment = align_bust(bust, head.landmarks, head.landmark_index, surface, frame)
    return bust, alignment


def probe(bust_path, head, out):
    from PIL import Image
    from twintex.camera import OrthoCamera

    from .hy3d import vertex_colour_render

    bust, aligned = aligned_bust(bust_path, head)
    p = aligned.positions
    eye = head.eyes.mean(0)
    region = (np.abs(p[:, 0] - eye[0]) < 0.095) & (np.abs(p[:, 1] - eye[1]) < 0.055) & (p[:, 2] > eye[2] - 0.02)
    camera = OrthoCamera.azimuth("front", 0).fit_bounds(p[region], 1000, 700, margin=0.05)
    image, _ = vertex_colour_render(
        p, bust.faces[region[bust.faces].any(1)], aligned.normals, bust.colours, camera, 1000, 700, lights=None
    )
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out / "bust_glasses_front.png")
    np.savez_compressed(
        out / "aligned_bust.npz", positions=p, normals=aligned.normals, faces=bust.faces, colours=bust.colours
    )
    (out / "alignment.json").write_text(json.dumps(aligned.report, indent=2), encoding="utf8")
    (out / "head_landmarks.json").write_text(
        json.dumps(
            {
                "eyes": head.eyes.tolist(),
                "nose": head.nose.tolist(),
                "ears": head.ears.tolist(),
                "inverse_head": head.inverse_head.tolist(),
            },
            indent=2,
        ),
        encoding="utf8",
    )
    return bust, aligned


def measure_rims(
    positions,
    faces,
    normals,
    colours,
    eyes,
    out: Path | None = None,
    *,
    image_camera=None,
    image_name="bust_rim_fit.png",
):
    """Fit two ellipse silhouettes to front-render edges in physical metres.

    Bounded deterministic optimisation samples complete ellipses, preventing an
    eyelid, brow or headphone edge from being mistaken for a rim. The measured
    vertical drop is kept for UV cleanup; final lenses are centred on FLAME eyes.
    """
    import cv2
    from scipy.optimize import differential_evolution
    from twintex.camera import OrthoCamera

    from .hy3d import vertex_colour_render

    centre = eyes.mean(0)
    if image_camera is None:
        region = (np.abs(positions[:, 0] - centre[0]) < 0.095) & (np.abs(positions[:, 1] - centre[1]) < 0.065)
        region &= positions[:, 2] > centre[2] - 0.01
        camera = OrthoCamera.azimuth("front", 0).fit_bounds(positions[region], 1000, 760, margin=0.05)
        image, _ = vertex_colour_render(positions, faces[region[faces].any(1)], normals, colours, camera, 1000, 760)
    else:
        image, camera = image_camera
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 35, 85)
    distances = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    fits, metrics = [], []
    for eye in eyes:

        def score(params, eye=eye):
            x, y, rx, ry = params
            p = np.column_stack((x + rx * np.cos(angles), y + ry * np.sin(angles), np.full(len(angles), centre[2])))
            xy = camera.project(p)[:, :2]
            values = cv2.remap(
                distances,
                xy[:, 0].astype(np.float32).reshape(1, -1),
                xy[:, 1].astype(np.float32).reshape(1, -1),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=30,
            ).ravel()
            # A complete silhouette, with a weak roundness/eye-midline prior.
            return np.mean(np.minimum(values, 20)) + 1.5 * abs(rx / ry - 1) + abs(x - eye[0]) * 100

        result = differential_evolution(
            score,
            [(eye[0] - 0.008, eye[0] + 0.008), (eye[1] - 0.026, eye[1] + 0.004), (0.020, 0.033), (0.020, 0.034)],
            seed=17,
            popsize=14,
            maxiter=130,
            tol=0.001,
            polish=True,
        )
        fits.append(result.x)
        metrics.append(float(result.fun))
    fits = np.asarray(fits)
    if out is not None:
        from PIL import Image

        overlay = image.copy()
        for x, y, rx, ry in fits:
            p = np.column_stack((x + rx * np.cos(angles), y + ry * np.sin(angles), np.full(len(angles), centre[2])))
            xy = np.rint(camera.project(p)[:, :2]).astype(np.int32)
            cv2.polylines(overlay, [xy], True, (0, 180, 255), 2)
        Image.fromarray(overlay).save(out / image_name)
    return fits, {
        "rim_fit_edge_error_pixels": metrics,
        "rim_centres_xy_m": fits[:, :2].tolist(),
        "rim_radii_xy_mm": (fits[:, 2:] * 1000).tolist(),
        "measured_vertical_drop_mm": ((fits[:, 1] - eyes[:, 1]) * 1000).tolist(),
    }


def measure_baked_rims(twin, head, folder):
    """Refit the removal curves to the BAKED texture, never to original photos.

    Hunyuan's reconstructed lens shape differs from the photographed frames.
    Thus bust dimensions fit the accessory, while baked-texture outlines drive
    cleanup. Rendering uses the actual GLB atlas with its alpha/UV mapping.
    """
    import io

    from glbio import read_glb as read_scene
    from PIL import Image
    from twinrefine.render import DEFAULT_LIGHTS
    from twintex.camera import OrthoCamera

    from .previews import render_cutout

    scene = read_scene(str(twin))
    prim = scene.prims[0]
    eye = head.eyes.mean(0)
    bounds = np.array([[eye[0] - 0.090, eye[1] - 0.060, eye[2]], [eye[0] + 0.090, eye[1] + 0.045, eye[2]]])
    camera = OrthoCamera.azimuth("front", 0).fit_bounds(bounds, 1000, 760, margin=0.06)
    mat = scene.materials[prim.material]
    tex = mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    image = scene.images[scene.textures[tex]["source"]]
    atlas = np.array(Image.open(io.BytesIO(image["data"])).convert("RGBA"))
    keep = (prim.positions[prim.indices, 1] > eye[1] - 0.07).any(1)
    render = render_cutout(
        prim.positions, prim.indices[keep], prim.normals, camera, 1000, 760, prim.uv, atlas, lights=DEFAULT_LIGHTS, ss=1
    )
    return measure_rims(
        None, None, None, None, head.eyes, folder, image_camera=(render, camera), image_name="baked_rim_fit.png"
    )


def removal_report_from_bake(texture, positions, faces, uv, eyes, *, folder=None):
    """Measure fresh painted-frame guides before the hybrid atlas is written.

    Inputs are the current deformed head and baked face crop, never an earlier run
    or original photographs. The accessory stage reuses these original guides
    once the hybrid face has already been cleaned.
    """
    from twinrefine.render import DEFAULT_LIGHTS
    from twinrefine.scan import welded_vertex_normals
    from twintex.camera import OrthoCamera

    from .previews import render_cutout

    eyes = np.asarray(eyes, float)
    eyes = eyes[np.argsort(eyes[:, 0])]
    eye = eyes.mean(0)
    bounds = np.array([[eye[0]-.090, eye[1]-.060, eye[2]], [eye[0]+.090, eye[1]+.045, eye[2]]])
    camera = OrthoCamera.azimuth("front", 0).fit_bounds(bounds, 1000, 760, margin=.06)
    render = render_cutout(positions, faces, welded_vertex_normals(positions, faces), camera,
                           1000, 760, uv, texture, lights=DEFAULT_LIGHTS, ss=1)
    rims, measurement = measure_rims(None, None, None, None, eyes, folder,
                                    image_camera=(render, camera), image_name="baked_rim_fit.png")
    z = float(max(positions[:, 2].max(), eye[2]+.025))
    curves = measured_curves(rims, eyes, z)
    # Projection uses arm Y/Z only and separately gates the lateral skin.
    arm_z = np.linspace(z, np.percentile(positions[:, 2], 30), 160)
    arms = np.vstack([np.column_stack((np.full(len(arm_z), eye[0]+sign*.080),
                     np.full(len(arm_z), eye[1]+.011), arm_z)) for sign in (-1, 1)])
    return {"placement": {"eyes_m": eyes.tolist(), "rim_radii_mm": (rims[:, 2:].mean(0)*1000).tolist()},
            "painted_frame_measurement": measurement,
            "removal_curves_m": {"front": curves.tolist(), "temples": arms.tolist()},
            "source": "current hybrid bake, measured before deglass"}


def segment_glasses(positions, normals, colours, eyes, rims, *, corridor_m=0.0025):
    """Geometry/colour diagnostic segmentation: thin rim ridges and lens plates.

    Hair, headphones, brows and the back of the head are excluded. Skin-filled
    lens regions are counted explicitly: they must never become accessory lenses.
    This is a candidate mask, not a claim of semantic separation of a fused mesh.
    """
    p, n, rgb = np.asarray(positions), np.asarray(normals), np.asarray(colours) / 255.0
    chroma = rgb.max(1) - rgb.min(1)
    blue = (rgb[:, 2] > rgb[:, 0] * 1.14) & (rgb[:, 2] > rgb[:, 1] * 1.07)
    dark = rgb.mean(1) < 0.20
    skin = (rgb[:, 0] > rgb[:, 2] + 0.055) & (rgb[:, 0] > rgb[:, 1] + 0.025)
    neutral_metal = (chroma < 0.20) & (rgb.mean(1) > 0.32)
    frame, lens = np.zeros(len(p), bool), np.zeros(len(p), bool)
    for eye, (x, y, rx, ry) in zip(eyes, rims, strict=True):
        radial = np.sqrt(((p[:, 0] - x) / rx) ** 2 + ((p[:, 1] - y) / ry) ** 2)
        front = p[:, 2] > eye[2] + 0.005
        ridge = np.abs(n[:, 2]) < 0.86
        frame |= (np.abs(radial - 1) * min(rx, ry) < corridor_m) & front & (ridge | neutral_metal)
        lens |= (radial < 0.88) & front & (np.abs(n[:, 2]) > 0.65)
    midpoint = eyes.mean(0)
    bridge = (np.abs(p[:, 0] - midpoint[0]) < 0.018) & (
        np.abs(p[:, 1] - (rims[:, 1] + rims[:, 3] * 0.75).mean()) < 0.006
    )
    frame |= bridge & (p[:, 2] > eyes[:, 2].mean() + 0.015) & neutral_metal
    # Temples are behind the frame, beside the skull, near the eye line.
    arms = (np.abs(p[:, 0] - midpoint[0]) > 0.055) & (np.abs(p[:, 0] - midpoint[0]) < 0.095)
    arms &= (np.abs(p[:, 1] - midpoint[1]) < 0.012) & (p[:, 2] > eyes[:, 2].mean() - 0.11)
    frame |= arms & neutral_metal & ~skin
    frame &= ~blue & ~dark
    lens &= ~blue & ~dark
    return (
        frame,
        lens,
        {
            "candidate_frame_vertices": int(frame.sum()),
            "candidate_lens_vertices": int(lens.sum()),
            "skin_coloured_lens_fraction": float(skin[lens].mean()) if lens.any() else 0,
            "excluded_blue_vertices": int(blue.sum()),
        },
    )


def pack_pieces(pieces):
    positions, normals, faces, offset = [], [], [], 0
    for p, n, f in pieces:
        positions.append(p)
        normals.append(n)
        faces.append(f + offset)
        offset += len(p)
    return np.vstack(positions), np.vstack(normals), np.vstack(faces)


def measure_source_temples(positions, frame_mask, eyes):
    """Visible metal-only arm spans; mark headphone-occluded arms as incomplete."""
    centre = eyes.mean(0)
    result = {}
    for side, sign in (("right", -1), ("left", 1)):
        region = frame_mask & (sign * (positions[:, 0] - centre[0]) > 0.060)
        region &= (positions[:, 2] < centre[2] - 0.010) & (np.abs(positions[:, 1] - centre[1]) < 0.015)
        points = positions[region]
        metric = {
            "samples": int(len(points)),
            "complete_arm": False,
            "note": "visible candidate span only; blue headphones obscure the ear hook",
        }
        if len(points) >= 12 and np.ptp(points[:, 2]) > 0.005:
            low, high = np.percentile(points[:, 2], [5, 95])
            slope = np.polyfit(points[:, 2], sign * (points[:, 0] - centre[0]), 1)[0]
            metric.update(
                visible_depth_span_mm=float((high - low) * 1000), visible_splay_deg=float(np.degrees(np.arctan(slope)))
            )
        result[side] = metric
    return result


def fit_accessory(head: TwinHead, radii: np.ndarray, colour: str = "#b9b4ad"):
    """Symmetric ellipse rims and ear-guided arms using the existing wire builder.

    Fit a plane parallel to the eye axis, then move it forward until ALL frame
    vertices and triangle midpoints clear the skin by >=1 mm. Only symmetric
    global dimensions change; noisy bust triangles never enter the final GLB.
    """
    from scipy.interpolate import PchipInterpolator

    from .hairgen import HeadSurface

    centre = head.eyes.mean(0)
    rx, ry = np.asarray(radii, float)
    if not np.isfinite(radii).all() or not 0.012 < rx < 0.04 or not 0.012 < ry < 0.04:
        raise ValueError("Implausible lens radii")
    half_ipd = np.linalg.norm(head.eyes[1, :2] - head.eyes[0, :2]) / 2
    # Do not let measured rims collide at the nose when centring them on the eyes.
    rx = min(rx, half_ipd - 0.004)
    centre[2] = max(head.nose[2] + 0.0018, centre[2] + 0.015)
    wire = 0.00055
    bridge_width = 2 * (half_ipd - rx)
    ear_x = float(np.abs(head.ears[:, 0] - centre[0]).mean())
    ear_y = float(head.ears[:, 1].mean())
    ear_z = float(head.ears[:, 2].mean())
    surface = HeadSurface(head.positions, head.faces)
    roll = np.arctan2(head.eyes[1, 1] - head.eyes[0, 1], head.eyes[1, 0] - head.eyes[0, 0])
    rotation = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    angle = np.linspace(0, 2 * np.pi, 64, endpoint=False)

    def build(front_z, side_x):
        base = np.array([centre[0], centre[1], front_z])
        pieces, fronts, arms = [], [], []
        for sign in (-1, 1):
            path = np.column_stack((sign * half_ipd + rx * np.cos(angle), ry * np.sin(angle), np.zeros(len(angle))))
            fronts.append(path @ rotation.T + base)
            pieces.append(tube(fronts[-1], wire, True))
            hinge_x = max(half_ipd + rx + 0.004, side_x - 0.007)
            hinge = np.array([sign * hinge_x, 0.004, -0.004]) @ rotation.T + base
            end = np.array([sign * (half_ipd + rx), 0, 0]) @ rotation.T + base
            fronts.append(np.linspace(end, hinge, 16))
            pieces.append(tube(fronts[-1], wire))
            # Smooth curve stays outside the temples and passes ABOVE the ear top.
            zs = np.array([ear_z - 0.030, ear_z - 0.012, ear_z + 0.002, front_z - 0.028, hinge[2]])
            xs = np.array([side_x + 0.003, side_x + 0.005, side_x + 0.004, side_x, hinge_x])
            ys = np.array([ear_y - 0.014, ear_y + 0.001, ear_y + 0.003, centre[1] + 0.006, hinge[1]])
            z = np.linspace(zs[-1], zs[0], 28)
            path = np.column_stack((centre[0] + sign * PchipInterpolator(zs, xs)(z), PchipInterpolator(zs, ys)(z), z))
            arms.append(path)
            pieces.append(tube(path, wire))
            # Fitted stalks/pads sit forward of the nose, not inside its surface.
            pad = np.array([sign * (bridge_width / 2 - 0.001), -0.008, -0.001]) @ rotation.T + base
            stalk = np.array([sign * bridge_width / 2, -0.003, 0]) @ rotation.T + base
            fronts.append(np.linspace(stalk, pad, 12))
            pieces.append(tube(fronts[-1], wire * 0.65))
            pieces.append(nose_pad(pad))
        t = np.linspace(0, np.pi, 24)
        bridge = np.column_stack((-bridge_width / 2 * np.cos(t), 0.004 * np.sin(t), np.zeros(len(t))))
        fronts.append(bridge @ rotation.T + base)
        pieces.append(tube(fronts[-1], wire))
        return pack_pieces(pieces), np.vstack(fronts), np.vstack(arms)

    front_z, side_x = float(centre[2]), max(ear_x + 0.0015, half_ipd + rx + 0.008)
    for _ in range(30):
        geometry, front, arms = build(front_z, side_x)
        p, _, f = geometry
        samples = np.vstack(
            (
                p,
                p[f].mean(1),
                (p[f[:, 0]] + p[f[:, 1]]) / 2,
                (p[f[:, 1]] + p[f[:, 2]]) / 2,
                (p[f[:, 2]] + p[f[:, 0]]) / 2,
            )
        )
        distance, _, normal = surface.signed_distance(samples, candidates=24)
        bad = distance < 0.0011
        if not bad.any():
            break
        face_bad = bad & (samples[:, 2] > front_z - 0.022)
        if face_bad.any():
            front_z += min(0.002, max(0.0002, 0.0012 - float(distance[face_bad].min())))
        if (bad & ~face_bad).any():
            side_x += 0.0005
    else:
        raise ValueError("Cannot fit symmetric glasses with 1 mm surface clearance")
    # Lenses are optional: omitted for clear visibility, permitted by the contract.
    local_p = transform(p, head.inverse_head)
    local_n = geometry[1] @ head.inverse_head[:3, :3].T
    centres = np.array([[-half_ipd, 0, 0], [half_ipd, 0, 0]]) @ rotation.T + [centre[0], centre[1], front_z]
    eye_errors = np.linalg.norm(centres[:, :2] - head.eyes[:, :2], axis=1) * 1000
    bridge_points = front[(np.abs(front[:, 0] - centre[0]) < bridge_width / 2 + 0.0001) & (front[:, 1] >= centre[1])]
    bridge_clearance = surface.signed_distance(bridge_points)[0].min() * 1000 - wire * 1000
    params = {
        **DEFAULTS,
        "outerRadius": float(rx),
        "lensAspect": float(ry / rx),
        "bridgeWidth": float(bridge_width),
        "frameWidth": float(2 * side_x),
        "thickness": 2 * wire,
        "colour": colour,
        "templeLength": float(front_z - (ear_z - 0.030)),
        "clearLenses": False,
    }
    metrics = {
        "coordinateSpace": "world metres (+Y up, +Z front)",
        "eyes_m": head.eyes.tolist(),
        "nose_bridge_m": head.nose.tolist(),
        "ear_tops_m": head.ears.tolist(),
        "lens_centres_m": centres.tolist(),
        "eye_xy_error_mm": eye_errors.tolist(),
        "lens_eye_depth_mm": ((centres[:, 2] - head.eyes[:, 2]) * 1000).tolist(),
        "head_local_lens_centres_m": transform(centres, head.inverse_head).tolist(),
        "bridge_clearance_mm": float(bridge_clearance),
        "min_surface_clearance_mm": float(distance.min() * 1000),
        "triangle_samples_checked": int(len(samples)),
        "triangles": int(len(f)),
        "rim_radii_mm": [float(rx * 1000), float(ry * 1000)],
        "interpupillary_mm": float(2 * half_ipd * 1000),
        "temple_endpoints_m": arms[[27, -1]].tolist(),
        "temple_path_length_mm": [
            float(np.linalg.norm(np.diff(arm, axis=0), axis=1).sum() * 1000) for arm in np.split(arms, 2)
        ],
        "temple_splay_deg": float(np.degrees(np.arctan2(side_x - (half_ipd + rx), front_z - ear_z))),
        "temple_side_x_mm": float(side_x * 1000),
        "temple_ear_top_clearance_design_mm": 3.0,
        "eye_axis_roll_deg": float(np.degrees(roll)),
        "lenses": "omitted (clear)",
        "metal": colour,
    }
    return params, (local_p, local_n, geometry[2]), front, arms, metrics


def measured_curves(rims, eyes, z):
    """Dense original frame outlines for texture projection (not the new fit)."""
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    curves = [np.column_stack((x + rx * np.cos(t), y + ry * np.sin(t), np.full(len(t), z))) for x, y, rx, ry in rims]
    x_left = rims[0, 0] + rims[0, 2] * 0.90
    x_right = rims[1, 0] - rims[1, 2] * 0.90
    t = np.linspace(0, np.pi, 100)
    y = float(eyes[:, 1].mean())
    curves.append(
        np.column_stack(
            ((x_left + x_right) / 2 - (x_right - x_left) / 2 * np.cos(t), y + 0.003 * np.sin(t), np.full(len(t), z))
        )
    )
    # Near-horizontal upper eyewires and endpieces in the photographs can differ
    # from an ellipse (Hunyuan turned them into a swollen, lower round outline).
    for sign in (-1, 1):
        x = eyes[:, 0].mean() + sign * np.linspace(0.016, 0.080, 160)
        curves.append(
            np.column_stack(
                (x, y + 0.011 - 0.003 * ((np.abs(x - eyes[:, 0].mean()) - 0.040) / 0.040) ** 2, np.full(len(x), z))
            )
        )
        # Small dense pad guide: removes photographed pads/stalks as well as wire.
        gx, gy = np.meshgrid(np.linspace(0.012, 0.016, 12), np.linspace(-0.014, -0.007, 18))
        curves.append(np.column_stack((eyes[:, 0].mean() + sign * gx.ravel(), y + gy.ravel(), np.full(gx.size, z))))
    return np.vstack(curves)


def encode_document(document, blob):
    import struct

    document["buffers"][0]["byteLength"] = len(blob)
    encoded = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary = bytes(blob) + b"\0" * (-len(blob) % 4)
    return (
        struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(binary))
        + struct.pack("<I4s", len(encoded), b"JSON")
        + encoded
        + struct.pack("<I4s", len(binary), b"BIN\0")
        + binary
    )


def clean_baked_texture(twin, out, hybrid_dir, head, curves, arms, rims, preview_dir):
    """Create a separate deglassed rigged GLB; preserve every other GLB resource.

    Appended PNG bufferView replaces only the body's base colour image reference.
    The raw hybrid/rig outputs and the concurrent hair work remain untouched.
    """
    import io

    from accessory_glb import view_bytes
    from PIL import Image
    from twintex.raster import rasterize_uv

    from .deglass_tex import FrameProjection, remove_glasses_frames

    document, original_blob = read_glb(twin)
    hair_name = document.get("asset", {}).get("extras", {}).get("dtHairNode")
    node = next(n for n in document["nodes"] if "mesh" in n and n.get("name") != hair_name)
    primitive = document["meshes"][node["mesh"]]["primitives"][0]
    material = document["materials"][primitive["material"]]
    image_id = document["textures"][material["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["source"]
    atlas = np.array(
        Image.open(io.BytesIO(view_bytes(document, original_blob, document["images"][image_id]["bufferView"]))).convert(
            "RGBA"
        )
    )
    asset = json.loads((hybrid_dir / "face_asset/face-asset.json").read_text(encoding="utf8"))
    window = asset["texture"]["pixelWindow"]
    x0, y0, w, h = [window[key] for key in ("x", "y", "width", "height")]
    size = head.report["texture"]["square_uv_size"]
    uv_pixels = head.uv * size - [x0, y0]
    # Only the head UV island, never eye cavities with independent UV islands.
    keep = (
        (uv_pixels[head.faces, 0] >= -1).all(1)
        & (uv_pixels[head.faces, 0] <= w + 1).all(1)
        & (uv_pixels[head.faces, 1] >= -1).all(1)
        & (uv_pixels[head.faces, 1] <= h + 1).all(1)
    )
    faces = head.faces[keep]
    fid, bary = rasterize_uv(uv_pixels, faces, w, h)
    ty, tx = np.nonzero(fid >= 0)
    points = np.einsum("ij,ijk->ik", bary[ty, tx], head.positions[faces[fid[ty, tx]]])
    covered = fid >= 0
    crop = atlas[y0 : y0 + h, x0 : x0 + w].copy()
    # Brows are above both the photographed rim and its search corridor.
    protected = np.zeros((h, w), bool)
    protect = points[:, 1] > head.eyes[:, 1].mean() + 0.022
    protected[ty[protect], tx[protect]] = True
    geometry = FrameProjection(points, ty, tx, curves, arms, head.eyes, rims[:, 2:].mean(0), covered, protected)
    clean, report, mask = remove_glasses_frames(crop, geometry, return_report=True, method="ns", radius=5)
    report["head_island_texels"] = int(covered.sum())
    report["fraction_of_head_island"] = float(mask.sum() / max(1, covered.sum()))
    report["bake_mapping"] = "deformed template barycentric positions in original MakeHuman UV; glTF v-down"
    report["pixel_window"] = window
    atlas[y0 : y0 + h, x0 : x0 + w] = clean
    Image.fromarray(clean).save(hybrid_dir / "face_asset/face-texture-deglassed.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(preview_dir / "deglass_uv_mask.png")
    png = io.BytesIO()
    Image.fromarray(atlas).save(png, format="PNG")
    blob = bytearray(original_blob)
    blob.extend(b"\0" * (-len(blob) % 4))
    new_view = len(document["bufferViews"])
    document["bufferViews"].append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(png.getvalue())})
    blob.extend(png.getvalue())
    document["images"][image_id] = {"bufferView": new_view, "mimeType": "image/png"}
    document["asset"].setdefault("extras", {})["dtDeglass"] = report
    out.write_bytes(encode_document(document, blob))
    return report


def write_previews(twin, head, geometry, colour, folder):
    """All head views of the accessory alone and on the actual textured twin."""
    import io

    import cv2
    from glbio import read_glb as read_scene
    from PIL import Image
    from twinrefine.render import DEFAULT_LIGHTS
    from twintex.colorspace import linear_to_srgb

    from .hy3d import vertex_colour_render
    from .previews import head_camera, render_cutout

    p, n, f = geometry
    world = np.linalg.inv(head.inverse_head)
    p, n = transform(p, world), n @ world[:3, :3].T
    rgb = np.array([int(colour[i : i + 2], 16) for i in (1, 3, 5)])
    colours = np.tile(rgb, (len(p), 1))
    from .previews import HEAD_VIEWS

    for name, angle in HEAD_VIEWS:
        camera = head_camera(name, angle, p, 800, margin=0.12)
        image, _ = vertex_colour_render(p, f, n, colours, camera, 800, 800, ss=2, spec=0.45)
        Image.fromarray(image).save(folder / f"accessory_{name}.png")
    scene = read_scene(str(twin))
    for name, angle in HEAD_VIEWS:
        camera = head_camera(name, angle, head.positions, 800, margin=0.08)
        frame_image, frame_depth = vertex_colour_render(p, f, n, colours, camera, 800, 800, ss=2, spec=0.45)
        base = None
        base_depth = np.full((1600, 1600), np.inf)
        for prim in scene.prims:
            keep = (prim.positions[prim.indices, 1] > head.positions[:, 1].min() - 0.01).any(1)
            if not keep.any() or prim.uv is None or prim.material is None:
                continue
            mat = scene.materials[prim.material]
            tex = mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]
            image = scene.images[scene.textures[tex]["source"]]
            atlas = np.array(Image.open(io.BytesIO(image["data"])).convert("RGBA"))
            linear, depth = render_cutout(
                prim.positions,
                prim.indices[keep],
                prim.normals,
                camera,
                800,
                800,
                prim.uv,
                atlas,
                lights=DEFAULT_LIGHTS,
                ss=2,
                raw=True,
            )
            depth = depth.reshape(1600, 1600)
            if base is None:
                base = linear
            else:
                nearer = depth < base_depth
                base[nearer] = linear[nearer]
            base_depth = np.minimum(base_depth, depth)
        image = np.clip(
            np.rint(linear_to_srgb(cv2.resize(base, (800, 800), interpolation=cv2.INTER_AREA)) * 255), 0, 255
        ).astype(np.uint8)
        frame_visible = cv2.resize(
            (frame_depth < base_depth).astype(np.float32), (800, 800), interpolation=cv2.INTER_AREA
        )
        image = np.rint(image * (1 - frame_visible[..., None]) + frame_image * frame_visible[..., None]).astype(
            np.uint8
        )
        Image.fromarray(image).save(folder / f"head_{name}.png")


def build_glasses(twin, hybrid_dir, bust_path, fit, assets, out, cleaned_twin, *, previews=True):
    private_output(out)
    private_output(cleaned_twin)
    if out.resolve() in {twin.resolve(), bust_path.resolve()} or cleaned_twin.resolve() in {
        twin.resolve(),
        bust_path.resolve(),
        out.resolve(),
    }:
        raise ValueError("Outputs must not overwrite inputs or each other")
    out.parent.mkdir(parents=True, exist_ok=True)
    folder = out.parent / "previews_glasses"
    folder.mkdir(parents=True, exist_ok=True)
    head = load_twin_head(twin, hybrid_dir, fit, assets)
    bust, alignment = aligned_bust(bust_path, head)
    rims, measured = measure_rims(alignment.positions, bust.faces, alignment.normals, bust.colours, head.eyes, folder)
    frame, lens, segmentation = segment_glasses(alignment.positions, alignment.normals, bust.colours, head.eyes, rims)
    measured["temples"] = measure_source_temples(alignment.positions, frame, head.eyes)
    colours = bust.colours[frame]
    # Bust colours contain reconstructed flesh; only neutral metal samples decide
    # the material. The original asset's silver bridge is the strongest evidence.
    neutral = colours[(colours.max(1) - colours.min(1) < 18) & (colours.mean(1) > 140)]
    metal_rgb = np.clip(np.median(neutral, axis=0), 135, 215) if len(neutral) else np.array([185, 180, 173])
    colour = "#" + "".join(f"{int(v):02x}" for v in metal_rgb)
    params, geometry, front, arms, placement = fit_accessory(head, rims[:, 2:].mean(0), colour)
    out.write_bytes(
        encode_glasses(
            params,
            np.zeros(3),
            "Hunyuan silhouette fitted procedural wires + FLAME landmarks",
            geometry=geometry,
            placement=placement,
        )
    )
    hybrid_deglass = head.report.get("texture", {}).get("deglass")
    if hybrid_deglass is not None:
        # Pipeline already applied the hook (or explicitly opted out). Preserve
        # the rigged bytes; never inpaint a cleaned face for a second time.
        cleaned_twin.write_bytes(twin.read_bytes())
        deglass = {**hybrid_deglass, "accessory_stage": "pass through hybrid deglass decision"}
        guides = hybrid_deglass.get("glasses_report", {})
        curves = np.asarray(guides.get("removal_curves_m", {}).get("front", front))
        old_arms = np.asarray(guides.get("removal_curves_m", {}).get("temples", arms))
        painted_measurement = guides.get("painted_frame_measurement", {})
        from PIL import Image

        face_image = hybrid_dir / "face_asset/face-texture.png"
        (hybrid_dir / "face_asset/face-texture-deglassed.png").write_bytes(face_image.read_bytes())
        audit = hybrid_dir / "deglass_audit.npz"
        if hybrid_deglass.get("enabled") and audit.is_file():
            with np.load(audit) as data:
                mask = data["mask"].astype(np.uint8) * 255
            Image.fromarray(mask).save(folder / "deglass_uv_mask.png")
        else:
            with Image.open(face_image) as image:
                Image.fromarray(np.zeros((image.height, image.width), np.uint8)).save(folder / "deglass_uv_mask.png")
    else:
        painted_rims, painted_measurement = measure_baked_rims(twin, head, folder)
        curves = measured_curves(painted_rims, head.eyes, float(front[:, 2].max()))
        # Original photographed temples lie near the outer upper rim, not necessarily
        # on the new centred rims. Keep their removal guides independent of the fit.
        old_arms = arms.copy()
        old_arms[:, 1] = head.eyes[:, 1].mean() + 0.011
        deglass = clean_baked_texture(twin, cleaned_twin, hybrid_dir, head, curves, old_arms, painted_rims, folder)
    deglass["painted_frame_measurement"] = painted_measurement
    report = {
        "path": "fitted procedural generator",
        "reason": "generated rims are swollen/skin coloured; lens plates merge into the face; a raw cut would carry skin and noise",
        "source": str(bust_path),
        "alignment": alignment.report,
        "measurement": measured,
        "segmentation": segmentation,
        "placement": placement,
        "params": params,
        "deglass": deglass,
        "removal_curves_m": {"front": curves.tolist(), "temples": old_arms.tolist()},
        "accessory": str(out),
        "cleaned_rigged": str(cleaned_twin),
        "license": "Private Hunyuan/FLAME derived measurements; keep all outputs under user-data",
    }
    (out.parent / "glasses_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    if previews:
        write_previews(cleaned_twin, head, geometry, colour, folder)
    print("Glasses: fitted generator; validated rigid head accessory; UV deglass complete", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twin", type=Path, required=True)
    parser.add_argument("--hybrid-dir", type=Path, required=True)
    parser.add_argument("--bust", type=Path, required=True)
    parser.add_argument("--fit", type=Path, default=REPO / "user-data/twin/head/flame/fit")
    parser.add_argument("--flame-assets", type=Path, default=REPO / "user-data/flame")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--cleaned-twin", type=Path)
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args()
    private_output(args.out)
    if args.probe:
        head = load_twin_head(args.twin, args.hybrid_dir, args.fit, args.flame_assets)
        probe(args.bust, head, args.out.parent / "previews_glasses")
    else:
        build_glasses(
            args.twin,
            args.hybrid_dir,
            args.bust,
            args.fit,
            args.flame_assets,
            args.out,
            args.cleaned_twin or args.out.parent / "glasses_rigged.glb",
            previews=not args.no_previews,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
