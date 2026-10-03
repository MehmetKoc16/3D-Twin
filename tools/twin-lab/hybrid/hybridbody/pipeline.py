"""Template-character twin: solved MakeHuman body, FLAME-deformed head, photo-baked face, CC0 parts."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from flamehead.assets import read_mesh
from flamehead.camera import load_cameras
from flamehead.colour import to_lab
from mh import BODY_DIR, MHModel
from twin_export import evaluate_measure

from . import BODY_ASSETS, PARTS_ASSETS, REPO, log
from .assemble import assemble, eye_offsets, mesh_validity, write_glb
from .body import neck_definition, solved_body
from .faceasset import write_face_asset
from .facetex import bake_face, blend_unobserved, build_head_mesh, load_photos, vertex_photo_check
from .headfit import build_template, fit_head, load_flame_fit, read_face_map, seam_vertices
from .partstex import (
    DEFAULT_HAIR_LINEAR,
    card_tile,
    eye_tile,
    lab_to_srgb255,
    make_tile,
    pack_strip,
    srgb_hex,
    triangle_coverage,
)
from .photocolours import hair_colour, iris_colour
from .register import Surface
from .skin import Underwear, match_mean, pad_texture, paint_body, seam_blend
from .template import load_part

# Bones whose weight marks "waist and upper legs": the painted boxer shorts stay off arms and hands.
UNDERWEAR_BONES = ("pelvis", "thigh_l", "thigh_r", "spine_01")
PART_IDS = {"eyes": "eyes-default", "eyebrows": "eyebrows-default", "eyelashes": "eyelashes-default"}
PRUNE_COVERAGE = {"hair": 0.35, "eyelashes": 0.2}
TILE_WIDTHS = {"hair": 1024, "eyes": 512, "eyebrows": 1024, "eyelashes": 1024}
FALLBACK_SKIN_LAB = np.array([58.35, 7.98, 10.71])


def private_output(path):
    if not Path(path).resolve().is_relative_to(REPO / "user-data"):
        raise ValueError("Hybrid outputs and previews must stay under user-data/")


def vertex_gate(model: MHModel, bones) -> np.ndarray:
    weights = np.zeros((model.nr, len(model.bones)))
    for k in range(4):
        np.add.at(weights, (np.arange(model.nr), model.skin_j[:, k]), model.skin_w[:, k])
    return weights[:, [model.bone_index[b] for b in bones]].sum(1)


def surface_clearance(points: np.ndarray, head_positions: np.ndarray, head_faces: np.ndarray) -> dict:
    """Signed distance of part vertices to the head surface in mm (negative = inside the head)."""
    surface = Surface(head_positions, head_faces)
    point, _, _, normal = surface.closest(points)
    signed = np.einsum("ij,ij->i", points - point, normal) * 1000
    return {
        "vertices": int(len(signed)),
        "min_mm": float(signed.min()),
        "p05_mm": float(np.percentile(signed, 5)),
        "median_mm": float(np.median(signed)),
        "p95_mm": float(np.percentile(signed, 95)),
        "deeper_than_4mm": float((signed < -4).mean()),
        "deeper_than_10mm": float((signed < -10).mean()),
    }


def seam_report(atlas, mesh_uv, head_ring_render, torso_ring_render, atlas_size, tone_lab):
    """Mean Lab on the head and torso copies of the neck seam vertices, and against the photo tone."""
    h, w = atlas.shape[:2]

    def sample(render_ids):
        xy = np.clip((mesh_uv[render_ids] * atlas_size).astype(int), 0, [w - 1, h - 1])
        return to_lab(atlas[xy[:, 1], xy[:, 0]])

    head, torso = sample(head_ring_render), sample(torso_ring_render)
    head_mean, torso_mean = head.mean(0, dtype=np.float64), torso.mean(0, dtype=np.float64)
    delta = head_mean - torso_mean
    return {
        "ring_vertices": int(len(head_ring_render)),
        "head_side_mean_lab": head_mean.tolist(),
        "torso_side_mean_lab": torso_mean.tolist(),
        "delta_lab": delta.tolist(),
        "delta_e76": float(np.linalg.norm(delta)),
        "torso_vs_photo_delta_e76": float(np.linalg.norm(torso_mean - tone_lab)),
        "head_vs_photo_delta_e76": float(np.linalg.norm(head_mean - tone_lab)),
    }


@dataclass
class PartsBuild:
    names: tuple  # parts that enter the mesh, in assembly order
    loaded: dict
    bound: dict
    tiles: dict
    strip: np.ndarray
    colours: dict
    report: dict


def build_parts(
    final, tone, size, flame_eyes, head_surface, head_faces, *, hair, brows, hair_photo, iris
) -> PartsBuild:
    """Load, bind (MHCLO binding on the deformed body), recolour and tile the eyes, lashes, hair and optional brows."""
    strip_height = size // 4
    ids = {**PART_IDS, "hair": hair}
    names = tuple(n for n in ("eyes", "eyebrows", "eyelashes", "hair") if brows or n != "eyebrows")
    loaded = {n: load_part(PARTS_ASSETS, ids[n]) for n in names}
    # The single opaque material has no cut-outs: drop cards that carry (almost) no strands.
    pruned = {}
    for category, threshold in PRUNE_COVERAGE.items():
        keep = triangle_coverage(loaded[category]) >= threshold
        pruned[category] = {"triangles_before": int(len(keep)), "triangles_after": int(keep.sum())}
        loaded[category] = loaded[category].compact(keep)
    bound = {n: part.bind(final) for n, part in loaded.items()}
    shift, eye_report = eye_offsets(bound["eyes"], flame_eyes)
    bound["eyes"] = bound["eyes"] + shift

    skin_srgb = lab_to_srgb255(tone)
    hair_linear = (
        np.power(np.asarray(hair_photo["srgb"]) / 255.0, 2.2) if hair_photo["from_photo"] else DEFAULT_HAIR_LINEAR
    )
    hair_srgb = np.power(hair_linear, 1 / 2.2) * 255
    brow_srgb = np.power(hair_linear * 0.9, 1 / 2.2) * 255
    iris_srgb = np.asarray(iris["srgb"], float)
    images = {
        "hair": card_tile(loaded["hair"], hair_linear, hair_srgb * 0.45),
        "eyes": eye_tile(loaded["eyes"], iris_srgb),
        "eyelashes": card_tile(loaded["eyelashes"], None, skin_srgb * 0.55),
    }
    if brows:
        images["eyebrows"] = card_tile(loaded["eyebrows"], hair_linear * 0.9, skin_srgb)
    tiles = {
        n: make_tile(n, images[n], loaded[n].uv, TILE_WIDTHS[n] * size // 4096, strip_height)
        for n in ("hair", "eyes", "eyebrows", "eyelashes")
        if n in images
    }
    strip = pack_strip(list(tiles.values()), size, strip_height, y_offset=size)
    clearance = {
        n: surface_clearance(bound[n], head_surface, head_faces)
        for n in names
        if n in ("hair", "eyelashes", "eyebrows")
    }
    report = {
        "hair": {
            "id": ids["hair"],
            "colour_hex": srgb_hex(hair_srgb),
            "from_photo": hair_photo["from_photo"],
            "samples": hair_photo["samples"],
            "clearance_to_head": clearance["hair"],
        },
        "eyes": {
            "id": ids["eyes"],
            "iris_hex": srgb_hex(iris_srgb),
            "from_photo": iris["from_photo"],
            "samples": iris["samples"],
            "placement": eye_report,
        },
        "eyelashes": {"id": ids["eyelashes"], "clearance_to_head": clearance["eyelashes"]},
        "eyebrows": (
            {"id": ids["eyebrows"], "colour_hex": srgb_hex(brow_srgb), "clearance_to_head": clearance["eyebrows"]}
            if brows
            else {"enabled": False, "reason": "the photographed brows are baked into the face texture"}
        ),
        "pruned_cards": pruned,
    }
    colours = {"hair": srgb_hex(hair_srgb), "eyebrows": srgb_hex(brow_srgb), "iris": srgb_hex(iris_srgb)}
    return PartsBuild(names, loaded, bound, tiles, strip, colours, report)


def run(
    bodyfix,
    out,
    *,
    measurements=None,
    fit=None,
    photos=None,
    flame_assets=None,
    hair="hair-short",
    brows=False,
    texture_size=4096,
    previews=True,
    preview_dir=None,
):
    started = time.perf_counter()
    out = Path(out)
    private_output(out)
    preview_dir = Path(preview_dir) if preview_dir else out.parent / "previews"
    if previews:
        private_output(preview_dir)
    fit = Path(fit or REPO / "user-data/twin/head/flame/fit")
    photos = Path(photos or REPO / "user-data/twin/head/colab_upload")
    flame_assets = Path(flame_assets or REPO / "user-data/flame")
    out.parent.mkdir(parents=True, exist_ok=True)

    model = MHModel()
    log("solving the MakeHuman body")
    combined, solution, body_report = solved_body(model, bodyfix, measurements)
    neck = neck_definition()
    face_map = read_face_map(BODY_ASSETS / "face-map.json")
    template = build_template(combined[: model.nr], model.faces, face_map, neck["verts"])
    flame = load_flame_fit(fit, flame_assets)

    # ----------------------------------------------------------------------------------- head deformation
    log("registering the template head to the FLAME surface")

    def neck_measure(moved):
        p = combined.copy()
        p[: model.nr] = moved[template.inverse]
        return {
            "girth_cm": evaluate_measure(neck, p, model.nr) * 100,
            "solved_girth_cm": evaluate_measure(neck, combined, model.nr) * 100,
            "target_cm": solution["targetsCm"].get("neck"),
        }

    head_fit = fit_head(template, flame, neck_measure=neck_measure)
    if abs(head_fit.report["neck"]["girth_cm"] - head_fit.report["neck"]["solved_girth_cm"]) > 1e-6:
        raise ValueError("Neck girth changed during the head deformation")
    offsets = head_fit.displacement[template.inverse]  # (nr, 3) per render vertex: the face-asset morph target
    final = combined.copy()
    final[: model.nr] = combined[: model.nr] + offsets

    # ------------------------------------------------------------------------------------- face texture
    size = texture_size
    head = build_head_mesh(template)
    log("baking the face from the photos")
    face = bake_face(template, head_fit, flame, head, model.uv, photos, size, log=log)
    photo = face.report["photo_colour"]
    tone = np.asarray(photo["photo_mean_lab"] if photo else FALLBACK_SKIN_LAB, np.float64)
    log(f"skin tone Lab {tone.round(2).tolist()}")

    cameras = load_cameras(fit / "cameras.json")
    photo_images = load_photos(photos, cameras)
    front_view, _ = read_mesh(fit / "fitted_views/front.ply")
    right_view, _ = read_mesh(fit / "fitted_views/right.ply")
    iris = iris_colour(flame, front_view, cameras["front"], photo_images["front"])
    hair_photo = hair_colour(flame, {"front": front_view, "right": right_view}, cameras, photo_images)

    canvas = np.zeros((size, size, 3), np.uint8)
    covered = np.zeros((size, size), bool)
    inseam = next(d for d in json.loads((BODY_ASSETS / "measures.json").read_text())["measures"] if d["id"] == "inseam")
    pelvis_y = final[model.bones[model.bone_index["pelvis"]].head_vert, 1]
    underwear = Underwear(bottom_y=float(final[inseam["vert"], 1] - 0.11), top_y=float(pelvis_y + 0.09))
    log("painting body skin and underwear")
    gate = vertex_gate(model, UNDERWEAR_BONES)
    skin_report = paint_body(canvas, covered, final[: model.nr], model.faces, model.uv, gate, tone, underwear, size)

    texture, observed = blend_unobserved(face.texture, face.covered, face.confidence, tone)
    texture, match = match_mean(texture, face.covered, face.skin_mask, tone)
    torso_island = int(template.islands[neck["verts"][0]])
    seam_w = seam_vertices(template, torso_island)
    moved = template.base + head_fit.displacement
    texture, seam_blur = seam_blend(texture, face.texel_points, face.texel_y, face.texel_x, moved[seam_w], tone)
    x0, y0 = face.origin
    h, w = texture.shape[:2]
    canvas[y0 : y0 + h, x0 : x0 + w][face.covered] = texture[face.covered]
    covered[y0 : y0 + h, x0 : x0 + w] |= face.covered
    body_atlas = pad_texture(canvas, covered)
    photo_check = vertex_photo_check(face, head, template, model.uv[head.ids], body_atlas, size)
    log(f"vertex/photo check: {photo_check}")

    # ------------------------------------------------------------------------------------------ parts
    log("binding the MakeHuman parts to the deformed head")
    aligned = head_fit.aligned_flame
    flame_eyes = {"left": aligned[flame.masks["left_eyeball"]], "right": aligned[flame.masks["right_eyeball"]]}
    parts = build_parts(
        final,
        tone,
        size,
        flame_eyes,
        final[: model.nr][head.ids],
        head.faces,
        hair=hair,
        brows=brows,
        hair_photo=hair_photo,
        iris=iris,
    )
    atlas = np.vstack((body_atlas, parts.strip))
    atlas_height = atlas.shape[0]
    body_uv = model.uv * np.array([1.0, size / atlas_height])
    cavity = parts.loaded["eyes"].delete_verts
    mesh = assemble(
        final[: model.nr],
        model.faces,
        body_uv,
        [(n, parts.loaded[n], parts.bound[n], parts.tiles[n]) for n in parts.names],
        (size, atlas_height),
        drop_vertices=cavity[cavity < model.nr],
    )
    bare = mesh.without("hair")

    # -------------------------------------------------------------------------------------------- output
    head_ring = np.flatnonzero(np.isin(template.inverse, seam_w) & (template.islands == template.head_island))
    torso_ring = np.flatnonzero(np.isin(template.inverse, seam_w) & (template.islands == torso_island))
    seam = seam_report(atlas, body_uv, head_ring, torso_ring, np.array([size, atlas_height]), tone)
    # Everything below the cut is exactly the solved body; parts and the deformed head sit above it.
    cut = float(min(template.anchor_y, mesh.positions[mesh.kind > 0][:, 1].min()) - 0.002)
    manifest_sha = hashlib.sha256((Path(BODY_DIR) / "manifest.json").read_bytes()).hexdigest()
    marker = {
        "version": 1,
        "frame": "MakeHuman-grounded-A-pose",
        "method": "MakeHuman template head deformed to the FLAME fit; no graft",
        "bodySource": "MakeHuman CC0",
        "bodyManifestSha256": manifest_sha,
        "cutHeightM": cut,
        "ownsHands": True,
        "ownsFeet": True,
        "nativeResolve": body_report["nativeResolve"],
        "hair": hair,
        "license": "MakeHuman CC0 + private non-commercial FLAME/Pixel3DMM fit; never redistribute",
    }
    extras = {"dtHybrid": marker, "dtBodyfix": solution, "dtScanHandsRemoved": False, "dtHasMakeHumanHands": True}
    log("writing the hybrid GLB")
    glb = write_glb(out, mesh, atlas, extras)

    report = {
        "version": 2,
        "method": "template head deformation (Avaturn-style): same vertices, same UVs, same rig",
        "body": body_report,
        "bodyfixSolution": solution,
        "head": head_fit.report,
        "texture": {
            "atlas_size": [int(atlas.shape[1]), int(atlas.shape[0])],
            "square_uv_size": size,
            "photo_observed_ratio": face.report["photo_observed_ratio"],
            "head_island_texels": face.report["island_texels"],
            "confident_fraction": observed["observed_fraction"],
            "fill_ratio": 1.0,
            "photo_colour": photo,
            "vertex_photo_check": photo_check,
            "mean_match": match,
            "body": skin_report,
            "seam_blend": seam_blur,
            "views": face.report["view_weight_share"],
            "gains": face.report["gains"],
        },
        "seam": seam,
        "parts": {**parts.report, "slices": mesh.parts},
        "mesh": {
            "vertices": int(len(mesh.positions)),
            "triangles": int(len(mesh.faces)),
            "height_m": float(mesh.positions[:, 1].max()),
            "bare_head_top_m": float(final[: model.nr][:, 1].max()),
            "target_height_cm": solution["targetsCm"].get("height"),
            "neck": head_fit.report["neck"],
            "validity": mesh_validity(mesh),
        },
        "hands": {
            "native": True,
            "dtScanHandsRemoved": False,
            "dtHasMakeHumanHands": True,
            "appReplacementPolicy": "the current app always draws its own MakeHuman hands over the twin",
        },
        "glb": {**glb, "path": str(out)},
    }
    asset_dir = out.parent / "face_asset"
    asset = write_face_asset(
        asset_dir,
        offsets=offsets,
        manifest_sha256=manifest_sha,
        solved_body={
            "macros": solution["fittedMacros"],
            "modifiers": solution["fittedModifiers"],
            "targetsCm": solution["targetsCm"],
        },
        head_texture=texture,
        window_px=(x0, y0, w, h),
        atlas_size=(int(atlas.shape[1]), int(atlas.shape[0])),
        skin={"lab": tone.tolist(), "srgbHex": srgb_hex(lab_to_srgb255(tone))},
        parts={
            "hair": {"id": hair, "colourHex": parts.colours["hair"]},
            "eyebrows": {"id": PART_IDS["eyebrows"], "colourHex": parts.colours["eyebrows"], "enabled": bool(brows)},
            "eyelashes": {"id": PART_IDS["eyelashes"]},
            "eyes": {
                "id": PART_IDS["eyes"],
                "irisHex": parts.colours["iris"],
                "translationMm": parts.report["eyes"]["placement"],
            },
        },
        metrics={
            "registrationResidualMm": head_fit.report["residual_to_flame_surface"],
            "maxDisplacementMm": head_fit.report["displacement_mm"]["max"],
            "neckGirthCm": head_fit.report["neck"]["girth_cm"],
        },
    )
    report["face_asset"] = {"folder": str(asset_dir), "offsets": asset["headOffsets"]["count"]}
    if previews:
        from .previews import write_previews

        head_y = float(final[: model.nr][head.ids][:, 1].min())
        report["previews"] = write_previews(preview_dir, mesh, bare, atlas, head_y)
    report["seconds"] = round(time.perf_counter() - started, 2)
    (out.parent / "hybrid_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False, default=float) + "\n", encoding="utf8"
    )
    return report


def verify_report(report: dict) -> list[str]:
    """Human-readable list of failed numeric acceptance checks (empty when everything passes)."""
    problems = []
    neck = report["head"]["neck"]
    if abs(neck["girth_cm"] - neck["solved_girth_cm"]) > 1e-4:
        problems.append("neck girth changed")
    if report["seam"]["delta_e76"] >= 2:
        problems.append("neck seam colour step >= 2")
    if report["head"]["triangle_quality"]["flipped_triangles"]:
        problems.append("flipped triangles in the deformed head")
    if report["mesh"]["validity"]["body_nonmanifold_edges"]:
        problems.append("non-manifold body edges")
    return problems
