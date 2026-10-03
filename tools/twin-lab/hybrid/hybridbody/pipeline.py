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
from twinrefine.scan import welded_vertex_normals

from . import BODY_ASSETS, PARTS_ASSETS, REPO, log
from .assemble import assemble, eye_offsets, mesh_validity, write_glb
from .body import neck_definition, solved_body
from .faceasset import write_face_asset
from .facetex import (
    bake_face,
    blend_unobserved,
    build_head_mesh,
    head_island_triangles,
    load_photos,
    neck_weights,
    vertex_photo_check,
)
from .hair import PROCEDURAL, build_procedural_hair
from .haircheck import skin_weight_report
from .headfit import build_template, fit_head, load_flame_fit, read_face_map, seam_vertices
from .partstex import (
    DEFAULT_HAIR_LINEAR,
    DEFAULT_HAIR_SRGB,
    card_tile,
    eye_tile,
    lab_to_srgb255,
    make_tile,
    pack_strip,
    plausible_iris,
    srgb_hex,
)
from .photocolours import hair_colour, iris_colour
from .register import Surface, smoothstep
from .skin import (
    Underwear,
    fade_to_tone,
    hair_cover,
    match_mean,
    neck_luminance,
    pad_texture,
    paint_body,
    sample_hair_surface,
    seam_blend,
    tint_scalp,
)
from .template import load_part

# Bones whose weight marks "waist and upper legs": the painted boxer shorts stay off arms and hands.
UNDERWEAR_BONES = ("pelvis", "thigh_l", "thigh_r", "spine_01")
PART_IDS = {"eyes": "eyes-default", "eyebrows": "eyebrows-default", "eyelashes": "eyelashes-default"}
DEFAULT_HAIR = PROCEDURAL  # procedural cards in their own dtHair node: short sides and back, volume on top, no fringe
HAIR_FALLBACK_ID = "hair-short"  # what the app's part library mounts for a face asset that says "procedural"
SCALP_TINT = 0.45  # dark scalp under the hair: hair colour x this (linear-ish factor on sRGB)
TILE_WIDTHS = {"hair": 1024, "eyes": 512, "eyebrows": 1024, "eyelashes": 1024}
HAIR_ATLAS_SIZE = (2048, 1024)  # the strand data atlas of the dtHair node
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
    names: tuple  # parts that enter the body mesh, in assembly order (the procedural hair is a separate node)
    loaded: dict
    bound: dict
    tiles: dict
    strip: np.ndarray
    colours: dict
    report: dict
    hair_srgb: np.ndarray
    hair_build: object | None = None  # ``HairBuild`` of the procedural hair (None for a MakeHuman hair part)


def choose_hair_colour(hair_photo: dict, override: str | None) -> dict:
    """Base colour of the procedural hair: the default #2a1e18, ``--hair-hex #rrggbb``, or ``--hair-hex photo``.

    The photographed colour is lit and washed out (near-grey in dim photos), so it is only used when asked for; then
    it is made a natural dark brown (``plausible_hair``).
    """
    from .partstex import plausible_hair

    measured = hair_photo["srgb"] if hair_photo["from_photo"] else None
    if override and override.strip().lower() == "photo":
        choice = plausible_hair(measured, None)
        return {**choice, "method": "photo: " + choice["method"]}
    if override:
        return plausible_hair(measured, override)
    return {
        "srgb": DEFAULT_HAIR_SRGB.copy(),
        "method": "default",
        "measured_hex": srgb_hex(measured) if measured is not None else None,
    }


def build_parts(
    final,
    tone,
    size,
    flame_eyes,
    head_surface,
    head_faces,
    *,
    hair,
    brows,
    hair_photo,
    iris,
    hair_hex=None,
    hair_style=None,
) -> PartsBuild:
    """Load, bind (MHCLO binding on the deformed body), recolour and tile the eyes, lashes, hair and optional brows.

    ``hair`` is a MakeHuman hair part id or ``"procedural"`` (cards grown on the deformed head, see ``hairgen``).
    """
    strip_height = size // 4
    procedural = hair == PROCEDURAL
    ids = {**PART_IDS, "hair": hair}
    names = tuple(
        n
        for n in ("eyes", "eyebrows", "eyelashes", "hair")
        if (brows or n != "eyebrows") and not (procedural and n == "hair")
    )
    loaded = {n: load_part(PARTS_ASSETS, ids[n]) for n in names}
    bound = {n: part.bind(final) for n, part in loaded.items()}
    shift, eye_report = eye_offsets(bound["eyes"], flame_eyes)
    bound["eyes"] = bound["eyes"] + shift

    hair_choice = None
    if procedural:
        hair_choice = choose_hair_colour(hair_photo, hair_hex)
        hair_srgb = np.asarray(hair_choice["srgb"], float)
        hair_linear = np.power(hair_srgb / 255.0, 2.2)
    else:
        hair_linear = (
            np.power(np.asarray(hair_photo["srgb"]) / 255.0, 2.2) if hair_photo["from_photo"] else DEFAULT_HAIR_LINEAR
        )
        hair_srgb = np.power(hair_linear, 1 / 2.2) * 255
    brow_srgb = np.power(hair_linear * 0.9, 1 / 2.2) * 255
    iris_srgb = np.asarray(iris["srgb"], float)
    images = {
        "eyes": eye_tile(loaded["eyes"], iris_srgb),
        "eyelashes": card_tile(loaded["eyelashes"], None),
    }
    hair_build = None
    if procedural:
        eye_y = float(np.mean([v[:, 1].mean() for v in flame_eyes.values()]))
        hair_build = build_procedural_hair(
            head_surface, head_faces, eye_y, hair_srgb, style=hair_style, atlas_size=HAIR_ATLAS_SIZE
        )
    else:
        images["hair"] = card_tile(loaded["hair"], hair_linear)
    if brows:
        images["eyebrows"] = card_tile(loaded["eyebrows"], hair_linear * 0.9)
    tiles = {
        n: make_tile(n, images[n], loaded[n].uv, TILE_WIDTHS[n] * size // 4096, strip_height)
        for n in ("hair", "eyes", "eyebrows", "eyelashes")
        if n in images
    }
    strip = pack_strip(list(tiles.values()), size, strip_height, y_offset=size)
    clearance = {
        n: surface_clearance(bound[n], head_surface, head_faces)
        for n in names
        if n in ("eyelashes", "eyebrows") or n == "hair"
    }
    if procedural:
        clearance["hair"] = hair_build.report["penetration"]
    report = {
        "hair": {
            "id": ids["hair"],
            "colour_hex": srgb_hex(hair_srgb),
            "from_photo": hair_photo["from_photo"],
            "samples": hair_photo["samples"],
            "clearance_to_head": clearance["hair"],
            **(
                {
                    "colour_method": hair_choice["method"],
                    "measured_hex": hair_choice["measured_hex"],
                    "node": hair_build.mesh.node,
                    "colours": hair_build.mesh.colours,
                    "procedural": hair_build.report,
                }
                if procedural
                else {}
            ),
        },
        "eyes": {
            "id": ids["eyes"],
            "iris_hex": srgb_hex(iris_srgb),
            "from_photo": iris["from_photo"],
            "samples": iris["samples"],
            "method": iris.get("method"),
            "measured_hex": iris.get("measured_hex"),
            "placement": eye_report,
        },
        "eyelashes": {"id": ids["eyelashes"], "clearance_to_head": clearance["eyelashes"]},
        "eyebrows": (
            {"id": ids["eyebrows"], "colour_hex": srgb_hex(brow_srgb), "clearance_to_head": clearance["eyebrows"]}
            if brows
            else {"enabled": False, "reason": "the photographed brows are baked into the face texture"}
        ),
        "triangles": {n: int(len(loaded[n].faces)) for n in names},
        "pruned_cards": "none: eyelash (and brow) cards keep their texture alpha as real cut-outs",
    }
    colours = {"hair": srgb_hex(hair_srgb), "eyebrows": srgb_hex(brow_srgb), "iris": srgb_hex(iris_srgb)}
    return PartsBuild(names, loaded, bound, tiles, strip, colours, report, hair_srgb, hair_build)


def run(
    bodyfix,
    out,
    *,
    measurements=None,
    fit=None,
    photos=None,
    flame_assets=None,
    hair=DEFAULT_HAIR,
    brows=False,
    texture_size=4096,
    previews=True,
    preview_dir=None,
    iris_hex=None,
    hair_hex=None,
    hair_style=None,
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
    iris_measured = iris_colour(flame, front_view, cameras["front"], photo_images["front"])
    choice = plausible_iris(iris_measured["srgb"] if iris_measured["from_photo"] else None, iris_hex)
    iris = {**iris_measured, "srgb": choice["srgb"].tolist(), "method": choice["method"]}
    iris["measured_hex"] = srgb_hex(iris_measured["srgb"]) if iris_measured["from_photo"] else None
    log(f"iris {iris['measured_hex']} -> {srgb_hex(choice['srgb'])} ({choice['method']})")
    hair_photo = hair_colour(flame, {"front": front_view, "right": right_view}, cameras, photo_images)

    # ------------------------------------------------------------------------------------------ parts
    log("binding the MakeHuman parts to the deformed head")
    aligned = head_fit.aligned_flame
    flame_eyes = {"left": aligned[flame.masks["left_eyeball"]], "right": aligned[flame.masks["right_eyeball"]]}
    head_positions = final[: model.nr][head.ids]
    parts = build_parts(
        final,
        tone,
        size,
        flame_eyes,
        head_positions,
        head.faces,
        hair=hair,
        brows=brows,
        hair_photo=hair_photo,
        iris=iris,
        hair_hex=hair_hex,
        hair_style=hair_style,
    )

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

    # Under the jaw the photos only carry the chin shadow (and beard stubble): take the body skin there.
    ty, tx, points = face.texel_y, face.texel_x, face.texel_points
    nw = neck_weights(template, head_fit, flame, head, face)
    neck_sets = {
        "flame_neck_region": nw["raw"] > 0.5,
        "below_chin": points[:, 1] < nw["chin_y"],
    }
    neck_before = {k: neck_luminance(texture, ty, tx, points, v, tone) for k, v in neck_sets.items()}
    hand_over = np.maximum(
        smoothstep((nw["smooth"] - 0.25) / 0.5), smoothstep((nw["chin_y"] - 0.002 - points[:, 1]) / 0.006)
    )
    texture = fade_to_tone(texture, ty, tx, hand_over, tone)
    neck_after = {k: neck_luminance(texture, ty, tx, points, v, tone) for k, v in neck_sets.items()}
    neck_report = {
        "chin_y_m": nw["chin_y"],
        "flame_neck_vertices": nw["vertices"],
        "body_skin_lab": tone.tolist(),
        "before": neck_before,
        "after": neck_after,
    }
    log(
        f"neck luminance vs body: {neck_report['before']['below_chin'].get('delta_l_vs_body')} -> "
        f"{neck_report['after']['below_chin'].get('delta_l_vs_body')}"
    )

    x0, y0 = face.origin
    h, w = texture.shape[:2]

    def paste(source):
        out = canvas.copy()
        out[y0 : y0 + h, x0 : x0 + w][face.covered] = source[face.covered]
        return out

    covered[y0 : y0 + h, x0 : x0 + w] |= face.covered
    photo_check = vertex_photo_check(
        face, head, template, model.uv[head.ids], pad_texture(paste(texture), covered), size
    )
    log(f"vertex/photo check: {photo_check}")

    # Dark scalp under the hair: the soft gaps between hair cards read as hair, not as skin.
    head_normals = welded_vertex_normals(head_positions, head.faces)
    tri = head_island_triangles(template, head)[face.texel_face]
    texel_normals = np.einsum("ij,ijk->ik", face.texel_bary, head_normals[tri])
    texel_normals /= np.maximum(np.linalg.norm(texel_normals, axis=1, keepdims=True), 1e-9)
    if parts.hair_build is not None:
        cover = parts.hair_build.field.cover(points)  # the hair density field itself: exact hairline, stubble shadow
    else:
        hair_surface = sample_hair_surface(parts.bound["hair"], parts.loaded["hair"].faces)
        cover = hair_cover(points, texel_normals, hair_surface)
    scalp_lab = to_lab(np.clip(parts.hair_srgb * SCALP_TINT, 1, 255).astype(np.float32).reshape(1, 3) / 255.0)[0]
    texture_under_hair = tint_scalp(texture, ty, tx, cover, scalp_lab)
    eye_y = float(np.mean([v[:, 1].mean() for v in flame_eyes.values()]))
    front = (texel_normals[:, 2] > 0.6) & (points[:, 1] > eye_y + 0.03) & (np.abs(points[:, 0]) < 0.02)
    forehead = {}
    if front.any():
        open_y = points[front & (cover < 0.5), 1]
        hairline_y = float(open_y.max()) if len(open_y) else float(points[front, 1].min())
        forehead = {
            "midline_texels": int(front.sum()),
            "hair_cover_fraction": float((cover[front] > 0.5).mean()),
            "open_skin_above_eyes_mm": (hairline_y - eye_y) * 1000,
        }
    scalp_report = {
        "forehead": forehead,
        "method": "procedural hair density field" if parts.hair_build is not None else "ray march to the hair surface",
        "tint_lab": scalp_lab.tolist(),
        "covered_texel_fraction": float((cover > 0.5).mean()),
        "head_island_texels": int(len(cover)),
    }
    body_atlas = np.dstack((pad_texture(paste(texture_under_hair), covered), np.full((size, size), 255, np.uint8)))
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
    hair_mesh = parts.hair_build.mesh if parts.hair_build is not None else None
    if hair_mesh is not None:
        # the weights the rig stage will transfer to the cards (closest body vertices): they must follow the head bone
        parts.report["hair"]["skin_weights"] = skin_weight_report(
            model, final[: model.nr], hair_mesh.positions, hair_mesh.faces
        )
        log(
            f"hair skin weights on the head chain: {parts.report['hair']['skin_weights']['fraction_head_chain_ge_0_95']:.3f}"
        )

    # -------------------------------------------------------------------------------------------- output
    head_ring = np.flatnonzero(np.isin(template.inverse, seam_w) & (template.islands == template.head_island))
    torso_ring = np.flatnonzero(np.isin(template.inverse, seam_w) & (template.islands == torso_island))
    seam = seam_report(atlas[..., :3], body_uv, head_ring, torso_ring, np.array([size, atlas_height]), tone)
    # Everything below the cut is exactly the solved body; parts and the deformed head sit above it.
    part_lowest = [mesh.positions[mesh.kind > 0][:, 1].min()] if (mesh.kind > 0).any() else []
    if hair_mesh is not None:
        part_lowest.append(hair_mesh.positions[:, 1].min())
    cut = float(min(template.anchor_y, *part_lowest) - 0.002)
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
        "hairProcedural": parts.hair_build is not None,
        **(
            {
                "hairAtlas": {
                    "format": "rcov-groot-bvar/1",
                    "size": [int(hair_mesh.atlas.shape[1]), int(hair_mesh.atlas.shape[0])],
                    "channels": "R coverage, G root-to-tip (0 root), B variation, A min(1, 2.5 R)",
                    "shader": "creategamecharacters/threejs-hair-shader (MIT), compact atlas",
                }
            }
            if hair_mesh is not None
            else {}
        ),
        "license": "MakeHuman CC0 + private non-commercial FLAME/Pixel3DMM fit; never redistribute",
    }
    extras = {"dtHybrid": marker, "dtBodyfix": solution, "dtScanHandsRemoved": False, "dtHasMakeHumanHands": True}
    log("writing the hybrid GLB")
    glb = write_glb(out, mesh, atlas, extras, hair=hair_mesh)

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
            "neck": neck_report,
            "scalp_tint": scalp_report,
            "alpha": {
                "channel": "RGBA PNG; skin texels alpha 255, eyelash (and brow) cards carry their strip texture alpha; "
                "the hair is a separate node with its own strand data atlas",
                "opaque_fraction": float((atlas[..., 3] == 255).mean()),
                "cutout_texels": int((atlas[..., 3] < 128).sum()),
            },
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
            "hair": {
                "id": hair,
                "colourHex": parts.colours["hair"],
                **(
                    {
                        "kind": "procedural-cards",
                        "fallbackId": HAIR_FALLBACK_ID,
                        "style": parts.hair_build.style.to_dict(),
                        "format": "rcov-groot-bvar/1",
                        "colours": hair_mesh.colours,
                        "note": "cards grown on the deformed head (hybrid/hairgen.py), a separate dtHair node with a strand "
                        "data atlas for the hair shader; the app's part library has no such part, so it mounts "
                        "fallbackId until the cards are loaded from the twin GLB",
                    }
                    if parts.hair_build is not None
                    else {}
                ),
            },
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
        report["previews"] = write_previews(preview_dir, mesh, bare, atlas, head_y, hair=hair_mesh)
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
    chin = report["texture"]["neck"]["after"]["below_chin"]
    if chin.get("texels") and abs(chin["delta_l_vs_body"]) >= 3:
        problems.append("neck below the jaw differs from the body skin by dL >= 3")
    if report["head"]["triangle_quality"]["flipped_triangles"]:
        problems.append("flipped triangles in the deformed head")
    if report["mesh"]["validity"]["body_nonmanifold_edges"]:
        problems.append("non-manifold body edges")
    hair = report["parts"]["hair"]
    procedural = hair.get("procedural")
    if procedural:
        penetration = procedural["penetration"]
        if penetration["vertices_inside_head"] or penetration["vertices_below_clearance"]:
            problems.append("hair card vertices closer to the head than the clearance")
        if not 8000 <= procedural["triangles"] <= 60000:
            problems.append("hair triangle count outside 8k..60k")
        coverage = procedural.get("coverage")
        if coverage and coverage["min_hidden_fraction"] < 0.8:
            problems.append("hair leaves more than 20 percent of the scalp visible from a main view")
        weights = hair.get("skin_weights")
        if weights and weights["fraction_head_chain_ge_0_95"] < 0.8:
            problems.append("hair vertices are not weighted to the head/neck bones")
    return problems
