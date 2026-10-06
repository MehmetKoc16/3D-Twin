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
from .hair import FORMAT_SHELL, HY3D, PROCEDURAL, build_procedural_hair
from .haircheck import skin_weight_report
from .hairhy3d import DEFAULT_BUST, ShellField, ShellStyle, build_hy3d_hair
from .headfit import build_template, fit_head, load_flame_fit, read_face_map, seam_vertices
from .neckhair import BLEND_M, flame_vertex_groups, region_stats, restrict_photo, restriction_weights, texel_weights
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
    border_audit,
    fade_to_tone,
    finish_texture,
    hair_cover,
    match_mean,
    neck_luminance,
    pad_texture,
    paint_body,
    repair_border_colours,
    sample_hair_surface,
    seam_blend,
    tint_scalp,
)
from .skin_source import load_skin
from .template import load_part

# Bones whose weight marks "waist and upper legs": the painted boxer shorts stay off arms and hands.
UNDERWEAR_BONES = ("pelvis", "thigh_l", "thigh_r", "spine_01")
PART_IDS = {"eyes": "eyes-default", "eyebrows": "eyebrows-default", "eyelashes": "eyelashes-default"}
DEFAULT_HAIR = PROCEDURAL  # procedural cards in their own dtHair node: short sides and back, volume on top, no fringe
BUST_RELATIVE = DEFAULT_BUST.relative_to(REPO)  # user-data/twin/hy3d/hy3d.glb: the user's own hair, when it exists


def resolve_hair(hair: str | None, repo: Path | None = None) -> str:
    """``None`` (the default) picks the user's own hair (``hy3d``) when the bust exists in ``user-data/``, else cards."""
    if hair:
        return hair
    return HY3D if (Path(repo or REPO) / BUST_RELATIVE).is_file() else DEFAULT_HAIR
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


def shell_scalp_tint(field, texture, covered, ty, tx, points, normals, distance, cover, scalp_lab):
    """The skin texture under and around the hy3d shell, plus the numbers of the temple / sideburn band.

    Round 3: a graded stubble tint (full under the shell, a few mm of dithered fade at the forehead hairline, a longer
    dark band down the temple and sideburn) in a smooth colour. ``legacy`` (round 2: ear-protected stubble fringe in the
    nearest shell sample's colour) is rebuilt only to put numbers on the improvement.
    """
    legacy_cover, _ = field.tint_cover(points, normals, cover)
    legacy = finish_texture(tint_scalp(texture, ty, tx, legacy_cover, field.tint_lab_nearest(points)), covered)
    weight = field.tint_weight(points, distance)
    grain = np.random.default_rng(23).normal(size=len(weight)).astype(np.float32)
    grain = np.clip(grain, -2.0, 2.0) / 1.2
    new = finish_texture(
        tint_scalp(texture, ty, tx, weight, scalp_lab, blur_px=1.2, grain=grain, dither=field.DITHER), covered
    )
    band = field.band_weight(points) > 0.5
    ring = band & (distance > 0.004) & (distance <= 0.010)  # the full-stubble skin just outside the shell edge, in the band
    after_lab, before_lab = to_lab(new[ty[ring], tx[ring]]), to_lab(legacy[ty[ring], tx[ring]])
    skin_lab = to_lab(texture[ty[ring], tx[ring]])
    d = np.linspace(0.0, 0.03, 3001)
    widths = {}
    for name, curve in (("legacy", field.front_profile(d, legacy=True)), ("graded", field.front_profile(d))):
        hi, lo = d[np.argmax(curve < 0.9)], d[np.argmax(curve < 0.1)]
        widths[name] = {"d90_mm": float(hi * 1000), "d10_mm": float(lo * 1000), "width_mm": float((lo - hi) * 1000)}
    report = {
        "texels": int(ring.sum()),
        "pale_threshold_l": 35.0,
        "region": "temple/sideburn band (az 42..112 deg, eye-9..eye+62 mm, away from the ears), skin 4..10 mm outside the shell",
        "pale_texels_before": int((before_lab[:, 0] > 35).sum()) if ring.any() else 0,
        "pale_texels_after": int((after_lab[:, 0] > 35).sum()) if ring.any() else 0,
        "mean_l_before": float(before_lab[:, 0].mean()) if ring.any() else None,
        "mean_l_after": float(after_lab[:, 0].mean()) if ring.any() else None,
        "l_std_before": float(before_lab[:, 0].std()) if ring.any() else None,
        "l_std_after": float(after_lab[:, 0].std()) if ring.any() else None,
        "chroma_std_before": float(np.hypot(before_lab[:, 1], before_lab[:, 2]).std()) if ring.any() else None,
        "chroma_std_after": float(np.hypot(after_lab[:, 1], after_lab[:, 2]).std()) if ring.any() else None,
        "mean_l_untinted_skin": float(skin_lab[:, 0].mean()) if ring.any() else None,
        "fade_width_mm": widths,
        "method": "graded stubble tint with grain: full under the shell, 2 mm + 7 mm fade at the hairline, "
        "full to 9 mm and fading to 15 mm in the sideburn band; smoothed shell colour",
    }
    return new, report


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
    bust=None,
) -> PartsBuild:
    """Load, bind (MHCLO binding on the deformed body), recolour and tile the eyes, lashes, hair and optional brows.

    ``hair`` is a MakeHuman hair part id, ``"procedural"`` (cards grown on the deformed head, see ``hairgen``) or
    ``"hy3d"`` (the shell cut out of the user's bust, see ``hairhy3d``; ``bust`` carries its inputs).
    """
    strip_height = size // 4
    shell = hair == HY3D
    procedural = hair == PROCEDURAL or shell  # a separate dtHair node, no MakeHuman hair part
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
    hair_build = None
    if shell:
        eye_y = float(np.mean([v[:, 1].mean() for v in flame_eyes.values()]))
        hair_build = build_hy3d_hair(
            bust["path"],
            head_surface,
            head_faces,
            eye_y,
            bust["landmark_points"],
            bust["landmark_index"],
            style=bust.get("style"),
            colour_hex=hair_hex or "#2a1e18",
        )
        hair_srgb = np.array([int(hair_build.mesh.colours["colorHex"][i : i + 2], 16) for i in (1, 3, 5)], float)
        hair_choice = {"srgb": hair_srgb, "method": "Lab chroma grade with preserved luminance", "measured_hex": None}
        hair_linear = np.power(hair_srgb / 255.0, 2.2)
    elif procedural:
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
    if procedural and not shell:
        eye_y = float(np.mean([v[:, 1].mean() for v in flame_eyes.values()]))
        hair_build = build_procedural_hair(
            head_surface, head_faces, eye_y, hair_srgb, style=hair_style, atlas_size=HAIR_ATLAS_SIZE
        )
    elif not procedural:
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
                    ("shell" if shell else "procedural"): hair_build.report,
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
    hair=None,
    brows=False,
    texture_size=4096,
    previews=True,
    preview_dir=None,
    iris_hex=None,
    hair_hex=None,
    hair_style=None,
    deglass=True,
    glasses_bust=None,
    fetch_skin=False,
    restrict_neck_hair=True,
):
    started = time.perf_counter()
    hair = resolve_hair(hair)
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
    if size > 4096:
        raise ValueError("Body atlas width must not exceed 4096")
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
    bust = None
    if hair == HY3D:
        bust_path = REPO / BUST_RELATIVE
        if not bust_path.is_file():
            raise ValueError(f"--hair hy3d needs the Hunyuan3D bust at {bust_path}")
        moved_landmarks = np.asarray(template.landmark_bary @ (template.base + head_fit.displacement))
        bust = {
            "path": bust_path,
            "landmark_points": moved_landmarks,
            "landmark_index": template.landmark_index,
            "style": hair_style if isinstance(hair_style, ShellStyle) else None,
        }
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
        hair_style=None if hair == HY3D else hair_style,
        bust=bust,
    )

    canvas = np.zeros((size, size, 3), np.uint8)
    covered = np.zeros((size, size), bool)
    inseam = next(d for d in json.loads((BODY_ASSETS / "measures.json").read_text())["measures"] if d["id"] == "inseam")
    pelvis_y = final[model.bones[model.bone_index["pelvis"]].head_vert, 1]
    underwear = Underwear(bottom_y=float(final[inseam["vert"], 1] - 0.11), top_y=float(pelvis_y + 0.09))
    log("painting body skin and underwear")
    gate = vertex_gate(model, UNDERWEAR_BONES)
    source, provenance = load_skin(fetch=fetch_skin)
    normal_canvas = np.full((size, size, 3), [128, 128, 255], np.uint8)
    cavity = parts.loaded["eyes"].delete_verts
    deleted = np.zeros(model.nr, bool)
    deleted[cavity[cavity < model.nr]] = True
    skin_faces = model.faces[~deleted[model.faces].any(1)]
    skin_report = paint_body(canvas, covered, final[: model.nr], skin_faces, model.uv, gate, tone, underwear, size,
                             source=source, normal_canvas=normal_canvas, neck_y=template.anchor_y)
    skin_report["detail"] = provenance
    padding_steps = {"body": border_audit(canvas, covered)}
    canvas, padding_steps["body_repair"] = repair_border_colours(canvas, covered)
    canvas = pad_texture(canvas, covered)
    normal_canvas = pad_texture(normal_canvas, covered)

    texture, observed = blend_unobserved(face.texture, face.covered, face.confidence, tone)
    texture = finish_texture(texture, face.covered)
    texture, match = match_mean(texture, face.covered, face.skin_mask, tone)
    texture = finish_texture(texture, face.covered)
    torso_island = int(template.islands[neck["verts"][0]])
    seam_w = seam_vertices(template, torso_island)
    moved = template.base + head_fit.displacement
    texture, seam_blur = seam_blend(texture, face.texel_points, face.texel_y, face.texel_x, moved[seam_w], tone)
    texture = finish_texture(texture, face.covered)

    accessory_bust = Path(glasses_bust) if glasses_bust else REPO / BUST_RELATIVE
    deglass_report = {"enabled": False, "reason": "opt out" if not deglass else "no glasses accessory bust"}
    if deglass and accessory_bust.is_file():
        from PIL import Image

        from .deglass_tex import _lab as _lab_of
        from .deglass_tex import (
            correct_lens_stage,
            projection_from_report,
            protected_report,
            regrain,
            remove_glasses_frames,
            remove_rim_traces,
            temple_streak,
        )
        from .glasses_hy3d import removal_report_from_bake

        log("removing photographed glasses frames from the current face bake")
        h, w = texture.shape[:2]
        guides = removal_report_from_bake(
            texture, head_positions, head_island_triangles(template, head),
            (model.uv[head.ids] * size - face.origin) / [w, h],
            np.array([v.mean(0) for v in flame_eyes.values()]),
            folder=preview_dir if previews else None,
        )
        before = texture.copy()
        projection = projection_from_report(face, guides)
        from .deglass_tex import FrameProjection

        has_geometry = isinstance(projection, FrameProjection)  # tests may inject an explicit UV mask instead
        lens_ctx, lens_masks = None, {}
        if has_geometry:
            # 1. The photographed lens AREA (tint / reflections inside the rims): a smooth Lab gain/offset field measured
            #    across each snapped rim outline, applied before the wire is inpainted. Eyes, eyelids, lashes and brows
            #    are protected (FLAME eye regions + brow detection).
            lens_groups = flame_vertex_groups(head_fit, flame, head)
            lens_tri = head_island_triangles(template, head)[face.texel_face]
            for key in ("eye", "ear"):
                flag = np.zeros(face.covered.shape, bool)
                flag[face.texel_y, face.texel_x] = (
                    texel_weights(lens_groups[key].astype(float), lens_tri, face.texel_bary) > 0.5
                )
                lens_masks[key] = flag
            texture, lens_report, lens_ctx = correct_lens_stage(
                texture, projection, reference=before, eye_region=lens_masks["eye"],
            )
            # eyes, eyelids and lashes stay exactly as photographed for the thin-line pass too
            projection.protected = lens_ctx["prot"]["eyes"] | (
                projection.protected if projection.protected is not None else False
            )
        lens_corrected = texture
        # 2. The painted frame lines (thin-line NS inpaint).
        texture, deglass_report, mask = remove_glasses_frames(
            texture, projection, method="ns", radius=5, return_report=True,
        )
        deglass_report.update(enabled=True, glasses_report=guides,
                              fraction_of_head_island=float(mask.sum() / face.covered.sum()))
        if has_geometry:
            # NS fills are smooth: give the filled texels the fine grain of the skin around them
            texture = regrain(texture, mask, _lab_of(lens_corrected))
            # 2b. Rim wire left over along the snapped outline (e.g. between the eye and the brow).
            texture, trace_report, _ = remove_rim_traces(texture, projection, lens_ctx)
            deglass_report["rim_traces"] = trace_report
            # 3. The temple-arm streak: tracked along the photographed arm, NS inpainted along a thin band.
            texture, arm_report, arm_band = temple_streak(
                texture, projection, lens_ctx["prot"], ear=lens_masks["ear"], reference=before,
            )
            deglass_report["lens_residue"] = {
                "texel_mm": lens_report["texel_mm"],
                "lens": lens_report,
                "temple": arm_report,
                "protected_regions": protected_report(before, texture, lens_ctx["prot"]),
            }
            lens_report = deglass_report["lens_residue"]
            lens_qa = {"lens_weight": lens_ctx["weight"], "temple_band": arm_band,
                       "protected": lens_ctx["prot"]["protected"]}
            log("lens residue: " + json.dumps({
                "eyes": [e.get("interior_vs_ring_before") and
                         {"before": e["interior_vs_ring_before"], "after": e["interior_vs_ring_after"]}
                         for e in lens_report["lens"]["eyes"]],
                "protected_max_abs_rgb_diff": {k: v["max_abs_rgb_diff"]
                                               for k, v in lens_report["protected_regions"].items()},
            }))
        audit = {"before": before, "lens_corrected": lens_corrected, "after": texture, "mask": mask,
                 "origin": np.asarray(face.origin)}
        if has_geometry:
            audit.update(lens_weight=lens_qa["lens_weight"], temple_band=lens_qa["temple_band"],
                         protected=lens_qa["protected"])
            np.savez_compressed(out.parent / "deglass_geometry.npz", texel_points=face.texel_points,
                                texel_y=face.texel_y, texel_x=face.texel_x, covered=face.covered,
                                skin_mask=face.skin_mask, eye_region=lens_masks["eye"], ear=lens_masks["ear"],
                                front=projection.front_curves, temples=projection.temple_curves,
                                eyes=projection.eyes, radii=projection.radii, protected=projection.protected)
        np.savez_compressed(out.parent / "deglass_audit.npz", **audit)
        if previews:
            preview_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(preview_dir / "deglass_uv_mask.png")
            Image.fromarray(before).save(preview_dir / "deglass_before.png")
            Image.fromarray(texture).save(preview_dir / "deglass_after.png")
            if has_geometry:
                Image.fromarray(np.maximum(lens_qa["lens_weight"] * 255, lens_qa["temple_band"] * 255).astype(np.uint8)).save(
                    preview_dir / "deglass_lens_field.png")
        texture = finish_texture(texture, face.covered)

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
    texture = finish_texture(texture, face.covered)

    # Behind the ear and on the side / back of the neck the photos only show the user's real hair and stubble (which
    # belongs to the hair shell): take the clean body skin of the same texels there, blended into the beard.
    log("removing photographed hair behind the ear and from the neck sides")
    ear_x0, ear_y0 = face.origin
    vertex_groups = flame_vertex_groups(head_fit, flame, head)
    restriction = restriction_weights(head_positions, head.faces, head.welded, vertex_groups)
    island_tri = head_island_triangles(template, head)[face.texel_face]
    restrict_w = texel_weights(restriction["weight"], island_tri, face.texel_bary)
    if not restrict_neck_hair:
        restrict_w = np.zeros_like(restrict_w)
    clean_skin = canvas[ear_y0 + ty, ear_x0 + tx]
    texture_before_restrict = texture
    texture = restrict_photo(texture, ty, tx, restrict_w, clean_skin)
    texture = finish_texture(texture, face.covered)
    face_texel = vertex_groups["face"][island_tri[np.arange(len(island_tri)), face.texel_bary.argmax(1)]]
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

    # Keep the rendered-body coverage: the head bake can still contain triangles
    # removed for the eye apertures. Those must become padding, not fill sources.
    photo_check = vertex_photo_check(
        face, head, template, model.uv[head.ids], pad_texture(paste(texture), covered), size
    )
    log(f"vertex/photo check: {photo_check}")

    # Dark scalp under the hair: the soft gaps between hair cards read as hair, not as skin.
    head_normals = welded_vertex_normals(head_positions, head.faces)
    tri = head_island_triangles(template, head)[face.texel_face]
    texel_normals = np.einsum("ij,ijk->ik", face.texel_bary, head_normals[tri])
    texel_normals /= np.maximum(np.linalg.norm(texel_normals, axis=1, keepdims=True), 1e-9)
    field = parts.hair_build.field if parts.hair_build is not None else None
    shell_distance = None
    if isinstance(field, ShellField):
        # the solid shell: distance along the head normals / radial rays, and the scalp takes the colour of the shell
        # around it (averaged over ~12 mm, so the tint carries no per-sample streaks)
        shell_distance = field.distance(points, texel_normals)
        cover = field.cover_from(shell_distance)
        scalp_lab = field.tint_lab(points)
        scalp_method = (
            "hy3d shell coverage (distance along the head normals), graded stubble tint beyond the edge, "
            "tinted with the smoothed colour of the shell around it"
        )
    elif field is not None:
        cover = field.cover(points)  # the hair density field itself: exact hairline, stubble shadow
        scalp_method = "procedural hair density field"
    else:
        hair_surface = sample_hair_surface(parts.bound["hair"], parts.loaded["hair"].faces)
        cover = hair_cover(points, texel_normals, hair_surface)
        scalp_method = "ray march to the hair surface"
    if not isinstance(field, ShellField):
        scalp_lab = to_lab(np.clip(parts.hair_srgb * SCALP_TINT, 1, 255).astype(np.float32).reshape(1, 3) / 255.0)[0]
    texture_under_hair = tint_scalp(texture, ty, tx, cover, scalp_lab)
    texture_under_hair = finish_texture(texture_under_hair, face.covered)
    temple_report = {"texels": 0}
    if isinstance(field, ShellField):
        texture_under_hair, temple_report = shell_scalp_tint(
            field, texture, face.covered, ty, tx, points, texel_normals, shell_distance, cover, scalp_lab
        )
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
        "method": scalp_method,
        "tint_lab": (scalp_lab[cover > 0.5].mean(0) if (cover > 0.5).any() else scalp_lab.mean(0)).tolist()
        if scalp_lab.ndim == 2
        else scalp_lab.tolist(),
        "covered_texel_fraction": float((cover > 0.5).mean()),
        "head_island_texels": int(len(cover)),
        "temple_gap": temple_report,
    }
    ear_top_y = float(head_positions[vertex_groups["ear"], 1].max()) if vertex_groups["ear"].any() else np.inf
    behind = (restrict_w > 0.5) & (points[:, 1] < ear_top_y) & (cover < 0.5)
    beard = face_texel & (restrict_w < 1e-3) & (points[:, 1] < float(np.mean([v[:, 1].mean() for v in flame_eyes.values()])) - 0.03)
    blend_zone = (restrict_w > 1e-3) & (restrict_w < 0.999)
    neckhair_report = {
        "method": "photo hair behind the ear and on the neck sides replaced by the clean body skin, 13 mm geodesic blend",
        "blend_mm": BLEND_M * 1000,
        "ear_z_mid_m": {"left": restriction["z_ear_mid"][1], "right": restriction["z_ear_mid"][-1]},
        "core_vertices": int(restriction["core"].sum()),
        "region_texels_weight_gt_0_5": int((restrict_w > 0.5).sum()),
        "behind_ear_neck_visible": {
            "before": region_stats(texture_before_restrict, ty, tx, behind, tone),
            "after": region_stats(texture, ty, tx, behind, tone),
        },
        "beard_untouched": {
            "before": region_stats(texture_before_restrict, ty, tx, beard, tone),
            "after": region_stats(texture, ty, tx, beard, tone),
        },
        "blend_zone": {
            "before": region_stats(texture_before_restrict, ty, tx, blend_zone, tone),
            "after": region_stats(texture, ty, tx, blend_zone, tone),
        },
    }
    log(f"neck hair: {neckhair_report['behind_ear_neck_visible']}")
    unpadded = paste(texture_under_hair)
    padding_steps["final_before"] = border_audit(unpadded, covered)
    repaired, padding_steps["final_repair"] = repair_border_colours(unpadded, covered)
    padded = pad_texture(repaired, covered)
    padding_steps["final_after"] = border_audit(padded, covered)
    body_atlas = np.dstack((padded, np.full((size, size), 255, np.uint8)))
    atlas = np.vstack((body_atlas, parts.strip))
    normal_atlas = np.vstack((normal_canvas, np.full((*parts.strip.shape[:2], 3), [128, 128, 255], np.uint8)))
    atlas_height = atlas.shape[0]
    body_uv = model.uv * np.array([1.0, size / atlas_height])
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
        "hairProcedural": parts.hair_build is not None and hair != HY3D,
        **(
            {
                "hairShell": {
                    "format": FORMAT_SHELL,
                    "source": "Hunyuan3D bust of the user (user-data/twin/hy3d/hy3d.glb), cut out and fitted to this head",
                    "colourSize": [int(hair_mesh.atlas.shape[1]), int(hair_mesh.atlas.shape[0])],
                    "normalSize": [int(hair_mesh.normal_atlas.shape[1]), int(hair_mesh.normal_atlas.shape[0])],
                    "triangles": int(len(hair_mesh.faces)),
                }
            }
            if hair == HY3D and hair_mesh is not None
            else {}
        ),
        **(
            {
                "hairAtlas": {
                    "format": "rcov-groot-bvar/1",
                    "size": [int(hair_mesh.atlas.shape[1]), int(hair_mesh.atlas.shape[0])],
                    "channels": "R coverage, G root-to-tip (0 root), B variation, A min(1, 2.5 R)",
                    "shader": "creategamecharacters/threejs-hair-shader (MIT), compact atlas",
                }
            }
            if hair_mesh is not None and hair != HY3D
            else {}
        ),
        "license": "MakeHuman CC0 + private non-commercial FLAME/Pixel3DMM fit; never redistribute",
    }
    extras = {"dtHybrid": marker, "dtBodyfix": solution, "dtScanHandsRemoved": False, "dtHasMakeHumanHands": True}
    extras["dtDeglass"] = {k: v for k, v in deglass_report.items() if k not in ("glasses_report", "lens_residue")}
    log("writing the hybrid GLB")
    glb = write_glb(out, mesh, atlas, extras, hair=hair_mesh, normal_atlas=normal_atlas)

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
            "border_padding": padding_steps,
            "seam_blend": seam_blur,
            "deglass": deglass_report,
            "neck": neck_report,
            "scalp_tint": scalp_report,
            "neck_hair": neckhair_report,
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
                        "kind": "bust-shell" if hair == HY3D else "procedural-cards",
                        "fallbackId": HAIR_FALLBACK_ID,
                        "style": parts.hair_build.style.to_dict(),
                        "format": hair_mesh.format,
                        "colours": hair_mesh.colours,
                        "note": (
                            "a textured shell cut out of the own Hunyuan3D bust of the user and fitted to this head "
                            "(hybrid/hairhy3d.py), a separate dtHair node of format shell/1; private to the user"
                            if hair == HY3D
                            else "cards grown on the deformed head (hybrid/hairgen.py), a separate dtHair node with a "
                            "strand data atlas for the hair shader"
                        )
                        + "; the part library of the app has no such part, so it mounts fallbackId until the hair is "
                        "loaded from the twin GLB",
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
        from .previews import write_ear_neck_closeups

        ear_points = {
            "left": head_positions[vertex_groups["ear"] & (head_positions[:, 0] > 0)],
            "right": head_positions[vertex_groups["ear"] & (head_positions[:, 0] < 0)],
        }
        report["previews"].update(write_ear_neck_closeups(preview_dir, mesh, atlas, ear_points, hair=hair_mesh))
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
    shell = hair.get("shell")
    procedural = hair.get("procedural") or shell
    if procedural:
        penetration = procedural["penetration"]
        if penetration["vertices_inside_head"] or penetration["vertices_below_clearance"]:
            problems.append("hair vertices closer to the head than the clearance")
        if penetration.get("samples_inside_head"):
            problems.append("hair triangles cut into the head")
        if not 8000 <= procedural["triangles"] <= 60000:
            problems.append("hair triangle count outside 8k..60k")
        coverage = procedural.get("coverage")
        if coverage and coverage["min_hidden_fraction"] < 0.8:
            problems.append("hair leaves more than 20 percent of the scalp visible from a main view")
        weights = hair.get("skin_weights")
        if weights and weights["fraction_head_chain_ge_0_95"] < 0.8:
            problems.append("hair vertices are not weighted to the head/neck bones")
    if shell:
        if not 15000 <= shell["triangles"] <= 30000:
            problems.append("hy3d hair: triangle count outside 15k..30k")
        clearance = shell["fit"]["clearance"]
        if clearance["min_sample_mm"] < 1.5 or clearance["samples_inside_head"]:
            problems.append("hy3d hair: triangle interiors do not keep 1.5 mm clearance")
        front = shell["hairline"]["front_mm_above_eye_shell"]
        if front is None or not 40.0 <= front <= 100.0:
            problems.append("hy3d hair: the front hairline is not 4 to 10 cm above the eye line")
        edge = shell["edge_gap"]
        if edge.get("vertices") and edge["median_mm"] > 4.0:
            problems.append("hy3d hair: the hairline edge floats more than 4 mm above the scalp")
        visible = shell["visible_clearance"]["min_mm"]
        if visible is not None and visible < 1.5:
            problems.append("hy3d hair: the visible shell is closer than 1.5 mm to the head")
    return problems
