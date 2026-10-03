"""Procedural hair cards on the generic CC0 MakeHuman head (no person, no photo, no FLAME data)."""

import numpy as np
import pytest
from hybridbody import PARTS_ASSETS
from hybridbody.hair import DEFAULT_HAIR_HEX, build_procedural_hair, hair_colours
from hybridbody.haircheck import penetration_report, scalp_coverage, skin_weight_report
from hybridbody.hairdemo import build_generic_twin, generic_head, write_generic_previews
from hybridbody.hairgen import (
    LAYERS,
    HairField,
    HairStyle,
    measure_head,
    poisson_select,
    ribbon_faces,
)
from hybridbody.hairtex import FORMAT, GAIN, hair_atlas, make_layout, prefiltered_alpha, strip_statistics
from hybridbody.partstex import DEFAULT_HAIR_SRGB, plausible_hair, srgb_hex
from hybridbody.pipeline import surface_clearance
from hybridbody.template import load_part
from synth import sphere_template

TARGET = 12000


@pytest.fixture(scope="module")
def generic(model):
    _, combined, _, head = generic_head(1.0)
    body = combined[: model.nr]
    eye_y = float(load_part(PARTS_ASSETS, "eyes-default").bind(combined)[:, 1].mean())
    style = HairStyle.with_overrides({"triangle_target": TARGET})
    build = build_procedural_hair(
        body[head.ids], head.faces, eye_y, DEFAULT_HAIR_SRGB, style=style, atlas_size=(512, 256), coverage=False
    )
    return {"build": build, "body": body, "combined": combined, "eye_y": eye_y, "style": style}


# ----------------------------------------------------------------------------------------------- style, colour
def test_style_overrides_are_validated_and_in_millimetres():
    style = HairStyle.with_overrides({"hairline_front": 70, "seed": 3, "triangle_target": 20000})
    assert style.hairline_front == 70.0 and style.seed == 3 and style.triangle_target == 20000
    assert HairStyle().hairline_front == 68.0 and HairStyle().length_side == 8.0  # defaults of the requested cut
    with pytest.raises(ValueError, match="Unknown hair parameter"):
        HairStyle.with_overrides({"hairline": 1})
    with pytest.raises(ValueError, match="non-negative"):
        HairStyle.with_overrides({"length_side": -1})
    assert set(HairStyle().to_dict()) >= {"hairline_front", "length_front", "clearance", "seed"}


def test_plausible_hair_makes_a_grey_measurement_dark_brown_and_honours_overrides():
    grey = plausible_hair([0x49, 0x41, 0x3D])  # the washed-out photo colour
    r, g, b = grey["srgb"]
    assert r > g > b and r - b > 10 and r < 110 and grey["method"].startswith("near-grey")
    assert plausible_hair(None)["method"] == "default" and (plausible_hair(None)["srgb"] == DEFAULT_HAIR_SRGB).all()
    assert srgb_hex(plausible_hair(None, "#3b2a1e")["srgb"]) == "#3b2a1e"
    light = plausible_hair([200, 160, 100])  # chromatic and too light: lightness clamped, hue kept
    assert light["srgb"][0] > light["srgb"][2] and light["srgb"].max() < 140 and light["method"] == "lightness clamped"
    with pytest.raises(ValueError):
        plausible_hair(None, "#12")


def test_default_hair_colour_is_the_requested_dark_brown_and_the_colours_are_graded():
    assert DEFAULT_HAIR_HEX == "#2a1e18" and srgb_hex(DEFAULT_HAIR_SRGB) == "#2a1e18"
    assert plausible_hair(None)["method"] == "default"
    colours = hair_colours(DEFAULT_HAIR_SRGB)
    assert colours["colorHex"] == "#2a1e18"
    root, base, tip = (
        np.array([int(colours[k][i : i + 2], 16) for i in (1, 3, 5)]) for k in ("rootHex", "colorHex", "tipHex")
    )
    assert (root < base).all() and (base < tip).all() and root[0] > root[2]  # darker root, lighter tip, still brown


# --------------------------------------------------------------------------------------------------- texture
def test_strand_atlas_has_the_contract_channels_dense_roots_and_tapered_tips():
    atlas, layout = hair_atlas(512, 256, seed=1)
    assert atlas.shape == (256, 512, 4) and atlas.dtype == np.uint8
    r, g, b, a = (atlas[..., k].astype(float) / 255 for k in range(4))
    stats = strip_statistics(atlas, layout)
    for kind in ("long", "mid", "short"):
        mean = stats[kind]["mean_r"]
        assert mean["root"] > 0.6 and mean["root"] > mean["middle"] > 0.25  # dense solid root, strands in the body
        assert mean["tip"] < 0.5 * mean["middle"]  # strands taper out towards the tip
    # R coverage is soft (many in-between values), A is the plain-viewer fallback min(1, 2.5 R)
    assert 0.2 < ((r > 0.05) & (r < 0.95)).mean() / (r > 0.05).mean() and 0.15 < r.mean() < 0.6
    np.testing.assert_allclose(a, np.minimum(1.0, GAIN * r), atol=1.5 / 255)
    # G is the root to tip position: 0 on the slot's first (root) row, 1 on its last, rising monotonically in between
    for slot in (*layout.long, *layout.mid, *layout.short):
        column = g[slot.y0 : slot.y1, slot.x0 + slot.width // 2]
        assert column[0] < 0.02 and column[-1] > 0.98 and (np.diff(column) >= -1.5 / 255).all()
        u0, v_root, u1, v_tip = slot.uv_box(layout.width, layout.height)
        assert 0 <= u0 < u1 <= 1 and 0 <= v_root < v_tip <= 1  # the root is the top (v small), glTF v down
    # B is the per-strand variation, centred on 0.5 where there are strands
    covered = r > 0.5
    assert 0.4 < b[covered].mean() < 0.6 and 0.05 < b[covered].std() < 0.3
    again, _ = hair_atlas(512, 256, seed=1)
    assert (again == atlas).all()  # deterministic
    assert not (hair_atlas(512, 256, seed=2)[0] == atlas).all()
    layout_big = make_layout(2048, 1024)
    assert (len(layout_big.long), len(layout_big.mid), len(layout_big.short)) == (20, 16, 32)
    assert layout_big.long[0].height == 1024 and layout_big.mid[0].height == 512 and layout_big.short[0].height == 128
    assert FORMAT == "rcov-groot-bvar/1"


def test_prefiltered_alpha_is_the_gained_mip_filtered_coverage():
    atlas, _ = hair_atlas(512, 256, seed=1)
    alpha = prefiltered_alpha(atlas, level=3)
    assert alpha.shape == (32, 64) and alpha.min() >= 0 and alpha.max() <= 1.0
    assert alpha.mean() > atlas[..., 0].mean() / 255  # the gain of 2.5 makes the filtered strips nearly solid


def test_ribbon_faces_have_two_triangles_per_segment():
    faces = ribbon_faces(4)
    assert faces.shape == (6, 3) and faces.max() == 7 and faces.min() == 0


def test_poisson_select_keeps_the_minimum_distance_and_fills_the_area():
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(1)
    points = np.column_stack((rng.random(6000), rng.random(6000), np.zeros(6000)))
    radius = 0.04
    chosen = poisson_select(points, radius, rng)
    picked = points[chosen]
    nearest = cKDTree(picked).query(picked, k=2)[0][:, 1]
    assert nearest.min() >= radius - 1e-12 and len(chosen) > 0.4 / radius**2 * 0.6
    far = cKDTree(picked).query(points)[0]
    assert far.max() <= radius + 1e-9  # maximal: nothing left that could be added
    assert len(poisson_select(np.zeros((0, 3)), 0.1, rng)) == 0


# --------------------------------------------------------------------------------------- head frame, field
def test_head_frame_finds_the_ears_and_the_nape_crease(generic):
    build = generic["build"]
    frame, eye = build.frame, generic["eye_y"]
    for side in (1, -1):
        ear = frame.ears[side]
        assert ear.detected and ear.vertices > 60
        assert 0.0 < ear.y_top - eye < 0.025 and -0.065 < ear.y_bottom - eye < -0.03
        assert 70 < ear.a_front < 92 and ear.a_front + 8 < ear.a_back < 115
    assert -0.085 <= frame.crease_y - eye <= -0.045
    assert abs(frame.x0) < 0.01 and frame.top_y - eye > 0.09
    assert len(frame.ear_points) > 100


def test_field_places_the_hairline_leaves_the_ears_bare_and_grades_the_lengths(generic):
    build, eye = generic["build"], generic["eye_y"]
    field, surface = build.field, build.surface
    v = surface.vertices
    centre_line = np.abs(v[:, 0] - build.frame.x0) < 0.004
    forehead = v[centre_line & (v[:, 2] > build.frame.zc)]
    above = forehead[(forehead[:, 1] > eye + 0.068 + 0.025) & (forehead[:, 1] < eye + 0.068 + 0.04)]
    below = forehead[(forehead[:, 1] < eye + 0.068 - 0.025) & (forehead[:, 1] > eye + 0.02)]
    assert len(above) and len(below)
    assert field.weight(above).min() > 0.9 and field.weight(below).max() < 0.05
    assert abs(float(field.hairline[1](0.0)) - 0.068) < 1e-9
    assert field.weight(build.frame.ear_points).max() < 1e-6  # the ears carry no hair
    assert field.cover(below).max() < 0.4 and field.cover(above).min() > 0.9
    # lengths: long at the front of the top, medium at the crown, short on the sides and back
    top = build.cards.roots[(field.top_weight(build.cards.roots) > 0.9) & (field.ap_position(build.cards.roots) > 0.8)]
    assert len(top) and np.median(field.lengths(top)) > 0.036
    sides = v[(np.abs(field.azimuth(v)) > 60) & (np.abs(field.azimuth(v)) < 120) & (field.weight(v) > 0.99)]
    sides = sides[field.top_weight(sides) < 0.05]
    assert len(sides) and 0.004 < np.median(field.lengths(sides)) < 0.010  # about 8 mm, less next to the ears
    # the flow field gives unit tangents
    normals = surface.signed_distance(build.cards.roots)[2]
    flow = field.flow(build.cards.roots, normals)
    assert np.allclose(np.linalg.norm(flow, axis=1), 1.0, atol=1e-6)
    assert np.abs(np.einsum("ij,ij->i", flow, normals)).max() < 1e-6
    # on the top the hair is swept back (-z); at the back of the head it is combed down
    topmost = flow[field.top_weight(build.cards.roots) > 0.9]
    assert (topmost[:, 2] < 0).mean() > 0.9
    back = flow[np.abs(field.azimuth(build.cards.roots)) > 150]
    assert back[:, 1].mean() < -0.6


# ------------------------------------------------------------------------------------------------- the cards
def test_cards_are_valid_triangles_with_uvs_inside_the_tile_and_a_triangle_budget(generic):
    build = generic["build"]
    cards = build.cards
    assert np.isfinite(cards.positions).all() and np.isfinite(cards.uv).all()
    assert cards.faces.min() == 0 and cards.faces.max() == len(cards.positions) - 1
    assert cards.uv.min() >= 0 and cards.uv.max() <= 1
    assert abs(len(cards.faces) / TARGET - 1) < 0.2  # the spacing is tuned to the triangle target
    tri = cards.positions[cards.faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    assert area.min() > 1e-9  # no degenerate triangles
    assert set(np.unique(cards.layer_of_face)) == set(range(len(LAYERS)))
    hair = build.mesh
    assert hair.node == "dtHair" and len(hair.faces) == len(cards.faces) and hair.card_count == len(cards.roots)
    assert hair.normals.shape == hair.positions.shape and np.allclose(np.linalg.norm(hair.normals, axis=1), 1.0)
    assert hair.atlas.shape == (256, 512, 4) and set(hair.colours) == {"colorHex", "rootHex", "tipHex"}
    # root to tip: along every card the atlas G channel (and so the v of the UV) rises from the root to the tip
    atlas_g = hair.atlas[..., 1] / 255.0
    ix = np.clip((hair.uv[:, 0] * 512).astype(int), 0, 511)
    iy = np.clip((hair.uv[:, 1] * 256).astype(int), 0, 255)
    g_vertex = atlas_g[iy, ix]
    root_v = np.full(len(cards.roots), np.inf)
    tip_v = np.full(len(cards.roots), -np.inf)
    np.minimum.at(root_v, cards.card_of_face, hair.uv[cards.faces][:, :, 1].min(1))
    np.maximum.at(tip_v, cards.card_of_face, hair.uv[cards.faces][:, :, 1].max(1))
    assert (tip_v > root_v).all()
    assert g_vertex.min() < 0.05 and g_vertex.max() > 0.9
    # the cards face away from the scalp: mean of (card normal . surface normal at the root) is clearly positive
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    root_normal = cards.root_normals[cards.card_of_face]
    assert np.einsum("ij,ij->i", normal, root_normal).mean() > 0.4


def test_no_card_vertex_penetrates_the_head_and_all_keep_the_two_millimetre_offset(generic):
    build = generic["build"]
    report = penetration_report(build.surface, build.cards.positions, build.cards.faces, 0.002)
    assert report["vertices_inside_head"] == 0 and report["vertices_below_clearance"] == 0
    assert report["min_mm"] >= 1.99
    assert report["samples_inside_head"] == 0
    assert report["samples_min_mm"] > 1.0  # the interior of every triangle stays off the skin (to 1 mm)
    assert build.report["penetration"]["vertices_below_clearance"] == 0
    # independent check with the pipeline's own surface_clearance (unwelded render head, interpolated normals)
    head = generic_head(1.0)[3]
    independent = surface_clearance(build.cards.positions, generic["body"][head.ids], head.faces)
    assert independent["min_mm"] > 1.5 and independent["deeper_than_4mm"] == 0  # plane distance along the normal


def test_hair_stays_above_the_skin_even_with_a_larger_clearance_setting(generic):
    style = HairStyle.with_overrides({"triangle_target": 6000, "clearance": 3.5})
    build = build_procedural_hair(
        generic["body"][generic_head(1.0)[3].ids],
        generic_head(1.0)[3].faces,
        generic["eye_y"],
        DEFAULT_HAIR_SRGB,
        style=style,
        atlas_size=(256, 128),
        coverage=False,
    )
    assert build.report["penetration"]["min_mm"] >= 3.49
    assert build.report["penetration"]["vertices_below_clearance"] == 0


def test_hairline_and_lengths_in_the_report_match_the_requested_cut(generic):
    report = generic["build"].report
    assert 60 < report["hairline"]["front_root_p02_mm_above_eye"] < 72
    lengths = report["lengths_mm"]
    assert lengths["top_front"]["median"] > 25 and lengths["top_middle"]["median"] > 35
    assert 18 < lengths["crown"]["median"] < 40
    assert 5 < lengths["sides"]["median"] < 10 and 5 < lengths["back"]["median"] < 10  # the body layer of the 8 mm cut
    assert lengths["hairline_edge"]["median"] < 8
    assert report["frame"]["ears"]["left"]["detected"] and report["frame"]["ears"]["right"]["detected"]
    assert report["triangles"] == len(generic["build"].cards.faces)


def test_hairline_height_follows_the_style_parameter(generic):
    style = HairStyle.with_overrides({"triangle_target": 6000, "hairline_front": 62})
    head = generic_head(1.0)[3]
    build = build_procedural_hair(
        generic["body"][head.ids], head.faces, generic["eye_y"], DEFAULT_HAIR_SRGB, style=style,
        atlas_size=(256, 128), coverage=False,
    )  # fmt: skip
    assert 54 < build.report["hairline"]["front_root_p02_mm_above_eye"] < 66


def test_the_same_seed_gives_the_same_hair(generic):
    head = generic_head(1.0)[3]
    kwargs = {"atlas_size": (256, 128), "coverage": False}
    style = HairStyle.with_overrides({"triangle_target": 5000})
    a = build_procedural_hair(
        generic["body"][head.ids], head.faces, generic["eye_y"], DEFAULT_HAIR_SRGB, style=style, **kwargs
    )
    b = build_procedural_hair(
        generic["body"][head.ids], head.faces, generic["eye_y"], DEFAULT_HAIR_SRGB, style=style, **kwargs
    )
    np.testing.assert_array_equal(a.cards.positions, b.cards.positions)
    other = HairStyle.with_overrides({"triangle_target": 5000, "seed": 9})
    c = build_procedural_hair(
        generic["body"][head.ids], head.faces, generic["eye_y"], DEFAULT_HAIR_SRGB, style=other, **kwargs
    )
    assert len(c.cards.positions) != len(a.cards.positions) or not np.allclose(c.cards.positions, a.cards.positions)


def test_scalp_is_hidden_from_every_main_view_and_the_hair_follows_the_head_bone(generic, model):
    build = generic["build"]
    coverage = scalp_coverage(
        build.surface,
        build.field,
        build.cards.positions,
        build.cards.faces,
        build.cards.uv,
        prefiltered_alpha(build.mesh.atlas),
        pixel_mm=1.0,
        samples=30000,
    )
    for view in ("front", "left", "right", "back", "top"):
        assert coverage[view]["visible_samples"] > 200 and coverage[view]["hidden_fraction"] > 0.7, (
            view
        )  # 12k triangles
    assert coverage["min_hidden_fraction"] > 0.7
    skin = skin_weight_report(model, generic["body"], build.cards.positions, build.cards.faces)
    assert skin["head_chain_weight_min"] > 0.3  # worst vertex (the nape) still mostly on head/neck bones
    assert skin["fraction_head_chain_ge_0_95"] > 0.95 and skin["head_bone_weight_mean"] > 0.7
    assert next(iter(skin["dominant_bones"])) == "head"


def test_a_sphere_head_without_ears_still_gets_hair_with_default_ear_boxes():
    positions, faces, _, _ = sphere_template(radius=0.095, subdivisions=4, centre_y=1.65)
    style = HairStyle.with_overrides({"triangle_target": 3000})
    build = build_procedural_hair(
        positions, faces, 1.65 - 0.02, DEFAULT_HAIR_SRGB, style=style, atlas_size=(256, 128), coverage=False
    )
    assert build.frame.notes and not build.frame.ears[1].detected
    assert build.report["penetration"]["vertices_below_clearance"] == 0 and build.report["triangles"] > 500
    field = build.field
    assert field.weight(positions).max() > 0.9 and field.weight(positions).min() < 0.01


def test_measure_head_accepts_an_explicit_ear_mask(generic):
    surface = generic["build"].surface
    default = measure_head(surface, generic["eye_y"], generic["style"])
    ears = np.zeros(len(surface.vertices), bool)
    ears[
        np.flatnonzero(np.isin(np.arange(len(surface.vertices)), np.argsort(np.abs(surface.vertices[:, 0]))[-300:]))
    ] = True
    given = measure_head(surface, generic["eye_y"], generic["style"], ears=ears)
    assert default.ears[1].detected and given.ear_points.shape[0] == 300
    field = HairField(given, generic["style"])
    assert field.weight(given.ear_points).max() < 1e-6


def test_generic_demo_writes_head_views_and_numbers(tmp_path):
    twin = build_generic_twin(size=512, style=HairStyle.with_overrides({"triangle_target": 4000}), coverage=False)
    assert (
        twin["atlas"].shape == (640, 512, 4)
        and "hair" not in twin["mesh"].parts  # the hair is a separate node, not part of the body mesh
        and len(twin["build"].mesh.faces) > 500
    )
    paths = write_generic_previews(tmp_path, twin, size=160, variants=(False,))
    assert set(paths) >= {"generic_head_front", "generic_head_side", "generic_head_back", "generic_head_top"}
    assert all((tmp_path / f"{name}.png").is_file() for name in paths)
