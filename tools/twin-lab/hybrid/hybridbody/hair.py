"""Procedural hair for the hybrid twin: the driver that ties head frame, hair field, cards, atlas and checks together.

The result is a separate hair mesh (the ``dtHair`` node of the twin GLB) with its own strand data atlas; the body mesh and
its texture do not contain hair cards. The hair field also darkens the scalp texture under the hair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .haircheck import penetration_report, scalp_coverage
from .hairgen import (
    LAYERS,
    MM,
    Cards,
    HairField,
    HairStyle,
    HeadFrame,
    HeadSurface,
    grow_cards,
    measure_head,
    plan_cards,
)
from .hairtex import FORMAT, GAIN, StripLayout, hair_atlas, prefiltered_alpha, strip_statistics

PROCEDURAL = "procedural"
DEFAULT_HAIR_HEX = "#2a1e18"  # dark brown, nearly black-brown
ROOT_FACTOR = 0.74  # the root colour relative to the base colour (sRGB, per channel, rounded down)
TIP_FACTOR = np.array([1.38, 1.40, 1.34])  # the tip colour: lighter and a little warmer
NODE_NAME = "dtHair"


@dataclass
class HairMesh:
    """What the GLB writer needs of the hair: geometry, atlas and the colour data for the material extras."""

    positions: np.ndarray  # (n, 3) metres, body frame
    normals: np.ndarray  # (n, 3)
    uv: np.ndarray  # (n, 2) atlas UV
    faces: np.ndarray  # (m, 3)
    atlas: np.ndarray  # (h, w, 4) uint8: R coverage, G root to tip, B variation, A = min(1, 2.5 R)
    colours: dict  # colorHex, rootHex, tipHex
    card_count: int
    node: str = NODE_NAME


def hex_of(rgb) -> str:
    c = np.clip(np.rint(np.asarray(rgb, float)), 0, 255).astype(int)
    return "#" + "".join(f"{int(v):02x}" for v in c)


def hair_colours(base_srgb) -> dict:
    """``colorHex`` (the shader base colour), ``rootHex`` (darker) and ``tipHex`` (lighter), all sRGB hex."""
    base = np.asarray(base_srgb, float)
    return {
        "colorHex": hex_of(base),
        "rootHex": hex_of(np.floor(base * ROOT_FACTOR + 1e-6)),
        "tipHex": hex_of(np.minimum(base * TIP_FACTOR, 255.0)),
    }


@dataclass
class HairBuild:
    mesh: HairMesh
    layout: StripLayout
    field: HairField
    surface: HeadSurface
    frame: HeadFrame
    cards: Cards
    style: HairStyle
    report: dict

    @property
    def tile(self) -> np.ndarray:  # the strand data atlas (kept under its old name for the callers that read alpha)
        return self.mesh.atlas


def tune_plan(surface, field, style, layout, tolerance: float = 0.12, attempts: int = 5):
    """Plan the cards and rescale their spacing until the triangle count is within ``tolerance`` of the target."""
    scale = 1.0
    plan = plan_cards(surface, field, style, layout, scale)
    if plan.cards == 0:
        raise ValueError("The hairline leaves no scalp for hair cards; check --hairline-mm and the eye line")
    for _ in range(attempts):
        ratio = plan.triangles / max(style.triangle_target, 1)
        if abs(ratio - 1.0) <= tolerance:
            break
        scale *= float(np.clip(np.sqrt(ratio), 0.5, 2.0))
        plan = plan_cards(surface, field, style, layout, scale)
    return plan


def region_lengths(field: HairField, cards: Cards, layer_index: int = 1) -> dict:
    """Median card length (mm) per region of the head, from the roots of one layer."""
    sel = cards.layer_of_card == layer_index
    roots, length = cards.roots[sel], cards.lengths[sel] * 1000
    tau = field.top_weight(roots)
    u = field.ap_position(roots)
    az = np.abs(field.azimuth(roots))
    above = field.height_above_hairline(roots)
    regions = {
        "top_front": (tau > 0.8) & (u > 0.8),
        "top_middle": (tau > 0.8) & (u > 0.4) & (u < 0.7),
        "crown": (tau > 0.8) & (u < 0.25),
        "sides": (tau < 0.1) & (az > 60) & (az < 120) & (above > 0.03),
        "back": (tau < 0.1) & (az > 140) & (above > 0.03),
        "hairline_edge": (tau < 0.1) & (above < 0.004),
    }
    return {
        name: (
            {
                "median": float(np.median(length[m])),
                "p05": float(np.percentile(length[m], 5)),
                "p95": float(np.percentile(length[m], 95)),
                "cards": int(m.sum()),
            }
            if m.any()
            else {"cards": 0}
        )
        for name, m in regions.items()
    }


def build_procedural_hair(
    head_positions: np.ndarray,
    head_faces: np.ndarray,
    eye_y: float,
    colour_srgb,
    *,
    style: HairStyle | None = None,
    atlas_size: tuple[int, int] = (2048, 1024),
    ears: np.ndarray | None = None,
    texture_seed: int = 3,
    coverage: bool = True,
    coverage_pixel_mm: float = 0.5,
) -> HairBuild:
    """Hair cards for a head surface (render vertices and faces of the head region, metres, +Y up, +Z front).

    ``eye_y`` is the height of the eye centre line (the hairline is measured from it); ``colour_srgb`` the base
    strand colour (0..255, only travels in the material extras: the atlas holds no colour). The result carries the hair
    mesh with its strand data atlas, the hair field (also used to darken the scalp texture) and a report of numbers.
    """
    style = style or HairStyle()
    surface = HeadSurface(head_positions, head_faces)
    frame = measure_head(surface, eye_y, style, ears)
    field = HairField(frame, style)
    atlas, layout = hair_atlas(atlas_size[0], atlas_size[1], texture_seed)
    plan = tune_plan(surface, field, style, layout)
    cards = grow_cards(surface, field, style, plan)
    colours = hair_colours(colour_srgb)
    mesh = HairMesh(cards.positions, cards.normals, cards.uv, cards.faces, atlas, colours, int(len(cards.roots)))
    clearance = style.clearance * MM
    penetration = penetration_report(surface, cards.positions, cards.faces, clearance)
    n_cards = len(cards.roots)
    layers = {}
    for index, layer in enumerate(LAYERS):
        sel = cards.layer_of_card == index
        layers[layer.name] = {"cards": int(sel.sum()), "triangles": int((cards.layer_of_face == index).sum())}
    az_roots = np.abs(field.azimuth(cards.roots))
    front_roots = cards.roots[az_roots < 8.0]
    ears_report = {
        ("left" if side > 0 else "right"): {
            "detected": box.detected,
            "vertices": box.vertices,
            "top_mm_above_eye": (box.y_top - eye_y) * 1000,
            "bottom_mm_above_eye": (box.y_bottom - eye_y) * 1000,
            "azimuth_front_deg": box.a_front,
            "azimuth_back_deg": box.a_back,
        }
        for side, box in frame.ears.items()
    }
    tri = cards.positions[cards.faces]
    card_area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    report = {
        "kind": "procedural-cards",
        "format": FORMAT,
        "style": style.to_dict(),
        "cards": int(n_cards),
        "vertices": int(len(cards.positions)),
        "triangles": int(len(cards.faces)),
        "triangle_target": style.triangle_target,
        "spacing_scale": plan.scale,
        "layers": layers,
        "card_area_cm2": float(card_area.sum() * 1e4),
        "colours": colours,
        "frame": {
            "eye_y_m": eye_y,
            "skull_centre_xz_m": [frame.x0, frame.zc],
            "head_top_mm_above_eye": (frame.top_y - eye_y) * 1000,
            "neck_crease_mm_below_eye": (eye_y - frame.crease_y) * 1000,
            "nape_hairline_mm_below_eye": (eye_y - frame.crease_y - style.nape_above_crease * MM) * 1000,
            "ears": ears_report,
            "notes": frame.notes,
        },
        "hairline": {
            "front_mm_above_eye_analytic": float(field.hairline[1](0.0)) * 1000,
            "front_root_p02_mm_above_eye": float(np.percentile((front_roots[:, 1] - eye_y) * 1000, 2))
            if len(front_roots)
            else None,
            "scalp_hair_texels_note": "see texture.scalp_tint for the forehead measurement on the baked texture",
        },
        "lengths_mm": region_lengths(field, cards),
        "penetration": penetration,
        "strip": strip_statistics(atlas, layout),
        "atlas": {
            "format": FORMAT,
            "size": [int(atlas.shape[1]), int(atlas.shape[0])],
            "alpha": f"min(1, {GAIN} * R) for plain glTF viewers",
            "slots": {k: len(layout.slots(k)) for k in ("long", "mid", "short")},
        },
    }
    if coverage:
        report["coverage"] = scalp_coverage(
            surface, field, cards.positions, cards.faces, cards.uv, prefiltered_alpha(atlas), pixel_mm=coverage_pixel_mm
        )
    return HairBuild(mesh, layout, field, surface, frame, cards, style, report)
