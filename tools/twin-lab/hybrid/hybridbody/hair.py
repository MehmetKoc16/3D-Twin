"""Procedural hair for the hybrid twin: the driver that ties head frame, hair field, cards, texture and checks together."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .haircheck import penetration_report, scalp_coverage
from .hairgen import (
    MM,
    CardPart,
    Cards,
    HairField,
    HairStyle,
    HeadFrame,
    HeadSurface,
    grow_cards,
    measure_head,
    plan_cards,
)
from .hairtex import StripLayout, hair_tile, strip_statistics

PROCEDURAL = "procedural"


@dataclass
class HairBuild:
    part: CardPart
    positions: np.ndarray
    tile: np.ndarray  # (h, w, 4) uint8 RGBA strip tile
    layout: StripLayout
    field: HairField
    surface: HeadSurface
    frame: HeadFrame
    cards: Cards
    style: HairStyle
    report: dict


def tune_plan(surface, field, style, layout, tolerance: float = 0.12, attempts: int = 4):
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
    tile_size: tuple[int, int] = (2048, 1024),
    ears: np.ndarray | None = None,
    texture_seed: int = 3,
    coverage: bool = True,
    coverage_pixel_mm: float = 0.5,
) -> HairBuild:
    """Hair cards for a head surface (render vertices and faces of the head region, metres, +Y up, +Z front).

    ``eye_y`` is the height of the eye centre line (the hairline is measured from it); ``colour_srgb`` the base
    strand colour (0..255). The result carries the part for ``assemble`` (positions already in the body frame), the
    RGBA strip tile, the hair field (also used to darken the scalp texture) and a report of numbers.
    """
    style = style or HairStyle()
    surface = HeadSurface(head_positions, head_faces)
    frame = measure_head(surface, eye_y, style, ears)
    field = HairField(frame, style)
    tile, layout = hair_tile(tile_size[0], tile_size[1], colour_srgb, texture_seed)
    plan = tune_plan(surface, field, style, layout)
    cards = grow_cards(surface, field, style, plan)
    part = CardPart(
        "procedural",
        "hair",
        cards.faces,
        cards.uv,
        cards.positions,
        "MASK",
        {"material": {"alphaMode": "MASK", "alphaCutoff": 0.5, "doubleSided": True, "tintable": False}},
    )
    clearance = style.clearance * MM
    penetration = penetration_report(surface, cards.positions, cards.faces, clearance)
    n_cards = len(cards.roots)
    layers = {}
    from .hairgen import LAYERS

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
    report = {
        "kind": "procedural-cards",
        "style": style.to_dict(),
        "cards": int(n_cards),
        "vertices": int(len(cards.positions)),
        "triangles": int(len(cards.faces)),
        "triangle_target": style.triangle_target,
        "spacing_scale": plan.scale,
        "layers": layers,
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
        "strip": strip_statistics(tile, layout),
        "tile_size": [int(tile.shape[1]), int(tile.shape[0])],
    }
    if coverage:
        report["coverage"] = scalp_coverage(
            surface, field, cards.positions, cards.faces, cards.uv, tile[..., 3] / 255.0, pixel_mm=coverage_pixel_mm
        )
    return HairBuild(part, cards.positions, tile, layout, field, surface, frame, cards, style, report)
