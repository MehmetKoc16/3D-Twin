"""Numeric hair-bake diagnostics; no source image or source crop is exported."""

import numpy as np
from flamehead.colour import skin_samples, to_lab


def colour_stats(lab):
    if len(lab) == 0:
        return {"texels": 0, "skin_fraction": None, "dark_fraction": None}
    return {"texels": int(len(lab)), "skin_fraction": float(skin_samples(lab).mean()),
            "dark_fraction": float((lab[:, 0] < 38).mean()),
            "mean_lab": lab.mean(0, dtype=np.float64).tolist(),
            "std_lab": lab.std(0, dtype=np.float64).tolist()}


def bake_hair_audit(texture, diagnostics, masks, tri, projection=None):
    """Keep raw projection, retained evidence and repaired bake clearly separate.

    Before/after baked checks use the exact same per-side source visibility
    support. Retained-source checks include their reduced sample counts.
    """
    y, x = diagnostics["y"], diagnostics["x"]
    lab = to_lab(texture[y, x])
    member = np.zeros(max(int(tri.max()), max(masks["face"], default=0)) + 1, bool)
    member[masks["face"]] = True
    face = member[tri].all(1)
    result = {**diagnostics["hair"], "projection_skin_check": {}, "temple_band": {},
              "beard": colour_stats(lab[diagnostics["beard_allow"]])}
    for side in ("left", "right"):
        raw = diagnostics["source_labs"][side] if projection is None else projection[0][side]
        regions = {"ear": diagnostics["ear_members"][side], "face": face}
        result["projection_skin_check"][side] = {}
        for name, region in regions.items():
            if projection is None:
                weight = diagnostics["source_weights"][side]
                accepted = diagnostics["accepted_weights"][side]
            else:
                weight = projection[2 if name == "ear" else 1][side]
                accepted = weight * ~diagnostics["rejected_samples"][side]
            support = region & (weight > 0.08)
            result["projection_skin_check"][side][name] = {
                "raw_photo": colour_stats(raw[support]),
                "accepted_photo": colour_stats(raw[support & (accepted > 0.08)]),
                "repaired_bake_same_support": colour_stats(lab[support]),
            }
        temple = diagnostics["temple"] & ((diagnostics["points"][:, 0] > 0) == (side == "left"))
        result["temple_band"][side] = colour_stats(lab[temple])
    result["method"] = "raw photo / accepted photo / repaired bake; same raw per-side visibility support > 0.08"
    return result


def compare_bakes(before, after, geometry):
    """Compare final private head atlases on fixed anatomical/visibility masks."""
    if before.shape != after.shape:
        raise ValueError("Before/after head atlas dimensions differ")
    y, x = geometry["y"], geometry["x"]
    a, b = to_lab(before[y, x]), to_lab(after[y, x])

    def compare(selection):
        return {"before": colour_stats(a[selection]), "after": colour_stats(b[selection])}

    result = {"projection_skin_check": {}, "temple_band": {}, "beard": compare(geometry["beard"])}
    for side in ("left", "right"):
        support = geometry[f"{side}_support"]
        face_support = geometry[f"{side}_face_support"] if f"{side}_face_support" in geometry else support
        result["projection_skin_check"][side] = {
            "ear": compare(geometry[f"{side}_ear"] & support),
            "face": compare(geometry["face_region"] & face_support),
        }
        side_mask = (geometry["points"][:, 0] > 0) == (side == "left")
        result["temple_band"][side] = compare(geometry["temple"] & side_mask)
    if geometry["beard"].any():
        result["beard"]["mean_delta_lab"] = (b[geometry["beard"]].mean(0, dtype=np.float64)
                                               - a[geometry["beard"]].mean(0, dtype=np.float64)).tolist()
        result["beard"]["std_delta_lab"] = (b[geometry["beard"]].std(0, dtype=np.float64)
                                              - a[geometry["beard"]].std(0, dtype=np.float64)).tolist()
    result["method"] = "final baked atlas before/after on identical geometry, semantic masks and per-side source support"
    return result
