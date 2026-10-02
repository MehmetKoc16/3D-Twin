"""Bounded MakeHuman solve, preserving the scan fit as the shape prior."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import brentq, least_squares, minimize_scalar

from mh import BODY_DIR, MHModel
from twin_export import CLOTHING_ALLOWANCE_CM, evaluate_measure

FIELDS = {key + "Cm": key for key in (
    "height", "shoulder", "neck", "chest", "waist", "hip", "armLength",
    "inseam", "thigh", "upperArm")}
EU_FOOT_CM = [22.0, 22.7, 23.3, 24.0, 24.7, 25.3, 26.0,
              26.7, 27.3, 28.0, 28.7, 29.3, 30.0, 30.7]


def positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def parse_measurements(data: object) -> tuple[dict[str, float], float | None]:
    if not isinstance(data, dict):
        raise ValueError("measurements must be a JSON object")
    unknown = set(data) - set(FIELDS) - {"weightKg", "shoe"}
    if unknown:
        raise ValueError(f"unknown measurement fields: {', '.join(sorted(unknown))}")
    targets = {measure: positive(data[field], field) for field, measure in FIELDS.items() if field in data}
    if "shoe" in data:
        shoe = data["shoe"]
        if not isinstance(shoe, dict) or shoe.get("system") != "EU" or "size" not in shoe:
            raise ValueError("shoe must contain system EU and a positive size")
        size = positive(shoe["size"], "shoe.size")
        # Same piecewise interpolation/extrapolation as avatar-core/src/shoe.ts.
        i = max(0, min(12, int(np.floor(size)) - 35))
        targets["footLength"] = EU_FOOT_CM[i] + (size - (35 + i)) * (EU_FOOT_CM[i + 1] - EU_FOOT_CM[i])
        positive(targets["footLength"], "converted foot length")
    mass = positive(data["weightKg"], "weightKg") if "weightKg" in data else None
    return targets, mass


@lru_cache(maxsize=1)
def definitions() -> list[dict]:
    return json.loads((Path(BODY_DIR) / "measures.json").read_text(encoding="utf8"))["measures"]


def measurements(model: MHModel, points: np.ndarray, floor_vertices: np.ndarray | None = None) -> dict[str, float]:
    result = {d["id"]: evaluate_measure(d, points, model.nr) * 100 for d in definitions()}
    if floor_vertices is not None:
        floor = floor_vertices[:, 1].min()
        result["height"] = float(np.ptp(floor_vertices[:, 1]) * 100)
        for d in definitions():
            if d["type"] == "vertexHeight":
                result[d["id"]] = float((points[d["vert"], 1] - floor) * 100)
    return result


def mass_kg(model: MHModel, points: np.ndarray) -> float:
    p = points[:model.nr] - points[:model.nr].mean(axis=0)
    triangles = p[model.faces]
    return float(abs(np.einsum("ij,ij->", triangles[:, 0],
                              np.cross(triangles[:, 1], triangles[:, 2]))) / 6 * 1010)


def solve(model: MHModel, fitted_macro: dict[str, float], fitted_mods: dict[str, float],
          targets: dict[str, float], weight: float | None,
          evaluate: Callable[[np.ndarray], dict[str, float]] | None = None,
          allowance: dict[str, float] = CLOTHING_ALLOWANCE_CM) -> tuple[dict, dict, np.ndarray, dict]:
    """Height first, then provided measure drivers and optional mass macro.

    Height is re-rooted inside every local residual, like avatar-core. Mass is a
    soft volume term (10 kg scale); tape measures take priority (0.25 cm scale).
    Gender, muscle and all non-driver modifiers retain their fitted character.
    """
    defs = {d["id"]: d for d in definitions()}
    drivers = sorted({driver for key in targets if key != "height" for driver in defs[key]["drivers"]})
    variables = [("mod", key) for key in drivers]
    if weight is not None:
        variables.append(("macro", "weight"))
    bounds = [(model.modifiers[model.mod_index[key]]["min"], model.modifiers[model.mod_index[key]]["max"])
              if kind == "mod" else (model.macro_vars[key]["min"], model.macro_vars[key]["max"])
              for kind, key in variables]
    initial = np.array([fitted_mods.get(key, 0) if kind == "mod" else fitted_macro.get(key, 0.5)
                        for kind, key in variables])
    if bounds:
        initial = np.clip(initial, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
    evaluate = evaluate or (lambda points: measurements(model, points))
    count = 0

    def state(x: np.ndarray) -> tuple[dict, dict, np.ndarray, dict]:
        nonlocal count
        macro, mods = dict(fitted_macro), dict(fitted_mods)
        for (kind, key), value in zip(variables, x):
            (mods if kind == "mod" else macro)[key] = float(value)

        def at_height(value: float) -> tuple[np.ndarray, dict]:
            macro["height"] = value
            points = model.shape(macro, mods, ground=False)
            return points, evaluate(points)

        if "height" in targets:
            wanted = targets["height"] + allowance.get("height", 0)
            lo, hi = model.macro_vars["height"]["min"], model.macro_vars["height"]["max"]
            # Root if reachable; bounded closest value otherwise.
            _, a = at_height(lo)
            _, b = at_height(hi)
            if a["height"] <= wanted <= b["height"]:
                value = brentq(lambda value: at_height(value)[1]["height"] - wanted,
                               lo, hi, xtol=1e-7)
                points, measured = at_height(value)
            else:
                best = minimize_scalar(lambda value: (at_height(value)[1]["height"] - wanted)**2,
                                       bounds=(lo, hi), method="bounded")
                candidates = [lo, hi, float(best.x)]
                value = min(candidates, key=lambda v: abs(at_height(v)[1]["height"] - wanted))
                points, measured = at_height(value)
        else:
            points = model.shape(macro, mods, ground=False)
            measured = evaluate(points)
        count += 1
        return macro, mods, points, measured

    # Establish the requested height before solving the local drivers.
    state(initial)

    def residual(x: np.ndarray) -> np.ndarray:
        _, _, points, measured = state(x)
        rows = [(measured[key] - (value + allowance.get(key, 0))) / 0.25
                for key, value in targets.items()]
        if weight is not None:
            # Strip clothing mathematically by an independent naked-body solve is
            # underdetermined; use the fitted body volume and explicitly report it.
            rows.append((mass_kg(model, points) - weight) / 10)
        return np.concatenate([rows, 0.025 * (x - initial)])

    def jacobian(x: np.ndarray) -> np.ndarray:
        columns = []
        for i, (lo, hi) in enumerate(bounds):
            a, b = x.copy(), x.copy()
            a[i], b[i] = max(lo, x[i] - 0.001), min(hi, x[i] + 0.001)
            columns.append((residual(b) - residual(a)) / (b[i] - a[i]))
        return np.column_stack(columns)

    result = least_squares(residual, initial, jac=jacobian,
                           bounds=(np.array(bounds)[:, 0], np.array(bounds)[:, 1]),
                           max_nfev=50, ftol=1e-7, xtol=1e-6, gtol=1e-6) if variables else None
    macro, mods, points, measured = state(result.x if result is not None else initial)
    return macro, mods, points, {"evaluations": count, "converged": result is None or bool(result.success),
                                 "rawCm": measured, "estimatedMassKg": mass_kg(model, points)}
