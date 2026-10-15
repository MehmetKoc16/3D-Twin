"""Reuse bodyfix's canonical morph model and optional native tape solve."""

import json
import math
import sys
from pathlib import Path

import numpy as np
from bodyfix_solution import read_solution
from glbio import _split
from twin_export import measure_body

from . import LAB


def _macro_range(model, key: str) -> tuple[float, float]:
    if hasattr(model, "macro_vars") and key in model.macro_vars:
        defn = model.macro_vars[key]
        if isinstance(defn, dict) and "min" in defn and "max" in defn:
            return float(defn["min"]), float(defn["max"])
    return 0.0, 1.0


def _modifier_range(model, key: str) -> tuple[float, float]:
    if hasattr(model, "modifiers") and hasattr(model, "mod_index") and key in model.mod_index:
        defn = model.modifiers[model.mod_index[key]]
        if isinstance(defn, dict) and "min" in defn and "max" in defn:
            return float(defn["min"]), float(defn["max"])
    return -1.0, 1.0


def solved_body(model, bodyfix, measurements=None, shape_override=None):
    original = None
    if bodyfix is not None:
        path = Path(bodyfix)
        path = path / "bodyfixed.glb" if path.is_dir() else path
        js, _ = _split(path.read_bytes())
        original = read_solution(js.get("asset", {}).get("extras", {}), model)
        if original is None:
            raise ValueError("Bodyfix input has no dtBodyfix solution")
    if measurements is None:
        if original is None:
            raise ValueError("Provide --bodyfix or --measurements")
        solution = original
        report = {"mode": "saved-bodyfix-exact", "nativeResolve": False}
    else:
        sys.path.insert(0, str(LAB / "bodyfix"))
        from solver import parse_measurements, solve

        targets, weight = parse_measurements(json.loads(Path(measurements).read_text(encoding="utf8")))
        if not targets:
            raise ValueError("Native body solve requires tape measurements")
        macro = original["fittedMacros"] if original else {k: v["default"] for k, v in model.macro_vars.items()}
        mods = original["fittedModifiers"] if original else {}
        macro, mods, points, stats = solve(model, macro, mods, targets, weight, allowance={})
        actual = measure_body(model, points)
        solution = {
            "version": 1,
            "fittedMacros": macro,
            "fittedModifiers": mods,
            "targetsCm": targets,
            "achievedCm": actual,
            "achievedRawCm": actual,
            "residualsCm": {k: actual[k] - v for k, v in targets.items()},
            "clothingAllowanceCm": {},
            "measurementBasis": "native MakeHuman tape solve; skin only, no clothing allowance",
        }
        read_solution({"dtBodyfix": solution}, model)
        report = {"mode": "bodyfix-solver-native-tape", "nativeResolve": True, "solve": stats}

    if shape_override is not None:
        if isinstance(shape_override, (str, Path)):
            override_path = Path(shape_override)
            override_file_name = override_path.name
            override_data = json.loads(override_path.read_text(encoding="utf8"))
            label = str(override_data.get("label", override_path.stem))
        elif isinstance(shape_override, dict):
            override_data = shape_override
            override_file_name = Path(override_data["path"]).name if "path" in override_data else ""
            label = str(override_data.get("label", "override"))
        else:
            raise ValueError(f"Unsupported shape_override type: {type(shape_override)}")

        if not isinstance(override_data, dict):
            raise ValueError("Shape override must be a JSON object")

        override_macros = override_data.get("macros", {})
        if not isinstance(override_macros, dict):
            raise ValueError("macros in shape override must be an object")
        allowed_macros = set(model.macro_vars) if hasattr(model, "macro_vars") else set()
        unknown_macros = set(override_macros) - allowed_macros
        if unknown_macros:
            raise ValueError(f"Unknown macro variable: {', '.join(sorted(unknown_macros))}")
        for k, v in override_macros.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"Invalid macro value for {k}: {v!r}")

        allowed_mods = set(model.mod_index) if hasattr(model, "mod_index") else set()
        override_mods = override_data.get("modifiers", {})
        if not isinstance(override_mods, dict):
            raise ValueError("modifiers in shape override must be an object")
        unknown_mods = set(override_mods) - allowed_mods
        if unknown_mods:
            raise ValueError(f"Unknown modifier: {', '.join(sorted(unknown_mods))}")
        for k, v in override_mods.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"Invalid modifier value for {k}: {v!r}")

        override_add_mods = override_data.get("addModifiers", {})
        if not isinstance(override_add_mods, dict):
            raise ValueError("addModifiers in shape override must be an object")
        unknown_add_mods = set(override_add_mods) - allowed_mods
        if unknown_add_mods:
            raise ValueError(f"Unknown modifier in addModifiers: {', '.join(sorted(unknown_add_mods))}")
        for k, v in override_add_mods.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"Invalid addModifiers value for {k}: {v!r}")

        solution = dict(solution)
        solution["fittedMacros"] = dict(solution["fittedMacros"])
        solution["fittedModifiers"] = dict(solution["fittedModifiers"])

        applied = {}
        for k, val in override_macros.items():
            lo, hi = _macro_range(model, k)
            clamped = float(np.clip(val, lo, hi))
            solution["fittedMacros"][k] = clamped
            applied[k] = clamped

        for k, val in override_mods.items():
            lo, hi = _modifier_range(model, k)
            clamped = float(np.clip(val, lo, hi))
            solution["fittedModifiers"][k] = clamped
            applied[k] = clamped

        for k, delta in override_add_mods.items():
            base = solution["fittedModifiers"].get(k, 0.0)
            lo, hi = _modifier_range(model, k)
            clamped = float(np.clip(base + delta, lo, hi))
            solution["fittedModifiers"][k] = clamped
            applied[k] = clamped

        positions = model.shape(solution["fittedMacros"], solution["fittedModifiers"], ground=True)
        actual = measure_body(model, positions)
        solution["achievedCm"] = actual
        solution["achievedRawCm"] = actual
        solution["clothingAllowanceCm"] = {}
        solution["targetsCm"] = {}
        solution["residualsCm"] = {}
        prev_basis = solution.get("measurementBasis", "")
        if "tape solve" in prev_basis.lower():
            solution["measurementBasis"] = f"{prev_basis} plus shape override '{label}'"
        else:
            solution["measurementBasis"] = f"tape solve plus shape override '{label}'"
        read_solution({"dtBodyfix": solution}, model)
        report["shapeOverride"] = {
            "label": label,
            "path": override_file_name,
            "applied": applied,
        }
    else:
        positions = model.shape(solution["fittedMacros"], solution["fittedModifiers"], ground=True)

    report["bodyMeasurementsCm"] = measure_body(model, positions)
    if original:
        before = model.shape(original["fittedMacros"], original["fittedModifiers"], ground=True)
        report["savedSolutionBodyMeasurementsCm"] = measure_body(model, before)
    return positions, solution, report


def neck_definition():
    from mh import BODY_DIR

    return next(d for d in json.loads((Path(BODY_DIR) / "measures.json").read_text())["measures"] if d["id"] == "neck")
