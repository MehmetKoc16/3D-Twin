"""Reuse bodyfix's canonical morph model and optional native tape solve."""

import json
import sys
from pathlib import Path

from bodyfix_solution import read_solution
from glbio import _split
from twin_export import measure_body

from . import LAB


def solved_body(model, bodyfix, measurements=None):
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
    positions = model.shape(solution["fittedMacros"], solution["fittedModifiers"], ground=True)
    report["bodyMeasurementsCm"] = measure_body(model, positions)
    if original:
        before = model.shape(original["fittedMacros"], original["fittedModifiers"], ground=True)
        report["savedSolutionBodyMeasurementsCm"] = measure_body(model, before)
    return positions, solution, report


def neck_definition():
    from mh import BODY_DIR

    return next(d for d in json.loads((Path(BODY_DIR) / "measures.json").read_text())["measures"] if d["id"] == "neck")
