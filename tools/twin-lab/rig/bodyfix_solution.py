"""Validate the solved shape and scan measurements embedded by bodyfix.

Missing metadata keeps legacy fitting; present but invalid metadata must fail
instead of silently replacing the tape-constrained body with a fresh fit.
"""

from __future__ import annotations

import math

from mh import MHModel
from twin_export import MACRO_KEYS, measure_body


def read_solution(extras: dict, model: MHModel) -> dict | None:
    if "dtBodyfix" not in extras:
        return None
    solution = extras["dtBodyfix"]
    if not isinstance(solution, dict) or type(solution.get("version")) is not int or solution["version"] != 1:
        raise ValueError("Unsupported dtBodyfix solution")

    def numbers(name: str, allowed: set[str], *, positive: bool = False) -> dict:
        values = solution.get(name)
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError(f"Invalid dtBodyfix {name}")
        for value in values.values():
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or (positive and value <= 0)):
                raise ValueError(f"Invalid dtBodyfix {name} value")
        return values

    macro = numbers("fittedMacros", set(model.macro_vars))
    if not set(MACRO_KEYS).issubset(macro):
        raise ValueError("dtBodyfix needs all fitted macros")
    mods = numbers("fittedModifiers", set(model.mod_index))
    for key, value in macro.items():
        definition = model.macro_vars[key]
        if not definition["min"] <= value <= definition["max"]:
            raise ValueError(f"dtBodyfix macro out of bounds: {key}")
    for key, value in mods.items():
        definition = model.modifiers[model.mod_index[key]]
        if not definition["min"] <= value <= definition["max"]:
            raise ValueError(f"dtBodyfix modifier out of bounds: {key}")
    measures = set(measure_body(model, model.shape(macro, mods, ground=False)))
    targets = numbers("targetsCm", measures, positive=True)
    achieved = numbers("achievedCm", measures, positive=True)
    raw = numbers("achievedRawCm", measures, positive=True)
    residuals = numbers("residualsCm", measures)
    allowance = numbers("clothingAllowanceCm", measures)
    if set(achieved) != measures or set(raw) != measures or set(residuals) != set(targets):
        raise ValueError("Incomplete dtBodyfix measurements")
    for key in measures:
        if not math.isclose(achieved[key], raw[key] - allowance.get(key, 0), abs_tol=1e-8):
            raise ValueError("Inconsistent dtBodyfix clothing allowance")
    for key in targets:
        if not math.isclose(residuals[key], achieved[key] - targets[key], abs_tol=1e-8):
            raise ValueError("Inconsistent dtBodyfix residual")
    if not isinstance(solution.get("measurementBasis"), str):
        raise ValueError("Missing dtBodyfix measurement basis")
    return solution
