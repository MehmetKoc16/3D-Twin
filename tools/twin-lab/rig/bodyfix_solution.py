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


def hybrid_fit(extras: dict, model: MHModel, solution: dict | None, vertices):
    """Reuse an exactly solved native A-pose body: no scan pose fit and no shape refit.

    The hybrid stage writes ``asset.extras.dtHybrid`` (frame, manifest hash, ``cutHeightM``) next to ``dtBodyfix``.
    Every vertex below ``cutHeightM`` must then be the solved MakeHuman body itself (2e-5 m); the head above it may
    be deformed and carries the bound parts. Returns ``None`` when the marker is absent (the legacy scan path).
    """
    if "dtHybrid" not in extras:
        return None
    import hashlib
    from pathlib import Path

    import numpy as np
    from scipy.spatial import cKDTree

    from mh import BODY_DIR
    from rigfit import POSE_BONES, FitResult

    marker = extras["dtHybrid"]
    if not isinstance(marker, dict) or marker.get("version") != 1 or marker.get("frame") != "MakeHuman-grounded-A-pose":
        raise ValueError("Unsupported dtHybrid frame")
    if solution is None:
        raise ValueError("Hybrid requires its exact dtBodyfix solution")
    digest = hashlib.sha256((Path(BODY_DIR) / "manifest.json").read_bytes()).hexdigest()
    if marker.get("bodyManifestSha256") != digest:
        raise ValueError("Hybrid MakeHuman manifest differs from the rig's assets")
    cut = marker.get("cutHeightM")
    if isinstance(cut, bool) or not isinstance(cut, (int, float)) or not math.isfinite(cut):
        raise ValueError("Hybrid needs a finite neck cut")
    rest = model.shape(solution["fittedMacros"], solution["fittedModifiers"], ground=False)
    root_t = np.array([0.0, -rest[: model.nr, 1].min(), 0.0])
    posed = rest[: model.nr] + root_t
    body = vertices[vertices[:, 1] < cut - 0.001]
    if len(body) < 8:
        raise ValueError("Hybrid body is missing")
    deviation = cKDTree(posed).query(body)[0].max()
    if deviation > 0.00002:
        raise ValueError("Hybrid body is not the exact solved MakeHuman A-pose")
    stats = {
        "shape_source": "hybrid-bodyfix",
        "pose_source": "verified-native-A-pose",
        "native_body_max_deviation_mm": float(deviation * 1000),
    }
    return FitResult(
        dict(solution["fittedMacros"]),
        dict(solution["fittedModifiers"]),
        rest,
        {name: np.zeros(3) for name in POSE_BONES},
        root_t,
        posed,
        stats,
    )
