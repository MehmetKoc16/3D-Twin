"""Correct a clothed scan to tape measurements using the existing rig fitter."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# These modules remain owned by rig; use precisely its morph/fitting semantics.
HERE = Path(__file__).resolve().parent
RIG = HERE.parent / "rig"
sys.path.insert(0, str(RIG))

import numpy as np
import trimesh

from glb import Document
from mh import MHModel
from rigfit import Fitter
from solver import measurements, parse_measurements, solve
from transfer import Transfer
from twin_export import CLOTHING_ALLOWANCE_CM, canonicalize_fit


def private_output(source: Path, measurements_path: Path, destination: Path) -> None:
    repo = HERE.parents[2]
    personal = repo / "user-data"
    if any(path.resolve().is_relative_to(personal) for path in (source, measurements_path)):
        if not destination.resolve().is_relative_to(personal):
            raise ValueError("Personal outputs must stay under user-data/")
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output must be different files")


def run(source: Path, measurements_path: Path, destination: Path, *,
        allowance: dict[str, float] | None = None) -> dict:
    private_output(source, measurements_path, destination)
    allowances = dict(CLOTHING_ALLOWANCE_CM if allowance is None else allowance)
    if measurements_path.is_file():
        targets, weight = parse_measurements(json.loads(measurements_path.read_text(encoding="utf8")))
    else:
        targets, weight = {}, None
    destination.parent.mkdir(parents=True, exist_ok=True)
    report_path = destination.parent / "bodyfix_report.json"
    if not targets and weight is None:
        reason = "missing measurements file" if not measurements_path.is_file() else "no measurements provided"
        print(f"[bodyfix] NO-OP: {reason}; copying input unchanged", flush=True)
        shutil.copyfile(source, destination)
        report = {"version": 1, "status": "no-op", "reason": reason, "targetsCm": {}, "residualsCm": {}}
    else:
        document = Document(source)
        model = MHModel()
        normals = trimesh.Trimesh(document.vertices, document.faces, process=False).vertex_normals
        print("[bodyfix] Fitting MakeHuman to scan (CPU, rig environment)", flush=True)
        fitted = Fitter(model, document.vertices, normals).run()
        canonicalize_fit(model, fitted)
        transfer = Transfer(model, fitted, document.vertices, document.faces)

        def evaluate(target: np.ndarray) -> dict[str, float]:
            rest, _ = transfer.deform(target)
            return measurements(model, transfer.landmarks(rest, target), rest)

        before_raw = evaluate(fitted.rest_positions)
        # First solve the MH proxy, then close transfer/smoothing residuals using
        # the SAME bounded parameters against actual scan surface landmarks.
        macro, mods, _, proxy = solve(model, fitted.macro, fitted.mods, targets, weight, allowance=allowances)
        macro, mods, target, solved = solve(model, macro, mods, targets, weight,
                                          evaluate=evaluate, allowance=allowances)
        rest, corrected = transfer.deform(target)
        corrected[:, 1] -= corrected[:, 1].min()
        document.write(destination, corrected)
        after_raw = measurements(model, transfer.landmarks(rest, target), rest)
        before = {key: value - allowances.get(key, 0) for key, value in before_raw.items()}
        after = {key: value - allowances.get(key, 0) for key, value in after_raw.items()}
        residuals = {key: after[key] - value for key, value in targets.items()}
        report = {
            "version": 1, "status": "corrected", "targetsCm": targets,
            "beforeCm": before, "afterCm": after, "beforeRawCm": before_raw,
            "afterRawCm": after_raw, "residualsCm": residuals,
            "unreachable": [key for key, value in residuals.items()
                            if abs(value) > (0.5 if key == "height" else 1.5)],
            "clothingAllowanceCm": allowances,
            "heightAllowance": {"totalCm": allowances.get("height", 0),
                                "soleCm": 2.0 if allowance is None else 0.0,
                                "hairCm": 1.0 if allowance is None else 0.0},
            "weightKg": weight, "estimatedMassKg": solved["estimatedMassKg"],
            "massResidualKg": solved["estimatedMassKg"] - weight if weight is not None else None,
            "massBasis": "MakeHuman signed volume at 1.01 kg/L; clothed proxy, soft term",
            "measurementBasis": "avatar-core definitions at barycentric scan landmarks in fitted rest pose; scan bbox floor/height",
            "scanHeightCm": float(np.ptp(corrected[:, 1]) * 100),
            "headPolicy": "rigid translation with neck, original shape retained above neck; 5 cm smooth transition",
            "fittedMacros": fitted.macro, "fittedModifiers": fitted.mods,
            "targetMacros": macro, "targetModifiers": mods,
            "fit": fitted.stats, "proxySolve": proxy, "scanSolve": solved,
            "correspondenceCm": {"median": float(np.median(transfer.forward.distance) * 100),
                                 "p99": float(np.percentile(transfer.forward.distance, 99) * 100)},
        }
        print(f"[bodyfix] Corrected mesh written; {len(report['unreachable'])} targets outside tolerance", flush=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, default=Path("user-data/twin/measurements.json"))
    parser.add_argument("--out", dest="destination", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run(args.source, args.measurements, args.destination)
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        print(f"[bodyfix] Failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
