import json

import numpy as np
import pytest
from hybridbody.body import neck_definition, solved_body
from synth_mh import bodyfix_solution, write_bodyfix_glb


def test_saved_solution_is_reproduced_exactly_and_marked_as_such(model, tmp_path):
    solution = bodyfix_solution(model)
    write_bodyfix_glb(tmp_path / "bodyfixed.glb", model, solution)
    positions, used, report = solved_body(model, tmp_path, None)
    assert report["mode"] == "saved-bodyfix-exact" and report["nativeResolve"] is False
    expected = model.shape(solution["fittedMacros"], solution["fittedModifiers"], ground=True)
    np.testing.assert_allclose(positions, expected)
    assert used["fittedMacros"] == solution["fittedMacros"]
    assert positions[: model.nr, 1].min() == pytest.approx(0, abs=1e-12)


def test_native_tape_solve_hits_the_skin_measurements_without_clothing_allowance(model, tmp_path):
    solution = bodyfix_solution(model)
    write_bodyfix_glb(tmp_path / "bodyfixed.glb", model, solution)
    tape = tmp_path / "measurements.json"
    height, neck = solution["achievedRawCm"]["height"] + 1.5, solution["achievedRawCm"]["neck"] + 1.0
    tape.write_text(json.dumps({"heightCm": height, "neckCm": neck}))
    positions, used, report = solved_body(model, tmp_path / "bodyfixed.glb", tape)
    assert report["mode"] == "bodyfix-solver-native-tape" and report["nativeResolve"] is True
    assert report["bodyMeasurementsCm"]["height"] == pytest.approx(height, abs=0.02)
    assert report["bodyMeasurementsCm"]["neck"] == pytest.approx(neck, abs=0.05)
    assert used["clothingAllowanceCm"] == {} and used["residualsCm"]["neck"] == pytest.approx(0, abs=0.05)
    assert "savedSolutionBodyMeasurementsCm" in report


def test_missing_inputs_fail_loudly(model, tmp_path):
    with pytest.raises(ValueError, match="--bodyfix or --measurements"):
        solved_body(model, None, None)
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    with pytest.raises(ValueError, match="tape measurements"):
        solved_body(model, None, empty)


def test_neck_definition_is_the_circumference_loop():
    definition = neck_definition()
    assert definition["id"] == "neck" and definition["type"] == "circumference" and len(definition["verts"]) >= 20
