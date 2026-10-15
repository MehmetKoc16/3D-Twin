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


def test_no_shape_override_leaves_output_identical(model, tmp_path):
    solution = bodyfix_solution(model)
    write_bodyfix_glb(tmp_path / "bodyfixed.glb", model, solution)
    pos1, used1, report1 = solved_body(model, tmp_path / "bodyfixed.glb", None)
    pos2, used2, report2 = solved_body(model, tmp_path / "bodyfixed.glb", None, shape_override=None)
    np.testing.assert_array_equal(pos1, pos2)
    assert used1 == used2
    assert report1 == report2
    assert "shapeOverride" not in report1


def test_shape_override_sets_and_adds_values_and_remeasures(model, tmp_path):
    solution = bodyfix_solution(model)
    write_bodyfix_glb(tmp_path / "bodyfixed.glb", model, solution)
    override = tmp_path / "muscular.json"
    override.write_text(
        json.dumps({
            "label": "muscular",
            "macros": {"muscle": 0.85, "weight": 0.45},
            "modifiers": {"torso/torso-muscle-pectoral": 0.6},
            "addModifiers": {"measure/measure-upperarm-length": 0.1},
        }),
        encoding="utf8",
    )
    positions, used, report = solved_body(model, tmp_path / "bodyfixed.glb", None, shape_override=override)

    assert used["fittedMacros"]["muscle"] == pytest.approx(0.85)
    assert used["fittedMacros"]["weight"] == pytest.approx(0.45)
    assert used["fittedModifiers"]["torso/torso-muscle-pectoral"] == pytest.approx(0.6)
    # in synth_mh bodyfix_solution, measure/measure-upperarm-length defaults to 0.2, so 0.2 + 0.1 = 0.3
    assert used["fittedModifiers"]["measure/measure-upperarm-length"] == pytest.approx(0.3)
    assert used["targetsCm"] == {}
    assert used["residualsCm"] == {}
    assert used["clothingAllowanceCm"] == {}
    assert used["achievedCm"] == used["achievedRawCm"]
    assert "tape solve" in used["measurementBasis"]
    assert "muscular" in used["measurementBasis"]

    expected_pos = model.shape(used["fittedMacros"], used["fittedModifiers"], ground=True)
    np.testing.assert_allclose(positions, expected_pos)
    assert report["bodyMeasurementsCm"] == used["achievedCm"]
    assert "savedSolutionBodyMeasurementsCm" in report

    override_rep = report["shapeOverride"]
    assert override_rep["label"] == "muscular"
    assert override_rep["path"] == "muscular.json"
    assert override_rep["applied"]["muscle"] == pytest.approx(0.85)
    assert override_rep["applied"]["weight"] == pytest.approx(0.45)
    assert override_rep["applied"]["torso/torso-muscle-pectoral"] == pytest.approx(0.6)
    assert override_rep["applied"]["measure/measure-upperarm-length"] == pytest.approx(0.3)


def test_shape_override_clamps_out_of_range_values(model, tmp_path):
    solution = bodyfix_solution(model)
    write_bodyfix_glb(tmp_path / "bodyfixed.glb", model, solution)
    override = tmp_path / "extreme.json"
    override.write_text(
        json.dumps({
            "label": "extreme",
            "macros": {"muscle": 2.5, "weight": -0.8},
            "modifiers": {"torso/torso-muscle-pectoral": 3.0},
            "addModifiers": {"head/head-scale-horiz": -5.0},
        }),
        encoding="utf8",
    )
    positions, used, report = solved_body(model, tmp_path / "bodyfixed.glb", None, shape_override=override)

    assert used["fittedMacros"]["muscle"] == pytest.approx(1.0)
    assert used["fittedMacros"]["weight"] == pytest.approx(0.0)
    assert used["fittedModifiers"]["torso/torso-muscle-pectoral"] == pytest.approx(1.0)
    assert used["fittedModifiers"]["head/head-scale-horiz"] == pytest.approx(-1.0)

    applied = report["shapeOverride"]["applied"]
    assert applied["muscle"] == pytest.approx(1.0)
    assert applied["weight"] == pytest.approx(0.0)
    assert applied["torso/torso-muscle-pectoral"] == pytest.approx(1.0)
    assert applied["head/head-scale-horiz"] == pytest.approx(-1.0)


def test_shape_override_rejects_unknown_names(model, tmp_path):
    solution = bodyfix_solution(model)
    write_bodyfix_glb(tmp_path / "bodyfixed.glb", model, solution)

    bad_macro = tmp_path / "bad_macro.json"
    bad_macro.write_text(json.dumps({"macros": {"fake_macro_key": 0.5}}), encoding="utf8")
    with pytest.raises(ValueError, match="Unknown macro variable"):
        solved_body(model, tmp_path / "bodyfixed.glb", None, shape_override=bad_macro)

    bad_mod = tmp_path / "bad_mod.json"
    bad_mod.write_text(json.dumps({"modifiers": {"fake_mod_key": 0.5}}), encoding="utf8")
    with pytest.raises(ValueError, match="Unknown modifier"):
        solved_body(model, tmp_path / "bodyfixed.glb", None, shape_override=bad_mod)

    bad_add_mod = tmp_path / "bad_add_mod.json"
    bad_add_mod.write_text(json.dumps({"addModifiers": {"fake_add_mod_key": 0.5}}), encoding="utf8")
    with pytest.raises(ValueError, match="Unknown modifier in addModifiers"):
        solved_body(model, tmp_path / "bodyfixed.glb", None, shape_override=bad_add_mod)


def test_shape_override_with_native_tape_solve(model, tmp_path):
    solution = bodyfix_solution(model)
    write_bodyfix_glb(tmp_path / "bodyfixed.glb", model, solution)
    tape = tmp_path / "measurements.json"
    height, neck = solution["achievedRawCm"]["height"] + 1.5, solution["achievedRawCm"]["neck"] + 1.0
    tape.write_text(json.dumps({"heightCm": height, "neckCm": neck}))

    override = tmp_path / "variant.json"
    override.write_text(
        json.dumps({
            "label": "athletic",
            "macros": {"muscle": 0.8},
        }),
        encoding="utf8",
    )
    positions, used, report = solved_body(model, tmp_path / "bodyfixed.glb", tape, shape_override=override)
    assert report["mode"] == "bodyfix-solver-native-tape" and report["nativeResolve"] is True
    assert used["fittedMacros"]["muscle"] == pytest.approx(0.8)
    assert used["targetsCm"] == {}
    assert used["residualsCm"] == {}
    assert report["shapeOverride"]["label"] == "athletic"
    assert report["shapeOverride"]["path"] == "variant.json"
    assert report["shapeOverride"]["applied"]["muscle"] == pytest.approx(0.8)

