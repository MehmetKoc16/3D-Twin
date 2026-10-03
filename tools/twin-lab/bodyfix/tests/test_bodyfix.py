"""Only CC0-derived MakeHuman geometry and generated textures; no user data."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rig"))

from bodyfix import run
from glb import Document
from glbio import GlbScene, Prim, _accessor, _split, write_static_glb
from mh import MHModel
from solver import mass_kg, measurements, parse_measurements, solve
from transfer import surface_map
from twin_export import CLOTHING_ALLOWANCE_CM


@pytest.fixture(scope="module")
def model():
    return MHModel()


@pytest.fixture(scope="module")
def standin(tmp_path_factory, model):
    folder = tmp_path_factory.mktemp("cc0")
    subprocess.run([sys.executable, str(HERE.parent / "rig/make_standin.py"), str(folder), "--textured"], check=True)
    truth = json.loads((folder / "truth.json").read_text())
    # Keep the official stand-in shape/UVs/texture but remove its known pose, so
    # independent tape evaluation uses the original vertex indices directly.
    positions = model.shape(truth["macro"], truth["mods"])
    source = folder / "rest.glb"
    Document(folder / "mesh.glb").write(source, positions[:model.nr])
    return folder, source, truth, positions


def test_changed_parameters_recover_on_output_mesh(standin, model):
    folder, source, truth, _ = standin
    macro = {**truth["macro"], "height": 0.62}
    mods = {**truth["mods"], "measure/measure-waist-circ": 0.05,
            "measure/measure-bust-circ": -0.2, "measure/measure-hips-circ": -0.15,
            "measure/measure-thigh-circ": -0.1, "measure/measure-neck-circ": 0.1,
            "measure/measure-upperarm-circ": -0.1,
            "measure/measure-shoulder-dist": 0.15,
            "measure/measure-upperarm-length": 0.2,
            "measure/measure-lowerarm-length": -0.1,
            "measure/measure-upperleg-height": 0.1,
            "measure/measure-lowerleg-height": 0.1}
    known = measurements(model, model.shape(macro, mods))
    keys = ("height", "chest", "waist", "hip", "thigh", "neck", "upperArm", "shoulder", "armLength", "inseam")
    data = {key + "Cm": known[key] - CLOTHING_ALLOWANCE_CM[key] for key in keys}
    data["shoe"] = {"system": "EU", "size": 42.5}
    tape = folder / "measurements.json"
    tape.write_text(json.dumps(data))
    out = folder / "full/bodyfixed.glb"
    report = run(source, tape, out, remove_scan_hands=False)
    output = Document(out).vertices
    actual = measurements(model, np.vstack([output, model.shape(macro, mods)[model.nr:]]))
    for key in keys:
        tolerance = 0.5 if key == "height" else 1.5
        assert abs(actual[key] - known[key]) <= tolerance, (key, actual[key], known[key])
        assert abs(report["residualsCm"][key]) <= tolerance
    assert not report["unreachable"]
    assert abs(report["afterCm"]["footLength"] - 27.0) < 1.5
    original_json, original_bin = _split(source.read_bytes())
    result_json, result_bin = _split(out.read_bytes())
    solution = result_json["asset"]["extras"]["dtBodyfix"]
    assert solution["fittedMacros"] == report["targetMacros"]
    assert solution["fittedModifiers"] == report["targetModifiers"]
    assert solution["targetsCm"] == report["targetsCm"]
    assert solution["achievedCm"] == report["afterCm"]
    assert solution["achievedRawCm"] == report["afterRawCm"]
    assert solution["residualsCm"] == report["residualsCm"]
    assert original_json["materials"] == result_json["materials"]
    assert original_json["textures"] == result_json["textures"]
    assert original_json["images"] == result_json["images"]
    assert result_bin[:len(original_bin)] == original_bin
    for original, result in zip(original_json["meshes"], result_json["meshes"]):
        a = original["primitives"][0]["attributes"]["TEXCOORD_0"]
        b = result["primitives"][0]["attributes"]["TEXCOORD_0"]
        np.testing.assert_array_equal(_accessor(original_json, original_bin, a), _accessor(result_json, result_bin, b))
    # Pairwise offsets of vertices well above the neck remain identical.
    original = Document(source).vertices
    mask = original[:, 1] > original[:, 1].max() - 0.12
    displacement = output[mask] - original[mask]
    assert np.max(np.ptp(displacement, axis=0)) < 2e-5


def test_missing_file_is_byte_exact_noop(standin, capsys):
    folder, source, _, _ = standin
    out = folder / "missing/bodyfixed.glb"
    report = run(source, folder / "absent.json", out)
    assert out.read_bytes() == source.read_bytes()
    assert report["status"] == "no-op"
    assert "NO-OP: missing measurements file" in capsys.readouterr().out
    assert json.loads((out.parent / "bodyfix_report.json").read_text())["status"] == "no-op"


def test_partial_height_and_driver_solve(model, standin):
    _, _, truth, positions = standin
    wanted = measurements(model, positions)["waist"] - 4
    macro, mods, result, stats = solve(model, truth["macro"], truth["mods"], {"waist": wanted}, None, allowance={})
    assert abs(measurements(model, result)["waist"] - wanted) < 0.1
    assert macro == truth["macro"]
    for key, value in truth["mods"].items():
        if key != "measure/measure-waist-circ":
            assert mods[key] == value
    wanted_height = measurements(model, positions)["height"] + 5
    _, _, result, _ = solve(model, truth["macro"], truth["mods"], {"height": wanted_height}, None, allowance={})
    assert abs(measurements(model, result)["height"] - wanted_height) < 0.01
    assert stats["converged"]


def test_partial_pipeline_scan(standin, model):
    folder, source, _, positions = standin
    wanted = measurements(model, positions)["waist"] - CLOTHING_ALLOWANCE_CM["waist"] - 3
    tape = folder / "partial.json"
    tape.write_text(json.dumps({"waistCm": wanted}))
    out = folder / "partial/bodyfixed.glb"
    report = run(source, tape, out, remove_scan_hands=False)
    actual = measurements(model, np.vstack([Document(out).vertices, positions[model.nr:]]))
    assert abs(actual["waist"] - CLOTHING_ALLOWANCE_CM["waist"] - wanted) < 1.5
    assert report["targetsCm"] == {"waist": wanted}
    assert report["targetMacros"]["height"] == report["fittedMacros"]["height"]


def test_remeshed_surface_correspondence(standin, model):
    folder, _, truth, positions = standin
    vertices, faces = trimesh.remesh.subdivide(positions[:model.nr], model.faces)
    source = folder / "subdivided.glb"
    write_static_glb(str(source), GlbScene([Prim(vertices, faces)]))
    target = model.shape({**truth["macro"], "height": 0.60},
                         {**truth["mods"], "measure/measure-waist-circ": 0.2,
                          "measure/measure-bust-circ": -0.15,
                          "measure/measure-hips-circ": -0.1})
    known = measurements(model, target)
    keys = ("height", "chest", "waist", "hip", "thigh")
    tape = folder / "remeshed.json"
    tape.write_text(json.dumps({key + "Cm": known[key] - CLOTHING_ALLOWANCE_CM[key] for key in keys}))
    out = folder / "remeshed/bodyfixed.glb"
    report = run(source, tape, out, remove_scan_hands=False)
    result = Document(out).vertices
    assert len(result) == len(vertices)
    # Subdivision appends midpoints: the first nr vertices remain independent
    # anatomical ground-truth anchors, unknown to body's surface mapper.
    actual = measurements(model, np.vstack([result[:model.nr], target[model.nr:]]), result)
    for key in keys:
        assert abs(actual[key] - known[key]) < (0.5 if key == "height" else 1.5), (key, actual[key], known[key])
    assert not report["unreachable"]


@pytest.mark.parametrize("data", [{"heightCm": 0}, {"waistCm": True}, {"heightCm": float("nan")},
                                 {"heightCm": "178"}, {"shoe": {"system": "US", "size": 9}}, {"typoCm": 50}, []])
def test_invalid_measurements(data):
    with pytest.raises(ValueError):
        parse_measurements(data)


def test_shoe_and_mass(model, standin):
    targets, weight = parse_measurements({"shoe": {"system": "EU", "size": 42.5}, "weightKg": 70})
    assert targets == {"footLength": pytest.approx(27.0)}
    assert weight == 70
    _, _, truth, positions = standin
    known = mass_kg(model, positions)
    _, _, _, stats = solve(model, truth["macro"], truth["mods"], {}, known - 3, allowance={})
    assert abs(stats["estimatedMassKg"] - (known - 3)) < 0.1


def test_barycentric_triangle_interior():
    v = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]])
    mapping = surface_map(v, np.array([[0, 1, 2]]), np.array([[0.25, 0.5, 0.1]]))
    np.testing.assert_allclose(mapping.sample(v), [[0.25, 0.5, 0]], atol=1e-9)
    np.testing.assert_allclose(mapping.barycentric.sum(axis=1), 1)


def test_launcher_order_and_missing_measurements(tmp_path):
    spec = importlib.util.spec_from_file_location("bodyfix_launcher", HERE.parent / "run_all.py")
    launcher = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = launcher
    spec.loader.exec_module(launcher)
    lab = tmp_path / "lab"
    for name, script in (("refine", "refine.py"), ("bodyfix", "bodyfix.py")):
        (lab / name).mkdir(parents=True)
        (lab / name / script).write_text("")
    stages = launcher.build_stages(tmp_path / "input", tmp_path / "out", 178, lab=lab)
    names = [stage.name for stage in stages]
    assert names.index("refine") < names.index("bodyfix") < names.index("rig")
    body = next(stage for stage in stages if stage.name == "bodyfix")
    assert str(launcher.interpreter("rig", lab)) == body.command[0]
    assert tmp_path / "input/measurements.json" not in body.inputs
    (tmp_path / "input").mkdir()
    (tmp_path / "input/measurements.json").write_text("{}")
    stages = launcher.build_stages(tmp_path / "input", tmp_path / "out", 178, lab=lab)
    assert tmp_path / "input/measurements.json" in next(s for s in stages if s.name == "bodyfix").inputs
    assert "bodyfix" in launcher.STAGES
