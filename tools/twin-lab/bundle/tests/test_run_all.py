import json
import os
import sys
import time

import pytest
from conftest import runner


@pytest.fixture
def lab(tmp_path):
    folder = tmp_path / "lab"
    for stage, script in (
        ("shape", "generate.py"),
        ("texture", "project.py"),
        ("rig", "rig_scan.py"),
        ("bundle", "write_twin_glb.py"),
    ):
        (folder / stage).mkdir(parents=True)
        (folder / stage / script).write_text("# Synthetic CLI stand-in\n")
    return folder


def test_dry_run_prints_commands_without_creating_outputs(
    lab, tmp_path, monkeypatch, capsys
):
    inputs, out = tmp_path / "input", tmp_path / "out"
    inputs.mkdir()
    for name in ("front", "back", "left", "right"):
        (inputs / f"{name}.png").write_bytes(b"synthetic placeholder; never decoded")
    monkeypatch.setattr(runner, "LAB", lab)
    monkeypatch.setattr(runner, "REPO", tmp_path / "repo")
    assert (
        runner.main(
            [
                "--input-dir",
                str(inputs),
                "--out-dir",
                str(out),
                "--height-cm",
                "182",
                "--dry-run",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    for stage in ("shape", "texture", "rig", "bundle"):
        assert f"[{stage}] RUN" in text
    assert "--back" in text and "--left" in text and "--right" in text
    assert "--variant auto" in text and "--height-cm 182.0" in text
    assert "--views-dir" in text and "--rigged" in text and "twin.glb" in text
    assert "[refine] SKIP" in text
    assert not out.exists()


def test_optional_refine_cli_and_own_interpreters(lab, tmp_path):
    refine = lab / "refine/refine.py"
    refine.parent.mkdir()
    refine.write_text("# Optional synthetic stage\n")
    stages = runner.build_stages(tmp_path / "input", tmp_path / "out", 178, lab=lab)
    assert [stage.name for stage in stages] == list(runner.STAGES)
    command = stages[2].command
    assert command[2:] == [
        "--in",
        str(tmp_path / "out/texture/textured.glb"),
        "--out",
        str(tmp_path / "out/refine/refined.glb"),
    ]
    assert str(lab / "refine/.venv") in command[0]
    assert stages[3].command[2] == str(tmp_path / "out/refine/refined.glb")
    assert str(lab / "rig/.venv") in stages[-1].command[0]
    python = (
        lab
        / "bundle/.venv"
        / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    python.parent.mkdir(parents=True)
    python.touch()
    assert runner.interpreter("bundle", lab) == python


def test_from_and_to_stage_dry_run(lab, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "LAB", lab)
    monkeypatch.setattr(runner, "REPO", tmp_path / "repo")
    assert (
        runner.main(
            [
                "--input-dir",
                str(tmp_path / "input"),
                "--out-dir",
                str(tmp_path / "out"),
                "--from-stage",
                "rig",
                "--to-stage",
                "bundle",
                "--dry-run",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "[shape]" not in text and "[texture]" not in text
    assert "[rig] RUN" in text and "[bundle] RUN" in text


def synthetic_stage(tmp_path, name="synthetic", exit_code=0):
    script, source, out = (
        tmp_path / f"{name}.py",
        tmp_path / f"{name}.in",
        tmp_path / f"{name}.out",
    )
    source.write_text("synthetic")
    script.write_text(
        f"import pathlib, sys\nprint('synthetic stage log')\nsys.exit({exit_code})\n"
        if exit_code
        else "import pathlib, sys\nprint('synthetic stage log')\n"
        "pathlib.Path(sys.argv[1]).write_text('synthetic output')\n"
    )
    return runner.Stage(
        name, [sys.executable, str(script), str(out)], [source, script], [out]
    )


def make_fresh(stage):
    timestamp = time.time_ns()
    for path in stage.inputs:
        os.utime(path, ns=(timestamp - 2_000_000_000, timestamp - 2_000_000_000))
    for path in stage.outputs:
        path.write_text("synthetic cached output")
        os.utime(path, ns=(timestamp - 1_000_000_000, timestamp - 1_000_000_000))


def test_cache_timestamps_and_command_changes(tmp_path):
    stage = synthetic_stage(tmp_path)
    stamp = tmp_path / "state.json"
    make_fresh(stage)
    assert runner.is_fresh(stage, stamp)
    stamp.write_text(json.dumps(runner.signature(stage)))
    assert runner.is_fresh(stage, stamp)
    stage.command += ["--height-cm", "190"]
    assert not runner.is_fresh(stage, stamp)
    stamp.unlink()
    stage.inputs[0].write_text("new synthetic input")
    assert not runner.is_fresh(stage, stamp)


def test_execute_logs_skip_and_force(tmp_path, capsys):
    stage = synthetic_stage(tmp_path)
    out_dir = tmp_path / "out"
    make_fresh(stage)
    runner.execute([stage], out_dir)
    assert "SKIP" in capsys.readouterr().out
    assert "SKIP" in (out_dir / "logs/synthetic.log").read_text()
    runner.execute([stage], out_dir, force=True)
    assert "DONE" in capsys.readouterr().out
    assert stage.outputs[0].read_text() == "synthetic output"
    assert "synthetic stage log" in (out_dir / "logs/synthetic.log").read_text()
    assert runner.is_fresh(stage, out_dir / "logs/synthetic.state.json")


def test_fail_fast_invalidates_cache_and_does_not_run_next_stage(tmp_path):
    failing = synthetic_stage(tmp_path, name="first", exit_code=7)
    next_stage = synthetic_stage(tmp_path, name="second")
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="exit code 7"):
        runner.execute([failing, next_stage], out_dir)
    assert not next_stage.outputs[0].exists()
    assert json.loads((out_dir / "logs/first.state.json").read_text()) == {}


def test_missing_inputs_are_readable_and_dry_run_does_not_require_them(tmp_path):
    stage = synthetic_stage(tmp_path)
    stage.inputs[0].unlink()
    out_dir = tmp_path / "out"
    runner.execute([stage], out_dir, dry_run=True)
    assert not out_dir.exists()
    with pytest.raises(ValueError, match="Missing input"):
        runner.execute([stage], out_dir)


def test_dry_run_propagates_upstream_rebuild_to_cached_downstream(tmp_path, capsys):
    first = synthetic_stage(tmp_path, name="first")
    second = synthetic_stage(tmp_path, name="second")
    make_fresh(second)
    runner.execute([first, second], tmp_path / "out", dry_run=True)
    text = capsys.readouterr().out
    assert "[first] RUN" in text and "[second] RUN" in text


@pytest.mark.parametrize(
    "args",
    [
        ["--from-stage", "bundle", "--to-stage", "shape"],
        ["--height-cm", "nan"],
        ["--height-cm", "0"],
    ],
)
def test_invalid_cli_arguments(args):
    with pytest.raises(SystemExit) as error:
        runner.main(args)
    assert error.value.code == 2


def test_private_output_guards_without_reading_personal_data(tmp_path):
    with pytest.raises(ValueError, match="user-data"):
        runner.ensure_private_output(runner.REPO / "user-data/twin", tmp_path / "out")
    with pytest.raises(ValueError, match="under user-data"):
        runner.ensure_private_output(tmp_path / "input", runner.REPO / "docs/out")


def test_existing_shape_cache_checks_height_and_view_set(tmp_path):
    image, mesh, meta = [
        tmp_path / name for name in ("front.png", "mesh.glb", "meta.json")
    ]
    image.write_bytes(b"synthetic placeholder")
    stage = runner.Stage(
        "shape",
        [sys.executable, "generate.py", "--front", str(image), "--height-cm", "178.0"],
        [image],
        [mesh, meta],
    )
    make_fresh(stage)
    meta.write_text(
        json.dumps({"normalization": {"heightCm": 178}, "inputs": {"front": {}}})
    )
    stamp = tmp_path / "shape.state.json"
    assert runner.is_fresh(stage, stamp)
    stage.command[-1] = "180.0"
    assert not runner.is_fresh(stage, stamp)
    stage.command[-1] = "178.0"
    stage.command.extend(["--back", str(tmp_path / "back.png")])
    assert not runner.is_fresh(stage, stamp)
