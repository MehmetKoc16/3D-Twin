"""Run the local twin stages with their own Python environments (no uploads)."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[1]
STAGES = ("shape", "texture", "refine", "rig", "bundle")


@dataclass
class Stage:
    name: str
    command: list[str]
    inputs: list[Path]
    outputs: list[Path]


def interpreter(stage: str, lab: Path) -> Path:
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    candidate = lab / stage / ".venv" / relative
    # Bundle can share the existing CPU rig environment, as documented.
    if stage == "bundle" and not candidate.is_file():
        return lab / "rig" / ".venv" / relative
    return candidate


def source_inputs(folder: Path) -> list[Path]:
    """Track local stage implementation changes without traversing caches/venvs."""
    result = []
    for root, directories, files in os.walk(folder):
        directories[:] = [
            name
            for name in directories
            if not name.startswith(".")
            and name not in {"weights", "outputs", "tests", "__pycache__"}
        ]
        result.extend(
            Path(root) / name
            for name in files
            if name.endswith(".py") or name == "requirements.txt"
        )
    return sorted(result)


def build_stages(
    input_dir: Path, out_dir: Path, height_cm: float, *, lab: Path = LAB
) -> list[Stage]:
    views = [
        (name, input_dir / f"{name}.png")
        for name in ("front", "back", "left", "right")
        if name == "front" or (input_dir / f"{name}.png").is_file()
    ]
    shape = out_dir / "shape"
    texture = out_dir / "texture"
    refine = out_dir / "refine"
    rig = out_dir / "rig"
    stages = []

    def add(
        name: str, script: str, args: list[str], inputs: list[Path], outputs: list[Path]
    ) -> None:
        command = [str(interpreter(name, lab)), str(lab / name / script), *args]
        stages.append(
            Stage(name, command, [*inputs, *source_inputs(lab / name)], outputs)
        )

    view_args = [part for name, path in views for part in (f"--{name}", str(path))]
    add(
        "shape",
        "generate.py",
        [
            *view_args,
            "--variant",
            "auto",
            "--height-cm",
            str(height_cm),
            "--out",
            str(shape),
        ],
        [path for _, path in views],
        [
            shape / "mesh.glb",
            shape / "meta.json",
            *[shape / f"cutout_{name}.png" for name, _ in views],
        ],
    )
    add(
        "texture",
        "project.py",
        [
            "--mesh",
            str(shape / "mesh.glb"),
            "--views-dir",
            str(input_dir),
            "--out",
            str(texture),
        ],
        [*stages[-1].outputs, *[path for _, path in views]],
        [texture / "textured.glb"],
    )
    scan = texture / "textured.glb"
    if (lab / "refine/refine.py").is_file():
        add(
            "refine",
            "refine.py",
            ["--in", str(scan), "--out", str(refine / "refined.glb")],
            [scan],
            [refine / "refined.glb"],
        )
        scan = refine / "refined.glb"
    body_assets = REPO / "apps/web/public/assets/body"
    add(
        "rig",
        "rig_scan.py",
        # Generated scans have fused fists touching the thighs: merge finger weights into the
        # hands and cut hand-thigh bridges (the web app shows MakeHuman hands instead).
        [str(scan), str(rig), "--fingers", "merge", "--cut-bridges"],
        [
            scan,
            *[
                body_assets / name
                for name in (
                    "base.glb",
                    "rig.json",
                    "manifest.json",
                    "morphs.bin",
                    "measures.json",
                )
            ],
        ],
        [rig / "rigged.glb", rig / "twin.json", rig / "mh2twin.bin"],
    )
    add(
        "bundle",
        "write_twin_glb.py",
        [
            "--rigged",
            str(rig / "rigged.glb"),
            "--twin",
            str(rig / "twin.json"),
            "--mh2twin",
            str(rig / "mh2twin.bin"),
            "--out",
            str(out_dir / "twin.glb"),
            "--shape",
            "Hunyuan3D-2",
            "--license",
            "Tencent Hunyuan 3D 2.0 Community License",
        ],
        [*stages[-1].outputs, body_assets / "rig.json"],
        [out_dir / "twin.glb"],
    )
    return stages


def signature(stage: Stage) -> dict:
    return {"command": stage.command, "inputs": [str(path) for path in stage.inputs]}


def is_fresh(stage: Stage, stamp: Path) -> bool:
    if not all(path.is_file() for path in [*stage.inputs, *stage.outputs]):
        return False
    if stamp.exists():
        try:
            if json.loads(stamp.read_text(encoding="utf-8")) != signature(stage):
                return False
        except (OSError, ValueError):
            return False
    elif stage.name == "shape":
        # Existing shape runs record these settings, even before the launcher manages them.
        try:
            meta_path = next(path for path in stage.outputs if path.name == "meta.json")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            height = float(stage.command[stage.command.index("--height-cm") + 1])
            if meta.get("normalization", {}).get("heightCm") != height:
                return False
            views = {
                flag[2:]
                for flag in stage.command
                if flag in {"--front", "--back", "--left", "--right"}
            }
            if set(meta.get("inputs", {})) != views:
                return False
        except (OSError, ValueError, StopIteration):
            return False
    return min(path.stat().st_mtime_ns for path in stage.outputs) > max(
        path.stat().st_mtime_ns for path in stage.inputs
    )


def display_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def ensure_private_output(input_dir: Path, out_dir: Path) -> None:
    if out_dir.is_relative_to(REPO):
        relative = out_dir.relative_to(REPO)
        if relative.parts[:1] != ("user-data",) and not (
            relative.parts[:2] == ("tools", "twin-lab") and "outputs" in relative.parts
        ):
            raise ValueError(
                "Output inside the repo must be under user-data/ or twin-lab/**/outputs/"
            )
    if input_dir.is_relative_to(REPO / "user-data") and not out_dir.is_relative_to(
        REPO / "user-data"
    ):
        raise ValueError("Personal outputs must stay under user-data/")


def execute(
    stages: list[Stage], out_dir: Path, *, force: bool = False, dry_run: bool = False
) -> None:
    logs = out_dir / "logs"
    dirty = False
    for stage in stages:
        started = time.perf_counter()
        stamp = logs / f"{stage.name}.state.json"
        fresh = not force and not dirty and is_fresh(stage, stamp)
        action = "SKIP (outputs newer than inputs)" if fresh else "RUN"
        print(f"[{stage.name}] {action}", flush=True)
        if dry_run:
            print(display_command(stage.command), flush=True)
            dirty |= not fresh
            continue
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / f"{stage.name}.log"
        if fresh:
            log_path.write_text(f"[{stage.name}] {action}\n", encoding="utf-8")
            stamp.write_text(
                json.dumps(signature(stage), indent=2) + "\n", encoding="utf-8"
            )
            continue
        missing = [path for path in stage.inputs if not path.is_file()]
        if missing:
            raise ValueError(
                f"[{stage.name}] Missing input: {missing[0]}; run the preceding stage first"
            )
        python_path, script = map(Path, stage.command[:2])
        if not python_path.is_file():
            raise ValueError(
                f"[{stage.name}] Missing interpreter: {python_path}; set up this stage's .venv"
            )
        if not script.is_file():
            raise ValueError(f"[{stage.name}] Missing script: {script}")
        for output in stage.outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
        # Invalidate the old cache before starting, so interrupted/failed stages are retried.
        stamp.write_text("{}\n", encoding="utf-8")
        with log_path.open("w", encoding="utf-8") as log:
            log.write(display_command(stage.command) + "\n")
            log.flush()
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
            with subprocess.Popen(
                stage.command,
                cwd=REPO,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            ) as process:
                assert process.stdout is not None
                for line in process.stdout:
                    print(f"[{stage.name}] {line.rstrip()}", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            elapsed = time.perf_counter() - started
            log.write(f"\nExit {code}; elapsed {elapsed:.2f}s\n")
        if code:
            raise ValueError(
                f"[{stage.name}] Failed with exit code {code} after {elapsed:.2f}s; log: {log_path}"
            )
        if not all(path.is_file() for path in stage.outputs):
            raise ValueError(
                f"[{stage.name}] Exited successfully but expected output is missing; log: {log_path}"
            )
        stamp.write_text(
            json.dumps(signature(stage), indent=2) + "\n", encoding="utf-8"
        )
        dirty = True
        print(f"[{stage.name}] DONE in {elapsed:.2f}s; log: {log_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("user-data/twin"))
    parser.add_argument("--out-dir", type=Path, default=Path("user-data/twin/out"))
    parser.add_argument("--from-stage", choices=STAGES, default="shape")
    parser.add_argument("--to-stage", choices=STAGES, default="bundle")
    parser.add_argument("--height-cm", type=float, default=178.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without running, creating files or requiring venvs",
    )
    args = parser.parse_args(argv)
    first, last = STAGES.index(args.from_stage), STAGES.index(args.to_stage)
    if first > last:
        parser.error("--from-stage must precede or equal --to-stage")
    if not math.isfinite(args.height_cm) or args.height_cm <= 0:
        parser.error("--height-cm must be finite and positive")
    input_dir, out_dir = args.input_dir.resolve(), args.out_dir.resolve()
    started = time.perf_counter()
    try:
        ensure_private_output(input_dir, out_dir)
        stages = build_stages(input_dir, out_dir, args.height_cm, lab=LAB)
        if first <= STAGES.index("refine") <= last and not any(
            stage.name == "refine" for stage in stages
        ):
            print(
                "[refine] SKIP (refine/refine.py does not exist); rig uses textured.glb",
                flush=True,
            )
        selected = [
            stage for stage in stages if first <= STAGES.index(stage.name) <= last
        ]
        execute(selected, out_dir, force=args.force, dry_run=args.dry_run)
    except (OSError, ValueError) as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Pipeline {'dry run' if args.dry_run else 'complete'} in {time.perf_counter() - started:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
