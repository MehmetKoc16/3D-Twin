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
STAGES = (
    "shape",
    "texture",
    "refine",
    "head",
    "bodyfix",
    "hybrid",
    "rig",
    "glasses",
    "bundle",
)


@dataclass
class Stage:
    name: str
    command: list[str]
    inputs: list[Path]
    outputs: list[Path]


def interpreter(stage: str, lab: Path) -> Path:
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    if stage == "bodyfix":
        return lab / "rig" / ".venv" / relative
    if stage in {
        "head",
        "hybrid",
        "glasses",
    }:  # shared geometry, textures and previews live in refine
        return lab / "refine" / ".venv" / relative
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
    input_dir: Path,
    out_dir: Path,
    height_cm: float,
    *,
    lab: Path = LAB,
    head: str = "auto",
    with_head: bool = False,
    body: str = "scan",
    bundle_out: Path | None = None,
    glasses_bust: Path | None = None,
    no_glasses: bool = False,
    no_deglass: bool = False,
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
    if body not in {"scan", "hybrid"}:
        raise ValueError(f"Unknown body mode: {body}")
    if glasses_bust is not None and (body != "hybrid" or no_glasses):
        raise ValueError(
            "--glasses-hy3d requires --body hybrid and conflicts with --no-glasses"
        )

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
    head_photos = input_dir / "head"
    if head not in {"auto", "flame", "recon", "none"}:
        raise ValueError(f"Unknown head mode: {head}")
    if with_head:
        if head not in {"auto", "recon"}:
            raise ValueError("--with-head conflicts with --head; use --head recon")
        head = "recon"
    fit = head_photos / "flame/fit"
    if head == "auto":
        head = "flame" if (fit / "head_neutral.obj").is_file() else "none"
    if body == "hybrid" and head != "flame":
        raise ValueError(
            "--body hybrid requires --head flame (or auto with a FLAME fit)"
        )
    if head == "flame":
        head_out = out_dir / "head/head.glb"
        photos = head_photos / "colab_upload"
        assets = REPO / "user-data/flame"
        neutral_name = (
            "head_neutral.obj"
            if (fit / "head_neutral.obj").is_file()
            or not (fit / "head_neutral.ply").is_file()
            else "head_neutral.ply"
        )
        fit_inputs = [
            fit / name
            for name in (
                neutral_name,
                "parameters.json",
                "cameras.json",
                "provenance.json",
                "fitted_views/front.ply",
                "fitted_views/right.ply",
            )
        ]
        # Also track optional fit files so later replacements invalidate the cache.
        fit_inputs += sorted(
            p for p in fit.rglob("*") if p.is_file() and p not in fit_inputs
        )
        asset_inputs = []
        for archive, filename in (
            ("FLAME_masks.zip", "FLAME_masks.pkl"),
            ("mediapipe_landmark_embedding.zip", "mediapipe_landmark_embedding.npz"),
        ):
            asset_inputs.append(
                assets / (filename if (assets / filename).is_file() else archive)
            )
        add(
            "head",
            "flame/flame_head.py",
            [
                "--in",
                str(scan),
                "--fit",
                str(fit),
                "--photos",
                str(photos),
                "--flame-assets",
                str(assets),
                "--out",
                str(head_out),
            ],
            [
                scan,
                *fit_inputs,
                *[photos / f"{name}.jpg" for name in ("front", "right")],
                *asset_inputs,
                *source_inputs(lab / "refine"),
                *source_inputs(lab / "texture"),
                lab / "rig/glbio.py",
            ],
            [head_out, head_out.parent / "flame_head_report.json"],
        )
        scan = head_out
    elif head == "recon":
        head_out = out_dir / "head" / "head.glb"
        add(
            "head",
            "recon/head.py",
            ["--in", str(scan), "--photos", str(head_photos), "--out", str(head_out)],
            [
                scan,
                *sorted(
                    p
                    for p in head_photos.rglob("*")
                    if p.suffix.lower() in {".jpg", ".png"}
                ),
            ],
            [head_out],
        )
        scan = head_out
    body_assets = REPO / "apps/web/public/assets/body"
    # Keep this immediately before rig, after any optional refine/head stages.
    if (lab / "bodyfix/bodyfix.py").is_file():
        measurements_file = input_dir / "measurements.json"
        corrected = out_dir / "bodyfix/bodyfixed.glb"
        measurement_inputs = [measurements_file] if measurements_file.is_file() else []
        add(
            "bodyfix",
            "bodyfix.py",
            [
                "--in",
                str(scan),
                "--measurements",
                str(measurements_file),
                "--out",
                str(corrected),
            ],
            [
                scan,
                *measurement_inputs,
                *source_inputs(lab / "rig"),
                *[
                    body_assets / name
                    for name in (
                        "base.glb",
                        "manifest.json",
                        "morphs.bin",
                        "rig.json",
                        "measures.json",
                    )
                ],
            ],
            [corrected, corrected.parent / "bodyfix_report.json"],
        )
        scan = corrected
    if body == "hybrid":
        if not any(stage.name == "bodyfix" for stage in stages):
            raise ValueError("--body hybrid requires the bodyfix stage")
        hybrid = out_dir / "hybrid/hybrid.glb"
        # Template-character twin: the MakeHuman head is deformed to the FLAME fit. The scan stages above only supply
        # bodyfix's shape prior; no scan or scan head geometry enters the hybrid mesh.
        photos = head_photos / "colab_upload"
        flame_assets = REPO / "user-data/flame"
        bust = glasses_bust or input_dir / "hy3d/hy3d.glb"
        cleanup_args = (["--no-deglass"] if no_glasses or no_deglass or not bust.is_file()
                        else ["--glasses-bust", str(bust)])
        add(
            "hybrid",
            "hybrid.py",
            [
                "--bodyfix",
                str(scan.parent),
                "--measurements",
                str(input_dir / "measurements.json"),
                "--fit",
                str(fit),
                "--photos",
                str(photos),
                "--flame-assets",
                str(flame_assets),
                "--out",
                str(hybrid),
                *cleanup_args,
            ],
            [
                scan,
                *([bust] if bust.is_file() else []),
                input_dir / "measurements.json",
                *sorted(p for p in fit.rglob("*") if p.is_file()),
                *[photos / f"{name}.jpg" for name in ("front", "right")],
                *[
                    flame_assets
                    / (filename if (flame_assets / filename).is_file() else archive)
                    for archive, filename in (
                        ("FLAME_masks.zip", "FLAME_masks.pkl"),
                        (
                            "mediapipe_landmark_embedding.zip",
                            "mediapipe_landmark_embedding.npz",
                        ),
                    )
                ],
                *source_inputs(lab / "head/flame"),
                *source_inputs(lab / "head/glasses"),
                *source_inputs(lab / "refine"),
                *source_inputs(lab / "texture"),
                *source_inputs(lab / "rig"),
                *source_inputs(lab / "bodyfix"),
                *[
                    body_assets / name
                    for name in (
                        "base.glb",
                        "manifest.json",
                        "morphs.bin",
                        "rig.json",
                        "measures.json",
                        "face-map.json",
                    )
                ],
                *sorted((body_assets.parent / "parts").glob("*")),
            ],
            [
                hybrid,
                hybrid.parent / "hybrid_report.json",
                hybrid.parent / "face_asset/face-asset.json",
            ],
        )
        scan = hybrid
        rig = out_dir / "hybrid/rig"
    bundled = bundle_out or (
        out_dir / "hybrid/twin.glb" if body == "hybrid" else out_dir / "twin.glb"
    )
    add(
        "rig",
        "rig_scan.py",
        # Generated scans have fused fists touching the thighs: merge finger weights into the
        # hands and cut hand-thigh bridges (the web app shows MakeHuman hands instead).
        (
            [str(scan), str(rig), "--fingers", "keep", "--smooth", "0"]
            if body == "hybrid"
            else [str(scan), str(rig), "--fingers", "merge", "--cut-bridges"]
        ),
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
    rig_outputs = stages[-1].outputs.copy()
    bundle_rigged = rig / "rigged.glb"
    glasses_args: list[str] = []
    bust = glasses_bust or input_dir / "hy3d/hy3d.glb"
    if (
        body == "hybrid"
        and not no_glasses
        and (glasses_bust is not None or bust.is_file())
    ):
        accessory = out_dir / "hybrid/glasses.glb"
        bundle_rigged = out_dir / "hybrid/glasses_rigged.glb"
        script = lab / "hybrid/hybridbody/glasses_hy3d.py"
        stages.append(
            Stage(
                "glasses",
                [
                    str(interpreter("glasses", lab)),
                    str(script),
                    "--twin",
                    str(rig_outputs[0]),
                    "--hybrid-dir",
                    str(out_dir / "hybrid"),
                    "--bust",
                    str(bust),
                    "--fit",
                    str(fit),
                    "--flame-assets",
                    str(REPO / "user-data/flame"),
                    "--out",
                    str(accessory),
                    "--cleaned-twin",
                    str(bundle_rigged),
                ],
                [
                    *rig_outputs,
                    bust,
                    out_dir / "hybrid/hybrid_report.json",
                    out_dir / "hybrid/face_asset/face-asset.json",
                    out_dir / "hybrid/face_asset/face-offsets.bin",
                    *sorted(p for p in fit.rglob("*") if p.is_file()),
                    *source_inputs(lab / "hybrid"),
                    *source_inputs(lab / "head/glasses"),
                    *source_inputs(lab / "head/flame"),
                    *source_inputs(lab / "texture"),
                    *source_inputs(lab / "refine"),
                    *source_inputs(lab / "rig"),
                    *[
                        REPO
                        / "user-data/flame"
                        / (
                            filename
                            if (REPO / "user-data/flame" / filename).is_file()
                            else archive
                        )
                        for archive, filename in (
                            ("FLAME_masks.zip", "FLAME_masks.pkl"),
                            (
                                "mediapipe_landmark_embedding.zip",
                                "mediapipe_landmark_embedding.npz",
                            ),
                        )
                    ],
                    *[
                        body_assets / name
                        for name in (
                            "base.glb",
                            "manifest.json",
                            "morphs.bin",
                            "rig.json",
                            "measures.json",
                            "face-map.json",
                        )
                    ],
                ],
                [
                    accessory,
                    bundle_rigged,
                    out_dir / "hybrid/glasses_report.json",
                    out_dir / "hybrid/face_asset/face-texture-deglassed.png",
                    *[
                        out_dir / "hybrid/previews_glasses" / name
                        for name in (
                            "accessory_front.png",
                            "accessory_side.png",
                            "accessory_back.png",
                            "accessory_three_quarter.png",
                            "accessory_three_quarter_back.png",
                            "accessory_top.png",
                            "head_front.png",
                            "head_three_quarter.png",
                            "head_side.png",
                            "head_back.png",
                            "head_three_quarter_back.png",
                            "head_top.png",
                            "deglass_uv_mask.png",
                        )
                    ],
                ],
            )
        )
        glasses_args = ["--glasses", str(accessory)]
    add(
        "bundle",
        "write_twin_glb.py",
        [
            "--rigged",
            str(bundle_rigged),
            "--twin",
            str(rig / "twin.json"),
            "--mh2twin",
            str(rig / "mh2twin.bin"),
            "--out",
            str(bundled),
            "--shape",
            "MakeHuman/FLAME hybrid" if body == "hybrid" else "Hunyuan3D-2",
            "--license",
            "CC0 MakeHuman + non-commercial FLAME/Pixel3DMM fit"
            if body == "hybrid"
            else "Tencent Hunyuan 3D 2.0 Community License",
            *glasses_args,
        ],
        [
            bundle_rigged,
            *rig_outputs[1:],
            *([accessory] if glasses_args else []),
            body_assets / "rig.json",
        ],
        [bundled],
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
    parser.add_argument("--body", choices=("scan", "hybrid"), default="scan")
    parser.add_argument(
        "--glasses-hy3d",
        type=Path,
        help="fit glasses from this private bust (hybrid only); defaults to input-dir/hy3d/hy3d.glb when present",
    )
    parser.add_argument(
        "--no-glasses",
        action="store_true",
        help="skip automatic hybrid glasses and UV cleanup",
    )
    parser.add_argument("--no-deglass", action="store_true", help="keep photographed frames in the hybrid texture")
    parser.add_argument(
        "--bundle-out",
        type=Path,
        help="override the final GLB destination; hybrid defaults to out/hybrid/twin.glb",
    )
    parser.add_argument(
        "--head",
        choices=("auto", "flame", "recon", "none"),
        default="auto",
        help="auto uses a local FLAME fit when present, otherwise skips head",
    )
    parser.add_argument(
        "--with-head",
        action="store_true",
        help="deprecated alias for --head recon",
    )
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
        if args.bundle_out:
            ensure_private_output(input_dir, args.bundle_out.resolve().parent)
        if args.with_head:
            print("[head] --with-head is deprecated; use --head recon", file=sys.stderr)
        stages = build_stages(
            input_dir,
            out_dir,
            args.height_cm,
            lab=LAB,
            head=args.head,
            with_head=args.with_head,
            body=args.body,
            bundle_out=args.bundle_out.resolve() if args.bundle_out else None,
            glasses_bust=args.glasses_hy3d.resolve() if args.glasses_hy3d else None,
            no_glasses=args.no_glasses,
            no_deglass=args.no_deglass,
        )
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
