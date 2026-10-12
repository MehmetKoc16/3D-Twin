"""Reproducible cached-catalogue visibility review (run with the pipeline venv)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .body import load_body, skin
from .geometry import prism_coverage, skin_weights
from .mesh_io import load_parts
from .pose_safety import POSES, body_regions, protected_vertices
from .validation import read_binding
from .visibility import diagnostics, hidden_triangles, contact_anatomy

PICKS = ("mh-jacket-t01", "mh-male-polo-shirt", "mh-m-trousers-01", "mh-m-trousers-02",
         "khronos-materials-variants-shoe", "existing-tshirt", "mh-denim-jacket")


def review(root, phase, identifiers=PICKS):
    from PIL import Image, ImageDraw
    from parts_debug import Scene
    body = load_body(Path(__file__).resolve().parents[3]/"apps/web/public/assets/body")
    regions = body_regions(body)
    protected = protected_vertices(body)
    all_reports, rows = {}, []
    for identifier in identifiers:
        out = root/identifier
        if phase == "before" and not (out/"before-assets").exists():
            import shutil
            snapshot = out/"before-assets"
            snapshot.mkdir()
            for file in out.iterdir():
                if file.is_file() and (file.suffix in (".glb", ".bin") or file.name == "index.json" or
                                       file.name.endswith(".report.json")):
                    shutil.copy2(file, snapshot/file.name)
        source = out/"before-assets" if phase == "before" and (out/"before-assets").exists() else out
        entry = next(e for e in json.loads((source/"index.json").read_text())["garments"] if e["id"] == identifier)
        part = load_parts(source/entry["mesh"])[0]
        bind = read_binding(source/entry["binding"])
        ji, jw = skin_weights(bind, body.joints, body.weights)
        deleted = np.frombuffer((source/entry["deleteVerts"]).read_bytes(), dtype="<u4") if entry.get("deleteVerts") else []
        # App coverage is computed on the fitted rest mesh, not recomputed each animation frame.
        coverage = prism_coverage(body.positions[:body.render_count], part.positions, part.faces)
        effective = np.union1d(deleted, np.flatnonzero(coverage)).astype(int)
        hidden = hidden_triangles(body.faces, body.render_count, effective)
        report = {"poses": {}, "mask": {"exported": len(deleted), "withAppCoverage": len(effective),
                  "exportedProtectedVertices": int(protected[np.asarray(deleted, int)].sum()),
                  "combinedAppProtectedVertices": int(protected[effective].sum()),
                  "combinedAppHiddenProtectedTriangles": int((hidden & protected[body.faces].any(axis=1)).sum())},
                  "geometrySource": str(source.resolve())}
        images = []
        category = entry["category"]
        bbox = (np.array([-.7, .65, -.3]), np.array([.7, 1.7, .4])) if category == "top" else (
            (np.array([-.4, -.03, -.3]), np.array([.4, 1.15, .4])) if category == "bottom" else
            (np.array([-.34, -.04, -.10]), np.array([.34, .23, .36])))
        for key, pose in POSES:
            transforms = body.transforms(body.pose(pose))
            bp = skin(body.positions[:body.render_count], body.joints, body.weights, transforms)
            gp = skin(part.positions, ji, jw, transforms)
            report["poses"][key] = diagnostics(gp, part.faces, bp, body.faces, effective, regions)
            report["poses"][key]['contactAnatomy'] = contact_anatomy(body, report["poses"][key])
            for view in ("front", "side"):
                texture = part.material.get("base")
                garment = {"pos": gp, "tris": part.faces, "uv": part.uv, "color": [90, 135, 195]}
                if texture:
                    garment["tex"] = np.array(texture["image"])
                    garment["color"] = [255, 255, 255]
                image = Scene([{"pos": bp, "tris": body.faces[~hidden], "color": [195, 154, 123]}],
                              view, bbox, 280).render([garment])
                panel = Image.new("RGB", (290, 330), "white")
                panel.paste(Image.fromarray(image), (0, 25))
                ImageDraw.Draw(panel).text((3, 3), f"{key} {view}", fill="black")
                images.append(panel)
        row = Image.new("RGB", (290*8, 355), "white")
        ImageDraw.Draw(row).text((5, 3), identifier+" | "+phase+" | body delete mask + app rest coverage", fill="black")
        for i, image in enumerate(images):
            row.paste(image, (i*290, 25))
        row.save(out/f"{phase}-preview.png")
        if phase == "after":
            row.save(out/"preview.png")
        rows.append(row)
        (out/f"{phase}-visibility.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        all_reports[identifier] = report
        print(identifier, phase, {k: (round(v['visiblePokeThroughPercent'], 3), round(v['selfIntersectionPercent'], 3))
                                  for k, v in report['poses'].items()}, flush=True)
    report_path = root/f"{phase}-visibility.json"
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    previous.update(all_reports)
    all_reports = previous
    report_path.write_text(json.dumps(all_reports, indent=2)+"\n", encoding="utf-8")
    order = [i for i in PICKS if i in all_reports]+[i for i in all_reports if i not in PICKS]
    rows = [Image.open(root/i/f"{phase}-preview.png") for i in order]
    canvas = Image.new("RGB", (2320, 355*len(rows)), "white")
    for i, row in enumerate(rows):
        canvas.paste(row, (0, 355*i))
    canvas.save(root/f"{phase}-review-preview.png")
    if phase == "after":
        canvas.save(root/"review-preview.png")
    return all_reports


def summarize(root, identifiers=PICKS):
    before = json.loads((root/"before-visibility.json").read_text())
    after = json.loads((root/"after-visibility.json").read_text())
    triangles = after[identifiers[0]]['poses']['A']['testedBodyTriangles']
    lines = ["# Pose-aware garment import review", "",
             "All exports and original snapshots are in this gitignored cache. App/package files and shipped garments were not edited by this pipeline job. No commit was made.", "",
             f"Visible poke-through uses the exported delete list plus the app's fitted-rest footprint, hiding a body triangle only when all corners are covered. It counts triangle intersections and guarded local exterior-centroid samples. The denominator is all body triangles ({triangles:,}). Self-intersection counts affected garment triangles, excluding welded adjacency; contacts within 0.1 micrometers and coplanar overlaps are included. Hidden containment counts garment vertices whose nearest body triangle is deleted. These percentages have different denominators and must not be compared to the old complete-body vertex percentages.", "",
             "Before = the garment exported by the previous importer (`before-assets/`). The denim jacket had never been imported (CC-BY-3.0 was not an accepted licence), so its before is the HEAD importer 147b3d2 run on the same source with the same arguments.", "",
             "## Neutral body, all four poses", "",
             "| Garment | Pose | Visible body % before / after | Self-intersecting cloth % before / after | Inside deleted-region vertices before / after |",
             "|---|---|---:|---:|---:|"]
    for identifier in identifiers:
        for key, _ in POSES:
            a, b = before[identifier]['poses'][key], after[identifier]['poses'][key]
            lines.append(f"| {identifier} | {key} | {a['visiblePokeThroughPercent']:.4f} / {b['visiblePokeThroughPercent']:.4f} | "
                         f"{a['selfIntersectionPercent']:.3f} / {b['selfIntersectionPercent']:.3f} | "
                         f"{a['insideDeletedRegionVertices']} / {b['insideDeletedRegionVertices']} |")
    lines += ["", "## Location and remaining contacts", ""]
    for identifier in identifiers:
        out = root/identifier
        report = json.loads((out/f"{identifier}.report.json").read_text())
        lines.append(f"- **{identifier}**: [serialized report]({identifier}/{identifier}.report.json), "
                     f"[before preview]({identifier}/before-preview.png), [after preview]({identifier}/after-preview.png), "
                     f"[original exports]({identifier}/before-assets/).")
        for key in ('relaxed', 'walk'):
            a, b = before[identifier]['poses'][key], after[identifier]['poses'][key]
            lines.append(f"  - {key}: visible body triangles {a['visiblePokeThroughTriangles']} -> {b['visiblePokeThroughTriangles']}; "
                         f"before regions {a.get('regions')}; after regions {b.get('regions')}; "
                         f"after self-intersection regions {b.get('selfIntersectionRegions')}.")
            if b['visiblePokeThroughTriangles']:
                lines.append(f"  - Remaining {key} contact anatomy (neutral rest bounds, meters): {b.get('contactAnatomy')}.")
    lines += ["", "## Rest reconstruction and three-shape grading", "",
              "Profiles: small 150 cm/45 kg neutral, medium 165 cm/58 kg female, large 180 cm/80 kg male. These are numerical grading checks, not collision-free quality approval.", "",
              "| Garment | Max rest round-trip mm | Small / medium / large | Max shape edge stretch P99 | Max shape edge growth mm |",
              "|---|---:|---|---:|---:|"]
    for identifier in identifiers:
        report = json.loads((root/identifier/f"{identifier}.report.json").read_text())
        shapes = report['validation']['shapes']
        lines.append(f"| {identifier} | {report['roundTrip']['maxMm']:.6f} | "
                     f"{' / '.join('pass' if shapes[n]['passed'] else 'FAIL' for n in ('small','medium','large'))} | "
                     f"{max(s['edgeStretchP99'] for s in shapes.values()):.3f} | "
                     f"{max(s['maxEdgeGrowthMm'] for s in shapes.values()):.3f} |")
    lines += ["", "## Licences and rigid shoe soles", "",
              "| Garment | Licence in catalogue entry | Layer | Rigid sidecar |", "|---|---|---:|---|"]
    for identifier in identifiers:
        report = json.loads((root/identifier/f"{identifier}.report.json").read_text())
        entry = next(e for e in json.loads((root/identifier/"index.json").read_text())["garments"] if e["id"] == identifier)
        rigid = report.get("rigidParts")
        runtime = report.get("validation", {}).get("rigidRuntime")
        note = "-" if not rigid else (f"`{rigid['file']}` v1: {rigid['vertices']} sole vertices on {', '.join(rigid['bones'])}; "
                                      f"avatar-core parse: {'accepted' if runtime and runtime['ok'] else 'NOT CHECKED'}"
                                      + (f", {runtime['bandVertices']} blend-band vertices, neutral rebuild error "
                                         f"{runtime['neutralRebuildMaxErrorM']*1e6:.2f} um" if runtime else ""))
        lines.append(f"| {identifier} | {entry['license']} | {entry['layer']} | {note} |")
    lines += ["", "## Importer changes", "",
              "- Smooth local clearance minima add up to 4 mm at armpits and 3 mm at elbows, fading out above the shoulder underside; exterior fabric already beyond the minimum is retained during clearance fitting.",
              "- Armpit/side-seam vertices choose nearby real triangles using neighbourhood skin-weight agreement and minimum posed triangle area. Changes in rest anchors are absorbed into world-axis offsets in the unchanged 36-byte records.",
              "- Top shells receive bounded separation over four poses (12 iterations, 12 mm displacement cap), with UV displacement welded. Regional body clearance is restored; the repair is accepted only if no pose's pair count increases relative to its repair input.",
              "- Delete masks union four-pose coverage and intersecting body-triangle corners within a bounded rest footprint. Hand/finger, neck and head guards override mask additions. Open-edge distance is sampled along edges, rather than just at edge endpoints.",
              "- Validation reports exact triangle contacts, local surface samples, hidden-region containment and anatomical counts separately, on neutral and all three graded bodies.", "",
              "## Precise runtime follow-up", "",
              "Remaining cloth self-intersections cannot be concealed by deleting body triangles. To support arbitrary animation, the runtime owner should evaluate current posed body/garment surfaces on skeleton changes in GarmentInstance, use a BVH broad phase for nonadjacent cloth pairs and non-hidden body triangles, and resolve those contacts in posed space. If the renderer continues to store rest positions in a SkinnedMesh, transfer corrections through the inverse blended skin matrix and recompute normals; skip deleted body triangles, and retain hands/neck/head. Current offline checks omit app length grading, layering and texture alpha. High source-shell intersection counts also require manual source cleanup; this import is not shipping approval.", "",
              "The importer excludes protected vertices from its delete file. The app's additional rest-footprint coverage can add them back; the final review measures that explicitly. To guarantee the same invariant in the app, coveredBodyVertices / WardrobeRig.hideBody must clear hand/finger/neck/head guard vertices from the combined mask before rebuilding body indices. The guard includes skin with any protected-bone influence above 0.001, plus the neck-height exclusion, so these counts include some adjacent shoulder/chest skin.", "",
              "| Garment | Protected vertices in export | Protected vertices after app footprint | Hidden triangles touching protected guard |",
              "|---|---:|---:|---:|"]
    for identifier in identifiers:
        mask = after[identifier]['mask']
        lines.append(f"| {identifier} | {mask['exportedProtectedVertices']} | {mask['combinedAppProtectedVertices']} | {mask['combinedAppHiddenProtectedTriangles']} |")
    lines += ["",
              "## Verification", "",
              "- Run commands and test results are recorded separately in [visibility verification](visibility-verification.md). Rendering a review does not run the test suites.",
              "- Every candidate was re-imported using its saved command.json; exact commands and logs stay in each candidate folder. Final preview metrics reload serialized GLBs and binding records.", "",
              "[Combined before preview](before-review-preview.png) · [Combined after preview](after-review-preview.png)", ""]
    path = root/"review-summary.md"
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


if __name__ == "__main__":
    from garment_import import require_venv
    require_venv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]/".cache/garment-import-out")
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("ids", nargs="*")
    args = parser.parse_args()
    results = review(args.root, args.phase, args.ids or PICKS)
    if args.phase == "after" and (args.root/"before-visibility.json").exists():
        summarize(args.root, tuple(i for i in PICKS if i in results))
