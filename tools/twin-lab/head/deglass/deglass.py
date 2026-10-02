"""Remove thin eyeglass frames locally; console output is counts only."""
from pathlib import Path
import argparse
import cv2
import numpy as np

from core import (ROOT, VIEWS, EYES, Landmarker, detect_glasses, inpaint, private_path,
                  pupil_centers, read_image, write_image, write_json)
from models import AccessorySegmenter,LamaInpainter
from quality import quality_stats,debug_composite


def main() -> None:
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=Path("user-data/twin/head"))
    parser.add_argument("--out-dir", type=Path, default=Path("user-data/twin/head/clean"))
    parser.add_argument("--model", type=Path, default=ROOT / "apps/web/public/models/face_landmarker.task")
    parser.add_argument("--method", choices=("auto", "lama", "telea", "ns"), default="auto")
    args = parser.parse_args()
    source, target = private_path(args.in_dir), private_path(args.out_dir)
    if source == target:
        parser.error("Output directory must differ from input directory")
    for name in VIEWS:
        if not (source / f"{name}.jpg").is_file():
            parser.error(f"Missing required input: {name}.jpg")
    target.mkdir(parents=True, exist_ok=True)
    debug_dir = private_path(target/"debug")
    debug_dir.mkdir(parents=True,exist_ok=True)
    detector = Landmarker(args.model)
    segmenter = AccessorySegmenter()
    cv2_method = "ns" if args.method=="ns" else "telea"
    lama, fallback_reason = None,None
    if args.method in ("auto","lama"):
        try:
            lama = LamaInpainter()
        except Exception as error:
            if args.method=="lama":
                raise
            fallback_reason = f"Big-LaMa initialization failed ({type(error).__name__}); check model/runtime setup"
    report = {"version": 3, "method": "big_lama_onnx" if lama else f"opencv_{cv2_method}",
              "segmentation": "selfie_multiclass_256x256 class 5 accessories, class 1 hair guard",
              "fallback_reason": fallback_reason, "qa_status":"running", "photos": {}}
    profile_ratio, eye_ratio = 0.35, 0.32
    try:
        for name in VIEWS:
            image = read_image(source / f"{name}.jpg")
            landmarks = None if name == "back" else detector.detect(image)
            accessory,hair = segmenter.segment(image,landmarks)
            if name == "front" and landmarks is not None:
                pupils = pupil_centers(landmarks)
                ipd = float(np.linalg.norm(pupils[1]-pupils[0]))
                height = float(np.linalg.norm(landmarks[10]-landmarks[152]))
                if height > 0 and ipd > 0:
                    profile_ratio = ipd / height
                    eye_ratio = max(float(np.linalg.norm(landmarks[e[0]]-landmarks[e[3]])) for e in EYES) / ipd
            detected = detect_glasses(image, landmarks, name, profile_ratio, eye_ratio,accessory,hair)
            method = "big_lama_onnx" if lama else f"opencv_{cv2_method}"
            photo_fallback = fallback_reason
            if lama is not None:
                try:
                    cleaned = lama.inpaint(image,detected.mask)
                except Exception as error:
                    if args.method=="lama":
                        raise
                    cleaned = inpaint(image,detected.mask,cv2_method)
                    method = f"opencv_{cv2_method}"
                    photo_fallback = f"Big-LaMa inference failed ({type(error).__name__}); check model/runtime setup"
            else:
                cleaned = inpaint(image,detected.mask,cv2_method)
            write_image(target / f"{name}.jpg",cleaned)
            write_image(target / f"{name}_mask.png", detected.mask)
            encoded_result = read_image(target/f"{name}.jpg")
            qa = quality_stats(image,encoded_result,detected.mask,detected.protected,landmarks)
            write_image(debug_dir/f"{name}.png",debug_composite(image,encoded_result,detected.mask))
            pixels = int((detected.mask > 0).sum())
            report["photos"][name] = {
                "mask_pixels": pixels, "total_pixels": int(detected.mask.size),
                "face_detected": landmarks is not None, "status": detected.status,
                "rims_detected": len(detected.rings),
                "protected_pixels_masked": int(((detected.mask > 0) & (detected.protected > 0)).sum()),
                "hair_pixels_masked": int(((detected.mask>0)&(hair>0)).sum()),
                "method": method if pixels else "unchanged",
                "fallback_reason": photo_fallback,
                "inference":lama.last_diagnostics if lama and pixels else None,
                "qa":qa,
                "review_required": True}
            write_json(target/"report.json",report)
            print(f"{name}: mask_pixels={pixels} inside_luminance={qa['inside_luminance_mean']} "
                  f"ring_luminance={qa['ring_luminance_mean']} eye_mask_fraction={qa['eye_mask_fraction']} "
                  f"flags={','.join(qa['flags']) or 'none'}",flush=True)
    finally:
        detector.close()
        segmenter.close()
    failed = any(not item["qa"]["checks_passed"] for item in report["photos"].values())
    report["qa_status"] = "failed" if failed else "passed"
    write_json(target / "report.json", report)
    if failed:
        raise SystemExit("Output QA failed; inspect the numeric report and private debug overlays")


if __name__ == "__main__":
    main()
