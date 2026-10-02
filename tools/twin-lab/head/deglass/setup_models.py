"""Download public models only. No personal inputs are accepted or transmitted."""
from pathlib import Path
import hashlib
import json
import urllib.request

CACHE = Path(__file__).resolve().parent / ".cache"
MODELS = {
    "selfie_multiclass_256x256.tflite": (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite",
        "Apache-2.0"),
    "lama_fp32.onnx": (
        "https://huggingface.co/Carve/LaMa-ONNX/resolve/c3c0c9e/lama_fp32.onnx",
        "Apache-2.0 (export publisher model card)"),
}
HASHES = {
    "selfie_multiclass_256x256.tflite": "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0",
    "lama_fp32.onnx": "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
}


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (url, license_name) in MODELS.items():
        destination = CACHE / name
        if not destination.exists():
            temporary = CACHE / (name + ".part")
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(destination)
        with destination.open("rb") as downloaded:
            digest = hashlib.file_digest(downloaded, "sha256").hexdigest()
        if digest!=HASHES[name]:
            raise ValueError(f"Model SHA-256 mismatch: {name}")
        manifest[name] = {"url": url, "license": license_name, "sha256": digest,
                          "bytes": destination.stat().st_size}
        print(f"{name}: bytes={destination.stat().st_size} sha256={digest}", flush=True)
    (CACHE / "models.json").write_text(json.dumps(manifest, indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
