# Local eyeglass removal and measurement

This standalone stage owns its Python environment and does not change the twin
GLB contract or the web app. Personal inputs, cleaned JPEGs, masks, reports and
measurements remain under gitignored `user-data/`. The processing scripts work
**offline**, enforce resolved private paths (including symlinks), and never
view, print or upload images. Tests use synthetic arrays only.

## Setup and run (repository root / Windows / Python 3.12)

```powershell
python -m venv tools/twin-lab/head/deglass/.venv
tools/twin-lab/head/deglass/.venv/Scripts/python.exe -m pip install --no-cache-dir -r tools/twin-lab/head/deglass/requirements.lock.txt
tools/twin-lab/head/deglass/.venv/Scripts/python.exe tools/twin-lab/head/deglass/setup_models.py
tools/twin-lab/head/deglass/.venv/Scripts/python.exe -m pytest tools/twin-lab/head/deglass/tests -p no:cacheprovider -q
tools/twin-lab/head/deglass/.venv/Scripts/python.exe tools/twin-lab/head/deglass/deglass.py --in-dir user-data/twin/head --out-dir user-data/twin/head/clean --method lama
tools/twin-lab/head/deglass/.venv/Scripts/python.exe tools/twin-lab/head/deglass/measure.py --in-dir user-data/twin/head --lens-shape rounded-polygon --ipd-mm 63
```

`requirements.txt` pins direct dependencies; `requirements.lock.txt` pins the
complete tested environment. MediaPipe requires `opencv-contrib-python`; do
not install a second conflicting `cv2` distribution. The existing landmark
model is `apps/web/public/models/face_landmarker.task` (`--model` override).

`setup_models.py` downloads only public weights into ignored `.cache/`: pinned
version 1 of MediaPipe's multiclass model and revision `c3c0c9e` of Carve's
Big-LaMa ONNX export. Pinned SHA-256 hashes are verified, and source/license/hash
metadata is recorded in `.cache/models.json`. Neither processing script performs
network requests. No downloaded model is added to tracked files.

`deglass.py --method auto` (default) tries Big-LaMa and falls back to OpenCV only
on model initialization/inference failure, recording the reason in the report.
`--method lama` fails instead of falling back; the revised real-photo run used
this strict mode successfully. `--method telea` and `ns` are explicit diagnostic
overrides. Measurement defaults to `--lens-shape rounded-polygon`; `round` is
also supported. Shape is an explicit hint, not an automatic classification.

The initial sandbox `ensurepip` failed because Python 3.12 mode-0700 temporary
directories received inaccessible Windows ACLs. The **same venv** was repaired
using ignored `.venv/Lib/site-packages/sitecustomize.py`, changing mkdir mode
0700 to 0777 (other modes unchanged), with `TEMP`/`TMP` in `.venv/tmp`. Pip's
outside-workspace cache also stalled; `--no-cache-dir` fixed it. No alternative
Python was used. If this venv cannot start, report it instead of switching
interpreters. These accommodations are unnecessary on unrestricted machines.

## Mask and inpainting

1. MediaPipe's 478 landmarks define pupils, eyelids and eyebrows. Crops, mirrored
   inference and small roll retries recover near profiles; every landmark is
   mapped back to the original, unmirrored photo.
2. `selfie_multiclass_256x256` segments the full portrait and a tighter head crop.
   Class **5 (others/accessories)** confirms ridge pixels. Class **1 (hair)**,
   or hair probability at least 0.35, supplies a strict guard enlarged 2 pixels.
   Confirmation needs probability at least 0.08 near a ridge within the
   eye/temple region. Coarse accessory areas are never added as filled masks.
3. Multi-scale black-hat/top-hat responses find bright/dark thin ridges at up to
   1024 pixels. Ellipse envelopes initialize a closed dynamic-programming contour
   with a varying radius, following rounded polygonal rims rather than imposing
   circles. Supported trace segments join short upper-rim gaps. A narrow bridge
   corridor, compact nose-pad ellipses and hinge corridors restrict candidate
   ridges. Near-ear ridges require nearby
   accessory evidence to reduce skin-edge false positives.
4. Profiles use front IPD-to-face-height and eye-width-to-IPD ratios to account
   for foreshortening. Small specks are discarded and small gaps closed. Wire
   masks use nominal **5-pixel centerline bands** at original resolution, keeping
   observed thin edges and a bounded antialiasing fringe at corners. Broad ridge
   responses cannot become filled cheek/bridge regions. Compact, confirmed nose
   pads and their inner-rim attachment retain their observed footprint rather
   than collapsing to wire centerlines; this does not fill accessory areas.
5. Expanded eye/eyebrow polygons and the hair guard are removed after closing
   and dilation. This guarantees zero overlap with the **computed** guards;
   inaccurate landmarks/segmentation can still miss real anatomy. Frames crossing
   those guards are retained to respect the protection requirement.
6. Big-LaMa runs offline using ONNX Runtime CPU, overlapping 512-pixel context
   tiles, reflection padding and weighted overlap. Generated pixels are copied
   only into the final mask; original unmasked pixels remain byte-identical
   before JPEG encoding. The fallback uses OpenCV Telea/NS and is reported.

The exact ONNX contract is named float32 NCHW `image` (RGB, 0..1), named float32
NCHW `mask` (binary 0/1), and named float32 NCHW `output` (RGB, **0..255**).
Fixed 512px tiles use reflection padding and are multiples of 8. Unlike the
reference PyTorch generator's 0..1 prediction, [Carve's export wrapper](https://github.com/Carve-Photos/lama/blob/main/export_LaMa_to_onnx.ipynb)
already multiplies by 255 inside the graph. The former extra postprocessing
multiply saturated pixels to white; it is removed. Output shape/range/finiteness
and the export's exact unmasked identity are checked before compositing. No
automatic output-range guessing is used.

Outputs are quality-95 `<name>.jpg`, original-resolution `<name>_mask.png`, and
`report.json` with counts, guard overlap, actual methods, fallback reasons and
numeric QA. Each `debug/<name>.png` shows original | red mask overlay | result
for **the lead to view locally**. The agent never views these personal composites.
Console output contains numeric counts, luminance statistics and flag names only.
The back receives a zero mask because no
face region is available; no-face views are explicitly skipped. This is not
proof that arms behind the ears are absent. Zero-mask JPEGs are still re-encoded,
so compression may change decoded pixels.

LaMa predicts appearance, not physically observed skin. Glare/refraction outside
the mask remains. Numeric counts and synthetic tests do not establish visual
success. The lead must review upper rims, pads, bridge, hinges, ears, protected
anatomy, retained frame/anatomy overlaps, generated skin and JPEG artifacts.

## Metric estimates and uncertainty

`glasses.json` uses **meters**. Default assumed IPD is 0.063 m (`--ipd-mm`, finite
range 40..85 mm). Scale = IPD / front pupil distance; two supported front rims
are required. Failure writes no invented dimensions.

- `lens_shape`: `round` or `rounded-polygon`, from the explicit hint.
- `lens_corner_count_approx`: convex contour hull simplified at 1.5% of its
  perimeter; zero for the explicit round hint. This is an approximate count.
- `lens_roundness`: `4*pi*area/perimeter^2`, from the convex contour hull;
  1 means an ideal circle. Noise and masking bias both shape descriptors.
- `lens_outer_radius_m`: mean horizontal ellipse-envelope radius; a proxy for
  a rounded polygon, not a claim that its boundary is circular.
- `bridge_width_m`: inner envelope gap. `frame_width_at_temples_m`: outer
  envelope span, a hinge-width proxy rather than independently recovered hardware.
- `frame_thickness_m`: twice the median local maximum of the **undilated
  ridge-only** distance transform. Accessory/lens-interior regions are excluded
  from the thickness statistic.
- `frame_colour_rgb` / `frame_colour_hex`: RGB channel medians from the final
  mask, as requested. Skin margins and highlights
  contaminate this statistic; it is not intrinsic metal colour.
- `temple_arm_length_m`: approximately horizontal visible profile segment,
  calibrated using eye width, plus an **assumed 0.025 m** hidden/curved ear end.
  Median across usable profiles; **0.14 m fallback** if none qualifies.
  `temple_arm_method` identifies measured projection versus assumption.
- `lens_vertical_offset_m`: mean rim-center y minus mean pupil y; positive down.

All dimensions scale linearly with assumed IPD. Perspective, refraction, pose
and landmarks are uncalibrated. Review allowances: roughly +/-15% main dimensions,
+/-50% thickness, +/-0.03 m arm length, +/-0.005 m vertical offset. These are
**heuristics, not statistical confidence intervals**. Hair guards may suppress
partially occluded arms; Hough lines can confuse other edges with metal. Confirm
shape/dimensions against the physical frame where available.

## Licenses and provenance

No photos, generated personal examples, models or other assets enter tracked
paths. Synthetic geometry is drawn in code. No SMPL, SMPL-X, FLAME or Mixamo data
is used. Installed distribution notices stay in the ignored venv; model caches
stay ignored. Preserve bundled notices if redistributing software later.

| Component                                                          | License / primary reference                                                                                                                 |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Multiclass selfie segmentation                                     | Apache-2.0; [official model documentation](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter)                   |
| Big-LaMa ONNX export                                               | [Apache-2.0 declared by Carve's model card](https://huggingface.co/Carve/LaMa-ONNX); export of the original Big-LaMa, fixed 512px, opset 17 |
| ONNX Runtime 1.22.1                                                | [MIT](https://github.com/microsoft/onnxruntime/blob/v1.22.1/LICENSE)                                                                        |
| MediaPipe, absl-py, flatbuffers                                    | [Apache-2.0](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)                                                               |
| Existing landmark model                                            | [Apache-2.0 FaceMesh model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)           |
| OpenCV 4.12                                                        | [Apache-2.0](https://opencv.org/license/)                                                                                                   |
| OpenCV Python packaging                                            | MIT; [bundled notices](https://github.com/opencv/opencv-python/blob/4.x/LICENSE-3RD-PARTY.txt)                                              |
| NumPy                                                              | BSD-3-Clause; bundled numerical-library notices                                                                                             |
| pytest, pluggy, iniconfig, colorama, sounddevice / PortAudio       | MIT                                                                                                                                         |
| cffi                                                               | MIT-0                                                                                                                                       |
| pycparser, contourpy, cycler, coloredlogs, protobuf, sympy, mpmath | BSD-3-Clause                                                                                                                                |
| humanfriendly                                                      | MIT                                                                                                                                         |
| pyreadline3                                                        | BSD-3-Clause                                                                                                                                |
| Pygments                                                           | BSD-2-Clause                                                                                                                                |
| packaging                                                          | Apache-2.0 OR BSD-2-Clause                                                                                                                  |
| fonttools, kiwisolver, pyparsing, six                              | MIT                                                                                                                                         |
| python-dateutil                                                    | Apache-2.0 OR BSD-3-Clause                                                                                                                  |
| Pillow                                                             | [MIT-CMU](https://github.com/python-pillow/Pillow/blob/main/LICENSE)                                                                        |
| Matplotlib (MediaPipe dependency)                                  | [PSF-based permissive license and bundled BSD-compatible notices](https://matplotlib.org/stable/project/license.html)                       |

Only ONNX Runtime is added for LaMa inference; the reference project's large
training dependencies are not installed. The export publisher's license is
recorded explicitly rather than inferring it from the LaMa code license.

## Validation

All **24 synthetic pytest tests passed** with both local models present.
The suite covers dark/light frames, resolutions, recall and
precision, protected anatomy, polygon upper rims, nose pads, hair guards, metric
geometry, IPD scaling, projected profiles, private paths, actual offline LaMa
removal, unmasked-pixel preservation, overlapping/non-square tiles, and actual
multiclass inference. Regression checks reject pure-white/wrong-range predictions,
wrong channel order, excessive eye-region area and coarse accessory area filling.
The high-resolution synthetic fixture keeps wire pixel thickness constant to
match the thin-wire requirement. Model smoke tests skip only when their local
cache is absent; both models were present for local validation.

Real-run QA is computed on the actual saved quality-95 JPEG. The original-photo
surrounding ring excludes the mask, a 3px safety band and protected anatomy,
extending 12px from the mask. Mean output luminance must lie within the ring mean
+/-max(30, 2*ring standard deviation), capped at 245 when ring mean is below 230.
Unnaturally near-white mean/fraction adds a separate flag. At most **12% of the
landmark eye region** may be masked; thresholds are fixed, not adjusted to accept
a run. `report.json` records means, spread, plausible bounds, near-white fraction,
eye-mask fraction, outlier flags and overall QA status. An outlier causes a
nonzero exit after diagnostic outputs are written. Empty masks have null
luminance statistics. These checks catch gross errors but do not prove likeness
or complete frame removal; the lead must still review the private composites.

`npm run typecheck` passed. Repository-wide lint fails on an existing inaccessible
root `.pytest_cache`; default npm tests fail on sandbox `spawn EPERM` / web Vite
configuration loading. Prior direct source ESLint and avatar-core tests with
`--pool=threads` passed. No out-of-scope files were edited for those failures.
