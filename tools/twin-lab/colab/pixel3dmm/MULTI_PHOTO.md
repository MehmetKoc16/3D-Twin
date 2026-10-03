# Additional head photos

Select up to **12 files together**: required `front`, optional `left`, `right`,
`back`, and up to eight extras named exactly `extra_1` through `extra_8`.
Use `.png`, `.jpg` or `.jpeg`. Gaps in extra numbering are allowed; duplicate
views, unknown names and path components are rejected. Processing order is
front, left, right, back, then extras in numeric order, regardless of upload order.
Failed optional views are recorded with a warning/reason and omitted; included
views get contiguous frame indices. Front must preprocess successfully.

All images must show the same person. Different occasions, cameras, lighting
and expressions are supported through independent cameras and facial parameters.
Sharp neutral images help the mouth texture; smiling/angled views can still
contribute to geometry. Consistent eyeglasses are accepted, but frames and
reflections can bias fitting and photographed textures. The fitter does not
remove glasses, normalize lighting or reconstruct hair.

## Fitting and T4 budget

The [pinned tracker](https://github.com/SimonGiebenhain/pixel3dmm/blob/fcd1fa973c7715b02a8948dfc679dff53cf85924/src/pixel3dmm/tracking/tracker.py)
initializes each image separately, then optimizes shared shape with per-image
expression, jaw, eyelids, pose and camera embeddings. `is_discontinuous=True`
removes temporal coupling; `global_camera=False` gives independent intrinsics.
`num_views=1` is the upstream camera-stream dimension; uploaded photographs are
discontinuous frames, not synchronized camera streams.

Keep `MAX_FIT_BATCH_SIZE=1` on T4. The successful two-view defaults of 1500 online
iterations per view and a 1500-iteration global floor are preserved. Joint
iterations become `max(GLOBAL_ITERS, ceil(GLOBAL_ITERS*N/(2*batch)))`, with `N`
the usable view count. At defaults this gives about 750 expected joint updates
per image, rather than diluting a fixed 1500-step budget across more photographs.

| Usable views | Online steps | Joint steps | Work versus two views |
| --- | ---: | ---: | ---: |
| 2 | 3000 | 1500 | 1× |
| 7 | 10500 | 5250 | 3.5× |
| 9 | 13500 | 6750 | 4.5× |
| 12 | 18000 | 9000 | 6× |

The ratios are planning estimates from iteration counts, **not measured timings**.
For example, if a warm two-view T4 run took 20 minutes, allow roughly 90 minutes
for nine usable images or 120 for twelve. Use your own successful run as the
baseline; installation, model downloads, detector retries and notebook/browser
interactions add time. Skipped views reduce work. For a quick preview, try
`ITERS=500`, `GLOBAL_ITERS=500`, then inspect overlays before a full-quality run.

Preprocessing and online tracking are approximately linear in N. The global
budget now scales with N/batch. Prediction already uses batch one; render/gradient
memory follows fitting batch size. Cached images, masks, normals and UVs still
grow with N. The main float32 image/mask buffers are about 5 MiB/view at 256²,
with temporary concatenation roughly doubling those buffers; this excludes
models, UV-loss state, optimizer buffers, native CUDA allocations and larger
preprocessing images. The full peak cannot be established without a GPU run.

`fit_runtime.json` in the ZIP records effective budgets, per-view online time,
joint/preprocessing/tracking time, and peak PyTorch allocated/reserved MiB.
The notebook prints phase times and allocated MiB after fitting. PyTorch counters
exclude some native/driver allocations. Up to twelve views remain T4-unbenchmarked.

## Profile retry

The [pinned crop/landmark implementation](https://github.com/SimonGiebenhain/pixel3dmm/blob/fcd1fa973c7715b02a8948dfc679dff53cf85924/src/pixel3dmm/preprocessing/pipnet_utils.py)
uses FaceBoxes for both initial crop and landmark sub-crop. The latter ordinarily
requires score 0.99, stricter than the initial crop's 0.75 gate. That can reject
profiles, though the owner's original left-view failure is not proven to have
this cause. After a known detection failure or missing/invalid crop/landmarks,
one retry lowers only this landmark gate to 0.75. Partial generated artifacts
are removed to defeat upstream's populated-crop early return. Unexpected CUDA,
import and native-process failures still abort with diagnostics.

No flip is performed: orientation, landmark indices and camera convention remain
the same, and actual bounds refer to the original EXIF-oriented photograph.
Successful retries are annotated `cropMethod=faceboxes_landmark_0.75_retry` and
warn to inspect the overlay. The retry is a quality tradeoff, not a guaranteed
profile fix. It can still fail if FaceBoxes finds no face or landmarks are invalid.

## Exports and compatibility

Every accepted view, including `extra_N`, has `fitted_views/<view>.ply`,
`overlays/<view>.png`, camera/parameter entries, original `originalSizeWH` and
`cropBoundsYminYmaxXminXmax`. Cameras retain `dt-flame-head-cameras/1` and all
existing projection/matrix fields. Additive `expression` metadata includes:

- `magnitudeL2`, `magnitudeRMS`: norms of the 100 fitted expression coefficients.
- `neutralityRank`: 1 + the number of views with strictly lower L2 magnitude;
  ties share a rank, and lower is preferred within this fit.
- `mouthTexturePreference`: `1/(1+magnitudeL2)`, a heuristic weight.
- `smileProxyOnly`: true; coefficient magnitude mixes expressions and fit error
  and cannot classify a smile or reliably measure mouth openness.

The same metadata is in `parameters.json.viewMetadata`, with scalar metrics in
NPZ. Neutral mesh generation still resets expression, jaw and eyelids. Shared
shape consistency is checked across final per-view checkpoints.

The unchanged current `head/flame/flamehead/camera.py` texture consumer requires
front **and** right and ignores additional camera entries. Its existing inputs
remain compatible, but it does not yet bake from extras or use mouth preferences.
That consumer extension is outside this task's scope. This notebook prepares
geometry, correspondence metadata and alignment overlays, not a texture atlas.

Photos/derived outputs remain private to the VM session. The lead's retained
`head_fit.zip` stays in the separate VM result directory until the final privacy
cell. Model cache reuse, traceback filtering, Drive model-only reads and final
cleanup remain in place. Clear saved diagnostics before sharing the notebook.

## Validation

On 2026-10-03, 65 synthetic pytest cases passed with `bundle/.venv`, including
canonical names/order, optional skips, retry behavior, budgets/expression proxies
and additive-schema loading through the actual unchanged consumer. No owner
images were opened. The owner confirmed the previous two-view T4 run end to end;
the new larger fit, relaxed retry and expression metadata still need Colab QA.
