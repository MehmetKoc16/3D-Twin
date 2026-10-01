# Twin stand-in fixture (NON-personal)

`rigged.glb`, `twin.json` and `mh2twin.bin` are the output of `tools/twin-lab/rig/rig_scan.py` run on a **stand-in
scan**: the CC0 MakeHuman base body (`apps/web/public/assets/body`) with another shape, another arm / leg pose, its own
UV layout and a procedurally generated texture (skin-toned gradient with a coloured grid). It is not a scan of a
person, nothing in it is derived from photos.

Regenerate (Windows, from `tools/twin-lab/rig`, see its README for the venv):

    .venv/Scripts/python make_standin.py --textured                                   # -> .cache/standin_textured/mesh.glb
    .venv/Scripts/python rig_scan.py .cache/standin_textured/mesh.glb .cache/standin_textured_out
    copy .cache/standin_textured_out/{rigged.glb,twin.json,mh2twin.bin} apps/web/e2e/fixtures/twin-standin/

Used by `e2e/twin.spec.ts` (picked through the twin panel's file input, like a user would) and by the unit test
`src/features/twin/twin.real.test.ts` (checks that the browser rebuilds the body the pipeline exported).
Licence: derived from the CC0 MakeHuman assets and generated code only.
