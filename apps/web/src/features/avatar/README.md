Avatar: R3F Avatar component and the avatar-core to three.js bridge (asset loading, worker results -> geometry, welded normals, rest-skeleton rebuild). Owner: Wave 2.

Realistic twin: `Avatar` creates a `TwinMode` (features/twin) next to the `WardrobeRig`. It owns the body parts (mounted only in standard mode) and, in twin mode, hides the MakeHuman body and shows the user's scan bound to the same skeleton; every solve result is forwarded to it (`onSolve`).
