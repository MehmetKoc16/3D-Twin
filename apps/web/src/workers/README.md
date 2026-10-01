Avatar worker (comlink): loads the body assets, keeps the BodySolver, returns grounded positions and joints; the client coalesces requests (latest wins). Owner: Wave 2.

`setFixedShape(shape | null)`: while set, `solve()` returns a fixed body (macro variables + net modifier values, `fixedShape.ts`) instead of solving for the params; used by the realistic twin (features/twin). Results of that mode carry `fixedShape: true`.
