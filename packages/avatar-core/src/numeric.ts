/** Small numeric helpers shared by the solver: safeguarded 1-D root finding and a dense linear solve. */

export interface ScalarSolution {
  /** Best abscissa found (smallest |f|). */
  x: number;
  fx: number;
  evaluations: number;
  /** True when |fx| <= tol. False when the root is outside [lo, hi] or the budget ran out. */
  converged: boolean;
}

/**
 * Finds x in [lo, hi] with |f(x)| <= tol for a NON-DECREASING f. Starts at x0, extrapolates with secant steps
 * until the root is bracketed, then refines with the Illinois method (regula falsi with bisection safeguard).
 * If the root is outside the interval the bound with the smallest |f| is returned with converged = false.
 */
export function solveScalar(
  f: (x: number) => number,
  x0: number,
  lo: number,
  hi: number,
  tol: number,
  maxEval = 40,
): ScalarSolution {
  let evals = 0;
  let bestX = lo;
  let bestF = Infinity;
  const ev = (x: number): number => {
    evals++;
    const fx = f(x);
    if (Math.abs(fx) < Math.abs(bestF)) {
      bestX = x;
      bestF = fx;
    }
    return fx;
  };
  const clampX = (x: number): number => Math.min(hi, Math.max(lo, x));
  const result = (): ScalarSolution => ({
    x: bestX,
    fx: bestF,
    evaluations: evals,
    converged: Math.abs(bestF) <= tol,
  });

  let x1 = clampX(x0);
  let f1 = ev(x1);
  if (Math.abs(f1) <= tol) return result();

  const step = 0.05 * (hi - lo);
  let x2 = clampX(x1 + (f1 > 0 ? -step : step));
  if (x2 === x1) return result(); // pinned at a bound and f pushes outward
  let f2 = ev(x2);

  // Phase 1: extrapolate until the sign changes.
  while (evals < maxEval && Math.abs(f2) > tol && f1 * f2 > 0) {
    const slope = (f2 - f1) / (x2 - x1);
    let x3 =
      slope > 0 && Number.isFinite(slope)
        ? x2 - f2 / slope
        : x2 + (f2 > 0 ? -1 : 1) * 2 * Math.abs(x2 - x1);
    const maxJump = 8 * Math.max(Math.abs(x2 - x1), step);
    x3 = clampX(Math.min(x2 + maxJump, Math.max(x2 - maxJump, x3)));
    if (x3 === x2) return result(); // stuck at a bound: unreachable
    x1 = x2;
    f1 = f2;
    x2 = x3;
    f2 = ev(x2);
  }
  if (Math.abs(f2) <= tol || f1 * f2 > 0) return result();

  // Phase 2: Illinois on the bracket [x1, x2].
  let a = x1;
  let fa = f1;
  let b = x2;
  let fb = f2;
  let side = 0;
  while (evals < maxEval) {
    const denom = fb - fa;
    let xc = denom !== 0 ? (a * fb - b * fa) / denom : 0.5 * (a + b);
    if (!(xc > Math.min(a, b) && xc < Math.max(a, b))) xc = 0.5 * (a + b);
    const fc = ev(xc);
    if (Math.abs(fc) <= tol) break;
    if (fc * fb > 0) {
      b = xc;
      fb = fc;
      if (side === -1) fa *= 0.5;
      side = -1;
    } else {
      a = xc;
      fa = fc;
      if (side === 1) fb *= 0.5;
      side = 1;
    }
    if (Math.abs(b - a) <= 1e-14 * (1 + Math.abs(a))) break;
  }
  return result();
}

/**
 * Solves A x = b in place (Gaussian elimination, partial pivoting). `a` is row-major n x n and is destroyed;
 * on success `b` holds the solution. Returns false when the matrix is (numerically) singular.
 */
export function solveLinear(a: Float64Array, b: Float64Array, n: number): boolean {
  for (let col = 0; col < n; col++) {
    let piv = col;
    let pivAbs = Math.abs(a[col * n + col]!);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(a[r * n + col]!);
      if (v > pivAbs) {
        pivAbs = v;
        piv = r;
      }
    }
    if (!(pivAbs > 1e-300)) return false;
    if (piv !== col) {
      for (let c = 0; c < n; c++) {
        const t = a[col * n + c]!;
        a[col * n + c] = a[piv * n + c]!;
        a[piv * n + c] = t;
      }
      const tb = b[col]!;
      b[col] = b[piv]!;
      b[piv] = tb;
    }
    const d = a[col * n + col]!;
    for (let r = col + 1; r < n; r++) {
      const m = a[r * n + col]! / d;
      if (m === 0) continue;
      for (let c = col; c < n; c++) a[r * n + c] = a[r * n + c]! - m * a[col * n + c]!;
      b[r] = b[r]! - m * b[col]!;
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r]!;
    for (let c = r + 1; c < n; c++) s -= a[r * n + c]! * b[c]!;
    b[r] = s / a[r * n + r]!;
  }
  return true;
}
