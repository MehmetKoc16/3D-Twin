import type { GarmentBinding } from './binding';
import { GarmentSections } from './centroid';

export interface GradeRing {
  planeY: number;
  deltaCircumferenceM: number;
}

export interface GradeGarmentOptions {
  outer?: boolean;
  /** Optional outer grading clearance in meters. Unclamped by default; layering resolves collisions. */
  minClearance?: number;
  /** Neighborhood search radius in meters for Laplacian smoothing (default 0.025 = 2.5 cm). */
  radius?: number;
  /** Number of Laplacian smoothing iterations (default 15). */
  iterations?: number;
  /** Weld tolerance in meters for coincident vertices (default 0.001 = 1 mm). */
  weldTolerance?: number;
}

/**
 * Circular-section approximation: a circumference change Δc shifts each vertex by Δc / 2π
 * along the horizontal ray from its bound body point's limb-section centroid. Rings interpolate
 * linearly by body height and hold their end values outside the given range. The offset is
 * clamped to leave at least 2 mm of projected clearance; this does not resolve mesh collisions.
 *
 * For outer garments (options.outer = true), displacement is smoothed over welded connections
 * (coincident vertices) and spatial neighbors (< radius) via multi-iteration Laplacian smoothing
 * without the 2 mm per-vertex clearance clamping that causes stacked parts (buttons, plackets,
 * flaps) and front opening edges to tear.
 */
export function gradeGarment(
  restPositions: ArrayLike<number>,
  binding: GarmentBinding,
  bodyPositions: ArrayLike<number>,
  rings: GradeRing[],
  out: Float32Array,
  sections: GarmentSections = new GarmentSections(bodyPositions),
  options?: GradeGarmentOptions,
): void {
  if (restPositions.length < binding.count * 3 || out.length < binding.count * 3)
    throw new Error('garment grade: position buffer too short');
  const sorted = [...rings].sort((a, b) => a.planeY - b.planeY);
  if (sorted.some((r) => !Number.isFinite(r.planeY) || !Number.isFinite(r.deltaCircumferenceM)))
    throw new Error('garment grade: invalid ring');

  const count = binding.count;
  if (count === 0) return;

  if (!options?.outer) {
    for (let v = 0; v < count; v++) {
      const o = v * 3;
      let bx = 0,
        by = 0,
        bz = 0;
      for (let j = 0; j < 3; j++) {
        const p = binding.indices[o + j]! * 3;
        const w = binding.weights[o + j]!;
        bx += w * bodyPositions[p]!;
        by += w * bodyPositions[p + 1]!;
        bz += w * bodyPositions[p + 2]!;
      }
      const { x: cx, z: cz } = sections.centroidAt(by, bx, bz);
      let dx = bx - cx,
        dz = bz - cz;
      let radius = Math.hypot(dx, dz);
      if (radius < 1e-9) {
        dx = restPositions[o]! - cx;
        dz = restPositions[o + 2]! - cz;
        radius = Math.hypot(dx, dz);
      }
      if (radius < 1e-9) {
        dx = 1;
        dz = 0;
        radius = 1;
      }
      dx /= radius;
      dz /= radius;
      let delta = 0;
      if (sorted.length) {
        let upper = 0;
        while (upper < sorted.length && sorted[upper]!.planeY < by) upper++;
        if (upper === 0) delta = sorted[0]!.deltaCircumferenceM;
        else if (upper === sorted.length) delta = sorted[sorted.length - 1]!.deltaCircumferenceM;
        else {
          const low = sorted[upper - 1]!;
          const high = sorted[upper]!;
          const t = high.planeY === low.planeY ? 1 : (by - low.planeY) / (high.planeY - low.planeY);
          delta = low.deltaCircumferenceM + t * (high.deltaCircumferenceM - low.deltaCircumferenceM);
        }
      }
      let shift = delta / (2 * Math.PI);
      const clearance = (restPositions[o]! - bx) * dx + (restPositions[o + 2]! - bz) * dz;
      shift = Math.max(shift, 0.002 - clearance);
      out[o] = restPositions[o]! + shift * dx;
      out[o + 1] = restPositions[o + 1]!;
      out[o + 2] = restPositions[o + 2]! + shift * dz;
    }
    return;
  }

  // Outer slot: preserve chart narrowing and stacked parts without a per-vertex body clamp,
  // then smooth the displacement over welds and spatial neighbours.
  const minClearance = options?.minClearance ?? -Infinity;
  const dispX = new Float32Array(count);
  const dispY = new Float32Array(count);
  const dispZ = new Float32Array(count);

  // A single 1 cm slice can contain only the front of a sparse body mesh, making its
  // centroid jump from the torso centre to the skin between adjacent garment rows.
  // Pool a +/-5 cm band before finding components; retain the limb separation.
  const pooledBody = new Float32Array(bodyPositions.length * 11);
  for (let sample = 0; sample < 11; sample++) {
    const offset = sample * bodyPositions.length;
    for (let p = 0; p < bodyPositions.length; p += 3) {
      pooledBody[offset + p] = bodyPositions[p]!;
      pooledBody[offset + p + 1] = bodyPositions[p + 1]! + (sample - 5) * 0.01;
      pooledBody[offset + p + 2] = bodyPositions[p + 2]!;
    }
  }
  const outerSections = new GarmentSections(pooledBody);

  for (let v = 0; v < count; v++) {
    const o = v * 3;
    let bx = 0,
      by = 0,
      bz = 0;
    for (let j = 0; j < 3; j++) {
      const p = binding.indices[o + j]! * 3;
      const w = binding.weights[o + j]!;
      bx += w * bodyPositions[p]!;
      by += w * bodyPositions[p + 1]!;
      bz += w * bodyPositions[p + 2]!;
    }
    const y = restPositions[o + 1]!;
    // Interpolate the pooled centres and use garment coordinates, so binding-triangle
    // changes cannot introduce steps along an otherwise smooth opening edge.
    const lowY = Math.floor(y * 100) / 100;
    const low = outerSections.centroidAt(lowY, restPositions[o]!, restPositions[o + 2]!);
    const high = outerSections.centroidAt(lowY + 0.01, restPositions[o]!, restPositions[o + 2]!);
    const t = (y - lowY) * 100;
    const cx = low.x + t * (high.x - low.x);
    const cz = low.z + t * (high.z - low.z);
    let dx = restPositions[o]! - cx,
      dz = restPositions[o + 2]! - cz;
    let radius = Math.hypot(dx, dz);
    if (radius < 1e-9) {
      dx = 1;
      dz = 0;
      radius = 1;
    }
    dx /= radius;
    dz /= radius;
    let delta = 0;
    if (sorted.length) {
      let upper = 0;
      while (upper < sorted.length && sorted[upper]!.planeY < by) upper++;
      if (upper === 0) delta = sorted[0]!.deltaCircumferenceM;
      else if (upper === sorted.length) delta = sorted[sorted.length - 1]!.deltaCircumferenceM;
      else {
        const low = sorted[upper - 1]!;
        const high = sorted[upper]!;
        const t = high.planeY === low.planeY ? 1 : (by - low.planeY) / (high.planeY - low.planeY);
        delta = low.deltaCircumferenceM + t * (high.deltaCircumferenceM - low.deltaCircumferenceM);
      }
    }
    let shift = delta / (2 * Math.PI);
    const clearance = (restPositions[o]! - bx) * dx + (restPositions[o + 2]! - bz) * dz;
    shift = Math.max(shift, minClearance - clearance);
    dispX[v] = shift * dx;
    dispY[v] = 0;
    dispZ[v] = shift * dz;
  }

  const searchRadius = options?.radius ?? 0.025;
  const radiusSq = searchRadius * searchRadius;
  const weldTol = options?.weldTolerance ?? 0.001;
  const weldTolSq = weldTol * weldTol;

  // Spatial hash grid over restPositions
  const cellSize = searchRadius;
  const grid = new Map<string, number[]>();
  for (let i = 0; i < count; i++) {
    const o = i * 3;
    const cx = Math.floor(restPositions[o]! / cellSize);
    const cy = Math.floor(restPositions[o + 1]! / cellSize);
    const cz = Math.floor(restPositions[o + 2]! / cellSize);
    const key = `${cx},${cy},${cz}`;
    const cell = grid.get(key);
    if (cell) cell.push(i);
    else grid.set(key, [i]);
  }

  // Disjoint set for weld components
  const parent = new Uint32Array(count);
  for (let i = 0; i < count; i++) parent[i] = i;
  function find(i: number): number {
    let root = i;
    while (parent[root] !== root) root = parent[root]!;
    let curr = i;
    while (curr !== root) {
      const next = parent[curr]!;
      parent[curr] = root;
      curr = next;
    }
    return root;
  }
  function union(i: number, j: number): void {
    const rootI = find(i);
    const rootJ = find(j);
    if (rootI !== rootJ) parent[rootI] = rootJ;
  }

  const adj: number[][] = Array.from({ length: count }, () => []);
  const adjWeights: number[][] = Array.from({ length: count }, () => []);

  for (let i = 0; i < count; i++) {
    const o = i * 3;
    const xi = restPositions[o]!;
    const yi = restPositions[o + 1]!;
    const zi = restPositions[o + 2]!;
    const cx = Math.floor(xi / cellSize);
    const cy = Math.floor(yi / cellSize);
    const cz = Math.floor(zi / cellSize);

    for (let gx = -1; gx <= 1; gx++) {
      for (let gy = -1; gy <= 1; gy++) {
        for (let gz = -1; gz <= 1; gz++) {
          const cell = grid.get(`${cx + gx},${cy + gy},${cz + gz}`);
          if (!cell) continue;
          for (let k = 0; k < cell.length; k++) {
            const j = cell[k]!;
            if (j <= i) continue;
            const jo = j * 3;
            const rx = restPositions[jo]! - xi;
            const ry = restPositions[jo + 1]! - yi;
            const rz = restPositions[jo + 2]! - zi;
            const dSq = rx * rx + ry * ry + rz * rz;
            if (dSq <= weldTolSq) {
              union(i, j);
            }
            if (dSq <= radiusSq) {
              const dist = Math.sqrt(dSq);
              const w = (1 - dist / searchRadius) / Math.max(dist, 0.001);
              if (w > 0) {
                adj[i]!.push(j);
                adjWeights[i]!.push(w);
                adj[j]!.push(i);
                adjWeights[j]!.push(w);
              }
            }
          }
        }
      }
    }
  }

  // Collect weld groups with > 1 vertices
  const groupMap = new Map<number, number[]>();
  for (let i = 0; i < count; i++) {
    const root = find(i);
    const list = groupMap.get(root);
    if (list) list.push(i);
    else groupMap.set(root, [i]);
  }
  const weldGroups: number[][] = [];
  for (const group of groupMap.values()) {
    if (group.length > 1) weldGroups.push(group);
  }

  // Initial synchronization of welded vertices
  for (let g = 0; g < weldGroups.length; g++) {
    const group = weldGroups[g]!;
    let sx = 0,
      sy = 0,
      sz = 0;
    for (let k = 0; k < group.length; k++) {
      const v = group[k]!;
      sx += dispX[v]!;
      sy += dispY[v]!;
      sz += dispZ[v]!;
    }
    sx /= group.length;
    sy /= group.length;
    sz /= group.length;
    for (let k = 0; k < group.length; k++) {
      const v = group[k]!;
      dispX[v] = sx;
      dispY[v] = sy;
      dispZ[v] = sz;
    }
  }

  // Multi-iteration Laplacian smoothing
  const iterations = options?.iterations ?? 15;
  let curDx = new Float32Array(dispX);
  let curDy = new Float32Array(dispY);
  let curDz = new Float32Array(dispZ);
  let nextDx = new Float32Array(count);
  let nextDy = new Float32Array(count);
  let nextDz = new Float32Array(count);

  for (let iter = 0; iter < iterations; iter++) {
    for (let i = 0; i < count; i++) {
      const neighbors = adj[i]!;
      const weights = adjWeights[i]!;
      if (neighbors.length === 0) {
        nextDx[i] = curDx[i]!;
        nextDy[i] = curDy[i]!;
        nextDz[i] = curDz[i]!;
        continue;
      }
      let sumW = 0;
      let nDx = 0;
      let nDy = 0;
      let nDz = 0;
      for (let k = 0; k < neighbors.length; k++) {
        const j = neighbors[k]!;
        const w = weights[k]!;
        sumW += w;
        nDx += w * curDx[j]!;
        nDy += w * curDy[j]!;
        nDz += w * curDz[j]!;
      }
      if (sumW < 1e-9) {
        nextDx[i] = curDx[i]!;
        nextDy[i] = curDy[i]!;
        nextDz[i] = curDz[i]!;
        continue;
      }
      const avgDx = nDx / sumW;
      const avgDy = nDy / sumW;
      const avgDz = nDz / sumW;
      nextDx[i] = 0.5 * curDx[i]! + 0.5 * avgDx;
      nextDy[i] = 0.5 * curDy[i]! + 0.5 * avgDy;
      nextDz[i] = 0.5 * curDz[i]! + 0.5 * avgDz;
    }

    // Re-synchronize weld groups
    for (let g = 0; g < weldGroups.length; g++) {
      const group = weldGroups[g]!;
      let gDx = 0,
        gDy = 0,
        gDz = 0;
      for (let k = 0; k < group.length; k++) {
        const v = group[k]!;
        gDx += nextDx[v]!;
        gDy += nextDy[v]!;
        gDz += nextDz[v]!;
      }
      gDx /= group.length;
      gDy /= group.length;
      gDz /= group.length;
      for (let k = 0; k < group.length; k++) {
        const v = group[k]!;
        nextDx[v] = gDx;
        nextDy[v] = gDy;
        nextDz[v] = gDz;
      }
    }

    const tmpX = curDx;
    curDx = nextDx;
    nextDx = tmpX;
    const tmpY = curDy;
    curDy = nextDy;
    nextDy = tmpY;
    const tmpZ = curDz;
    curDz = nextDz;
    nextDz = tmpZ;
  }

  // Write out positions
  for (let i = 0; i < count; i++) {
    const o = i * 3;
    out[o] = restPositions[o]! + curDx[i]!;
    out[o + 1] = restPositions[o + 1]! + curDy[i]!;
    out[o + 2] = restPositions[o + 2]! + curDz[i]!;
  }
}

export function gradeGarmentOuter(
  restPositions: ArrayLike<number>,
  binding: GarmentBinding,
  bodyPositions: ArrayLike<number>,
  rings: GradeRing[],
  out: Float32Array,
  sections: GarmentSections = new GarmentSections(bodyPositions),
  options?: Omit<GradeGarmentOptions, 'outer'>,
): void {
  gradeGarment(restPositions, binding, bodyPositions, rings, out, sections, {
    ...options,
    outer: true,
  });
}
