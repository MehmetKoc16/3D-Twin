import type { GarmentBinding } from './binding';
import { GarmentSections } from './centroid';

export interface GradeRing {
  planeY: number;
  deltaCircumferenceM: number;
}

export interface GradeInnerSurface {
  /** Graded positions of the inner garment (3 floats per vertex). */
  positions: ArrayLike<number>;
  /** Optional triangle index buffer of the inner garment. */
  index?: ArrayLike<number>;
}

/** Default outer grading clearance over an inner top garment (9 mm). */
export const DEFAULT_OUTER_INNER_CLEARANCE_M = 0.009;

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
  /** Inner garment surface (e.g. inner top) worn under this garment. */
  innerSurface?: GradeInnerSurface | ArrayLike<number>;
  /** Clearance to maintain outside the inner surface in meters (default 0.009 = 9 mm). */
  innerClearance?: number;
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
          delta =
            low.deltaCircumferenceM + t * (high.deltaCircumferenceM - low.deltaCircumferenceM);
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

  const rawInner = options?.innerSurface;
  const innerPositions: ArrayLike<number> | undefined =
    rawInner && typeof rawInner === 'object' && 'positions' in rawInner
      ? (rawInner as GradeInnerSurface).positions
      : (rawInner as ArrayLike<number> | undefined);
  const innerClearance = options?.innerClearance ?? DEFAULT_OUTER_INNER_CLEARANCE_M;

  const hasInner = !!(innerPositions && innerPositions.length >= 3);
  let jacketMinY = Infinity;
  let jacketMaxY = -Infinity;
  if (hasInner) {
    for (let i = 1; i < count * 3; i += 3) {
      const y = restPositions[i]!;
      if (y < jacketMinY) jacketMinY = y;
      if (y > jacketMaxY) jacketMaxY = y;
    }
  }

  const SECTORS = 48;
  const minSlab = hasInner ? Math.floor(jacketMinY * 100) - 1 : 0;
  const maxSlab = hasInner ? Math.ceil(jacketMaxY * 100) + 1 : 0;
  const numSlabs = hasInner ? Math.max(1, maxSlab - minSlab + 1) : 0;

  /**
   * Inner-top radii (distance from the pooled section centre) of one 1 cm slab and one body
   * component, bucketed by angular sector. The radii are kept (not reduced to a per-sector
   * maximum): a sleeve of the inner top shares the torso centre where the pooled body sections
   * merge arm and torso, so a maximum would make every jacket vertex of that sector chase the
   * sleeve tip. Each jacket vertex instead only sees radii near its own radius.
   */
  interface SlabComp {
    cx: number;
    cz: number;
    /** Per angular sector: flat (radius, angle) pairs of the inner vertices. */
    sectors: number[][];
  }
  const slabComponents: SlabComp[][] = hasInner ? Array.from({ length: numSlabs }, () => []) : [];

  if (hasInner) {
    const innerCount = Math.floor(innerPositions!.length / 3);
    for (let i = 0; i < innerCount; i++) {
      const iy = innerPositions![i * 3 + 1]!;
      if (iy < jacketMinY - 0.02 || iy > jacketMaxY + 0.02) continue;
      const s = Math.floor(iy * 100) - minSlab;
      if (s < 0 || s >= numSlabs) continue;

      const ix = innerPositions![i * 3]!;
      const iz = innerPositions![i * 3 + 2]!;

      const lowY = Math.floor(iy * 100) / 100;
      const low = outerSections.centroidAt(lowY, ix, iz);
      const high = outerSections.centroidAt(lowY + 0.01, ix, iz);
      const t = (iy - lowY) * 100;
      const icx = low.x + t * (high.x - low.x);
      const icz = low.z + t * (high.z - low.z);

      const idx = ix - icx;
      const idz = iz - icz;
      const iRadius = Math.hypot(idx, idz);
      let iAngle = Math.atan2(idz, idx);
      if (iAngle < 0) iAngle += 2 * Math.PI;
      const sec = Math.floor((iAngle / (2 * Math.PI)) * SECTORS) % SECTORS;

      const comps = slabComponents[s]!;
      let comp: SlabComp | undefined;
      for (let c = 0; c < comps.length; c++) {
        const candidate = comps[c]!;
        if (Math.hypot(candidate.cx - low.x, candidate.cz - low.z) < 0.05) {
          comp = candidate;
          break;
        }
      }
      if (!comp) {
        comp = {
          cx: low.x,
          cz: low.z,
          sectors: Array.from({ length: SECTORS }, () => []),
        };
        comps.push(comp);
      }
      comp.sectors[sec]!.push(iRadius, iAngle);
    }
  }

  // Inner radii further outside the jacket vertex than this are ignored (a sleeve tip, the far
  // side of a sleeve...); the contribution fades out between the two distances so the limit has
  // no steps. The jacket's rest surface already sits around the inner top, so this window only
  // has to cover the inner top's thickness over the rest surface.
  const INNER_NEAR_M = 0.03;
  const INNER_FAR_M = 0.05;
  const INNER_OFF_SHIFT_M = -0.1;
  // Angular taper: the limit falls by this much per metre of arc between a jacket vertex and an
  // inner vertex (a 2 cm local bulge fades out over 8 cm of arc) and is ignored beyond the reach.
  const INNER_TAPER_PER_ARC_M = 0.25;
  const INNER_ARC_REACH_M = 0.07;

  /** Minimum radial shift keeping a jacket vertex of radius r outside the local inner top. */
  function getInnerLimit(
    slab: number,
    angle: number,
    lowX: number,
    lowZ: number,
    r: number,
  ): number {
    const sectorWidth = (2 * Math.PI) / SECTORS;
    const sec = Math.floor(angle / sectorWidth) % SECTORS;
    const reach = Math.min(
      SECTORS / 2,
      Math.ceil(INNER_ARC_REACH_M / Math.max(r * sectorWidth, 1e-3)) + 1,
    );
    let limit = -Infinity;
    for (let ds = -1; ds <= 1; ds++) {
      const s = slab + ds;
      if (s < 0 || s >= numSlabs) continue;
      const comps = slabComponents[s]!;
      let bestDist = Infinity;
      let matchedComp: SlabComp | undefined;
      for (let c = 0; c < comps.length; c++) {
        const candidate = comps[c]!;
        const d = Math.hypot(candidate.cx - lowX, candidate.cz - lowZ);
        if (d < 0.08 && d < bestDist) {
          bestDist = d;
          matchedComp = candidate;
        }
      }
      if (!matchedComp) continue;
      for (let dsec = -reach; dsec <= reach; dsec++) {
        const entries = matchedComp.sectors[(sec + dsec + SECTORS) % SECTORS]!;
        for (let k = 0; k < entries.length; k += 2) {
          let da = Math.abs(entries[k + 1]! - angle);
          if (da > Math.PI) da = 2 * Math.PI - da;
          const arc = da * r;
          if (arc > INNER_ARC_REACH_M) continue;
          // The taper lowers the limit with angular distance, so a local bulge of the inner top
          // widens the jacket gradually instead of in a step at the sector edge.
          const excess = entries[k]! - r - INNER_TAPER_PER_ARC_M * arc;
          let lim: number;
          if (excess <= INNER_NEAR_M) lim = excess + innerClearance;
          else if (excess >= INNER_FAR_M) continue;
          else {
            const fade = (excess - INNER_NEAR_M) / (INNER_FAR_M - INNER_NEAR_M);
            lim =
              INNER_NEAR_M +
              innerClearance +
              fade * (INNER_OFF_SHIFT_M - INNER_NEAR_M - innerClearance);
          }
          if (lim > limit) limit = lim;
        }
      }
    }
    return limit;
  }

  const innerLimits = hasInner ? new Float32Array(count) : null;
  const dirX = new Float32Array(count);
  const dirZ = new Float32Array(count);

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
    dirX[v] = dx;
    dirZ[v] = dz;

    let delta = 0;
    if (sorted.length) {
      let upper = 0;
      while (upper < sorted.length && sorted[upper]!.planeY < by) upper++;
      if (upper === 0) delta = sorted[0]!.deltaCircumferenceM;
      else if (upper === sorted.length) delta = sorted[sorted.length - 1]!.deltaCircumferenceM;
      else {
        const lowR = sorted[upper - 1]!;
        const highR = sorted[upper]!;
        const tr =
          highR.planeY === lowR.planeY ? 1 : (by - lowR.planeY) / (highR.planeY - lowR.planeY);
        delta =
          lowR.deltaCircumferenceM + tr * (highR.deltaCircumferenceM - lowR.deltaCircumferenceM);
      }
    }
    let shift = delta / (2 * Math.PI);
    const clearance = (restPositions[o]! - bx) * dx + (restPositions[o + 2]! - bz) * dz;
    shift = Math.max(shift, minClearance - clearance);

    if (hasInner && delta < 0) {
      let angle = Math.atan2(dz, dx);
      if (angle < 0) angle += 2 * Math.PI;
      const s = Math.floor(y * 100) - minSlab;
      const minShift = getInnerLimit(s, angle, low.x, low.z, radius);
      innerLimits![v] = minShift;
      if (minShift > shift) shift = minShift;
    } else if (innerLimits) {
      innerLimits[v] = -Infinity;
    }

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

  // For stacked parts within 2.5 cm, share the inner limit of the innermost layer
  if (innerLimits) {
    const propLimits = new Float32Array(innerLimits);
    for (let i = 0; i < count; i++) {
      const limI = innerLimits[i]!;
      const neighbors = adj[i]!;
      const diX = dirX[i]!;
      const diZ = dirZ[i]!;
      let maxLim = limI;
      for (let k = 0; k < neighbors.length; k++) {
        const j = neighbors[k]!;
        const limJ = innerLimits[j]!;
        if (limJ === -Infinity) continue;
        const dot = diX * dirX[j]! + diZ * dirZ[j]!;
        if (dot > 0.8 && limJ > maxLim) maxLim = limJ;
      }
      propLimits[i] = maxLim;
    }
    for (let i = 0; i < count; i++) {
      innerLimits[i] = propLimits[i]!;
      const lim = innerLimits[i]!;
      if (lim !== -Infinity) {
        const curShift = dispX[i]! * dirX[i]! + dispZ[i]! * dirZ[i]!;
        if (curShift < lim) {
          dispX[i] = lim * dirX[i]!;
          dispZ[i] = lim * dirZ[i]!;
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

  // Post-smoothing: enforce inner surface limit per weld group so smoothing cannot pull vertices back inside
  if (innerLimits) {
    for (const group of groupMap.values()) {
      let maxDeficit = 0;
      for (let k = 0; k < group.length; k++) {
        const v = group[k]!;
        const limit = innerLimits[v]!;
        if (limit === -Infinity) continue;
        const radialDisp = curDx[v]! * dirX[v]! + curDz[v]! * dirZ[v]!;
        const deficit = limit - radialDisp;
        if (deficit > maxDeficit) maxDeficit = deficit;
      }
      if (maxDeficit > 0) {
        for (let k = 0; k < group.length; k++) {
          const v = group[k]!;
          curDx[v] = curDx[v]! + maxDeficit * dirX[v]!;
          curDz[v] = curDz[v]! + maxDeficit * dirZ[v]!;
        }
      }
    }
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
