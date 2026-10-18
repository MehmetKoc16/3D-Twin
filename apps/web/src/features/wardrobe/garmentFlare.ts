import type { Surface } from './garmentCollide';

/** Configuration options for flaring trouser hems radially over bulky shoes. */
export interface FlareOptions {
  /** Clearance margin outside shoe vertices, metres (default 0.005 m = 5 mm). */
  marginM?: number;
  /** Maximum radial outward flare distance beyond the trouser's natural radius, metres (default 0.015 m = 1.5 cm). */
  maxFlareM?: number;
  /** Height slab size for binning shoe vertices, metres (default 0.01 m = 1 cm). */
  slabHeightM?: number;
  /** Number of angular sectors around the leg axis (default 32). */
  sectorCount?: number;
  /** Distance below the trouser hem bottom to include shoe vertices, metres (default 0.03 m). */
  shoeMarginBelowHemM?: number;
  /** Vertical fade distance above the shoe top where flare fades to 0 via smoothstep, metres (default 0.12 m = 12 cm). */
  fadeBandM?: number;
  /** Height of the hem ring used to compute the leg axis centroid, metres (default 0.10 m = 10 cm). */
  hemBandM?: number;
  /** Minimum vertical penetration into the trouser tube for a shoe to trigger radial flare, metres (default 0.008 m = 8 mm). */
  minShoeReachM?: number;
}

/** Configuration options for hiding shoe vertices that sit above the trouser hem line. */
export interface ShoeAboveHemOptions {
  /** Radial reach beyond the trouser hem radius within which shoe vertices above the hem are hidden, metres (default 0.06 m = 6 cm). */
  reachM?: number;
  /** Vertical clearance above the local hem bottom Y before hiding begins, metres (default 0.003 m = 3 mm). */
  epsilonM?: number;
  /** Number of angular sectors around the leg axis (default 32). */
  sectorCount?: number;
  /** Height of the hem ring used to compute the leg axis centroid and hem radius, metres (default 0.10 m = 10 cm). */
  hemBandM?: number;
}

/** Minimum trouser vertex count per leg required to form a cylindrical tube around a leg axis. */
const MIN_LEG_VERTICES = 8;
/** Default radial reach beyond the trouser hem radius for hiding shoe tongue / lace vertices above the hem, metres (6 cm). */
const DEFAULT_HEM_HIDE_REACH_M = 0.06;
/** Default vertical clearance above the local hem bottom Y before a shoe vertex is hidden, metres (3 mm). */
const DEFAULT_HEM_HIDE_EPSILON_M = 0.003;
/** Maximum single-sector upward jump above its immediate neighbours before treating a sector's min-Y as a missed-ring spike, metres (1 cm). */
const HEM_SECTOR_SPIKE_M = 0.01;
/** Minimum vertical penetration into the trouser tube for a shoe to trigger radial flare, metres (8 mm). */
const DEFAULT_MIN_SHOE_REACH_M = 0.008;
/** Default vertical fade distance above the shoe top where flare fades to 0 via smoothstep, metres (12 cm). */
const DEFAULT_FADE_BAND_M = 0.12;
/** Default maximum radial outward flare distance beyond the trouser's natural radius, metres (1.5 cm). */
const DEFAULT_MAX_FLARE_M = 0.015;

interface LegPartition {
  shoeIndices: number[];
  trouserIndices: number[];
  trouserMinY: number;
  shoeMinY: number;
  shoeMaxY: number;
  axisX: number;
  axisZ: number;
}

/** Maps a radial offset `(dx, dz)` from the leg axis to an angular sector in `[0, sectorCount)`. */
function sectorIndex(dx: number, dz: number, sectorCount: number): number {
  let angle = Math.atan2(dz, dx);
  if (angle < 0) angle += 2 * Math.PI;
  return Math.floor((angle / (2 * Math.PI)) * sectorCount) % sectorCount;
}

/**
 * Fills unpopulated sectors (`populated[k] === 0`) in a 1D circular ring array by linear angular interpolation
 * between the nearest populated sectors on either side.
 */
function fillEmptySectors(
  values: Float32Array,
  populated: Uint8Array,
  sectorCount: number,
): void {
  let anyPopulated = false;
  for (let k = 0; k < sectorCount; k++) {
    if (populated[k] === 1) {
      anyPopulated = true;
      break;
    }
  }
  if (!anyPopulated) return;

  const snap = values.slice();
  for (let k = 0; k < sectorCount; k++) {
    if (populated[k] === 1) continue;
    let dLeft = 1;
    while (dLeft < sectorCount && populated[(k - dLeft + sectorCount) % sectorCount] === 0) {
      dLeft++;
    }
    let dRight = 1;
    while (dRight < sectorCount && populated[(k + dRight) % sectorCount] === 0) {
      dRight++;
    }
    const leftVal = snap[(k - dLeft + sectorCount) % sectorCount]!;
    const rightVal = snap[(k + dRight) % sectorCount]!;
    const t = dLeft / (dLeft + dRight);
    values[k] = leftVal + t * (rightVal - leftVal);
  }
}

/**
 * Splits trouser and relevant shoe vertices into per-leg partitions (1 or 2 legs via 2-means XZ clustering on the
 * shoes when both trouser span and shoe separation indicate two legs) and computes each leg's hem-band XZ axis.
 */
function partitionLegs(
  trouserPositions: ArrayLike<number>,
  trouserVertexCount: number,
  shoePositions: ArrayLike<number>,
  hemBandM: number,
  shoeMarginBelowHemM: number,
): LegPartition[] {
  const safeTrouserCount = Math.min(trouserVertexCount, Math.floor(trouserPositions.length / 3));
  if (safeTrouserCount < MIN_LEG_VERTICES || shoePositions.length < 3) return [];

  let trouserMinY = Infinity;
  for (let i = 0; i < safeTrouserCount; i++) {
    const y = trouserPositions[i * 3 + 1]!;
    if (y < trouserMinY) trouserMinY = y;
  }
  if (!Number.isFinite(trouserMinY)) return [];

  const shoeCutoffY = trouserMinY - shoeMarginBelowHemM;
  const shoeVertexCount = Math.floor(shoePositions.length / 3);

  const relevantShoeIndices: number[] = [];
  let shoeOverallMaxY = -Infinity;
  for (let v = 0; v < shoeVertexCount; v++) {
    const y = shoePositions[v * 3 + 1]!;
    if (y >= shoeCutoffY) {
      relevantShoeIndices.push(v);
      if (y > shoeOverallMaxY) shoeOverallMaxY = y;
    }
  }

  // If no shoe vertices exist above the cutoff, or none reach above the trouser hem, no leg interaction occurs.
  if (relevantShoeIndices.length === 0 || shoeOverallMaxY < trouserMinY) {
    return [];
  }

  let minTx = Infinity;
  let maxTx = -Infinity;
  let minTz = Infinity;
  let maxTz = -Infinity;
  for (let v = 0; v < safeTrouserCount; v++) {
    const y = trouserPositions[v * 3 + 1]!;
    if (y <= trouserMinY + hemBandM) {
      const x = trouserPositions[v * 3]!;
      const z = trouserPositions[v * 3 + 2]!;
      if (x < minTx) minTx = x;
      if (x > maxTx) maxTx = x;
      if (z < minTz) minTz = z;
      if (z > maxTz) maxTz = z;
    }
  }
  const trouserSpan = Math.max(maxTx - minTx, maxTz - minTz);

  // 2-means clustering in XZ to identify separate left and right shoes.
  let sumSx = 0;
  let sumSz = 0;
  for (const v of relevantShoeIndices) {
    sumSx += shoePositions[v * 3]!;
    sumSz += shoePositions[v * 3 + 2]!;
  }
  const meanSx = sumSx / relevantShoeIndices.length;
  const meanSz = sumSz / relevantShoeIndices.length;

  let bestD2 = -1;
  let seed0 = relevantShoeIndices[0]!;
  for (const v of relevantShoeIndices) {
    const dx = shoePositions[v * 3]! - meanSx;
    const dz = shoePositions[v * 3 + 2]! - meanSz;
    const d2 = dx * dx + dz * dz;
    if (d2 > bestD2) {
      bestD2 = d2;
      seed0 = v;
    }
  }

  bestD2 = -1;
  let seed1 = relevantShoeIndices[0]!;
  const s0x = shoePositions[seed0 * 3]!;
  const s0z = shoePositions[seed0 * 3 + 2]!;
  for (const v of relevantShoeIndices) {
    const dx = shoePositions[v * 3]! - s0x;
    const dz = shoePositions[v * 3 + 2]! - s0z;
    const d2 = dx * dx + dz * dz;
    if (d2 > bestD2) {
      bestD2 = d2;
      seed1 = v;
    }
  }

  let c0x = shoePositions[seed0 * 3]!;
  let c0z = shoePositions[seed0 * 3 + 2]!;
  let c1x = shoePositions[seed1 * 3]!;
  let c1z = shoePositions[seed1 * 3 + 2]!;

  for (let iter = 0; iter < 5; iter++) {
    let count0 = 0;
    let count1 = 0;
    let sum0x = 0;
    let sum0z = 0;
    let sum1x = 0;
    let sum1z = 0;
    for (const v of relevantShoeIndices) {
      const x = shoePositions[v * 3]!;
      const z = shoePositions[v * 3 + 2]!;
      const d0 = (x - c0x) ** 2 + (z - c0z) ** 2;
      const d1 = (x - c1x) ** 2 + (z - c1z) ** 2;
      if (d0 <= d1) {
        count0++;
        sum0x += x;
        sum0z += z;
      } else {
        count1++;
        sum1x += x;
        sum1z += z;
      }
    }
    if (count0 > 0) {
      c0x = sum0x / count0;
      c0z = sum0z / count0;
    }
    if (count1 > 0) {
      c1x = sum1x / count1;
      c1z = sum1z / count1;
    }
  }

  const shoeDist = Math.hypot(c1x - c0x, c1z - c0z);
  const isTwoLegs = trouserSpan > 0.18 && shoeDist > 0.12;

  const legShoeIndices: number[][] = [];
  const legTrouserIndices: number[][] = [];

  if (isTwoLegs) {
    legShoeIndices.push([], []);
    legTrouserIndices.push([], []);
    const midX = (c0x + c1x) / 2;
    const midZ = (c0z + c1z) / 2;
    const dirX = c1x - c0x;
    const dirZ = c1z - c0z;

    for (const v of relevantShoeIndices) {
      const x = shoePositions[v * 3]!;
      const z = shoePositions[v * 3 + 2]!;
      const side = (x - midX) * dirX + (z - midZ) * dirZ;
      legShoeIndices[side < 0 ? 0 : 1]!.push(v);
    }

    for (let v = 0; v < safeTrouserCount; v++) {
      const x = trouserPositions[v * 3]!;
      const z = trouserPositions[v * 3 + 2]!;
      const side = (x - midX) * dirX + (z - midZ) * dirZ;
      legTrouserIndices[side < 0 ? 0 : 1]!.push(v);
    }
  } else {
    legShoeIndices.push(relevantShoeIndices);
    const allTrouser: number[] = new Array(safeTrouserCount);
    for (let i = 0; i < safeTrouserCount; i++) allTrouser[i] = i;
    legTrouserIndices.push(allTrouser);
  }

  const legs: LegPartition[] = [];
  for (let leg = 0; leg < legShoeIndices.length; leg++) {
    const sIndices = legShoeIndices[leg]!;
    const tIndices = legTrouserIndices[leg]!;
    if (sIndices.length === 0 || tIndices.length < MIN_LEG_VERTICES) continue;

    let legTrouserMinY = Infinity;
    for (const v of tIndices) {
      const y = trouserPositions[v * 3 + 1]!;
      if (y < legTrouserMinY) legTrouserMinY = y;
    }
    if (!Number.isFinite(legTrouserMinY)) continue;

    let legShoeMinY = Infinity;
    let legShoeMaxY = -Infinity;
    for (const v of sIndices) {
      const y = shoePositions[v * 3 + 1]!;
      if (y < legShoeMinY) legShoeMinY = y;
      if (y > legShoeMaxY) legShoeMaxY = y;
    }
    if (legShoeMaxY < legTrouserMinY) continue;

    // Leg axis: vertical line through the XZ centroid of this leg's hem ring.
    let sumAx = 0;
    let sumAz = 0;
    let hemCount = 0;
    for (const v of tIndices) {
      const y = trouserPositions[v * 3 + 1]!;
      if (y <= legTrouserMinY + hemBandM) {
        sumAx += trouserPositions[v * 3]!;
        sumAz += trouserPositions[v * 3 + 2]!;
        hemCount++;
      }
    }
    const axisX = hemCount > 0 ? sumAx / hemCount : 0;
    const axisZ = hemCount > 0 ? sumAz / hemCount : 0;

    legs.push({
      shoeIndices: sIndices,
      trouserIndices: tIndices,
      trouserMinY: legTrouserMinY,
      shoeMinY: legShoeMinY,
      shoeMaxY: legShoeMaxY,
      axisX,
      axisZ,
    });
  }

  return legs;
}

/**
 * Fills empty (zero) bins in a `slabCount x sectorCount` trouser radius grid from neighbouring sectors and slabs:
 * 1. Local circular sector fill (±2 sectors) within each slab for coarse rings (e.g. 8-16 segments).
 * 2. Vertical fill/interpolation across slabs per sector so slabs between mesh rings inherit the leg profile.
 * 3. Full circular sector fill within each slab for any remaining empty sectors.
 */
function fillEmptyTrouserBins(
  rTrouser: Float32Array,
  maxSlab: number,
  sectorCount: number,
): void {
  const localSnap = rTrouser.slice();
  for (let s = 0; s <= maxSlab; s++) {
    const offset = s * sectorCount;
    for (let k = 0; k < sectorCount; k++) {
      if (localSnap[offset + k]! > 0) continue;
      for (let d = 1; d <= 2; d++) {
        const left = localSnap[offset + ((k - d + sectorCount) % sectorCount)]!;
        const right = localSnap[offset + ((k + d) % sectorCount)]!;
        if (left > 0 && right > 0) {
          rTrouser[offset + k] = 0.5 * (left + right);
          break;
        } else if (left > 0 || right > 0) {
          rTrouser[offset + k] = Math.max(left, right);
          break;
        }
      }
    }
  }

  for (let k = 0; k < sectorCount; k++) {
    for (let s = 0; s <= maxSlab; s++) {
      if (rTrouser[s * sectorCount + k]! > 0) continue;
      let sBelow = s - 1;
      while (sBelow >= 0 && rTrouser[sBelow * sectorCount + k]! === 0) sBelow--;
      let sAbove = s + 1;
      while (sAbove <= maxSlab && rTrouser[sAbove * sectorCount + k]! === 0) sAbove++;
      if (sBelow >= 0 && sAbove <= maxSlab) {
        const rBelow = rTrouser[sBelow * sectorCount + k]!;
        const rAbove = rTrouser[sAbove * sectorCount + k]!;
        const t = (s - sBelow) / (sAbove - sBelow);
        rTrouser[s * sectorCount + k] = rBelow + t * (rAbove - rBelow);
      } else if (sBelow >= 0) {
        rTrouser[s * sectorCount + k] = rTrouser[sBelow * sectorCount + k]!;
      } else if (sAbove <= maxSlab) {
        rTrouser[s * sectorCount + k] = rTrouser[sAbove * sectorCount + k]!;
      }
    }
  }

  const fullSnap = rTrouser.slice();
  const halfSectors = Math.floor(sectorCount / 2);
  for (let s = 0; s <= maxSlab; s++) {
    const offset = s * sectorCount;
    for (let k = 0; k < sectorCount; k++) {
      if (fullSnap[offset + k]! > 0) continue;
      for (let d = 1; d <= halfSectors; d++) {
        const left = fullSnap[offset + ((k - d + sectorCount) % sectorCount)]!;
        const right = fullSnap[offset + ((k + d) % sectorCount)]!;
        if (left > 0 && right > 0) {
          rTrouser[offset + k] = 0.5 * (left + right);
          break;
        } else if (left > 0 || right > 0) {
          rTrouser[offset + k] = Math.max(left, right);
          break;
        }
      }
    }
  }
}

/**
 * Computes the minimum trouser hem bottom Y per angular sector around the leg axis,
 * filling empty sectors and bridging isolated sector spikes.
 */
function computeSectorHemBottomY(
  trouserPositions: ArrayLike<number>,
  tIndices: readonly number[],
  axisX: number,
  axisZ: number,
  sectorCount: number,
): Float32Array {
  const hemBottomY = new Float32Array(sectorCount);
  hemBottomY.fill(Infinity);
  const yPopulated = new Uint8Array(sectorCount);

  for (const v of tIndices) {
    const dx = trouserPositions[v * 3]! - axisX;
    const dz = trouserPositions[v * 3 + 2]! - axisZ;
    const sector = sectorIndex(dx, dz, sectorCount);
    const y = trouserPositions[v * 3 + 1]!;
    yPopulated[sector] = 1;
    if (y < hemBottomY[sector]!) hemBottomY[sector] = y;
  }
  fillEmptySectors(hemBottomY, yPopulated, sectorCount);

  // Bridge isolated sector spikes where a sector missed the bottom hem ring and only caught a higher ring.
  const ySnap = hemBottomY.slice();
  for (let k = 0; k < sectorCount; k++) {
    const left = ySnap[(k - 1 + sectorCount) % sectorCount]!;
    const right = ySnap[(k + 1) % sectorCount]!;
    const neighborAvg = 0.5 * (left + right);
    if (ySnap[k]! > neighborAvg + HEM_SECTOR_SPIKE_M) {
      hemBottomY[k] = neighborAvg;
    }
  }

  return hemBottomY;
}

/**
 * Flares trouser hems radially outward over bulky shoes or sneakers so the fabric drapes smoothly
 * as a gentle cone instead of creating a horizontal lip or wide bell shape.
 *
 * Per leg (split into left/right by the midline between shoe centroids, or single leg if unseparated):
 * 1. Computes the vertical leg axis through the XZ centroid of the hem ring (lowest ~10 cm).
 * 2. Measures the trouser's own natural radius `r_trouser(slab, sector)` before moving anything,
 *    filling empty bins from neighbouring sectors and slabs.
 * 3. Identifies sectors where the shoe reaches at least `minShoeReachM` (8 mm) into the trouser tube.
 *    If the shoe reaches less than 8 mm in a sector, it produces no push there.
 * 4. Bins shoe vertices that lie within `r_trouser(slab, sector) + maxFlareM`, ignoring distant parts (toe box, sole rim).
 * 5. Propagates maximum radial clearance downwards per sector so the tube never narrows going down.
 * 6. Smooths the displacement profile around the ring with a light circular blur (no whole-ring unification).
 * 7. Displaces trouser vertices radially in XZ away from the leg axis (capped at `maxFlareM`),
 *    applied as a long vertical ramp: full push at the contact height, smoothly decreasing to 0 over 12 cm above it.
 *
 * @param positions Graded trouser vertex positions (3 floats per vertex, modified in-place).
 * @param vertexCount Number of trouser vertices.
 * @param shoe Graded surface of worn shoes.
 * @param options Optional tuning parameters.
 * @returns Number of vertices whose positions were moved.
 */
export function flareOverShoes(
  positions: Float32Array,
  vertexCount: number,
  shoe: Surface,
  options: FlareOptions = {},
): number {
  const shoeMarginBelowHem = options.shoeMarginBelowHemM ?? 0.03;
  const hemBandM = options.hemBandM ?? 0.10;
  const legs = partitionLegs(positions, vertexCount, shoe.positions, hemBandM, shoeMarginBelowHem);
  if (legs.length === 0) return 0;

  let movedCount = 0;
  const slabHeight = options.slabHeightM ?? 0.01;
  const sectorCount = options.sectorCount ?? 32;
  const marginM = options.marginM ?? 0.005;
  const maxFlareM = options.maxFlareM ?? DEFAULT_MAX_FLARE_M;
  const fadeBandM = options.fadeBandM ?? DEFAULT_FADE_BAND_M;
  const minShoeReachM = options.minShoeReachM ?? DEFAULT_MIN_SHOE_REACH_M;

  for (const leg of legs) {
    const {
      shoeIndices: sIndices,
      trouserIndices: tIndices,
      trouserMinY: legTrouserMinY,
      shoeMinY: legShoeMinY,
      shoeMaxY: legShoeMaxY,
      axisX,
      axisZ,
    } = leg;
    if (legShoeMinY > legTrouserMinY + hemBandM) continue;

    const hemBottomY = legTrouserMinY;
    const maxSlab = Math.max(0, Math.floor((legShoeMaxY - hemBottomY) / slabHeight));
    const slabCount = maxSlab + 1;

    // 1. Measure the trouser's own natural radius per (slab, sector) before moving anything.
    const rTrouser = new Float32Array(slabCount * sectorCount);
    for (const v of tIndices) {
      const y = positions[v * 3 + 1]!;
      const rawSlab = Math.floor((y - hemBottomY) / slabHeight);
      if (rawSlab < 0 || rawSlab > maxSlab) continue;
      const dx = positions[v * 3]! - axisX;
      const dz = positions[v * 3 + 2]! - axisZ;
      const r = Math.hypot(dx, dz);
      const sector = sectorIndex(dx, dz, sectorCount);
      const idx = rawSlab * sectorCount + sector;
      if (r > rTrouser[idx]!) rTrouser[idx] = r;
    }
    fillEmptyTrouserBins(rTrouser, maxSlab, sectorCount);

    // 2. Measure local hem bottom Y per sector and shoe penetration into the tube per sector.
    const sectorHemBottomY = computeSectorHemBottomY(
      positions,
      tIndices,
      axisX,
      axisZ,
      sectorCount,
    );

    const sectorShoeMaxY = new Float32Array(sectorCount);
    sectorShoeMaxY.fill(-Infinity);

    for (const v of sIndices) {
      const x = shoe.positions[v * 3]!;
      const y = shoe.positions[v * 3 + 1]!;
      const z = shoe.positions[v * 3 + 2]!;
      const dx = x - axisX;
      const dz = z - axisZ;
      const r = Math.hypot(dx, dz);
      const sector = sectorIndex(dx, dz, sectorCount);
      let slab = Math.floor((y - hemBottomY) / slabHeight);
      if (slab < 0) slab = 0;
      else if (slab > maxSlab) slab = maxSlab;
      const idx = slab * sectorCount + sector;
      if (r > rTrouser[idx]! + maxFlareM) continue;
      if (y > sectorShoeMaxY[sector]!) sectorShoeMaxY[sector] = y;
    }

    // 3. Bin shoe vertices only in sectors where the shoe reaches at least minShoeReachM (8 mm) into the tube.
    const rShoe = new Float32Array(slabCount * sectorCount);
    let contributingShoeMaxY = -Infinity;

    for (const v of sIndices) {
      const x = shoe.positions[v * 3]!;
      const y = shoe.positions[v * 3 + 1]!;
      const z = shoe.positions[v * 3 + 2]!;
      const dx = x - axisX;
      const dz = z - axisZ;
      const r = Math.hypot(dx, dz);
      const sector = sectorIndex(dx, dz, sectorCount);

      const reach = sectorShoeMaxY[sector]! - sectorHemBottomY[sector]!;
      if (reach < minShoeReachM) continue;

      let slab = Math.floor((y - hemBottomY) / slabHeight);
      if (slab < 0) slab = 0;
      else if (slab > maxSlab) slab = maxSlab;
      const idx = slab * sectorCount + sector;
      if (r > rTrouser[idx]! + maxFlareM) continue;
      if (r > rShoe[idx]!) rShoe[idx] = r;
      if (y > contributingShoeMaxY) contributingShoeMaxY = y;
    }

    if (contributingShoeMaxY < legTrouserMinY) continue;

    // Contact height per sector for the ramp transition.
    const contactY = new Float32Array(sectorCount);
    const contactPopulated = new Uint8Array(sectorCount);
    for (let k = 0; k < sectorCount; k++) {
      if (sectorShoeMaxY[k]! > -Infinity) {
        contactY[k] = sectorShoeMaxY[k]!;
        contactPopulated[k] = 1;
      }
    }
    fillEmptySectors(contactY, contactPopulated, sectorCount);

    // 4. Downward propagation: a ring must never narrow as it drapes down to the hem.
    for (let s = maxSlab - 1; s >= 0; s--) {
      const currOffset = s * sectorCount;
      const aboveOffset = (s + 1) * sectorCount;
      for (let k = 0; k < sectorCount; k++) {
        const higher = rShoe[aboveOffset + k]!;
        if (higher > rShoe[currOffset + k]!) {
          rShoe[currOffset + k] = higher;
        }
      }
    }

    // 5. Compute push per slab and apply a light circular blur (no whole-ring unification).
    const slabPush = new Float32Array(slabCount * sectorCount);
    const rawPush = new Float32Array(sectorCount);

    for (let s = 0; s <= maxSlab; s++) {
      const slabOffset = s * sectorCount;
      let anyPush = false;

      for (let k = 0; k < sectorCount; k++) {
        const shoeR = rShoe[slabOffset + k]!;
        const trouserR = rTrouser[slabOffset + k]!;
        if (shoeR > 0 && shoeR + marginM > trouserR) {
          const push = Math.min(maxFlareM, shoeR + marginM - trouserR);
          rawPush[k] = push;
          anyPush = true;
        } else {
          rawPush[k] = 0;
        }
      }

      if (!anyPush) continue;

      // 5-tap circular blur ([1, 2, 4, 2, 1] / 10): keeps the push localized to sectors needing it.
      for (let k = 0; k < sectorCount; k++) {
        const km2 = rawPush[(k - 2 + sectorCount) % sectorCount]!;
        const km1 = rawPush[(k - 1 + sectorCount) % sectorCount]!;
        const k0 = rawPush[k]!;
        const kp1 = rawPush[(k + 1) % sectorCount]!;
        const kp2 = rawPush[(k + 2) % sectorCount]!;
        const blurred = (km2 + 2 * km1 + 4 * k0 + 2 * kp1 + kp2) / 10;
        slabPush[slabOffset + k] = Math.min(maxFlareM, Math.max(k0, blurred));
      }
    }

    let maxCutoffY = -Infinity;
    for (let k = 0; k < sectorCount; k++) {
      const cutoff = contactY[k]! + fadeBandM;
      if (cutoff > maxCutoffY) maxCutoffY = cutoff;
    }

    // 6. Displace vertices radially in XZ away from the leg axis, clamped to maxFlareM.
    for (const v of tIndices) {
      const y = positions[v * 3 + 1]!;
      if (y >= maxCutoffY) continue;

      const x = positions[v * 3]!;
      const z = positions[v * 3 + 2]!;
      const dx = x - axisX;
      const dz = z - axisZ;
      const r = Math.hypot(dx, dz);
      if (r < 1e-6) continue;

      const sector = sectorIndex(dx, dz, sectorCount);
      const cY = contactY[sector]!;
      if (y >= cY + fadeBandM) continue;

      let push: number;
      if (y <= cY) {
        const slab = Math.min(maxSlab, Math.max(0, Math.floor((y - hemBottomY) / slabHeight)));
        push = slabPush[slab * sectorCount + sector]!;
      } else {
        const cSlab = Math.min(maxSlab, Math.max(0, Math.floor((cY - hemBottomY) / slabHeight)));
        push = slabPush[cSlab * sectorCount + sector]!;
        const t = (y - cY) / fadeBandM;
        const fade = 1 - t * t * (3 - 2 * t);
        push *= fade;
      }

      if (push > 1e-5) {
        const scale = (r + push) / r;
        positions[v * 3] = axisX + dx * scale;
        // Y coordinates are strictly unmodified.
        positions[v * 3 + 2] = axisZ + dz * scale;
        movedCount++;
      }
    }
  }

  return movedCount;
}

/**
 * Identifies shoe vertices that sit above the local trouser hem bottom line and within radial reach of the leg cuff,
 * so shoe tongues, lace loops, and ankle collars sticking out beyond the flared fabric can be hidden while preserving
 * the toe box and sole below the hem edge.
 *
 * @param shoe Graded shoe surface.
 * @param trouserPositions Graded trouser vertex positions (3 floats per vertex).
 * @param trouserVertexCount Number of trouser vertices.
 * @param options Optional tuning parameters.
 * @returns One byte (`0` or `1`) per shoe vertex: `1` when the vertex should be treated as covered/hidden.
 */
export function shoeVerticesAboveHem(
  shoe: Surface,
  trouserPositions: ArrayLike<number>,
  trouserVertexCount: number,
  options: ShoeAboveHemOptions = {},
): Uint8Array {
  const shoeVertexCount = Math.floor(shoe.positions.length / 3);
  const hidden = new Uint8Array(shoeVertexCount);
  if (trouserVertexCount < MIN_LEG_VERTICES || shoeVertexCount === 0) return hidden;

  const reachM = options.reachM ?? DEFAULT_HEM_HIDE_REACH_M;
  const epsilonM = options.epsilonM ?? DEFAULT_HEM_HIDE_EPSILON_M;
  const sectorCount = options.sectorCount ?? 32;
  const hemBandM = options.hemBandM ?? 0.10;
  const shoeMarginBelowHemM = 0.03;

  const legs = partitionLegs(
    trouserPositions,
    trouserVertexCount,
    shoe.positions,
    hemBandM,
    shoeMarginBelowHemM,
  );
  if (legs.length === 0) return hidden;

  for (const leg of legs) {
    const { shoeIndices: sIndices, trouserIndices: tIndices, axisX, axisZ } = leg;

    // 1. Compute local hem bottom Y per angular sector around the leg axis.
    const hemBottomY = computeSectorHemBottomY(
      trouserPositions,
      tIndices,
      axisX,
      axisZ,
      sectorCount,
    );

    // 2. Compute trouser radius near the hem in each sector, then fill empty sectors.
    const hemRadius = new Float32Array(sectorCount);
    const rPopulated = new Uint8Array(sectorCount);
    for (const v of tIndices) {
      const dx = trouserPositions[v * 3]! - axisX;
      const dz = trouserPositions[v * 3 + 2]! - axisZ;
      const sector = sectorIndex(dx, dz, sectorCount);
      const y = trouserPositions[v * 3 + 1]!;
      if (y <= hemBottomY[sector]! + hemBandM) {
        const r = Math.hypot(dx, dz);
        rPopulated[sector] = 1;
        if (r > hemRadius[sector]!) hemRadius[sector] = r;
      }
    }
    fillEmptySectors(hemRadius, rPopulated, sectorCount);

    // 3. Hide shoe vertices above the local hem bottom Y (+ epsilon) and within hemRadius + reachM.
    for (const v of sIndices) {
      const x = shoe.positions[v * 3]!;
      const y = shoe.positions[v * 3 + 1]!;
      const z = shoe.positions[v * 3 + 2]!;
      const dx = x - axisX;
      const dz = z - axisZ;
      const r = Math.hypot(dx, dz);
      const sector = sectorIndex(dx, dz, sectorCount);
      if (y > hemBottomY[sector]! + epsilonM && r <= hemRadius[sector]! + reachM) {
        hidden[v] = 1;
      }
    }
  }

  return hidden;
}
