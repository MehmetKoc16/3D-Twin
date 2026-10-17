import type { Surface } from './garmentCollide';

/** Configuration options for flaring trouser hems radially over bulky shoes. */
export interface FlareOptions {
  /** Clearance margin outside shoe vertices, metres (default 0.005 m = 5 mm). */
  marginM?: number;
  /** Maximum radial outward flare distance beyond the trouser's natural radius, metres (default 0.025 m = 2.5 cm). */
  maxFlareM?: number;
  /** Height slab size for binning shoe vertices, metres (default 0.01 m = 1 cm). */
  slabHeightM?: number;
  /** Number of angular sectors around the leg axis (default 32). */
  sectorCount?: number;
  /** Distance below the trouser hem bottom to include shoe vertices, metres (default 0.03 m). */
  shoeMarginBelowHemM?: number;
  /** Vertical fade distance above the shoe top where flare fades to 0 via smoothstep, metres (default 0.05 m = 5 cm). */
  fadeBandM?: number;
  /** Height of the hem ring used to compute the leg axis centroid, metres (default 0.10 m = 10 cm). */
  hemBandM?: number;
}

/** Minimum trouser vertex count per leg required to form a cylindrical tube around a leg axis. */
const MIN_LEG_VERTICES = 8;

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
 * Flares trouser hems radially outward over bulky shoes or sneakers so the fabric drapes smoothly
 * as a clean flared tube instead of creating jagged spikes or wide bell shapes.
 *
 * Per leg (split into left/right by the midline between shoe centroids, or single leg if unseparated):
 * 1. Computes the vertical leg axis through the XZ centroid of the hem ring (lowest ~10 cm).
 * 2. Measures the trouser's own natural radius `r_trouser(slab, sector)` before moving anything,
 *    filling empty bins from neighbouring sectors and slabs.
 * 3. Bins shoe vertices into `(slab, sector)` only when `r <= r_trouser(slab, sector) + maxFlareM`,
 *    ignoring distant parts of the shoe (toe box, sole rim).
 * 4. Propagates maximum radial clearance downwards so the tube never narrows going down.
 * 5. Smooths the displacement profile around the ring (circular max-filter + blur) to prevent spikes.
 * 6. Displaces trouser vertices radially in XZ away from the leg axis (clamped to `maxFlareM`),
 *    fading to 0 above the shoe top.
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
  if (vertexCount < MIN_LEG_VERTICES || shoe.positions.length < 3) return 0;

  let trouserMinY = Infinity;
  for (let i = 0; i < vertexCount; i++) {
    const y = positions[i * 3 + 1]!;
    if (y < trouserMinY) trouserMinY = y;
  }
  if (!Number.isFinite(trouserMinY)) return 0;

  const shoeMarginBelowHem = options.shoeMarginBelowHemM ?? 0.03;
  const shoeCutoffY = trouserMinY - shoeMarginBelowHem;
  const shoeVertexCount = Math.floor(shoe.positions.length / 3);

  const relevantShoeIndices: number[] = [];
  let shoeOverallMaxY = -Infinity;
  for (let v = 0; v < shoeVertexCount; v++) {
    const y = shoe.positions[v * 3 + 1]!;
    if (y >= shoeCutoffY) {
      relevantShoeIndices.push(v);
      if (y > shoeOverallMaxY) shoeOverallMaxY = y;
    }
  }

  // If no shoe vertices exist above the cutoff, or none reach above the trouser hem, no flare is needed.
  if (relevantShoeIndices.length === 0 || shoeOverallMaxY < trouserMinY) {
    return 0;
  }

  const hemBandM = options.hemBandM ?? 0.10;
  let minTx = Infinity;
  let maxTx = -Infinity;
  let minTz = Infinity;
  let maxTz = -Infinity;
  for (let v = 0; v < vertexCount; v++) {
    const y = positions[v * 3 + 1]!;
    if (y <= trouserMinY + hemBandM) {
      const x = positions[v * 3]!;
      const z = positions[v * 3 + 2]!;
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
    sumSx += shoe.positions[v * 3]!;
    sumSz += shoe.positions[v * 3 + 2]!;
  }
  const meanSx = sumSx / relevantShoeIndices.length;
  const meanSz = sumSz / relevantShoeIndices.length;

  let bestD2 = -1;
  let seed0 = relevantShoeIndices[0]!;
  for (const v of relevantShoeIndices) {
    const dx = shoe.positions[v * 3]! - meanSx;
    const dz = shoe.positions[v * 3 + 2]! - meanSz;
    const d2 = dx * dx + dz * dz;
    if (d2 > bestD2) {
      bestD2 = d2;
      seed0 = v;
    }
  }

  bestD2 = -1;
  let seed1 = relevantShoeIndices[0]!;
  const s0x = shoe.positions[seed0 * 3]!;
  const s0z = shoe.positions[seed0 * 3 + 2]!;
  for (const v of relevantShoeIndices) {
    const dx = shoe.positions[v * 3]! - s0x;
    const dz = shoe.positions[v * 3 + 2]! - s0z;
    const d2 = dx * dx + dz * dz;
    if (d2 > bestD2) {
      bestD2 = d2;
      seed1 = v;
    }
  }

  let c0x = shoe.positions[seed0 * 3]!;
  let c0z = shoe.positions[seed0 * 3 + 2]!;
  let c1x = shoe.positions[seed1 * 3]!;
  let c1z = shoe.positions[seed1 * 3 + 2]!;

  for (let iter = 0; iter < 5; iter++) {
    let count0 = 0;
    let count1 = 0;
    let sum0x = 0;
    let sum0z = 0;
    let sum1x = 0;
    let sum1z = 0;
    for (const v of relevantShoeIndices) {
      const x = shoe.positions[v * 3]!;
      const z = shoe.positions[v * 3 + 2]!;
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
      const x = shoe.positions[v * 3]!;
      const z = shoe.positions[v * 3 + 2]!;
      const side = (x - midX) * dirX + (z - midZ) * dirZ;
      legShoeIndices[side < 0 ? 0 : 1]!.push(v);
    }

    for (let v = 0; v < vertexCount; v++) {
      const x = positions[v * 3]!;
      const z = positions[v * 3 + 2]!;
      const side = (x - midX) * dirX + (z - midZ) * dirZ;
      legTrouserIndices[side < 0 ? 0 : 1]!.push(v);
    }
  } else {
    legShoeIndices.push(relevantShoeIndices);
    const allTrouser: number[] = new Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) allTrouser[i] = i;
    legTrouserIndices.push(allTrouser);
  }

  let movedCount = 0;
  const slabHeight = options.slabHeightM ?? 0.01;
  const sectorCount = options.sectorCount ?? 32;
  const marginM = options.marginM ?? 0.005;
  const maxFlareM = options.maxFlareM ?? 0.025;
  const fadeBandM = options.fadeBandM ?? 0.05;
  const maxFilterRadius = 2;

  for (let leg = 0; leg < legShoeIndices.length; leg++) {
    const sIndices = legShoeIndices[leg]!;
    const tIndices = legTrouserIndices[leg]!;
    if (sIndices.length === 0 || tIndices.length < MIN_LEG_VERTICES) continue;

    let legTrouserMinY = Infinity;
    for (const v of tIndices) {
      const y = positions[v * 3 + 1]!;
      if (y < legTrouserMinY) legTrouserMinY = y;
    }
    if (!Number.isFinite(legTrouserMinY)) continue;

    let legShoeMinY = Infinity;
    let legShoeMaxY = -Infinity;
    for (const v of sIndices) {
      const y = shoe.positions[v * 3 + 1]!;
      if (y < legShoeMinY) legShoeMinY = y;
      if (y > legShoeMaxY) legShoeMaxY = y;
    }
    if (legShoeMaxY < legTrouserMinY || legShoeMinY > legTrouserMinY + hemBandM) continue;

    // Leg axis: vertical line through the XZ centroid of this leg's hem ring.
    let sumAx = 0;
    let sumAz = 0;
    let hemCount = 0;
    for (const v of tIndices) {
      const y = positions[v * 3 + 1]!;
      if (y <= legTrouserMinY + hemBandM) {
        sumAx += positions[v * 3]!;
        sumAz += positions[v * 3 + 2]!;
        hemCount++;
      }
    }
    const axisX = hemCount > 0 ? sumAx / hemCount : 0;
    const axisZ = hemCount > 0 ? sumAz / hemCount : 0;

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
      let angle = Math.atan2(dz, dx);
      if (angle < 0) angle += 2 * Math.PI;
      const sector = Math.floor((angle / (2 * Math.PI)) * sectorCount) % sectorCount;
      const idx = rawSlab * sectorCount + sector;
      if (r > rTrouser[idx]!) rTrouser[idx] = r;
    }
    fillEmptyTrouserBins(rTrouser, maxSlab, sectorCount);

    // 2. Bin shoe vertices that lie within r_trouser(slab, sector) + maxFlareM.
    const rShoe = new Float32Array(slabCount * sectorCount);
    let contributingShoeMaxY = -Infinity;

    for (const v of sIndices) {
      const x = shoe.positions[v * 3]!;
      const y = shoe.positions[v * 3 + 1]!;
      const z = shoe.positions[v * 3 + 2]!;
      const dx = x - axisX;
      const dz = z - axisZ;
      const r = Math.hypot(dx, dz);
      let angle = Math.atan2(dz, dx);
      if (angle < 0) angle += 2 * Math.PI;
      const sector = Math.floor((angle / (2 * Math.PI)) * sectorCount) % sectorCount;
      let slab = Math.floor((y - hemBottomY) / slabHeight);
      if (slab < 0) slab = 0;
      else if (slab > maxSlab) slab = maxSlab;
      const idx = slab * sectorCount + sector;
      if (r > rTrouser[idx]! + maxFlareM) continue;
      if (r > rShoe[idx]!) rShoe[idx] = r;
      if (y > contributingShoeMaxY) contributingShoeMaxY = y;
    }

    if (contributingShoeMaxY < legTrouserMinY) continue;

    // 3. Downward propagation: a ring must never narrow as it drapes down to the hem.
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

    // 4. Smoothing per slab: circular max-filter + blur over sectors to keep the tube clean.
    const rRequired = new Float32Array(slabCount * sectorCount);
    const effectiveShoe = new Float32Array(sectorCount);
    for (let s = 0; s <= maxSlab; s++) {
      const slabOffset = s * sectorCount;
      let slabMax = 0;
      let anyNeedsFlare = false;
      for (let k = 0; k < sectorCount; k++) {
        const val = rShoe[slabOffset + k]!;
        if (val > slabMax) slabMax = val;
        if (val > 0 && val + marginM > rTrouser[slabOffset + k]!) {
          anyNeedsFlare = true;
        }
      }
      if (slabMax === 0 || !anyNeedsFlare) continue;

      for (let k = 0; k < sectorCount; k++) {
        const floorR = Math.max(0, rTrouser[slabOffset + k]! - marginM);
        effectiveShoe[k] = Math.max(rShoe[slabOffset + k]!, floorR);
      }

      const maxFiltered = new Float32Array(sectorCount);
      for (let k = 0; k < sectorCount; k++) {
        let maxR = effectiveShoe[k]!;
        for (let w = -maxFilterRadius; w <= maxFilterRadius; w++) {
          const neighbor = (k + w + sectorCount) % sectorCount;
          const val = effectiveShoe[neighbor]!;
          if (val > maxR) maxR = val;
        }
        maxFiltered[k] = maxR;
      }

      for (let k = 0; k < sectorCount; k++) {
        const km2 = maxFiltered[(k - 2 + sectorCount) % sectorCount]!;
        const km1 = maxFiltered[(k - 1 + sectorCount) % sectorCount]!;
        const k0 = maxFiltered[k]!;
        const kp1 = maxFiltered[(k + 1) % sectorCount]!;
        const kp2 = maxFiltered[(k + 2) % sectorCount]!;
        const blurred = (km2 + 2 * km1 + 4 * k0 + 2 * kp1 + kp2) / 10;
        const raw = effectiveShoe[k]!;
        const smoothed = Math.max(blurred, raw);
        const unified = 0.85 * smoothed + 0.15 * slabMax;
        const finalR = Math.max(unified, raw);
        rRequired[slabOffset + k] = finalR + marginM;
      }
    }

    const cutoffY = contributingShoeMaxY + fadeBandM;

    // 5. Displace vertices radially in XZ away from the leg axis, clamped to maxFlareM.
    for (const v of tIndices) {
      const y = positions[v * 3 + 1]!;
      if (y >= cutoffY) continue;

      const x = positions[v * 3]!;
      const z = positions[v * 3 + 2]!;
      const dx = x - axisX;
      const dz = z - axisZ;
      const r = Math.hypot(dx, dz);
      if (r < 1e-6) continue;

      let angle = Math.atan2(dz, dx);
      if (angle < 0) angle += 2 * Math.PI;
      const sector = Math.floor((angle / (2 * Math.PI)) * sectorCount) % sectorCount;

      const slab = Math.min(maxSlab, Math.max(0, Math.floor((y - hemBottomY) / slabHeight)));
      const required = rRequired[slab * sectorCount + sector]!;
      let push = Math.min(maxFlareM, Math.max(0, required - r));

      if (y > contributingShoeMaxY) {
        const t = (y - contributingShoeMaxY) / fadeBandM;
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
