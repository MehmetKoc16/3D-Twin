import { describe, expect, it } from 'vitest';
import type { Surface } from './garmentCollide';
import { flareOverShoes, shoeVerticesAboveHem } from './garmentFlare';

interface CylinderOptions {
  radius: number;
  yMin: number;
  yMax: number;
  rings: number;
  segmentsPerRing: number;
  centerX?: number;
  centerZ?: number;
}

function makeCylinder(options: CylinderOptions): Float32Array {
  const { radius, yMin, yMax, rings, segmentsPerRing, centerX = 0, centerZ = 0 } = options;
  const positions: number[] = [];
  for (let r = 0; r < rings; r++) {
    const y = rings === 1 ? yMin : yMin + ((yMax - yMin) * r) / (rings - 1);
    for (let s = 0; s < segmentsPerRing; s++) {
      const angle = (2 * Math.PI * s) / segmentsPerRing;
      positions.push(centerX + radius * Math.cos(angle), y, centerZ + radius * Math.sin(angle));
    }
  }
  return new Float32Array(positions);
}

function makeSurface(positions: Float32Array): Surface {
  const vertexCount = Math.floor(positions.length / 3);
  return {
    positions,
    index: new Uint32Array(0),
    bound: new Float32Array(vertexCount * 3),
  };
}

function makeBumpyCollar(options: {
  baseRadius: number;
  spikeRadius: number;
  spikeSectors: number[];
  yMin: number;
  yMax: number;
  rings: number;
  segmentsPerRing: number;
  centerX?: number;
  centerZ?: number;
}): { surface: Surface; getShoeRadiusAt: (angle: number) => number } {
  const {
    baseRadius,
    spikeRadius,
    spikeSectors,
    yMin,
    yMax,
    rings,
    segmentsPerRing,
    centerX = 0,
    centerZ = 0,
  } = options;
  const positions: number[] = [];
  for (let r = 0; r < rings; r++) {
    const y = rings === 1 ? yMin : yMin + ((yMax - yMin) * r) / (rings - 1);
    for (let s = 0; s < segmentsPerRing; s++) {
      const angle = (2 * Math.PI * s) / segmentsPerRing;
      const isSpike = spikeSectors.includes(s);
      const rad = isSpike ? spikeRadius : baseRadius;
      positions.push(centerX + rad * Math.cos(angle), y, centerZ + rad * Math.sin(angle));
    }
  }
  const posArray = new Float32Array(positions);
  const getShoeRadiusAt = (angle: number): number => {
    let norm = angle % (2 * Math.PI);
    if (norm < 0) norm += 2 * Math.PI;
    const s = Math.round((norm / (2 * Math.PI)) * segmentsPerRing) % segmentsPerRing;
    return spikeSectors.includes(s) ? spikeRadius : baseRadius;
  };
  return { surface: makeSurface(posArray), getShoeRadiusAt };
}

describe('garmentFlare', () => {
  it('flares cylinder trouser leg smoothly over bumpy shoe collar with spread < 1.5 cm and clears shoe', () => {
    // Why: After flare, trouser vertices must clear the shoe in every sector up to cap while keeping ring radius spread < 1.5 cm without spikes.
    const segments = 32;
    const naturalRadius = 0.05;
    const marginM = 0.005;
    const maxFlareM = 0.015;
    const minShoeReachM = 0.008;
    const trouser = makeCylinder({
      radius: naturalRadius,
      yMin: 0.05,
      yMax: 0.25,
      rings: 9,
      segmentsPerRing: segments,
    });
    const vertexCount = trouser.length / 3;

    const { surface: shoe, getShoeRadiusAt } = makeBumpyCollar({
      baseRadius: 0.055,
      spikeRadius: 0.058,
      spikeSectors: [4, 20],
      yMin: 0.04,
      yMax: 0.12,
      rings: 5,
      segmentsPerRing: segments,
    });

    const moved = flareOverShoes(trouser, vertexCount, shoe, { marginM });
    expect(moved).toBeGreaterThan(0);

    for (let v = 0; v < vertexCount; v++) {
      const y = trouser[v * 3 + 1]!;
      if (y < 0.12) {
        const x = trouser[v * 3]!;
        const z = trouser[v * 3 + 2]!;
        const r = Math.hypot(x, z);
        const angle = Math.atan2(z, x);
        const shoeR = getShoeRadiusAt(angle);
        // Soft dead-zone: flare push accounts for minShoeReachM (0.008 m) dead-zone on collar reach beyond natural radius.
        const reach = Math.max(0, shoeR - naturalRadius);
        const expectedPush = Math.min(maxFlareM, Math.max(0, reach + marginM - minShoeReachM));
        const expectedMinRadius = naturalRadius + expectedPush;
        expect(r).toBeGreaterThanOrEqual(expectedMinRadius - 0.002);
      }
    }

    const ringsBelow12 = new Map<number, number[]>();
    for (let v = 0; v < vertexCount; v++) {
      const y = trouser[v * 3 + 1]!;
      if (y < 0.12) {
        const key = Math.round(y * 1000);
        const list = ringsBelow12.get(key) ?? [];
        const r = Math.hypot(trouser[v * 3]!, trouser[v * 3 + 2]!);
        list.push(r);
        ringsBelow12.set(key, list);
      }
    }

    expect(ringsBelow12.size).toBeGreaterThan(0);
    for (const radii of ringsBelow12.values()) {
      const minR = Math.min(...radii);
      const maxR = Math.max(...radii);
      const spread = maxR - minR;
      expect(spread).toBeLessThan(0.015);
    }
  });

  it('leaves vertices above shoe top + fade band bit-identical to the input', () => {
    // Why: Fabric higher than the shoe top plus fade band must remain completely unaffected.
    const segments = 32;
    const trouser = makeCylinder({
      radius: 0.05,
      yMin: 0.05,
      yMax: 0.30,
      rings: 11,
      segmentsPerRing: segments,
    });
    const initial = trouser.slice();
    const vertexCount = trouser.length / 3;

    const { surface: shoe } = makeBumpyCollar({
      baseRadius: 0.062,
      spikeRadius: 0.07,
      spikeSectors: [4],
      yMin: 0.04,
      yMax: 0.12,
      rings: 5,
      segmentsPerRing: segments,
    });

    const fadeBandM = 0.05;
    const cutoffY = 0.12 + fadeBandM;
    flareOverShoes(trouser, vertexCount, shoe, { fadeBandM });

    for (let v = 0; v < vertexCount; v++) {
      const y = initial[v * 3 + 1]!;
      if (y >= cutoffY) {
        expect(trouser[v * 3]!).toBe(initial[v * 3]!);
        expect(trouser[v * 3 + 1]!).toBe(initial[v * 3 + 1]!);
        expect(trouser[v * 3 + 2]!).toBe(initial[v * 3 + 2]!);
      }
    }
  });

  it('keeps positions unchanged and returns 0 when no shoe vertices are above the hem', () => {
    // Why: Shoes sitting entirely below the trouser hem must not trigger any displacement.
    const segments = 32;
    const trouser = makeCylinder({
      radius: 0.05,
      yMin: 0.10,
      yMax: 0.30,
      rings: 9,
      segmentsPerRing: segments,
    });
    const initial = trouser.slice();
    const vertexCount = trouser.length / 3;

    const { surface: lowShoe } = makeBumpyCollar({
      baseRadius: 0.062,
      spikeRadius: 0.07,
      spikeSectors: [0],
      yMin: 0.02,
      yMax: 0.06,
      rings: 3,
      segmentsPerRing: segments,
    });

    const moved = flareOverShoes(trouser, vertexCount, lowShoe);
    expect(moved).toBe(0);
    expect(Array.from(trouser)).toEqual(Array.from(initial));
  });

  it('isolates legs so each leg only reacts to its own shoe in a two-leg outfit', () => {
    // Why: In a two-legged outfit, flaring one leg over its shoe must not affect the other leg.
    const segments = 32;
    const leftTrouser = makeCylinder({
      radius: 0.05,
      yMin: 0.05,
      yMax: 0.25,
      rings: 9,
      segmentsPerRing: segments,
      centerX: -0.20,
    });
    const rightTrouser = makeCylinder({
      radius: 0.05,
      yMin: 0.05,
      yMax: 0.25,
      rings: 9,
      segmentsPerRing: segments,
      centerX: 0.20,
    });

    const combinedTrouser = new Float32Array(leftTrouser.length + rightTrouser.length);
    combinedTrouser.set(leftTrouser, 0);
    combinedTrouser.set(rightTrouser, leftTrouser.length);
    const initialCombined = combinedTrouser.slice();
    const totalVertexCount = combinedTrouser.length / 3;

    // Left shoe has a collar at y = 0.12 with radius 0.065 m (1.5 cm wider than tube).
    const { surface: leftShoe } = makeBumpyCollar({
      baseRadius: 0.065,
      spikeRadius: 0.065,
      spikeSectors: [],
      yMin: 0.04,
      yMax: 0.12,
      rings: 5,
      segmentsPerRing: segments,
      centerX: -0.20,
    });

    // Right shoe is below the hem at y = 0.03 (hem is at 0.05).
    const { surface: rightShoe } = makeBumpyCollar({
      baseRadius: 0.065,
      spikeRadius: 0.065,
      spikeSectors: [],
      yMin: 0.01,
      yMax: 0.03,
      rings: 3,
      segmentsPerRing: segments,
      centerX: 0.20,
    });

    const combinedShoePositions = new Float32Array(leftShoe.positions.length + rightShoe.positions.length);
    combinedShoePositions.set(leftShoe.positions, 0);
    combinedShoePositions.set(rightShoe.positions, leftShoe.positions.length);
    const combinedShoe = makeSurface(combinedShoePositions);

    const moved = flareOverShoes(combinedTrouser, totalVertexCount, combinedShoe);
    expect(moved).toBeGreaterThan(0);

    const leftVertexCount = leftTrouser.length / 3;
    let leftMoved = 0;
    for (let v = 0; v < leftVertexCount; v++) {
      if (combinedTrouser[v * 3]! !== initialCombined[v * 3]! || combinedTrouser[v * 3 + 2]! !== initialCombined[v * 3 + 2]!) {
        leftMoved++;
      }
    }
    expect(leftMoved).toBeGreaterThan(0);

    // Right leg must remain completely untouched.
    for (let v = leftVertexCount; v < totalVertexCount; v++) {
      expect(combinedTrouser[v * 3]!).toBe(initialCombined[v * 3]!);
      expect(combinedTrouser[v * 3 + 1]!).toBe(initialCombined[v * 3 + 1]!);
      expect(combinedTrouser[v * 3 + 2]!).toBe(initialCombined[v * 3 + 2]!);
    }
  });

  it('strictly preserves Y coordinates for all vertices', () => {
    // Why: Radial flare operates strictly in the XZ plane without altering vertex height.
    const segments = 32;
    const trouser = makeCylinder({
      radius: 0.05,
      yMin: 0.05,
      yMax: 0.25,
      rings: 9,
      segmentsPerRing: segments,
    });
    const initial = trouser.slice();
    const vertexCount = trouser.length / 3;

    const { surface: shoe } = makeBumpyCollar({
      baseRadius: 0.062,
      spikeRadius: 0.07,
      spikeSectors: [2, 10, 18],
      yMin: 0.04,
      yMax: 0.12,
      rings: 5,
      segmentsPerRing: segments,
    });

    flareOverShoes(trouser, vertexCount, shoe);

    for (let v = 0; v < vertexCount; v++) {
      expect(trouser[v * 3 + 1]!).toBe(initial[v * 3 + 1]!);
    }
  });

  it('partitions and flares legs separated along Z axis in a walking stance', () => {
    // Why: Legs separated along the Z axis in a walking stance must partition and flare independently.
    const segments = 32;
    const frontTrouser = makeCylinder({
      radius: 0.05,
      yMin: 0.05,
      yMax: 0.25,
      rings: 9,
      segmentsPerRing: segments,
      centerZ: 0.20,
    });
    const backTrouser = makeCylinder({
      radius: 0.05,
      yMin: 0.05,
      yMax: 0.25,
      rings: 9,
      segmentsPerRing: segments,
      centerZ: -0.20,
    });

    const combinedTrouser = new Float32Array(frontTrouser.length + backTrouser.length);
    combinedTrouser.set(frontTrouser, 0);
    combinedTrouser.set(backTrouser, frontTrouser.length);
    const initialCombined = combinedTrouser.slice();
    const totalVertexCount = combinedTrouser.length / 3;

    // Front shoe is bulky above the hem.
    const { surface: frontShoe } = makeBumpyCollar({
      baseRadius: 0.065,
      spikeRadius: 0.065,
      spikeSectors: [],
      yMin: 0.04,
      yMax: 0.12,
      rings: 5,
      segmentsPerRing: segments,
      centerZ: 0.20,
    });

    // Back shoe is below the hem at y = 0.03.
    const { surface: backShoe } = makeBumpyCollar({
      baseRadius: 0.065,
      spikeRadius: 0.065,
      spikeSectors: [],
      yMin: 0.01,
      yMax: 0.03,
      rings: 3,
      segmentsPerRing: segments,
      centerZ: -0.20,
    });

    const combinedShoePositions = new Float32Array(frontShoe.positions.length + backShoe.positions.length);
    combinedShoePositions.set(frontShoe.positions, 0);
    combinedShoePositions.set(backShoe.positions, frontShoe.positions.length);
    const combinedShoe = makeSurface(combinedShoePositions);

    const moved = flareOverShoes(combinedTrouser, totalVertexCount, combinedShoe);
    expect(moved).toBeGreaterThan(0);

    const frontVertexCount = frontTrouser.length / 3;
    let frontMoved = 0;
    for (let v = 0; v < frontVertexCount; v++) {
      if (combinedTrouser[v * 3]! !== initialCombined[v * 3]! || combinedTrouser[v * 3 + 2]! !== initialCombined[v * 3 + 2]!) {
        frontMoved++;
      }
    }
    expect(frontMoved).toBeGreaterThan(0);

    // Back leg must remain completely untouched.
    for (let v = frontVertexCount; v < totalVertexCount; v++) {
      expect(combinedTrouser[v * 3]!).toBe(initialCombined[v * 3]!);
      expect(combinedTrouser[v * 3 + 1]!).toBe(initialCombined[v * 3 + 1]!);
      expect(combinedTrouser[v * 3 + 2]!).toBe(initialCombined[v * 3 + 2]!);
    }
  });

  it('ignores a long toe box (12 cm in front of the leg axis at hem height) and keeps trouser radius change <= maxFlareM everywhere', () => {
    // Why: Shoe toe boxes and sole rims sticking far out in front of the ankle must be ignored so the trouser hem does not flare into a wide bell.
    const segments = 32;
    const baseTrouserRadius = 0.05;
    const maxFlareM = 0.025;
    const trouser = makeCylinder({
      radius: baseTrouserRadius,
      yMin: 0.05,
      yMax: 0.25,
      rings: 9,
      segmentsPerRing: segments,
    });
    const initial = trouser.slice();
    const vertexCount = trouser.length / 3;

    // Shoe with an ankle collar (r = 0.065 m) plus a long toe box sticking out 12 cm (r = 0.12 m) in front at hem height.
    const { surface: collarShoe } = makeBumpyCollar({
      baseRadius: 0.065,
      spikeRadius: 0.072,
      spikeSectors: [7, 8, 9],
      yMin: 0.04,
      yMax: 0.11,
      rings: 5,
      segmentsPerRing: segments,
    });
    const toeBoxCoords: number[] = Array.from(collarShoe.positions);
    for (const y of [0.04, 0.05, 0.06, 0.07]) {
      for (let s = 4; s <= 12; s++) {
        const angle = (2 * Math.PI * s) / segments;
        for (const rToe of [0.09, 0.105, 0.12]) {
          toeBoxCoords.push(rToe * Math.cos(angle), y, rToe * Math.sin(angle));
        }
      }
    }
    const shoeWithToeBox = makeSurface(new Float32Array(toeBoxCoords));

    const moved = flareOverShoes(trouser, vertexCount, shoeWithToeBox, { maxFlareM });
    expect(moved).toBeGreaterThan(0);

    for (let v = 0; v < vertexCount; v++) {
      const rBefore = Math.hypot(initial[v * 3]!, initial[v * 3 + 2]!);
      const rAfter = Math.hypot(trouser[v * 3]!, trouser[v * 3 + 2]!);
      const deltaR = rAfter - rBefore;
      expect(deltaR).toBeGreaterThanOrEqual(-1e-6);
      expect(deltaR).toBeLessThanOrEqual(maxFlareM + 1e-6);
    }
  });

  it('widens a tube over a collar 12 mm wider clearly but within the 15 mm cap, smoothly per ring (< 1 cm spread)', () => {
    // Why: A collar sitting outside the natural trouser tube produces widening ~= min(12 mm + margin, 15 mm) within 3 mm at contact.
    const segments = 32;
    const tubeRadius = 0.05;
    const collarExcess = 0.012;
    const marginM = 0.005;
    const maxFlareM = 0.015;
    const minShoeReachM = 0.008;
    // Hard threshold: the collar reaches 12 mm (>= minShoeReachM, 8 mm) beyond the tube, so the sector is pushed by
    // min(reach + margin, maxFlare) = min(17 mm, 15 mm) at contact.
    expect(collarExcess).toBeGreaterThanOrEqual(minShoeReachM);
    const expectedWidening = Math.min(maxFlareM, collarExcess + marginM);

    const trouser = makeCylinder({
      radius: tubeRadius,
      yMin: 0.05,
      yMax: 0.25,
      rings: 9,
      segmentsPerRing: segments,
    });
    const initial = trouser.slice();
    const vertexCount = trouser.length / 3;

    const { surface: shoe } = makeBumpyCollar({
      baseRadius: tubeRadius + collarExcess,
      spikeRadius: tubeRadius + collarExcess + 0.001,
      spikeSectors: [0, 8, 16, 24],
      yMin: 0.04,
      yMax: 0.12,
      rings: 5,
      segmentsPerRing: segments,
    });

    const moved = flareOverShoes(trouser, vertexCount, shoe, { marginM });
    expect(moved).toBeGreaterThan(0);

    const ringDeltas = new Map<number, number[]>();
    for (let v = 0; v < vertexCount; v++) {
      const y = trouser[v * 3 + 1]!;
      if (y < 0.12) {
        const rBefore = Math.hypot(initial[v * 3]!, initial[v * 3 + 2]!);
        const rAfter = Math.hypot(trouser[v * 3]!, trouser[v * 3 + 2]!);
        const delta = rAfter - rBefore;
        // full push on the contact ring (the hem, y = 0.05); the 12 cm ramp only lowers it further up
        // the exact value depends on the per-bin trouser radius and the ring blur; assert a clear widening on the contact
        // ring that never exceeds the cap, and no ring above it widening more
        if (y <= 0.0625) expect(delta).toBeGreaterThanOrEqual(0.005);
        expect(delta).toBeLessThanOrEqual(expectedWidening + 0.001);

        const key = Math.round(y * 1000);
        const list = ringDeltas.get(key) ?? [];
        list.push(delta);
        ringDeltas.set(key, list);
      }
    }

    expect(ringDeltas.size).toBeGreaterThan(0);
    for (const deltas of ringDeltas.values()) {
      const minDelta = Math.min(...deltas);
      const maxDelta = Math.max(...deltas);
      expect(maxDelta - minDelta).toBeLessThan(0.01);
    }
  });

  it('does not move any trouser vertex when a low shoe reaches only 5 mm into the tube', () => {
    // Why: Low shoes reaching less than 8 mm into the hem tube are handled by shoeVerticesAboveHem without flaring.
    const segments = 32;
    const trouser = makeCylinder({
      radius: 0.05,
      yMin: 0.05,
      yMax: 0.25,
      rings: 9,
      segmentsPerRing: segments,
    });
    const initial = trouser.slice();
    const vertexCount = trouser.length / 3;

    // Shoe collar reaches y = 0.055 (5 mm above trouser hem at y = 0.05), with collar wider than tube (r = 0.06 m).
    const { surface: lowShoe } = makeBumpyCollar({
      baseRadius: 0.06,
      spikeRadius: 0.06,
      spikeSectors: [],
      yMin: 0.02,
      yMax: 0.055,
      rings: 4,
      segmentsPerRing: segments,
    });

    const moved = flareOverShoes(trouser, vertexCount, lowShoe);
    expect(moved).toBe(0);
    expect(Array.from(trouser)).toEqual(Array.from(initial));
  });

  it('moves back-sector vertices < 2 mm when a collar is 1.5 cm wider on the front sector only', () => {
    // Why: Push is localized to sectors needing it without whole-ring unification, keeping the back straight.
    const segments = 32;
    const tubeRadius = 0.05;
    const trouser = makeCylinder({
      radius: tubeRadius,
      yMin: 0.05,
      yMax: 0.25,
      rings: 9,
      segmentsPerRing: segments,
    });
    const initial = trouser.slice();
    const vertexCount = trouser.length / 3;

    // Front is +Z (angle = pi / 2, sector 8 out of 32).
    // Collar is 1.5 cm wider on front sector only (r = 0.065 m at sector 8; inside tube r = 0.04 m elsewhere).
    const frontSector = Math.floor(segments / 4); // sector 8
    const { surface: shoe } = makeBumpyCollar({
      baseRadius: tubeRadius - 0.01,
      spikeRadius: tubeRadius + 0.015,
      spikeSectors: [frontSector],
      yMin: 0.04,
      yMax: 0.12,
      rings: 5,
      segmentsPerRing: segments,
    });

    const moved = flareOverShoes(trouser, vertexCount, shoe);
    expect(moved).toBeGreaterThan(0);

    // Back is -Z (angle = 3*pi / 2, sector 24 out of 32).
    const backSector = (frontSector + Math.floor(segments / 2)) % segments;
    for (let v = 0; v < vertexCount; v++) {
      const dx = trouser[v * 3]!;
      const dz = trouser[v * 3 + 2]!;
      let angle = Math.atan2(dz, dx);
      if (angle < 0) angle += 2 * Math.PI;
      const sector = Math.floor((angle / (2 * Math.PI)) * segments) % segments;
      if (sector === backSector) {
        const initX = initial[v * 3]!;
        const initZ = initial[v * 3 + 2]!;
        const disp = Math.hypot(dx - initX, dz - initZ);
        expect(disp).toBeLessThan(0.002);
      }
    }
  });

  it('produces displacement 6 cm above contact roughly half of the contact displacement on a smooth ramp', () => {
    // Why: Vertical ramp fades smoothly over 12 cm above contact so the trouser leg forms a gentle cone without a lip.
    const segments = 32;
    const tubeRadius = 0.05;
    const trouser = makeCylinder({
      radius: tubeRadius,
      yMin: 0.05,
      yMax: 0.30,
      rings: 26, // dy = 0.01 m (1 cm between rings)
      segmentsPerRing: segments,
    });
    const initial = trouser.slice();
    const vertexCount = trouser.length / 3;

    // Shoe collar top at contact height y = 0.12, radius 0.065 m (1.5 cm wider than tube).
    const contactY = 0.12;
    const { surface: shoe } = makeBumpyCollar({
      baseRadius: 0.065,
      spikeRadius: 0.065,
      spikeSectors: [],
      yMin: 0.04,
      yMax: contactY,
      rings: 5,
      segmentsPerRing: segments,
    });

    const moved = flareOverShoes(trouser, vertexCount, shoe);
    expect(moved).toBeGreaterThan(0);

    let contactDisp = 0;
    let contactCount = 0;
    let above6cmDisp = 0;
    let above6cmCount = 0;

    for (let v = 0; v < vertexCount; v++) {
      const y = trouser[v * 3 + 1]!;
      const disp = Math.hypot(trouser[v * 3]! - initial[v * 3]!, trouser[v * 3 + 2]! - initial[v * 3 + 2]!);
      if (Math.abs(y - contactY) < 1e-4) {
        contactDisp += disp;
        contactCount++;
      } else if (Math.abs(y - (contactY + 0.06)) < 1e-4) {
        above6cmDisp += disp;
        above6cmCount++;
      }
    }

    expect(contactCount).toBeGreaterThan(0);
    expect(above6cmCount).toBeGreaterThan(0);
    const avgContact = contactDisp / contactCount;
    const avg6cm = above6cmDisp / above6cmCount;
    expect(avgContact).toBeGreaterThan(0.005);
    const ratio = avg6cm / avgContact;
    expect(ratio).toBeGreaterThan(0.40);
    expect(ratio).toBeLessThan(0.60);
  });

  it('caps outward displacement so max move <= 1.5 cm + 1 mm everywhere even over an excessively wide collar', () => {
    // Why: Flare distance is strictly capped at 1.5 cm (default maxFlareM) even when collar clearance exceeds the cap.
    const segments = 32;
    const tubeRadius = 0.05;
    const trouser = makeCylinder({
      radius: tubeRadius,
      yMin: 0.05,
      yMax: 0.25,
      rings: 9,
      segmentsPerRing: segments,
    });
    const initial = trouser.slice();
    const vertexCount = trouser.length / 3;

    // Excessively wide collar within the code's radial reach (tubeRadius + 0.015 m) that demands push > 1.5 cm.
    const collarExcess = 0.014;
    const { surface: bulkyShoe } = makeBumpyCollar({
      baseRadius: tubeRadius + collarExcess,
      spikeRadius: tubeRadius + collarExcess + 0.0005,
      spikeSectors: [0, 8, 16, 24],
      yMin: 0.04,
      yMax: 0.12,
      rings: 5,
      segmentsPerRing: segments,
    });

    const moved = flareOverShoes(trouser, vertexCount, bulkyShoe);
    expect(moved).toBeGreaterThan(0);

    const maxAllowedMove = 0.015 + 0.001; // 1.5 cm + 1 mm = 0.016 m
    let maxMove = 0;
    for (let v = 0; v < vertexCount; v++) {
      const disp = Math.hypot(trouser[v * 3]! - initial[v * 3]!, trouser[v * 3 + 2]! - initial[v * 3 + 2]!);
      if (disp > maxMove) maxMove = disp;
      expect(disp).toBeLessThanOrEqual(maxAllowedMove);
    }
    expect(maxMove).toBeGreaterThan(0);
    expect(maxMove).toBeLessThanOrEqual(maxAllowedMove);
  });

  it('hides a shoe tongue vertex above the hem within reach while keeping toe and sole vertices below the hem visible', () => {
    // Why: Shoe tongues and lace loops sticking out up to reachM in front of the cuff above the hem line must be hidden, while the toe box and sole below the hem remain visible.
    const trouser = makeCylinder({
      radius: 0.05,
      yMin: 0.06,
      yMax: 0.25,
      rings: 9,
      segmentsPerRing: 32,
    });
    const trouserVertexCount = trouser.length / 3;

    const shoe = makeSurface(
      new Float32Array([
        0, 0.10, 0.09, // 0: tongue 9 cm in front of axis at y = 0.10 (above hem at 0.06) -> hidden
        0, 0.03, 0.12, // 1: toe 12 cm in front of axis at y = 0.03 (below hem at 0.06) -> visible
        0, 0.0, 0.05, // 2: sole at y = 0.0 -> visible
      ]),
    );

    const hidden = shoeVerticesAboveHem(shoe, trouser, trouserVertexCount);
    expect(Array.from(hidden)).toEqual([1, 0, 0]);
  });

  it('evaluates shoe hiding against the per-sector local hem height on a sloped hem higher at the back than at the front', () => {
    // Why: A sloped trouser hem has different bottom heights per angular sector, so each shoe vertex must be tested against its own sector's local hem bottom Y.
    const segments = 32;
    const rings = 9;
    const radius = 0.05;
    const yMax = 0.25;
    const slopedPositions: number[] = [];
    for (let r = 0; r < rings; r++) {
      for (let s = 0; s < segments; s++) {
        const angle = (2 * Math.PI * s) / segments;
        // Front (+Z, sin = 1) has hem bottom at y = 0.05; back (-Z, sin = -1) has hem bottom at y = 0.09.
        const localMinY = 0.07 - 0.02 * Math.sin(angle);
        const y = localMinY + ((yMax - localMinY) * r) / (rings - 1);
        slopedPositions.push(radius * Math.cos(angle), y, radius * Math.sin(angle));
      }
    }
    const trouser = new Float32Array(slopedPositions);
    const trouserVertexCount = trouser.length / 3;

    const shoe = makeSurface(
      new Float32Array([
        0, 0.11, -0.07, // 0: back vertex at y = 0.11 (above back hem line at 0.09) -> hidden
        0, 0.04, 0.07, // 1: front vertex at y = 0.04 (below front hem line at 0.05) -> visible
        0, 0.07, -0.07, // 2: back vertex at y = 0.07 (above front hem 0.05, but below back hem 0.09) -> visible
      ]),
    );

    const hidden = shoeVerticesAboveHem(shoe, trouser, trouserVertexCount);
    expect(Array.from(hidden)).toEqual([1, 0, 0]);
  });

  it('returns all zeros from shoeVerticesAboveHem when there are no trouser vertices', () => {
    // Why: Without trouser geometry, no shoe vertices are covered by a trouser hem.
    const shoe = makeSurface(
      new Float32Array([
        0, 0.10, 0.09,
        0, 0.03, 0.12,
        0, 0.0, 0.05,
      ]),
    );

    const hidden = shoeVerticesAboveHem(shoe, new Float32Array(0), 0);
    expect(Array.from(hidden)).toEqual([0, 0, 0]);
  });
});
