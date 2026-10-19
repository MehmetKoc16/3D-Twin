import { describe, expect, it } from 'vitest';
import type { GarmentBinding } from './binding';
import { createGarmentSections } from './centroid';
import { gradeGarment, gradeGarmentOuter, type GradeRing } from './grade';

/** Cylinder body of radius 0.2 m at y = 1, 64 vertices. */
function cylinderBody(): Float32Array {
  const body = new Float32Array(64 * 3);
  for (let i = 0; i < 64; i++) {
    const angle = (2 * Math.PI * i) / 64;
    body[i * 3] = 0.2 * Math.cos(angle);
    body[i * 3 + 1] = 1;
    body[i * 3 + 2] = 0.2 * Math.sin(angle);
  }
  return body;
}

/** Two stacked vertices 3 mm apart in front of the body, bound to neighbouring body vertices 16 and 17. */
function stackedBinding(body: Float32Array): GarmentBinding {
  const targets: [number, number, number][] = [
    [0, 1, 0.203],
    [0, 1, 0.206],
  ];
  const bodyVerts = [16, 17];
  const indices = new Uint32Array(6);
  const weights = new Float32Array(6);
  const offsets = new Float32Array(6);
  targets.forEach((target, v) => {
    const b = bodyVerts[v]!;
    indices.fill(b, v * 3, v * 3 + 3);
    weights[v * 3] = 1;
    for (let k = 0; k < 3; k++) offsets[v * 3 + k] = target[k]! - body[b * 3 + k]!;
  });
  return { count: 2, indices, weights, offsets };
}

/** An open cylinder "jacket" covering angles from 0.4pi to 1.6pi around cylinderBody. */
function openCylinderJacket(body: Float32Array): {
  binding: GarmentBinding;
  rest: Float32Array;
  edgeVerts: number[];
} {
  const heights = [0.96, 0.98, 1.0, 1.02, 1.04];
  const angles: number[] = [];
  for (let a = 0.4; a <= 1.601; a += 0.05) {
    angles.push(a * Math.PI);
  }
  const targets: [number, number, number][] = [];
  const edgeVerts: number[] = [];

  for (const y of heights) {
    for (let ai = 0; ai < angles.length; ai++) {
      const angle = angles[ai]!;
      const x = 0.205 * Math.cos(angle);
      const z = 0.205 * Math.sin(angle);
      const vIdx = targets.length;
      targets.push([x, y, z]);
      if (ai === 0) {
        edgeVerts.push(vIdx);
      }
    }
  }

  const count = targets.length;
  const indices = new Uint32Array(count * 3);
  const weights = new Float32Array(count * 3);
  const offsets = new Float32Array(count * 3);
  const rest = new Float32Array(count * 3);

  for (let v = 0; v < count; v++) {
    const target = targets[v]!;
    rest[v * 3] = target[0];
    rest[v * 3 + 1] = target[1];
    rest[v * 3 + 2] = target[2];

    let bestB = 0;
    let bestDist = Infinity;
    for (let b = 0; b < 64; b++) {
      const bx = body[b * 3]!;
      const by = body[b * 3 + 1]!;
      const bz = body[b * 3 + 2]!;
      const d = (target[0] - bx) ** 2 + (target[1] - by) ** 2 + (target[2] - bz) ** 2;
      if (d < bestDist) {
        bestDist = d;
        bestB = b;
      }
    }
    indices.fill(bestB, v * 3, v * 3 + 3);
    weights[v * 3] = 1;
    for (let k = 0; k < 3; k++) {
      offsets[v * 3 + k] = target[k]! - body[bestB * 3 + k]!;
    }
  }

  return {
    binding: { count, indices, weights, offsets },
    rest,
    edgeVerts,
  };
}

describe('gradeGarment outer mode', () => {
  const body = cylinderBody();
  const sections = createGarmentSections(body);
  const binding = stackedBinding(body);
  const rest = new Float32Array([0, 1, 0.203, 0, 1, 0.206]);
  const shrinkRings: GradeRing[] = [{ planeY: 1, deltaCircumferenceM: -0.06 }];

  it('(1) preserves order and spacing within ±1 mm for stacked parts 3 mm apart bound to different body vertices', () => {
    const graded = new Float32Array(6);
    gradeGarment(rest, binding, body, shrinkRings, graded, sections, { outer: true });

    // Grading was applied
    expect(Array.from(graded)).not.toEqual(Array.from(rest));

    // Both shifted inward
    expect(graded[2]).toBeLessThan(rest[2]!);
    expect(graded[5]).toBeLessThan(rest[5]!);

    // Order preserved: target 1 is still in front of target 0 (greater Z)
    expect(graded[5]!).toBeGreaterThan(graded[2]!);

    // Spacing preserved within ±1 mm of initial 3 mm (2 mm .. 4 mm)
    const dist = Math.hypot(
      graded[3]! - graded[0]!,
      graded[4]! - graded[1]!,
      graded[5]! - graded[2]!,
    );
    expect(dist).toBeGreaterThanOrEqual(0.002);
    expect(dist).toBeLessThanOrEqual(0.004);
    expect(Math.abs(dist - 0.003)).toBeLessThanOrEqual(0.001);
  });

  it('(2) keeps displacement difference between neighboring edge vertices < 1 mm when narrowing an open cylinder jacket', () => {
    const openJacket = openCylinderJacket(body);
    const graded = new Float32Array(openJacket.binding.count * 3);
    gradeGarmentOuter(openJacket.rest, openJacket.binding, body, shrinkRings, graded, sections);

    const edge = openJacket.edgeVerts;
    expect(edge.length).toBeGreaterThan(1);
    for (let i = 0; i < edge.length - 1; i++) {
      const v0 = edge[i]!;
      const v1 = edge[i + 1]!;
      const u0x = graded[v0 * 3]! - openJacket.rest[v0 * 3]!;
      const u0y = graded[v0 * 3 + 1]! - openJacket.rest[v0 * 3 + 1]!;
      const u0z = graded[v0 * 3 + 2]! - openJacket.rest[v0 * 3 + 2]!;
      const u1x = graded[v1 * 3]! - openJacket.rest[v1 * 3]!;
      const u1y = graded[v1 * 3 + 1]! - openJacket.rest[v1 * 3 + 1]!;
      const u1z = graded[v1 * 3 + 2]! - openJacket.rest[v1 * 3 + 2]!;
      const diff = Math.hypot(u1x - u0x, u1y - u0y, u1z - u0z);
      expect(diff).toBeLessThan(0.001); // < 1 mm
    }
  });

  it('(3) leaves positions identical to rest when delta is 0', () => {
    const zeroRings: GradeRing[] = [{ planeY: 1, deltaCircumferenceM: 0 }];
    const graded = new Float32Array(6);
    gradeGarment(rest, binding, body, zeroRings, graded, sections, { outer: true });
    expect(Array.from(graded)).toEqual(Array.from(rest));

    const emptyGraded = new Float32Array(6);
    gradeGarment(rest, binding, body, [], emptyGraded, sections, { outer: true });
    expect(Array.from(emptyGraded)).toEqual(Array.from(rest));
  });

  it('(4) produces identical result to standard gradeGarment for non-outer garments', () => {
    const defaultGraded = new Float32Array(6);
    gradeGarment(rest, binding, body, shrinkRings, defaultGraded, sections);

    const explicitNonOuterGraded = new Float32Array(6);
    gradeGarment(rest, binding, body, shrinkRings, explicitNonOuterGraded, sections, {
      outer: false,
    });

    expect(Array.from(explicitNonOuterGraded)).toEqual(Array.from(defaultGraded));
  });

  it('keeps sparse body slices from rotating the opening displacement or clamping stacked fabric', () => {
    const sparse = Float32Array.from([
      ...body,
      // Only front samples in the next 1 cm slice: its unsmoothed centroid lies on the skin.
      -0.02,
      1.02,
      0.2,
      0.02,
      1.02,
      0.2,
    ]);
    const edgeRest = new Float32Array([0.02, 1, 0.204, 0.02, 1.02, 0.204, 0.02, 1.02, 0.207]);
    const edgeBinding: GarmentBinding = {
      count: 3,
      indices: new Uint32Array([16, 16, 16, 65, 65, 65, 65, 65, 65]),
      weights: new Float32Array([1, 0, 0, 1, 0, 0, 1, 0, 0]),
      offsets: new Float32Array(9),
    };
    const result = new Float32Array(9);
    gradeGarmentOuter(
      edgeRest,
      edgeBinding,
      sparse,
      [{ planeY: 1, deltaCircumferenceM: -0.12 }],
      result,
    );
    for (let v = 0; v < 3; v++)
      expect(edgeRest[v * 3 + 2]! - result[v * 3 + 2]!).toBeGreaterThan(0.015);
    expect(
      Math.hypot(
        result[0]! - edgeRest[0]! - result[3]! + edgeRest[3]!,
        result[2]! - edgeRest[2]! - result[5]! + edgeRest[5]!,
      ),
    ).toBeLessThan(0.003);
    expect(result[8]! - result[5]!).toBeCloseTo(0.003, 3);
  });
});

/** Cylinder garment at radius r around cylinderBody, closed ring covering all angles. */
function closedCylinderGarment(
  radius: number,
  body: Float32Array,
  heights = [0.96, 0.98, 1.0, 1.02, 1.04],
  numAngles = 48,
): {
  binding: GarmentBinding;
  rest: Float32Array;
  count: number;
} {
  const targets: [number, number, number][] = [];
  for (const y of heights) {
    for (let ai = 0; ai < numAngles; ai++) {
      const angle = (2 * Math.PI * ai) / numAngles;
      targets.push([radius * Math.cos(angle), y, radius * Math.sin(angle)]);
    }
  }
  const count = targets.length;
  const indices = new Uint32Array(count * 3);
  const weights = new Float32Array(count * 3);
  const offsets = new Float32Array(count * 3);
  const rest = new Float32Array(count * 3);

  const bodyCount = Math.floor(body.length / 3);
  for (let v = 0; v < count; v++) {
    const target = targets[v]!;
    rest[v * 3] = target[0];
    rest[v * 3 + 1] = target[1];
    rest[v * 3 + 2] = target[2];

    let bestB = 0;
    let bestDist = Infinity;
    for (let b = 0; b < bodyCount; b++) {
      const bx = body[b * 3]!;
      const by = body[b * 3 + 1]!;
      const bz = body[b * 3 + 2]!;
      const d = (target[0] - bx) ** 2 + (target[1] - by) ** 2 + (target[2] - bz) ** 2;
      if (d < bestDist) {
        bestDist = d;
        bestB = b;
      }
    }
    indices.fill(bestB, v * 3, v * 3 + 3);
    weights[v * 3] = 1;
    for (let k = 0; k < 3; k++) {
      offsets[v * 3 + k] = target[k]! - body[bestB * 3 + k]!;
    }
  }

  return {
    binding: { count, indices, weights, offsets },
    rest,
    count,
  };
}

/** Inner cylinder positions (and optional local radial bulge). */
function cylinderSurface(
  radius: number,
  heights = [0.95, 0.97, 0.99, 1.01, 1.03, 1.05],
  numAngles = 48,
  bulgeSector?: { sector: number; extraRadius: number },
): Float32Array {
  const positions = new Float32Array(heights.length * numAngles * 3);
  let idx = 0;
  for (const y of heights) {
    for (let ai = 0; ai < numAngles; ai++) {
      const angle = (2 * Math.PI * ai) / numAngles;
      let r = radius;
      if (bulgeSector && ai === bulgeSector.sector) {
        r += bulgeSector.extraRadius;
      }
      positions[idx * 3] = r * Math.cos(angle);
      positions[idx * 3 + 1] = y;
      positions[idx * 3 + 2] = r * Math.sin(angle);
      idx++;
    }
  }
  return positions;
}

describe('gradeGarment outer mode layer-aware shrink limit', () => {
  const body = cylinderBody();
  const sections = createGarmentSections(body);

  it('(1) limits outer shrink so every jacket vertex ends at r >= inner radius + clearance - 1 mm', () => {
    const jacketCyl = closedCylinderGarment(0.25, body);
    const innerPositions = cylinderSurface(0.21);
    const clearance = 0.009; // 9 mm
    const shrinkDelta = (0.19 - 0.25) * 2 * Math.PI;
    const shrinkRing: GradeRing[] = [{ planeY: 1, deltaCircumferenceM: shrinkDelta }];

    const graded = new Float32Array(jacketCyl.count * 3);
    gradeGarment(jacketCyl.rest, jacketCyl.binding, body, shrinkRing, graded, sections, {
      outer: true,
      innerSurface: { positions: innerPositions },
      innerClearance: clearance,
    });

    const minThreshold = 0.21 + clearance - 0.001;
    for (let v = 0; v < jacketCyl.count; v++) {
      const x = graded[v * 3]!;
      const z = graded[v * 3 + 2]!;
      const r = Math.hypot(x, z);
      expect(r).toBeGreaterThanOrEqual(minThreshold);
    }
  });

  it('(2) shrinks to r ~= 0.19 without an inner surface (unchanged behaviour)', () => {
    const jacketCyl = closedCylinderGarment(0.25, body);
    const shrinkDelta = (0.19 - 0.25) * 2 * Math.PI;
    const shrinkRing: GradeRing[] = [{ planeY: 1, deltaCircumferenceM: shrinkDelta }];

    const graded = new Float32Array(jacketCyl.count * 3);
    gradeGarment(jacketCyl.rest, jacketCyl.binding, body, shrinkRing, graded, sections, {
      outer: true,
    });

    for (let v = 0; v < jacketCyl.count; v++) {
      const x = graded[v * 3]!;
      const z = graded[v * 3 + 2]!;
      const r = Math.hypot(x, z);
      expect(r).toBeCloseTo(0.19, 3);
    }
  });

  it('(3) keeps two stacked jacket layers 3 mm apart still 3 mm (±1 mm) apart after the limit', () => {
    const stackedRest = new Float32Array([0, 1, 0.25, 0, 1, 0.253]);
    const stackedTargets: [number, number, number][] = [
      [0, 1, 0.25],
      [0, 1, 0.253],
    ];
    const bodyVerts = [16, 17];
    const indices = new Uint32Array(6);
    const weights = new Float32Array(6);
    const offsets = new Float32Array(6);
    stackedTargets.forEach((target, v) => {
      const b = bodyVerts[v]!;
      indices.fill(b, v * 3, v * 3 + 3);
      weights[v * 3] = 1;
      for (let k = 0; k < 3; k++) offsets[v * 3 + k] = target[k]! - body[b * 3 + k]!;
    });
    const stackedBind: GarmentBinding = { count: 2, indices, weights, offsets };

    const innerPositions = cylinderSurface(0.21);
    const clearance = 0.009;
    const shrinkDelta = (0.19 - 0.25) * 2 * Math.PI;
    const shrinkRing: GradeRing[] = [{ planeY: 1, deltaCircumferenceM: shrinkDelta }];

    const graded = new Float32Array(6);
    gradeGarment(stackedRest, stackedBind, body, shrinkRing, graded, sections, {
      outer: true,
      innerSurface: { positions: innerPositions },
      innerClearance: clearance,
    });

    expect(graded[2]!).toBeGreaterThanOrEqual(0.21 + clearance - 0.001);
    expect(graded[5]!).toBeGreaterThan(graded[2]!);

    const dist = Math.hypot(
      graded[3]! - graded[0]!,
      graded[4]! - graded[1]!,
      graded[5]! - graded[2]!,
    );
    expect(Math.abs(dist - 0.003)).toBeLessThanOrEqual(0.001);
  });

  it('(4) widens only smoothly around an inner surface with a local bulge (one sector 2 cm wider)', () => {
    const jacketCyl = closedCylinderGarment(0.25, body);
    const bulgeSector = 0;
    const innerPositions = cylinderSurface(0.21, undefined, 48, {
      sector: bulgeSector,
      extraRadius: 0.02,
    });
    const clearance = 0.009;
    const shrinkDelta = (0.19 - 0.25) * 2 * Math.PI;
    const shrinkRing: GradeRing[] = [{ planeY: 1, deltaCircumferenceM: shrinkDelta }];

    const graded = new Float32Array(jacketCyl.count * 3);
    gradeGarment(jacketCyl.rest, jacketCyl.binding, body, shrinkRing, graded, sections, {
      outer: true,
      innerSurface: { positions: innerPositions },
      innerClearance: clearance,
    });

    const midYIndex = Math.floor(jacketCyl.count / (5 * 48)) * 48;
    const bulgeVert = midYIndex + bulgeSector;
    const bulgeR = Math.hypot(graded[bulgeVert * 3]!, graded[bulgeVert * 3 + 2]!);
    expect(bulgeR).toBeGreaterThanOrEqual(0.23 + clearance - 0.001);

    const oppositeVert = midYIndex + 24;
    const oppositeR = Math.hypot(graded[oppositeVert * 3]!, graded[oppositeVert * 3 + 2]!);
    expect(oppositeR).toBeLessThan(0.225);

    for (let ai = 0; ai < 48; ai++) {
      const v0 = midYIndex + ai;
      const v1 = midYIndex + ((ai + 1) % 48);
      const r0 = Math.hypot(graded[v0 * 3]!, graded[v0 * 3 + 2]!);
      const r1 = Math.hypot(graded[v1 * 3]!, graded[v1 * 3 + 2]!);
      // The limit tapers by 0.25 m/m of arc: neighbours 3.27 cm apart (r = 0.25 m, 48 vertices)
      // differ by at most ~8.2 mm, instead of the full 2 cm bulge in a single step.
      expect(Math.abs(r1 - r0)).toBeLessThanOrEqual(0.0092);
    }
  });
  it('(5) ignores an inner-top part (a sleeve) reaching far outside the jacket: no spike towards it', () => {
    // Where the pooled body sections merge arm and torso, an inner sleeve shares the torso centre and
    // its tip sits 19 cm beyond the jacket radius. The limit must not drag the jacket out to it.
    const jacketCyl = closedCylinderGarment(0.25, body);
    const innerPositions = cylinderSurface(0.21, undefined, 48, { sector: 0, extraRadius: 0.19 });
    const shrinkDelta = (0.19 - 0.25) * 2 * Math.PI;
    const shrinkRing: GradeRing[] = [{ planeY: 1, deltaCircumferenceM: shrinkDelta }];

    const graded = new Float32Array(jacketCyl.count * 3);
    gradeGarment(jacketCyl.rest, jacketCyl.binding, body, shrinkRing, graded, sections, {
      outer: true,
      innerSurface: { positions: innerPositions },
      innerClearance: 0.009,
    });

    for (let v = 0; v < jacketCyl.count; v++) {
      const r = Math.hypot(graded[v * 3]!, graded[v * 3 + 2]!);
      expect(r).toBeLessThan(0.26);
      expect(r).toBeGreaterThanOrEqual(0.19 - 0.001);
    }
  });
});
