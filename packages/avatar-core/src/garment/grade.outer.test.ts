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
    gradeGarmentOuter(
      openJacket.rest,
      openJacket.binding,
      body,
      shrinkRings,
      graded,
      sections,
    );

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
});
