import { describe, expect, it } from 'vitest';
import type { GarmentBinding } from './binding';
import { GarmentSections } from './centroid';
import { gradeGarment, type GradeRing } from './grade';

/** Kernel radius the web app passes for outer-slot garments. */
const SMOOTH = { smoothRadiusM: 0.015 };

/**
 * Creates a synthetic cylindrical body of radius 0.2 m centered at (0, y, 0),
 * with 64 vertices at height y = 1.0 m so GarmentSections identifies a single
 * limb component centered at (0, 0).
 */
function createSyntheticCylinderBody(): Float32Array {
  const numVerts = 64;
  const body = new Float32Array(numVerts * 3);
  for (let i = 0; i < numVerts; i++) {
    const angle = (2 * Math.PI * i) / numVerts;
    body[i * 3] = 0.2 * Math.cos(angle);
    body[i * 3 + 1] = 1.0;
    body[i * 3 + 2] = 0.2 * Math.sin(angle);
  }
  return body;
}

/** Helper to create a GarmentBinding where each vertex is bound to a single body vertex. */
function createSimpleBinding(count: number, bodyVertexIndices: number[]): GarmentBinding {
  const indices = new Uint32Array(count * 3);
  const weights = new Float32Array(count * 3);
  const offsets = new Float32Array(count * 3);
  for (let v = 0; v < count; v++) {
    const b = bodyVertexIndices[v] ?? 0;
    indices[v * 3] = b;
    indices[v * 3 + 1] = b;
    indices[v * 3 + 2] = b;
    weights[v * 3] = 1.0;
    weights[v * 3 + 1] = 0.0;
    weights[v * 3 + 2] = 0.0;
  }
  return { count, indices, weights, offsets };
}

/** The grading ray of a body vertex: unit horizontal direction from its section centroid. */
function gradingRay(
  body: Float32Array,
  sections: GarmentSections,
  bodyVertex: number,
): { bx: number; bz: number; dx: number; dz: number } {
  const bx = body[bodyVertex * 3]!;
  const bz = body[bodyVertex * 3 + 2]!;
  const c = sections.centroidAt(body[bodyVertex * 3 + 1]!, bx, bz);
  const len = Math.hypot(bx - c.x, bz - c.z);
  return { bx, bz, dx: (bx - c.x) / len, dz: (bz - c.z) / len };
}

interface StackedSheets {
  rest: Float32Array;
  binding: GarmentBinding;
  sheetCount: number;
}

/**
 * Two parallel 5x5 sheets 3 mm apart: inner at z = 0.203 bound to body vertex 16, outer at z = 0.206
 * bound to the neighbouring body vertex 17 (a different body triangle, so a different clearance).
 */
function createStackedSheets(): StackedSheets {
  const gridSize = 5;
  const sheetCount = gridSize * gridSize;
  const totalCount = sheetCount * 2;
  const rest = new Float32Array(totalCount * 3);
  const bodyIndices: number[] = [];
  for (let layer = 0; layer < 2; layer++) {
    for (let i = 0; i < sheetCount; i++) {
      const v = layer * sheetCount + i;
      const gx = (i % gridSize) - Math.floor(gridSize / 2);
      const gy = Math.floor(i / gridSize) - Math.floor(gridSize / 2);
      rest[v * 3] = gx * 0.005;
      rest[v * 3 + 1] = 1.0 + gy * 0.005;
      rest[v * 3 + 2] = layer === 0 ? 0.203 : 0.206;
      bodyIndices.push(layer === 0 ? 16 : 17);
    }
  }
  return { rest, binding: createSimpleBinding(totalCount, bodyIndices), sheetCount };
}

describe('gradeGarment validation', () => {
  const body = createSyntheticCylinderBody();
  const binding = createSimpleBinding(2, [0, 1]);

  it('throws when position buffers are too short', () => {
    const rest = new Float32Array(3);
    const out = new Float32Array(6);
    expect(() => gradeGarment(rest, binding, body, [], out)).toThrow(/buffer too short/);
    expect(() => gradeGarment(new Float32Array(6), binding, body, [], new Float32Array(3))).toThrow(
      /buffer too short/,
    );
  });

  it('throws on non-finite ring parameters', () => {
    const rest = new Float32Array(6);
    const out = new Float32Array(6);
    expect(() =>
      gradeGarment(rest, binding, body, [{ planeY: NaN, deltaCircumferenceM: 0 }], out),
    ).toThrow(/invalid ring/);
    expect(() =>
      gradeGarment(rest, binding, body, [{ planeY: 1.0, deltaCircumferenceM: Infinity }], out),
    ).toThrow(/invalid ring/);
  });

  it('returns cleanly on an empty binding', () => {
    const emptyBinding = createSimpleBinding(0, []);
    const out = new Float32Array(0);
    expect(() => gradeGarment(new Float32Array(0), emptyBinding, body, [], out)).not.toThrow();
  });
});

describe('gradeGarment single-layer behaviour', () => {
  const body = createSyntheticCylinderBody();
  const sections = new GarmentSections(body);
  // Body vertex at angle pi/2: x ~ 0, z ~ 0.200, y = 1.0
  const frontBodyVertex = 16; // 64 * (pi/2) / (2*pi) = 16

  it('leaves rest positions unchanged when delta is zero and clearance exceeds 2 mm', () => {
    const count = 9;
    const rest = new Float32Array(count * 3);
    const bodyIndices: number[] = [];
    for (let i = 0; i < count; i++) {
      rest[i * 3] = ((i % 3) - 1) * 0.01;
      rest[i * 3 + 1] = 1.0 + (Math.floor(i / 3) - 1) * 0.01;
      rest[i * 3 + 2] = 0.208; // 8 mm clearance from body at 0.200
      bodyIndices.push(frontBodyVertex);
    }
    const binding = createSimpleBinding(count, bodyIndices);
    const out = new Float32Array(count * 3);
    const rings: GradeRing[] = [{ planeY: 1.0, deltaCircumferenceM: 0 }];

    gradeGarment(rest, binding, body, rings, out, sections);

    for (let i = 0; i < rest.length; i++) {
      expect(out[i]).toBeCloseTo(rest[i]!, 5);
    }
  });

  it('expands outward by approximately delta / 2pi for positive delta', () => {
    const count = 9;
    const rest = new Float32Array(count * 3);
    const bodyIndices: number[] = [];
    for (let i = 0; i < count; i++) {
      rest[i * 3] = ((i % 3) - 1) * 0.01;
      rest[i * 3 + 1] = 1.0;
      rest[i * 3 + 2] = 0.208;
      bodyIndices.push(frontBodyVertex);
    }
    const binding = createSimpleBinding(count, bodyIndices);
    const out = new Float32Array(count * 3);
    const deltaCircumferenceM = 0.05;
    const expectedShift = deltaCircumferenceM / (2 * Math.PI);
    const rings: GradeRing[] = [{ planeY: 1.0, deltaCircumferenceM }];

    gradeGarment(rest, binding, body, rings, out, sections);

    for (let i = 0; i < count; i++) {
      expect(out[i * 3 + 2]! - rest[i * 3 + 2]!).toBeCloseTo(expectedShift, 4);
      expect(out[i * 3 + 1]).toBe(rest[i * 3 + 1]);
    }
  });

  it('clamps at minimum 2 mm clearance under large negative delta', () => {
    const count = 1;
    const rest = new Float32Array([0, 1.0, 0.203]); // 3 mm clearance
    const binding = createSimpleBinding(count, [frontBodyVertex]);
    const out = new Float32Array(3);
    // Large shrinkage delta (-10 cm -> -1.59 cm shift)
    const rings: GradeRing[] = [{ planeY: 1.0, deltaCircumferenceM: -0.1 }];

    gradeGarment(rest, binding, body, rings, out, sections);

    const bodyZ = body[frontBodyVertex * 3 + 2]!;
    const finalClearance = out[2]! - bodyZ;
    expect(finalClearance).toBeGreaterThanOrEqual(0.00199);
    expect(finalClearance).toBeCloseTo(0.002, 4);
  });

  it('with a 1.5 cm kernel, keeps 2 mm clearance and inflates a mixed-clearance sheet by at most 1.2 mm over legacy', () => {
    const count = 25;
    const rest = new Float32Array(count * 3);
    const bodyIndices: number[] = [];

    // A smooth cylinder sheet arc with ~4 mm vertex spacing (well within the 1.5 cm kernel)
    // and mixed clearances (every third vertex 1 mm tight, the others 5 mm loose).
    for (let i = 0; i < count; i++) {
      const angle = Math.PI / 2 + (i - Math.floor(count / 2)) * 0.02; // ~4 mm arc spacing at r = 0.2 m
      const clearance = i % 3 === 0 ? 0.001 : 0.005;
      const r = 0.2 + clearance;
      rest[i * 3] = r * Math.cos(angle);
      rest[i * 3 + 1] = 1.0;
      rest[i * 3 + 2] = r * Math.sin(angle);
      bodyIndices.push(((Math.round((64 * angle) / (2 * Math.PI)) % 64) + 64) % 64);
    }

    const binding = createSimpleBinding(count, bodyIndices);
    const smoothed = new Float32Array(count * 3);
    const legacy = new Float32Array(count * 3);
    let maxInflation = 0;

    for (const delta of [0, 0.04, -0.04]) {
      const rings: GradeRing[] = [{ planeY: 1.0, deltaCircumferenceM: delta }];
      gradeGarment(rest, binding, body, rings, smoothed, sections, SMOOTH);
      gradeGarment(rest, binding, body, rings, legacy, sections);

      for (let i = 0; i < count; i++) {
        const o = i * 3;
        const { bx, bz, dx, dz } = gradingRay(body, sections, bodyIndices[i]!);
        // The re-clamp keeps the legacy body clearance guarantee.
        const clearance = (smoothed[o]! - bx) * dx + (smoothed[o + 2]! - bz) * dz;
        expect(clearance).toBeGreaterThanOrEqual(0.002 - 1e-6);
        // Loose vertices average in their tight neighbours' larger legacy shift: with this kernel the
        // tight share is at most ~26 % of the ~4 mm legacy-shift spread (~1.05 mm); tight vertices are
        // re-clamped to their own legacy shift.
        const inflation =
          (smoothed[o]! - legacy[o]!) * dx + (smoothed[o + 2]! - legacy[o + 2]!) * dz;
        expect(inflation).toBeLessThanOrEqual(0.0012);
        maxInflation = Math.max(maxInflation, inflation);
      }
    }
    // The smoothing really acts on this sheet, so the bound above is not vacuous.
    expect(maxInflation).toBeGreaterThan(0.0001);
  });
});

describe('gradeGarment legacy default', () => {
  const body = createSyntheticCylinderBody();
  const sections = new GarmentSections(body);

  it('without smoothRadiusM grades every vertex independently (exact legacy output)', () => {
    const { rest, binding, sheetCount } = createStackedSheets();
    const count = binding.count;
    const rings: GradeRing[] = [{ planeY: 1.0, deltaCircumferenceM: -0.06 }];

    const byDefault = new Float32Array(count * 3);
    const byZero = new Float32Array(count * 3);
    gradeGarment(rest, binding, body, rings, byDefault, sections);
    gradeGarment(rest, binding, body, rings, byZero, sections, { smoothRadiusM: 0 });
    expect(Array.from(byDefault)).toEqual(Array.from(byZero));

    // Legacy = each vertex graded on its own, bit for bit.
    for (let v = 0; v < count; v++) {
      const single = createSimpleBinding(1, [binding.indices[v * 3]!]);
      const singleOut = new Float32Array(3);
      gradeGarment(rest.subarray(v * 3, v * 3 + 3), single, body, rings, singleOut, sections);
      expect(Array.from(singleOut)).toEqual(Array.from(byDefault.subarray(v * 3, v * 3 + 3)));
    }

    // -9.5 mm raw shift: both sheets clamp to exactly 2 mm from their own body point.
    for (let v = 0; v < count; v++) {
      const { bx, bz, dx, dz } = gradingRay(body, sections, v < sheetCount ? 16 : 17);
      const clearance = (byDefault[v * 3]! - bx) * dx + (byDefault[v * 3 + 2]! - bz) * dz;
      expect(clearance).toBeCloseTo(0.002, 6);
    }
  });
});

describe('gradeGarment stacked parts and layer preservation (smoothed)', () => {
  const body = createSyntheticCylinderBody();
  const sections = new GarmentSections(body);
  const frontBodyVertex = 16;
  // A neighbouring body vertex representing a different triangle
  const nearbyBodyVertex = 17;

  it('keeps two stacked sheets 3 mm apart in order and at least 1 mm apart, where legacy collapses them', () => {
    const { rest, binding, sheetCount } = createStackedSheets();
    const totalCount = binding.count;
    const out = new Float32Array(totalCount * 3);
    const legacy = new Float32Array(totalCount * 3);
    const restSeparation = 0.003;
    const inner = gradingRay(body, sections, frontBodyVertex);

    for (const delta of [0.06, -0.06]) {
      const rings: GradeRing[] = [{ planeY: 1.0, deltaCircumferenceM: delta }];
      gradeGarment(rest, binding, body, rings, out, sections, SMOOTH);
      gradeGarment(rest, binding, body, rings, legacy, sections);

      let minSeparation = Infinity;
      let minLegacySeparation = Infinity;
      for (let i = 0; i < sheetCount; i++) {
        const innerZ = out[i * 3 + 2]!;
        const outerZ = out[(sheetCount + i) * 3 + 2]!;
        const separation = outerZ - innerZ;

        // Outer sheet stays outside the inner sheet, never closer than 1 mm, never pulled apart.
        expect(separation).toBeGreaterThanOrEqual(0.001);
        expect(separation).toBeLessThanOrEqual(restSeparation + 0.001);
        // The inner sheet keeps its 2 mm body clearance.
        const clearance = (out[i * 3]! - inner.bx) * inner.dx + (innerZ - inner.bz) * inner.dz;
        expect(clearance).toBeGreaterThanOrEqual(0.002 - 1e-6);

        minSeparation = Math.min(minSeparation, separation);
        minLegacySeparation = Math.min(
          minLegacySeparation,
          legacy[(sheetCount + i) * 3 + 2]! - legacy[i * 3 + 2]!,
        );
      }
      if (delta > 0) {
        // Pure expansion: no clamps, the stack moves as a whole.
        expect(Math.abs(minSeparation - restSeparation)).toBeLessThanOrEqual(0.001);
      } else {
        // Shrinkage: legacy clamps each sheet to its own body point and nearly collapses the stack.
        expect(minSeparation).toBeGreaterThan(minLegacySeparation + 0.0005);
      }
    }
  });

  it('keeps a small button piece 2 mm in front of a panel in front after grading', () => {
    const panelGrid = 5;
    const panelCount = panelGrid * panelGrid;
    const buttonCount = 4; // 2x2 button piece
    const totalCount = panelCount + buttonCount;
    const rest = new Float32Array(totalCount * 3);
    const bodyIndices: number[] = [];

    // Panel vertices at z = 0.204 (4 mm clearance)
    for (let i = 0; i < panelCount; i++) {
      const gx = (i % panelGrid) - Math.floor(panelGrid / 2);
      const gy = Math.floor(i / panelGrid) - Math.floor(panelGrid / 2);
      rest[i * 3] = gx * 0.005;
      rest[i * 3 + 1] = 1.0 + gy * 0.005;
      rest[i * 3 + 2] = 0.204;
      bodyIndices.push(frontBodyVertex);
    }

    // Button vertices sitting 2 mm in front of panel: z = 0.206, bound to nearby body triangle
    for (let i = 0; i < buttonCount; i++) {
      const v = panelCount + i;
      const bx = (i % 2) * 0.002 - 0.001;
      const by = Math.floor(i / 2) * 0.002 - 0.001;
      rest[v * 3] = bx;
      rest[v * 3 + 1] = 1.0 + by;
      rest[v * 3 + 2] = 0.206;
      bodyIndices.push(nearbyBodyVertex);
    }

    const binding = createSimpleBinding(totalCount, bodyIndices);
    const out = new Float32Array(totalCount * 3);

    // Test with both negative (shrinkage) and positive (expansion) deltas
    for (const delta of [-0.08, 0.06]) {
      const rings: GradeRing[] = [{ planeY: 1.0, deltaCircumferenceM: delta }];
      gradeGarment(rest, binding, body, rings, out, sections, SMOOTH);

      // Center panel vertex is index 12 (gx=0, gy=0)
      const centerPanelZ = out[12 * 3 + 2]!;
      for (let b = 0; b < buttonCount; b++) {
        const buttonZ = out[(panelCount + b) * 3 + 2]!;
        expect(buttonZ).toBeGreaterThan(centerPanelZ);
      }
    }
  });
});
