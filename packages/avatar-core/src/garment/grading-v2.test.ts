import { describe, expect, it } from 'vitest';
import type { GarmentBinding } from './binding';
import { createGarmentSections } from './centroid';
import { gradeGarment } from './grade';
import { garmentClearance } from './heatmap';
import { gradeGarmentLength } from './length';

function cylinders(
  centers: number[],
  radius = 0.08,
): { body: Float32Array; rest: Float32Array; binding: GarmentBinding } {
  const count = centers.length * 10 * 8;
  const body = new Float32Array(count * 3);
  const rest = new Float32Array(count * 3);
  const indices = new Uint32Array(count * 3);
  const weights = new Float32Array(count * 3);
  let v = 0;
  for (const cx of centers)
    for (let level = 0; level < 10; level++)
      for (let k = 0; k < 8; k++) {
        const angle = (k * Math.PI) / 4;
        const dx = Math.cos(angle),
          dz = Math.sin(angle);
        body.set([cx + radius * dx, level * 0.1, radius * dz], v * 3);
        rest.set([cx + (radius + 0.01) * dx, level * 0.1, (radius + 0.01) * dz], v * 3);
        indices[v * 3] = v;
        weights[v * 3] = 1;
        v++;
      }
  return { body, rest, binding: { count, indices, weights, offsets: new Float32Array(count * 3) } };
}

describe('per-limb garment sections', () => {
  it('moves the inner thighs away from each leg axis and reports their own clearance', () => {
    const { body, rest, binding } = cylinders([-0.1, 0.1]);
    const sections = createGarmentSections(body);
    const graded = new Float32Array(rest.length);
    gradeGarment(
      rest,
      binding,
      body,
      [{ planeY: 0.5, deltaCircumferenceM: 0.1 }],
      graded,
      sections,
    );
    for (const [v, cx] of [
      [44, -0.1],
      [124, 0.1],
    ] as const) {
      const before = Math.hypot(rest[v * 3]! - cx, rest[v * 3 + 2]!);
      const after = Math.hypot(graded[v * 3]! - cx, graded[v * 3 + 2]!);
      expect(after - before).toBeCloseTo(0.1 / (2 * Math.PI), 4);
    }
    const clearance = new Float32Array(binding.count);
    garmentClearance(graded, binding, body, clearance, sections);
    expect(Math.min(...clearance)).toBeGreaterThan(0.02);
  });

  it('does not let distant arms pull the torso centroid sideways', () => {
    const { body, rest, binding } = cylinders([-0.6, 0.1, 0.7]);
    const sections = createGarmentSections(body);
    const torso = sections.centroidAt(0.5, 0.18, 0);
    expect(torso.x).toBeCloseTo(0.1, 3);
    const graded = new Float32Array(rest.length);
    gradeGarment(
      rest,
      binding,
      body,
      [{ planeY: 0.5, deltaCircumferenceM: 0.1 }],
      graded,
      sections,
    );
    const innerTorso = 80 + 5 * 8 + 4;
    expect(graded[innerTorso * 3]!).toBeLessThan(rest[innerTorso * 3]!);
  });

  it('separates components at different depths when their X ranges overlap', () => {
    const { body, binding } = cylinders([0.1, 0.1]);
    for (let v = binding.count / 2; v < binding.count; v++)
      body[v * 3 + 2] = body[v * 3 + 2]! + 0.5;
    const sections = createGarmentSections(body);
    expect(sections.centroidAt(0.5, 0.1, 0).z).toBeCloseTo(0, 4);
    expect(sections.centroidAt(0.5, 0.1, 0.5).z).toBeCloseTo(0.5, 4);
  });
});

describe('length grading', () => {
  const { body, rest, binding } = cylinders([0.1]);
  it('moves the hem by the chart difference and leaves the anchor fixed', () => {
    const out = new Float32Array(rest.length);
    gradeGarmentLength(
      rest,
      binding,
      body,
      {
        chartCm: 73,
        referenceCm: 70,
        anchor: [0, 0.9, 0],
        hem: [0, 0, 0],
      },
      out,
    );
    expect(out[1]).toBeCloseTo(rest[1]! - 0.03, 5);
    expect(out[9 * 8 * 3 + 1]).toBeCloseTo(rest[9 * 8 * 3 + 1]!, 5);
  });

  it('clamps a longer trouser hem at the floor', () => {
    const out = new Float32Array(rest.length);
    gradeGarmentLength(
      rest,
      binding,
      body,
      {
        chartCm: 90,
        referenceCm: 70,
        anchor: [0, 0.9, 0],
        hem: [0, 0.01, 0],
        floorY: 0,
      },
      out,
    );
    expect(Math.min(...Array.from(out).filter((_, i) => i % 3 === 1))).toBeGreaterThanOrEqual(
      -1e-6,
    );
  });
});
