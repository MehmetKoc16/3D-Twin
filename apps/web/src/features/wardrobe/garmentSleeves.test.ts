import { describe, expect, it } from 'vitest';
import { createGarmentSections, type GarmentBinding, type Vec3 } from '@dt/avatar-core';
import { containInnerSleeves, pushOutside, sleeveWeight, type Surface } from './garmentCollide';
import { fitToChart } from './garmentGrading';

/** A separated arm with 1 cm rings; the inner short sleeve ends abruptly at y = 1.2. */
function fixture() {
  const sectors = 48;
  const rows = 51;
  const body: number[] = [];
  const rest: number[] = [];
  const inner: number[] = [];
  const innerBound: number[] = [];
  const index: number[] = [];
  const innerIndex: number[] = [];
  for (let row = 0; row < rows; row++) {
    const y = 0.9 + row * 0.01;
    for (let s = 0; s < sectors; s++) {
      const angle = (s * 2 * Math.PI) / sectors;
      const x = Math.cos(angle);
      const z = Math.sin(angle);
      body.push(0.4 + 0.035 * x, y, 0.035 * z);
      rest.push(0.4 + 0.075 * x, y, 0.075 * z);
      if (row >= 30) {
        inner.push(0.4 + 0.065 * x, y, 0.065 * z);
        innerBound.push(0.4 + 0.035 * x, y, 0.035 * z);
      }
      if (row > 0) {
        const a = (row - 1) * sectors + s;
        const b = (row - 1) * sectors + ((s + 1) % sectors);
        index.push(a, b, a + sectors, b, b + sectors, a + sectors);
        if (row > 30) innerIndex.push(...index.slice(-6).map((v) => v - 30 * sectors));
      }
    }
  }
  const count = rest.length / 3;
  const binding: GarmentBinding = {
    count,
    indices: Uint32Array.from({ length: count * 3 }, (_, i) => Math.floor(i / 3)),
    weights: Float32Array.from({ length: count * 3 }, (_, i) => (i % 3 === 0 ? 1 : 0)),
    offsets: Float32Array.from(rest, (value, i) => value - body[i]!),
  };
  const polo: Surface & { positions: Float32Array } = {
    positions: Float32Array.from(inner),
    bound: Float32Array.from(innerBound),
    index: innerIndex,
  };
  const fit = (innerSurface?: Surface, contain = false) => {
    const result = new Float32Array(rest.length);
    fitToChart(
      {
        def: {
          id: 'jacket',
          slot: 'outer',
          category: 'top',
          kind: 'longsleeve',
          label: { tr: 'Ceket', en: 'Jacket' },
          license: 'CC0-1.0',
          mesh: 'test.glb',
          nativeMeasures: { chest: 100 },
          defaultEase: {},
          layer: 3,
          baseColor: '#888888',
        },
        item: {
          id: 'jacket',
          templateId: 'jacket',
          name: 'Jacket',
          color: '#888888',
          sizes: ['M'],
          selectedSize: 'M',
          chart: { chest: [87.4] },
        },
        binding,
        body,
        sections: createGarmentSections(body),
        achievedCm: {},
        planes: { planeY: { chest: 1 } },
        joint: (name) =>
          name === 'upperarm_l'
            ? arms[0]!.shoulder
            : name === 'lowerarm_l'
              ? arms[0]!.elbow
              : undefined,
        innerSurface,
        containInnerSleeves: contain,
      },
      new Float32Array(rest.length),
      result,
    );
    if (innerSurface)
      pushOutside(result, count, [innerSurface], {
        orientation: 'surface',
        marginM: 0.012,
        frontDepthOnly: true,
        vertexWeights: contain
          ? Float32Array.from({ length: count }, (_, v) => 1 - sleeveWeight(body, v * 3, arms))
          : undefined,
      });
    return result;
  };
  const maxRingJump = (p: Float32Array) => {
    let max = 0;
    for (let row = 1; row < rows; row++) {
      for (let s = 0; s < sectors; s++) {
        const o = (row * sectors + s) * 3;
        const previous = o - sectors * 3;
        const radius = Math.hypot(p[o]! - 0.4, p[o + 2]!);
        const previousRadius = Math.hypot(p[previous]! - 0.4, p[previous + 2]!);
        max = Math.max(max, Math.abs(radius - previousRadius));
      }
    }
    return max;
  };
  return { polo, fit, maxRingJump, outer: { positions: fit(), index, bound: body } };
}

const arms: { shoulder: Vec3; elbow: Vec3 }[] = [{ shoulder: [0.4, 1.6, 0], elbow: [0.4, 1.1, 0] }];

describe('polo sleeves inside the denim chart sleeve', () => {
  it('reproduces the cuff step, then keeps adjacent 1 cm rings below 3 mm after both fit passes', () => {
    const { polo, fit, maxRingJump, outer } = fixture();
    const before = maxRingJump(fit(polo));
    expect(before).toBeGreaterThan(0.003);
    expect(containInnerSleeves(polo.positions, polo.bound, outer, arms)).toBeGreaterThan(0);
    const after = maxRingJump(fit(polo, true));
    expect(after).toBeLessThan(0.003);
    expect(after).toBeLessThan(before / 2);
  });

  it('keeps the polo sleeve inside the jacket with skin clearance and leaves torso vertices unchanged', () => {
    const { polo, outer } = fixture();
    const originalLength = polo.positions.length;
    const positions = Float32Array.from([...polo.positions, 0.1, 1.2, 0.08]);
    const bound = Float32Array.from([...Array.from(polo.bound), 0.08, 1.2, 0.07]);
    containInnerSleeves(positions, bound, outer, arms);
    for (let o = 0; o < originalLength; o += 3) {
      expect(Math.hypot(positions[o]! - 0.4, positions[o + 2]!)).toBeLessThan(0.055);
      const clearance = Math.hypot(...[0, 1, 2].map((k) => positions[o + k]! - bound[o + k]!));
      expect(clearance).toBeGreaterThanOrEqual(0.001999);
    }
    expect(Array.from(positions.slice(originalLength))).toEqual(
      Array.from(Float32Array.from([0.1, 1.2, 0.08])),
    );
  });

  it('leaves an uncovered inner sleeve unchanged when no jacket ray is hit', () => {
    const positions = Float32Array.from([0.4, 1.3, 0.08]);
    const original = positions.slice();
    expect(
      containInnerSleeves(
        positions,
        [0.4, 1.3, 0.04],
        {
          positions: [],
          bound: [],
          index: [],
        },
        arms,
      ),
    ).toBe(0);
    expect(positions).toEqual(original);
  });
});
