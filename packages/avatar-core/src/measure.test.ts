import { describe, expect, it } from 'vitest';
import type { MeasureDef } from './contracts';
import { estimateMassKg, measure, meshVolumeM3 } from './measure';
import { getSyntheticBody } from './testing/syntheticBody';

function ring(n: number, radius: number, y = 0): { pos: Float32Array; verts: number[] } {
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n;
    pos.set([radius * Math.cos(a), y, radius * Math.sin(a)], i * 3);
  }
  return { pos, verts: Array.from({ length: n }, (_, i) => i) };
}

const circ = (verts: number[]): MeasureDef => ({
  id: 'waist',
  type: 'circumference',
  verts,
  drivers: [],
});

/** Rotates every point about the x axis and then the z axis (tilted loop plane). */
function rotate(pos: Float32Array, ax: number, az: number): Float32Array {
  const out = new Float32Array(pos.length);
  for (let i = 0; i < pos.length; i += 3) {
    const [x, y, z] = [pos[i]!, pos[i + 1]!, pos[i + 2]!];
    const y1 = y * Math.cos(ax) - z * Math.sin(ax);
    const z1 = y * Math.sin(ax) + z * Math.cos(ax);
    const x2 = x * Math.cos(az) - y1 * Math.sin(az);
    const y2 = x * Math.sin(az) + y1 * Math.cos(az);
    out.set([x2, y2, z1], i);
  }
  return out;
}

describe('measure: circumference (convex-hull tape measure)', () => {
  it('equals the exact perimeter of a regular N-gon ring', () => {
    for (const n of [3, 4, 7, 24, 124]) {
      const r = 0.137;
      const { pos, verts } = ring(n, r, 1.1);
      const exact = n * 2 * r * Math.sin(Math.PI / n);
      expect(measure(circ(verts), pos)).toBeCloseTo(exact, 7);
    }
  });

  it('is invariant under rotation of the loop plane (tilted chest / hip loops)', () => {
    const { pos, verts } = ring(60, 0.15);
    const exact = measure(circ(verts), pos);
    for (const [ax, az] of [
      [0.6, 0],
      [0, 1.1],
      [-0.9, 0.7],
      [Math.PI / 2, 0],
    ] as const)
      expect(measure(circ(verts), rotate(pos, ax, az))).toBeCloseTo(exact, 6);
  });

  it('returns the hull perimeter (not the loop length) for a concave star loop', () => {
    const points = 5;
    const outer = 0.2;
    const inner = 0.08;
    const pos = new Float32Array(points * 2 * 3);
    for (let i = 0; i < points * 2; i++) {
      const a = (Math.PI * i) / points;
      const rad = i % 2 === 0 ? outer : inner;
      pos.set([rad * Math.cos(a), 0, rad * Math.sin(a)], i * 3);
    }
    const verts = Array.from({ length: points * 2 }, (_, i) => i);
    const hull = points * 2 * outer * Math.sin(Math.PI / points);
    const starLoop = measure(
      { id: 'x' as never, type: 'polyline', verts: [...verts, 0], drivers: [] },
      pos,
    );
    expect(starLoop).toBeGreaterThan(hull * 1.2);
    expect(measure(circ(verts), pos)).toBeCloseTo(hull, 6);
    expect(measure(circ(verts), rotate(pos, 0.8, 0.3))).toBeCloseTo(hull, 6);
  });

  it('ignores interior points and duplicated vertices', () => {
    const { pos, verts } = ring(32, 0.1);
    const more = new Float32Array(pos.length + 9);
    more.set(pos);
    more.set([0.01, 0, 0.02, 0, 0, 0, 0.05, 0, -0.03], pos.length);
    const v2 = [...verts, 32, 33, 34, 0, 5];
    expect(measure(circ(v2), more)).toBeCloseTo(measure(circ(verts), pos), 6);
  });

  it('measures the synthetic ring (vertical stack) like the analytic polygon', () => {
    const fx = getSyntheticBody();
    const r = fx.ringAt(0.62);
    const rad = fx.radiusAt(r / (fx.rings - 1));
    const n = fx.segments;
    const exact = n * 2 * rad * Math.sin(Math.PI / n);
    expect(
      measure(
        fx.measures.measures.find((m) => m.id === 'waist')!,
        fx.base,
      ),
    ).toBeCloseTo(exact, 5);
  });
});

describe('measure: other kinds', () => {
  const pos = Float32Array.from([0, 0, 0, 3, 4, 0, 3, 4, 12, -1, 2, 5, 0, 9, 0]);

  it('distance (euclidean and axis-only)', () => {
    expect(
      measure({ id: 'shoulder', type: 'distance', verts: [0, 1], drivers: [] }, pos),
    ).toBeCloseTo(5, 12);
    expect(
      measure({ id: 'shoulder', type: 'distance', verts: [0, 1], axis: 'x', drivers: [] }, pos),
    ).toBeCloseTo(3, 12);
    expect(
      measure({ id: 'shoulder', type: 'distance', verts: [1, 0], axis: 'y', drivers: [] }, pos),
    ).toBeCloseTo(4, 12);
    expect(
      measure({ id: 'footLength', type: 'distance', verts: [1, 2], axis: 'z', drivers: [] }, pos),
    ).toBeCloseTo(12, 12);
  });

  it('polyline length is open (no closing segment)', () => {
    expect(
      measure({ id: 'armLength', type: 'polyline', verts: [0, 1, 2], drivers: [] }, pos),
    ).toBeCloseTo(17, 12);
  });

  it('height is the bbox Y extent over render vertices only', () => {
    const def: MeasureDef = { id: 'height', type: 'height', drivers: [] };
    expect(measure(def, pos)).toBeCloseTo(9, 12);
    expect(measure(def, pos, 4)).toBeCloseTo(4, 12); // vertex 4 (y = 9) is a joint point
  });

  it('vertexHeight is the vertex y minus the render-vertex floor', () => {
    const p = Float32Array.from([0, 0.02, 0, 0, 1.5, 0, 0, 0.9, 0, 0, -5, 0]);
    expect(measure({ id: 'inseam', type: 'vertexHeight', vert: 2, drivers: [] }, p, 3)).toBeCloseTo(
      0.88,
      6,
    );
  });

  it('throws on invalid vertex indices and too-short loops', () => {
    expect(() => measure(circ([0, 1, 99]), pos)).toThrow(/out of range/);
    expect(() => measure(circ([0, 1]), pos)).toThrow(/at least 3/);
    expect(() =>
      measure({ id: 'x' as never, type: 'distance', verts: [0, -1], drivers: [] }, pos),
    ).toThrow(/out of range/);
  });
});

/** Closed prism with `n` sides, caps included; returns positions and triangle indices. */
function prism(n: number, radius: number, height: number, offset: [number, number, number]) {
  const pos: number[] = [];
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n;
    const x = radius * Math.cos(a) + offset[0];
    const z = radius * Math.sin(a) + offset[2];
    pos.push(x, offset[1], z, x, offset[1] + height, z);
  }
  const bottom = n * 2;
  const top = n * 2 + 1;
  pos.push(offset[0], offset[1], offset[2], offset[0], offset[1] + height, offset[2]);
  const idx: number[] = [];
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    idx.push(2 * i, 2 * j + 1, 2 * j, 2 * i, 2 * i + 1, 2 * j + 1);
    idx.push(bottom, 2 * i, 2 * j);
    idx.push(top, 2 * j + 1, 2 * i + 1);
  }
  return { pos: Float32Array.from(pos), idx: Uint32Array.from(idx) };
}

describe('mass estimate', () => {
  it('matches the analytic volume of a cylinder within 1%', () => {
    const r = 0.15;
    const h = 1.7;
    const { pos, idx } = prism(96, r, h, [0, 0, 0]);
    const analyticKg = Math.PI * r * r * h * 1000 * 1.01;
    const est = estimateMassKg(pos, idx);
    expect(Math.abs(est - analyticKg) / analyticKg).toBeLessThan(0.01);
    // exact polygon volume
    expect(meshVolumeM3(pos, idx)).toBeCloseTo(
      0.5 * 96 * r * r * Math.sin((2 * Math.PI) / 96) * h,
      7,
    );
  });

  it('is translation- and winding-invariant and honours the density argument', () => {
    const a = prism(48, 0.1, 1, [0, 0, 0]);
    const b = prism(48, 0.1, 1, [40, -7, 3]);
    expect(estimateMassKg(b.pos, b.idx)).toBeCloseTo(estimateMassKg(a.pos, a.idx), 3);
    const flipped = Uint32Array.from(a.idx);
    for (let i = 0; i < flipped.length; i += 3) {
      const t = flipped[i + 1]!;
      flipped[i + 1] = flipped[i + 2]!;
      flipped[i + 2] = t;
    }
    expect(estimateMassKg(a.pos, flipped)).toBeCloseTo(estimateMassKg(a.pos, a.idx), 9);
    expect(estimateMassKg(a.pos, a.idx, 2.02)).toBeCloseTo(
      2 * estimateMassKg(a.pos, a.idx, 1.01),
      9,
    );
  });

  it('rejects malformed index buffers', () => {
    const { pos } = prism(8, 0.1, 1, [0, 0, 0]);
    expect(() => estimateMassKg(pos, [0, 1])).toThrow(/multiple of 3/);
    expect(() => estimateMassKg(pos, [0, 1, 999])).toThrow(/outside/);
  });

  it('gives a plausible body mass for the synthetic fixture', () => {
    const fx = getSyntheticBody();
    const kg = estimateMassKg(fx.base, fx.indices, 1.01, fx.manifest.renderVertexCount);
    expect(kg).toBeGreaterThan(40);
    expect(kg).toBeLessThan(120);
  });
});
