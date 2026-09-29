import { describe, expect, it } from 'vitest';
import { measure } from '../measure';
import { parseMorphs, validateManifest } from '../morphs';
import { validateRig } from '../joints';
import { createSyntheticBody, getSyntheticBody, mulberry32, packMorphs } from './syntheticBody';

describe('synthetic fixture', () => {
  const fx = getSyntheticBody();

  it('has the advertised size and is internally consistent', () => {
    expect(fx.manifest.renderVertexCount).toBe(120 * 124 + 2);
    expect(fx.manifest.vertexCount).toBe(fx.manifest.renderVertexCount + 5);
    expect(fx.manifest.modifiers).toHaveLength(20);
    expect(fx.manifest.targets.filter((t) => t.group === 'macro')).toHaveLength(54);
    expect(() => validateManifest(fx.manifest)).not.toThrow();
    expect(() => parseMorphs(fx.morphs, fx.manifest)).not.toThrow();
    expect(() => validateRig(fx.rig, fx.manifest.vertexCount)).not.toThrow();
    expect(fx.base.length).toBe(fx.manifest.vertexCount * 3);
    expect(
      measure(
        { id: 'height', type: 'height', drivers: [] },
        fx.base,
        fx.manifest.renderVertexCount,
      ),
    ).toBeCloseTo(1.7, 6);
  });

  it('is a closed, consistently oriented (watertight) triangle mesh', () => {
    const directed = new Map<string, number>();
    for (let t = 0; t < fx.indices.length; t += 3)
      for (let k = 0; k < 3; k++) {
        const a = fx.indices[t + k]!;
        const b = fx.indices[t + ((k + 1) % 3)]!;
        directed.set(`${a}>${b}`, (directed.get(`${a}>${b}`) ?? 0) + 1);
      }
    for (const [edge, count] of directed) {
      expect(count).toBe(1);
      const [a, b] = edge.split('>');
      expect(directed.get(`${b}>${a}`)).toBe(1);
    }
  });

  it('is deterministic for a seed and configurable', () => {
    const a = createSyntheticBody({ rings: 30, segments: 16, seed: 5 });
    const b = createSyntheticBody({ rings: 30, segments: 16, seed: 5 });
    expect(new Uint8Array(a.morphs)).toEqual(new Uint8Array(b.morphs));
    expect(a.manifest.renderVertexCount).toBe(30 * 16 + 2);
  });

  it('packMorphs writes little-endian 16-byte entries', () => {
    const { buffer, targets } = packMorphs([
      { id: 'a', group: 'body', vertices: [5, 9], deltas: [1, 2, 3, 4, 5, 6] },
      { id: 'b', group: 'body', vertices: [], deltas: [] },
    ]);
    expect(buffer.byteLength).toBe(32);
    expect(targets.map((t) => [t.byteOffset, t.count])).toEqual([
      [0, 2],
      [32, 0],
    ]);
    const dv = new DataView(buffer);
    expect(dv.getUint32(16, true)).toBe(9);
    expect(dv.getFloat32(28, true)).toBe(6);
    expect(new Uint8Array(buffer)[0]).toBe(5); // low byte first
    expect(() => packMorphs([{ id: 'x', group: 'body', vertices: [1], deltas: [1] }])).toThrow();
  });

  it('mulberry32 is deterministic and in [0, 1)', () => {
    const r1 = mulberry32(1);
    const r2 = mulberry32(1);
    for (let i = 0; i < 100; i++) {
      const v = r1();
      expect(v).toBe(r2());
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });
});
