import { describe, expect, it } from 'vitest';
import type { BodyManifest } from './contracts';
import { AvatarCoreError } from './errors';
import {
  applyMorphs,
  buildBasePositions,
  getMorphSet,
  parseMorphs,
  validateManifest,
} from './morphs';
import { getSyntheticBody, mulberry32 } from './testing/syntheticBody';
import { rawMorphs, tinyManifest } from './testing/tiny';

const fx = getSyntheticBody();

describe('parseMorphs / manifest validation', () => {
  it('parses hand-built little-endian bytes and applies them', () => {
    const { buffer, offsets } = rawMorphs([
      [
        [0, 0.1, 0.2, 0.3],
        [3, 1, 1, 1],
      ],
      [[0, -0.5, 0, 0]],
    ]);
    const m = tinyManifest([
      { id: 'a', group: 'body', byteOffset: offsets[0]!, count: 2 },
      { id: 'b', group: 'body', byteOffset: offsets[1]!, count: 1 },
    ]);
    const base = new Float32Array(12).fill(1);
    const out = new Float32Array(12);
    applyMorphs(
      base,
      buffer,
      m,
      new Map([
        ['a', 2],
        ['b', 1],
      ]),
      out,
    );
    expect(out[0]).toBeCloseTo(1 + 0.2 - 0.5, 6);
    expect(out[1]).toBeCloseTo(1 + 0.4, 6);
    expect(out[2]).toBeCloseTo(1 + 0.6, 6);
    expect(out[9]).toBeCloseTo(3, 6); // joint point (vertex 3) is morphed like any vertex
    expect(out[3]).toBe(1);
  });

  it('round-trips the serialized synthetic morphs.bin', () => {
    const set = parseMorphs(fx.morphs, fx.manifest);
    expect(set.targetIds).toHaveLength(fx.manifest.targets.length);
    const dv = new DataView(fx.morphs);
    const t = fx.manifest.targets.findIndex((x) => x.id === 'measure-waist-circ-incr');
    const def = fx.manifest.targets[t]!;
    expect(def.count).toBeGreaterThan(100);
    for (const k of [0, 7, def.count - 1]) {
      const p = set.wordOffsets[t]! + 4 * k;
      expect(set.words[p]).toBe(dv.getUint32(def.byteOffset + 16 * k, true));
      expect(set.floats[p + 2]).toBe(dv.getFloat32(def.byteOffset + 16 * k + 8, true));
    }
  });

  it('accepts count = 0 targets', () => {
    const { buffer } = rawMorphs([[]]);
    const m = tinyManifest([{ id: 'empty', group: 'body', byteOffset: 0, count: 0 }]);
    expect(parseMorphs(buffer, m).counts[0]).toBe(0);
  });

  it('rejects out-of-range runs, misalignment, bad vertices and non-finite deltas', () => {
    const { buffer } = rawMorphs([[[0, 1, 1, 1]]]);
    const mk = (byteOffset: number, count: number): BodyManifest =>
      tinyManifest([{ id: 't', group: 'body', byteOffset, count }]);
    expect(() => parseMorphs(buffer, mk(0, 2))).toThrow(/exceeds morphs.bin size/);
    expect(() => parseMorphs(buffer, mk(16, 1))).toThrow(/exceeds morphs.bin size/);
    expect(() => parseMorphs(buffer, mk(2, 0))).toThrow(/not 4-byte aligned/);
    expect(() => parseMorphs(buffer, mk(-4, 1))).toThrow(/invalid byteOffset/);
    expect(() => parseMorphs(buffer, mk(0, 1.5))).toThrow(/invalid count/);

    const badVertex = rawMorphs([[[4, 0, 0, 0]]]);
    expect(() => parseMorphs(badVertex.buffer, mk(0, 1))).toThrow(
      /vertex index 4 >= vertexCount 4/,
    );
    const nan = rawMorphs([[[1, NaN, 0, 0]]]);
    expect(() => parseMorphs(nan.buffer, mk(0, 1))).toThrow(/non-finite delta/);
    const inf = rawMorphs([[[1, 0, Infinity, 0]]]);
    expect(() => parseMorphs(inf.buffer, mk(0, 1))).toThrow(AvatarCoreError);
  });

  it('validates the manifest structure', () => {
    const ok = tinyManifest([]);
    expect(() => validateManifest(ok)).not.toThrow();
    expect(() => validateManifest({ ...ok, vertexCount: 5 })).toThrow(/vertexCount 5/);
    expect(() => validateManifest({ ...ok, renderVertexCount: 0, vertexCount: 1 })).toThrow(
      /renderVertexCount/,
    );
    const dup = tinyManifest([
      { id: 'a', group: 'body', byteOffset: 0, count: 0 },
      { id: 'a', group: 'body', byteOffset: 0, count: 0 },
    ]);
    expect(() => validateManifest(dup)).toThrow(/duplicate target id "a"/);
    const badBucket = tinyManifest([
      {
        id: 'a',
        group: 'macro',
        byteOffset: 0,
        count: 0,
        macroConditions: [{ variable: 'weight', bucket: 'nope' }],
      },
    ]);
    expect(() => validateManifest(badBucket)).toThrow(/unknown bucket "nope"/);
    const badVar = tinyManifest([
      {
        id: 'a',
        group: 'macro',
        byteOffset: 0,
        count: 0,
        macroConditions: [{ variable: 'age', bucket: 'x' }],
      },
    ]);
    expect(() => validateManifest(badVar)).toThrow(/unknown macro variable "age"/);
    const badMod: BodyManifest = {
      ...ok,
      modifiers: [{ id: 'm', min: -1, max: 1, default: 0, incrTarget: 'ghost' }],
    };
    expect(() => validateManifest(badMod)).toThrow(/unknown target "ghost"/);
    const sameAt: BodyManifest = structuredClone(ok);
    sameAt.macroVariables[0]!.buckets[1]!.at = 0;
    expect(() => validateManifest(sameAt)).toThrow(/two buckets at 0/);
  });

  it('builds full base positions from glTF positions and joint points', () => {
    const m = tinyManifest([]);
    const full = buildBasePositions(new Float32Array(9).fill(2), m);
    expect(Array.from(full)).toEqual([2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 0]);
    expect(() => buildBasePositions(new Float32Array(6), m)).toThrow(/renderVertexCount/);
  });
});

describe('applyMorphs', () => {
  it('returns the base unchanged for zero / empty weights', () => {
    const out = new Float32Array(fx.base.length);
    applyMorphs(fx.base, fx.morphs, fx.manifest, new Map(), out);
    expect(out).toEqual(fx.base);
    const zeros = new Map(fx.manifest.targets.map((t) => [t.id, 0]));
    out.fill(7);
    applyMorphs(fx.base, fx.morphs, fx.manifest, zeros, out);
    expect(out).toEqual(fx.base);
    expect(out).not.toBe(fx.base);
  });

  it('matches a brute-force reference read straight from the bytes', () => {
    const rand = mulberry32(7);
    const weights = new Map<string, number>();
    const picks = fx.manifest.targets.filter(() => rand() < 0.2);
    for (const t of picks) weights.set(t.id, rand() * 2 - 1);
    const ref = Float64Array.from(fx.base);
    const dv = new DataView(fx.morphs);
    for (const t of picks) {
      const w = weights.get(t.id)!;
      for (let k = 0; k < t.count; k++) {
        const o = t.byteOffset + 16 * k;
        const vi = dv.getUint32(o, true) * 3;
        ref[vi] = ref[vi]! + w * dv.getFloat32(o + 4, true);
        ref[vi + 1] = ref[vi + 1]! + w * dv.getFloat32(o + 8, true);
        ref[vi + 2] = ref[vi + 2]! + w * dv.getFloat32(o + 12, true);
      }
    }
    const out = new Float32Array(fx.base.length);
    applyMorphs(fx.base, fx.morphs, fx.manifest, weights, out);
    for (let i = 0; i < out.length; i++) expect(out[i]).toBeCloseTo(ref[i]!, 5);
  });

  it('supports out === base and pre-parsed MorphSets', () => {
    const set = getMorphSet(fx.morphs, fx.manifest);
    expect(getMorphSet(fx.morphs, fx.manifest)).toBe(set); // cached
    const one = new Map([['measure-waist-circ-incr', 1]]);
    const a = new Float32Array(fx.base);
    applyMorphs(a, set, fx.manifest, one, a);
    const b = new Float32Array(fx.base.length);
    applyMorphs(fx.base, fx.morphs, fx.manifest, one, b);
    expect(a).toEqual(b);
    expect(a).not.toEqual(fx.base);
  });

  it('throws clear errors for unknown targets, bad weights and size mismatches', () => {
    const out = new Float32Array(fx.base.length);
    const nope = new Map([['nope', 1]]);
    expect(() => applyMorphs(fx.base, fx.morphs, fx.manifest, nope, out)).toThrow(
      /unknown target "nope"/,
    );
    const nan = new Map([['measure-waist-circ-incr', NaN]]);
    expect(() => applyMorphs(fx.base, fx.morphs, fx.manifest, nan, out)).toThrow(
      /non-finite weight/,
    );
    expect(() =>
      applyMorphs(fx.base, fx.morphs, fx.manifest, new Map(), new Float32Array(9)),
    ).toThrow(/out length/);
    expect(() => applyMorphs(new Float32Array(9), fx.morphs, fx.manifest, new Map(), out)).toThrow(
      /base length/,
    );
  });
});
