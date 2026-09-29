import { describe, expect, it } from 'vitest';
import type { RigDef } from './contracts';
import { boneOrder, computeJoints, validateRig } from './joints';
import { solveBody } from './solver';
import { getSyntheticBody } from './testing/syntheticBody';

const fx = getSyntheticBody();

const bone = (
  name: string,
  parent: string | null,
  extra: Partial<RigDef['bones'][number]> = {},
) => ({
  name,
  parent,
  head: { strategy: 'FIXED' as const, position: [0, 0, 0] as [number, number, number] },
  tail: { strategy: 'FIXED' as const, position: [0, 1, 0] as [number, number, number] },
  roll: 0,
  ...extra,
});

describe('computeJoints', () => {
  it('resolves MEAN / VERTEX / FIXED joints in rig order', () => {
    const joints = computeJoints(fx.rig, fx.base);
    expect([...joints.keys()]).toEqual(fx.rig.bones.map((b) => b.name));
    expect(joints.get('root')!.head).toEqual([0, 0, 0]); // FIXED
    const head = joints.get('head')!;
    expect(head.tail[1]).toBeCloseTo(fx.baseHeightM, 6); // VERTEX: top pole
    const leg = joints.get('leg')!;
    expect(leg.tail[0]).toBeCloseTo(0, 6); // MEAN of a ring lies on the axis
    expect(leg.tail[1]).toBeCloseTo(fx.baseHeightM * (fx.ringAt(0.04) / (fx.rings - 1)), 6);
    expect(joints.get('spine')!.head[1]).toBeCloseTo(0.5 * fx.baseHeightM, 6); // joint point
  });

  it('follows the mesh when the height increases', () => {
    const shoe = { system: 'EU', size: 42 } as const;
    const short = computeJoints(
      fx.rig,
      solveBody(fx.data, { gender: 0.5, heightCm: 160, weightKg: 65, shoe }).positions,
    );
    const tall = computeJoints(
      fx.rig,
      solveBody(fx.data, { gender: 0.5, heightCm: 190, weightKg: 65, shoe }).positions,
    );
    expect(short.get('head')!.tail[1]).toBeCloseTo(1.6, 3);
    expect(tall.get('head')!.tail[1]).toBeCloseTo(1.9, 3);
    for (const name of ['spine', 'neck', 'head', 'leg'])
      expect(tall.get(name)!.head[1]).toBeGreaterThan(short.get(name)!.head[1] + 0.02);
    // the pelvis joint sits at ~half of the body height in both cases
    expect(short.get('spine')!.head[1] / 1.6).toBeCloseTo(tall.get('spine')!.head[1] / 1.9, 1);
  });

  it('throws for vertices outside the position array', () => {
    const rig: RigDef = {
      version: 1,
      bones: [bone('a', null, { head: { strategy: 'VERTEX', vert: 10 } })],
    };
    expect(() => computeJoints(rig, new Float32Array(9))).toThrow(/invalid vertex 10/);
  });
});

describe('validateRig / boneOrder', () => {
  it('orders parents before children, keeping rig order otherwise', () => {
    const order = boneOrder(fx.rig);
    expect(new Set(order).size).toBe(fx.rig.bones.length);
    for (const b of fx.rig.bones)
      if (b.parent !== null) expect(order.indexOf(b.parent)).toBeLessThan(order.indexOf(b.name));
    expect(order[0]).toBe('root');
    const flat: RigDef = { version: 1, bones: [bone('b', null), bone('a', null), bone('c', 'b')] };
    expect(boneOrder(flat)).toEqual(['b', 'a', 'c']);
  });

  it('rejects unknown parents, duplicates, cycles and empty MEAN joints', () => {
    expect(() => validateRig({ version: 1, bones: [bone('a', 'ghost')] })).toThrow(
      /unknown parent "ghost"/,
    );
    expect(() => validateRig({ version: 1, bones: [bone('a', null), bone('a', null)] })).toThrow(
      /duplicate bone/,
    );
    expect(() => validateRig({ version: 1, bones: [bone('a', 'b'), bone('b', 'a')] })).toThrow(
      /cycle/,
    );
    expect(() => validateRig({ version: 1, bones: [bone('a', 'a')] })).toThrow(/own parent/);
    expect(() =>
      validateRig({
        version: 1,
        bones: [bone('a', null, { tail: { strategy: 'MEAN', verts: [] } })],
      }),
    ).toThrow(/empty MEAN/);
    expect(() =>
      validateRig({
        version: 1,
        bones: [bone('a', null, { head: { strategy: 'FIXED', position: [NaN, 0, 0] } })],
      }),
    ).toThrow(/non-finite/);
    expect(() => validateRig(fx.rig, fx.manifest.vertexCount)).not.toThrow();
  });
});
