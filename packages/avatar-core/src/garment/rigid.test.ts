import { describe, expect, it } from 'vitest';
import {
  fitGarmentRigidAttachments,
  garmentRigidSkinWeights,
  parseGarmentRigidAttachments,
  prepareGarmentRigidAttachments,
} from './rigid';
import type { GarmentRigidAttachments } from '../garmentContracts';
import type { Vec3 } from '../contracts';

const neutral = new Float32Array([
  0.1, 0, 0, 0.12, 0, 0, 0.1, 0, 0.02, 0.1, 0, 0.17, 0.12, 0, 0.17, 0.1, 0, 0.19, 0.1, 0.005, 0,
  0.1, 0.01, 0, 0.1, 0.02, 0, 0.1, 0.005, 0, 0.1, 0, 0,
]);
const index = new Uint32Array([0, 1, 2, 3, 4, 5, 0, 6, 1, 6, 7, 1, 7, 8, 1, 0, 9, 1]);
const heads: Vec3[] = [
  [0.1, 0.08, 0],
  [0.1, 0.03, 0.15],
];
const sidecar: GarmentRigidAttachments = {
  version: 1,
  soleHeightMm: 35,
  requiresRuntimeChange: true,
  warnings: ['Review only metadata is accepted.'],
  attachments: heads.map((head, i) => ({
    bone: i === 0 ? 'foot_l' : 'ball_l',
    vertices: [i * 3, i * 3 + 1, i * 3 + 2],
    localPositionsM: [i * 3, i * 3 + 1, i * 3 + 2].map((v): Vec3 => [
      neutral[v * 3]! - head[0],
      neutral[v * 3 + 1]! - head[1],
      neutral[v * 3 + 2]! - head[2],
    ]),
  })),
};

describe('rigid shoe attachments', () => {
  it('reads the importer version 1 arrays and ignores diagnostic metadata', () => {
    expect(parseGarmentRigidAttachments(sidecar, neutral.length / 3).attachments).toEqual(
      sidecar.attachments,
    );
    for (const bone of ['foot_l', 'ball_l', 'foot_r', 'ball_r']) {
      expect(
        parseGarmentRigidAttachments(
          { ...sidecar, attachments: [{ ...sidecar.attachments[0], bone }] },
          11,
        ).attachments[0]!.bone,
      ).toBe(bone);
    }
  });

  it.each([
    { ...sidecar, version: 2 },
    { ...sidecar, soleHeightMm: NaN },
    { ...sidecar, attachments: [{ bone: 'calf_l', vertices: [0], localPositionsM: [[0, 0, 0]] }] },
    { ...sidecar, attachments: [{ bone: 'foot_l', vertices: [11], localPositionsM: [[0, 0, 0]] }] },
    { ...sidecar, attachments: [{ bone: 'foot_l', vertices: [-1], localPositionsM: [[0, 0, 0]] }] },
    {
      ...sidecar,
      attachments: [
        {
          bone: 'foot_l',
          vertices: [0, 0],
          localPositionsM: [
            [0, 0, 0],
            [0, 0, 0],
          ],
        },
      ],
    },
    { ...sidecar, attachments: [{ bone: 'foot_l', vertices: [0], localPositionsM: [] }] },
    {
      ...sidecar,
      attachments: [{ bone: 'foot_l', vertices: [0], localPositionsM: [[0, Infinity, 0]] }],
    },
  ])('rejects malformed sidecars', (value) => {
    expect(() => parseGarmentRigidAttachments(value, 11)).toThrow(/garment rigid/);
  });

  it('rejects missing bones and mismatched local frames', () => {
    expect(() => prepareGarmentRigidAttachments(sidecar, neutral, index, ['foot_l'])).toThrow(
      /missing bone/,
    );
    const inconsistent = structuredClone(sidecar);
    inconsistent.attachments[0]!.localPositionsM[1]![0] += 0.01;
    expect(() =>
      prepareGarmentRigidAttachments(inconsistent, neutral, index, ['foot_l', 'ball_l']),
    ).toThrow(/disagree/);
    expect(() =>
      prepareGarmentRigidAttachments(sidecar, neutral, [99, 0, 1], ['foot_l', 'ball_l']),
    ).toThrow(/triangle/);
  });

  it('keeps authored soles rigid and smoothly fades only into the upper, including UV seams', () => {
    const rigid = prepareGarmentRigidAttachments(sidecar, neutral, index, ['foot_l', 'ball_l']);
    expect([...rigid.blend.slice(0, 6)]).toEqual([1, 1, 1, 1, 1, 1]);
    expect(rigid.blend[6]).toBeCloseTo(0.5, 6);
    expect(rigid.blend[7]).toBeCloseTo(0, 6);
    expect(rigid.blend[8]).toBe(0);
    expect(rigid.blend[9]).toBe(rigid.blend[6]);
    expect(rigid.blend[10]).toBe(1); // An unlisted UV copy of a sole vertex.

    const baseIndices = new Uint16Array(11 * 4).fill(2);
    const baseWeights = new Float32Array(11 * 4);
    for (let v = 0; v < 11; v++) baseWeights[v * 4] = 1;
    const skin = garmentRigidSkinWeights(rigid, baseIndices, baseWeights);
    expect([...skin.skinWeights.slice(0, 4)]).toEqual([1, 0, 0, 0]);
    expect(skin.skinIndices[12]).toBe(1);
    expect(skin.skinWeights[24]).toBeCloseTo(0.5, 6);
    expect(skin.skinWeights[25]).toBeCloseTo(0.5, 6);
    expect(skin.skinIndices.slice(32, 36)).toEqual(baseIndices.slice(32, 36));
    expect(skin.skinWeights.slice(32, 36)).toEqual(baseWeights.slice(32, 36));
  });

  it('rebuilds soles from scaled local geometry, leaves the distant upper intact and never accumulates scale', () => {
    const rigid = prepareGarmentRigidAttachments(sidecar, neutral, index, ['ball_l', 'foot_l']);
    const positions = neutral.map((n) => n + 0.03);
    const original = positions.slice();
    const scale: Vec3 = [1.2, 0.9, 1.1];
    const joint = (bone: number): Vec3 => heads[bone === 0 ? 1 : 0]!.map((n) => n + 0.2) as Vec3;
    fitGarmentRigidAttachments(rigid, scale, joint, positions);
    expect(positions[0]).toBeCloseTo(0.3, 6);
    expect(positions[1]).toBeCloseTo(0.28 - 0.08 * 0.9, 6);
    expect(positions.slice(24, 27)).toEqual(original.slice(24, 27));
    expect(positions[19]).toBeCloseTo((original[19]! + 0.28 - 0.075 * 0.9) / 2, 6);
    expect(positions.slice(30, 33)).toEqual(positions.slice(0, 3));
    const sole = positions.slice(0, 18);
    fitGarmentRigidAttachments(rigid, scale, joint, positions);
    expect(positions.slice(0, 18)).toEqual(sole);
    expect(() => fitGarmentRigidAttachments(rigid, scale, () => undefined, positions)).toThrow(
      /missing rest joint/,
    );
  });
});
