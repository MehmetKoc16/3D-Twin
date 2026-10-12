import {
  type GarmentBinding,
  type GarmentRigidAttachments,
  type GarmentTemplateDef,
  type StoreItemDef,
  type Vec3,
} from '@dt/avatar-core';

/** Synthetic geometry only; no imported garment or personal body data. */
export const shoeNeutral = new Float32Array([
  0.1, 0, 0, 0.12, 0, 0, 0.1, 0.01, 0, 0.1, 0, 0.17, 0.12, 0, 0.17, 0.1, 0.01, 0.19, -0.1, 0, 0,
  -0.12, 0, 0, -0.1, 0.01, 0, -0.1, 0, 0.17, -0.12, 0, 0.17, -0.1, 0.01, 0.19, 0.1, 0.015, 0, 0.1,
  0.05, 0.05, -0.1, 0.015, 0, -0.1, 0.05, 0.05,
]);
export const shoeIndex = new Uint32Array([
  0, 1, 2, 3, 4, 5, 6, 8, 7, 9, 11, 10, 2, 12, 1, 12, 13, 1, 8, 7, 14, 14, 7, 15,
]);
export const shoeBoneNames = ['foot_l', 'ball_l', 'calf_l', 'foot_r', 'ball_r', 'calf_r'];
export const shoeHeads: Vec3[] = [
  [0.1, 0.08, 0],
  [0.1, 0.03, 0.15],
  [0.1, 0.4, 0],
  [-0.1, 0.08, 0],
  [-0.1, 0.03, 0.15],
  [-0.1, 0.4, 0],
];
export const shoeSidecar: GarmentRigidAttachments = {
  version: 1,
  requiresRuntimeChange: true,
  soleHeightMm: 35,
  attachments: [0, 1, 3, 4].map((bone, part) => ({
    bone: shoeBoneNames[bone] as GarmentRigidAttachments['attachments'][number]['bone'],
    vertices: [part * 3, part * 3 + 1, part * 3 + 2],
    localPositionsM: [part * 3, part * 3 + 1, part * 3 + 2].map((v): Vec3 => [
      shoeNeutral[v * 3]! - shoeHeads[bone]![0],
      shoeNeutral[v * 3 + 1]! - shoeHeads[bone]![1],
      shoeNeutral[v * 3 + 2]! - shoeHeads[bone]![2],
    ]),
  })),
};
export const shoeDef: GarmentTemplateDef = {
  id: 'synthetic-shoe',
  category: 'shoes',
  kind: 'sneakers',
  label: { tr: 'Test', en: 'Test' },
  license: 'CC0-1.0',
  mesh: 'synthetic.glb',
  binding: 'synthetic.bind.bin',
  scaleRefs: { x: [16, 17, 1], y: [16, 17, 1], z: [16, 17, 1] },
  nativeMeasures: { footLength: 25 },
  defaultEase: { footLength: 2 },
  baseColor: '#888888',
  layer: 1,
};
export const shoeItem: StoreItemDef = {
  id: 'synthetic-item',
  name: 'Test',
  templateId: shoeDef.id,
  color: '#888888',
  sizes: ['M', 'L'],
  chart: { footLength: [25, 28] },
  selectedSize: 'M',
};
export function shoeBinding(): GarmentBinding {
  const count = shoeNeutral.length / 3;
  const indices = new Uint32Array(count * 3);
  const weights = new Float32Array(count * 3);
  for (let v = 0; v < count; v++) {
    indices.fill(v, v * 3, v * 3 + 3);
    weights[v * 3] = 1;
  }
  return { count, indices, weights, offsets: new Float32Array(count * 3) };
}
export function shoeBodySkin(): {
  skinIndex: Uint16Array;
  skinWeight: Float32Array;
  renderVertexCount: number;
} {
  const count = shoeNeutral.length / 3;
  const skinIndex = new Uint16Array(count * 4);
  const skinWeight = new Float32Array(count * 4);
  for (let v = 0; v < count; v++) {
    const right = shoeNeutral[v * 3]! < 0;
    skinIndex[v * 4] = right ? 3 : 0;
    skinIndex[v * 4 + 1] = right ? 5 : 2;
    skinWeight[v * 4] = 0.5;
    skinWeight[v * 4 + 1] = 0.5;
  }
  return { skinIndex, skinWeight, renderVertexCount: count };
}
