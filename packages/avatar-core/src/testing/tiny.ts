import type { BodyManifest, TargetDef } from '../contracts';

/** Hand-written little-endian morphs.bin bytes, independent of packMorphs. */
export function rawMorphs(entries: [number, number, number, number][][]): {
  buffer: ArrayBuffer;
  offsets: number[];
} {
  const total = entries.reduce((s, e) => s + e.length, 0);
  const buffer = new ArrayBuffer(total * 16);
  const dv = new DataView(buffer);
  const offsets: number[] = [];
  let o = 0;
  for (const run of entries) {
    offsets.push(o);
    for (const [vi, dx, dy, dz] of run) {
      dv.setUint32(o, vi, true);
      dv.setFloat32(o + 4, dx, true);
      dv.setFloat32(o + 8, dy, true);
      dv.setFloat32(o + 12, dz, true);
      o += 16;
    }
  }
  return { buffer, offsets };
}

/** Tiny manifest: 3 render vertices + 1 joint point, targets supplied by the caller. */
export function tinyManifest(targets: TargetDef[]): BodyManifest {
  return {
    version: 1,
    unit: 'm',
    renderVertexCount: 3,
    jointPoints: [{ name: 'j0', position: [0, 1, 0] }],
    vertexCount: 4,
    mesh: 'x.glb',
    morphs: 'morphs.bin',
    macroVariables: [
      {
        id: 'weight',
        min: 0,
        max: 1,
        default: 0.5,
        buckets: [
          { name: 'minweight', at: 0 },
          { name: 'averageweight', at: 0.5 },
          { name: 'maxweight', at: 1 },
        ],
      },
    ],
    targets,
    modifiers: [],
    license: { assets: 'CC0-1.0', source: 'test' },
  };
}
