/**
 * Synthetic, fully in-code test fixture that mimics the shape of the real asset-pipeline output:
 * a humanoid-ish column of horizontal vertex rings (closed and watertight), macro targets over
 * gender/muscle/weight/height, measure modifiers with decr/incr targets, joint points, a small rig,
 * measure definitions and a real serialized morphs.bin. Not exported from the package index; import from
 * `@dt/avatar-core/testing`.
 */
import type {
  BodyManifest,
  MeasureDef,
  MeasuresDef,
  MacroVar,
  ModifierDef,
  RigDef,
  TargetDef,
  Vec3,
} from '../contracts';
import { MORPH_ENTRY_BYTES } from '../morphs';
import type { SolverData } from '../types';

/** Deterministic PRNG (mulberry32) returning floats in [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface PackTargetInput {
  id: string;
  group: TargetDef['group'];
  macroConditions?: TargetDef['macroConditions'];
  /** Vertex indices. */
  vertices: ArrayLike<number>;
  /** 3 floats (dx, dy, dz) per vertex. */
  deltas: ArrayLike<number>;
}

/** Serializes targets into a real morphs.bin (little-endian, 16-byte entries) and the matching TargetDefs. */
export function packMorphs(inputs: PackTargetInput[]): {
  buffer: ArrayBuffer;
  targets: TargetDef[];
} {
  const total = inputs.reduce((s, t) => s + t.vertices.length, 0);
  const buffer = new ArrayBuffer(total * MORPH_ENTRY_BYTES);
  const view = new DataView(buffer);
  const targets: TargetDef[] = [];
  let offset = 0;
  for (const t of inputs) {
    if (t.deltas.length !== t.vertices.length * 3)
      throw new Error(`packMorphs: target "${t.id}" needs 3 deltas per vertex`);
    targets.push({
      id: t.id,
      group: t.group,
      byteOffset: offset,
      count: t.vertices.length,
      ...(t.macroConditions ? { macroConditions: t.macroConditions } : {}),
    });
    for (let k = 0; k < t.vertices.length; k++) {
      view.setUint32(offset, t.vertices[k]!, true);
      view.setFloat32(offset + 4, t.deltas[3 * k]!, true);
      view.setFloat32(offset + 8, t.deltas[3 * k + 1]!, true);
      view.setFloat32(offset + 12, t.deltas[3 * k + 2]!, true);
      offset += MORPH_ENTRY_BYTES;
    }
  }
  return { buffer, targets };
}

export interface SyntheticOptions {
  /** Number of horizontal rings (default 120). */
  rings?: number;
  /** Vertices per ring, must be even (default 124 -> 14 882 render vertices). */
  segments?: number;
  /** Height of the base mesh (default 1.7 m). */
  baseHeightM?: number;
  /** Extra modifiers that no measure uses (default 13 -> 20 modifiers in total). */
  decoyModifiers?: number;
  seed?: number;
}

export interface SyntheticBody {
  manifest: BodyManifest;
  /** Render vertices only (what base.glb would hold). */
  gltfPositions: Float32Array;
  /** Render vertices + joint points (what the solver takes as `base`). */
  base: Float32Array;
  morphs: ArrayBuffer;
  measures: MeasuresDef;
  rig: RigDef;
  /** Triangles over the render vertices (outward winding, watertight). */
  indices: Uint32Array;
  data: SolverData;
  rings: number;
  segments: number;
  baseHeightM: number;
  /** Vertex indices of one ring in loop order. */
  ringVerts(ring: number): number[];
  /** Ring index closest to a normalized height t in [0, 1]. */
  ringAt(t: number): number;
  /** Ring radius profile at normalized height t (meters). */
  radiusAt(t: number): number;
}

const PROFILE: ReadonlyArray<readonly [number, number]> = [
  [0.0, 0.055],
  [0.04, 0.05],
  [0.2, 0.06],
  [0.3, 0.07],
  [0.4, 0.11],
  [0.5, 0.14],
  [0.62, 0.115],
  [0.72, 0.14],
  [0.8, 0.15],
  [0.86, 0.058],
  [0.9, 0.056],
  [0.94, 0.09],
  [0.98, 0.085],
  [1.0, 0.03],
];

function profile(t: number): number {
  const c = Math.min(1, Math.max(0, t));
  for (let i = 0; i + 1 < PROFILE.length; i++) {
    const [t0, r0] = PROFILE[i]!;
    const [t1, r1] = PROFILE[i + 1]!;
    if (c <= t1) return r0 + ((r1 - r0) * (c - t0)) / (t1 - t0);
  }
  return PROFILE[PROFILE.length - 1]![1];
}

/** Raised-cosine window: 1 at u = 0, 0 for |u| >= 1. */
function bump(u: number): number {
  return Math.abs(u) >= 1 ? 0 : 0.5 * (1 + Math.cos(Math.PI * u));
}

type DeltaFn = (x: number, y: number, z: number) => Vec3;

interface MeasureMod {
  key: string;
  t0: number;
  w: number;
  incr: number;
  decr: number;
  /** 'radial' scales x and z, 'x' only x (shoulder width), 'lift' stretches the legs (inseam). */
  mode: 'radial' | 'x' | 'lift';
}

const MEASURE_MODS: MeasureMod[] = [
  { key: 'neck-circ', t0: 0.88, w: 0.06, incr: 0.5, decr: -0.4, mode: 'radial' },
  { key: 'chest-circ', t0: 0.72, w: 0.14, incr: 0.5, decr: -0.4, mode: 'radial' },
  { key: 'waist-circ', t0: 0.62, w: 0.14, incr: 0.5, decr: -0.4, mode: 'radial' },
  { key: 'hip-circ', t0: 0.5, w: 0.14, incr: 0.5, decr: -0.4, mode: 'radial' },
  { key: 'thigh-circ', t0: 0.4, w: 0.12, incr: 0.5, decr: -0.4, mode: 'radial' },
  { key: 'shoulder-width', t0: 0.8, w: 0.1, incr: 0.3, decr: -0.25, mode: 'x' },
  { key: 'inseam', t0: 0.46, w: 1, incr: 0.12, decr: -0.1, mode: 'lift' },
];

const GENDER = ['female', 'male'] as const;
const MUSCLE = ['minmuscle', 'averagemuscle', 'maxmuscle'] as const;
const WEIGHT = ['minweight', 'averageweight', 'maxweight'] as const;
const HEIGHT = ['minheight', 'averageheight', 'maxheight'] as const;

let cachedDefault: SyntheticBody | undefined;

/** Shared default fixture (built once per test worker). Treat it as read-only. */
export function getSyntheticBody(): SyntheticBody {
  cachedDefault ??= createSyntheticBody();
  return cachedDefault;
}

export function createSyntheticBody(opts: SyntheticOptions = {}): SyntheticBody {
  const R = opts.rings ?? 120;
  const S = opts.segments ?? 124;
  const H0 = opts.baseHeightM ?? 1.7;
  const decoys = opts.decoyModifiers ?? 13;
  const rand = mulberry32(opts.seed ?? 12345);
  if (S % 2 !== 0 || S < 8) throw new Error('segments must be even and >= 8');

  const ringAt = (t: number): number => Math.round(Math.min(1, Math.max(0, t)) * (R - 1));
  const ringVerts = (ring: number): number[] => Array.from({ length: S }, (_, s) => ring * S + s);

  // ---- render vertices: R rings of S vertices + bottom pole + top pole
  const renderCount = R * S + 2;
  const bottomPole = R * S;
  const topPole = R * S + 1;
  const gltf = new Float32Array(renderCount * 3);
  for (let r = 0; r < R; r++) {
    const t = r / (R - 1);
    const y = H0 * t;
    const rad = profile(t);
    for (let s = 0; s < S; s++) {
      const a = (2 * Math.PI * s) / S;
      const o = (r * S + s) * 3;
      gltf[o] = rad * Math.cos(a);
      gltf[o + 1] = y;
      gltf[o + 2] = rad * Math.sin(a);
    }
  }
  gltf[bottomPole * 3 + 1] = 0;
  gltf[topPole * 3 + 1] = H0;

  // ---- triangles (outward)
  const tris: number[] = [];
  for (let r = 0; r + 1 < R; r++) {
    for (let s = 0; s < S; s++) {
      const s1 = (s + 1) % S;
      const a = r * S + s;
      const b = r * S + s1;
      const c = (r + 1) * S + s1;
      const d = (r + 1) * S + s;
      tris.push(a, c, b, a, d, c);
    }
  }
  for (let s = 0; s < S; s++) {
    const s1 = (s + 1) % S;
    tris.push(bottomPole, s, s1);
    tris.push(topPole, (R - 1) * S + s1, (R - 1) * S + s);
  }
  const indices = Uint32Array.from(tris);

  // ---- joint points (appended after the render vertices)
  const jointDefs: { name: string; y: number }[] = [
    { name: 'pelvis', y: 0.5 },
    { name: 'spine-mid', y: 0.62 },
    { name: 'chest', y: 0.72 },
    { name: 'neck-base', y: 0.8 },
    { name: 'knee', y: 0.28 },
  ];
  const jointPoints = jointDefs.map((j) => ({
    name: j.name,
    position: [0, H0 * j.y, 0] as Vec3,
  }));
  const vertexCount = renderCount + jointPoints.length;
  const base = new Float32Array(vertexCount * 3);
  base.set(gltf);
  jointPoints.forEach((jp, i) => base.set(jp.position, (renderCount + i) * 3));

  // ---- targets
  const inputs: PackTargetInput[] = [];
  const addTarget = (
    id: string,
    group: TargetDef['group'],
    fn: DeltaFn,
    macroConditions?: TargetDef['macroConditions'],
  ): void => {
    const vertices: number[] = [];
    const deltas: number[] = [];
    for (let v = 0; v < vertexCount; v++) {
      const [dx, dy, dz] = fn(base[v * 3]!, base[v * 3 + 1]!, base[v * 3 + 2]!);
      if (dx !== 0 || dy !== 0 || dz !== 0) {
        vertices.push(v);
        deltas.push(dx, dy, dz);
      }
    }
    inputs.push({ id, group, vertices, deltas, ...(macroConditions ? { macroConditions } : {}) });
  };

  const hEffY = [-0.16, 0, 0.16];
  const hRad = [-0.05, 0, 0.06];
  const wRad = [-0.25, 0, 0.55];
  const mRad = [-0.03, 0, 0.08];
  GENDER.forEach((g, gi) => {
    MUSCLE.forEach((m, mi) => {
      WEIGHT.forEach((w, wi) => {
        HEIGHT.forEach((h, hi) => {
          const male = gi === 1;
          addTarget(
            `macro-${g}-${m}-${w}-${h}`,
            'macro',
            (x, y, z) => {
              const t = y / H0;
              const dyScale = hEffY[hi]! * (male ? 1.1 : 0.95) + (male ? 0.02 : -0.02);
              const chest = bump((t - 0.72) / 0.14);
              const hip = bump((t - 0.5) / 0.14);
              const waist = bump((t - 0.62) / 0.14);
              const k =
                wRad[wi]! * (0.6 + 0.8 * waist) +
                mRad[mi]! * (0.5 + chest) +
                hRad[hi]! +
                (male ? 0.06 * chest - 0.05 * hip : -0.06 * chest + 0.05 * hip);
              return [x * k, y * dyScale, z * k];
            },
            [
              { variable: 'gender', bucket: g },
              { variable: 'muscle', bucket: m },
              { variable: 'weight', bucket: w },
              { variable: 'height', bucket: h },
            ],
          );
        });
      });
    });
  });

  const mods: ModifierDef[] = [];
  const modFn = (mod: MeasureMod, amp: number): DeltaFn => {
    if (mod.mode === 'lift') {
      const yc = mod.t0 * H0;
      return (_x, y) => [0, (amp * Math.min(y, yc)) / yc, 0];
    }
    return (x, y, z) => {
      const k = amp * bump((y / H0 - mod.t0) / mod.w);
      return [x * k, 0, mod.mode === 'x' ? 0 : z * k];
    };
  };
  for (const mod of MEASURE_MODS) {
    addTarget(`measure-${mod.key}-decr`, 'measure', modFn(mod, mod.decr));
    addTarget(`measure-${mod.key}-incr`, 'measure', modFn(mod, mod.incr));
    mods.push({
      id: `measure/measure-${mod.key}-decr|incr`,
      min: -1,
      max: 1,
      default: 0,
      decrTarget: `measure-${mod.key}-decr`,
      incrTarget: `measure-${mod.key}-incr`,
    });
  }
  for (let i = 0; i < decoys; i++) {
    const mod: MeasureMod = {
      key: `decoy-${i}`,
      t0: 0.05 + 0.9 * rand(),
      w: 0.05 + 0.1 * rand(),
      incr: 0.05 + 0.15 * rand(),
      decr: -(0.05 + 0.1 * rand()),
      mode: 'radial',
    };
    const onlyIncr = i % 3 === 2;
    addTarget(`body-${mod.key}-incr`, 'body', modFn(mod, mod.incr));
    if (!onlyIncr) addTarget(`body-${mod.key}-decr`, 'body', modFn(mod, mod.decr));
    mods.push({
      id: `body/${mod.key}-decr|incr`,
      min: onlyIncr ? 0 : -1,
      max: 1,
      default: 0,
      incrTarget: `body-${mod.key}-incr`,
      ...(onlyIncr ? {} : { decrTarget: `body-${mod.key}-decr` }),
    });
  }

  const { buffer, targets } = packMorphs(inputs);

  const macroVar = (
    id: MacroVar,
    names: readonly string[],
  ): BodyManifest['macroVariables'][number] => ({
    id,
    min: 0,
    max: 1,
    default: 0.5,
    buckets: names.map((name, i) => ({ name, at: i / (names.length - 1) })),
  });
  const manifest: BodyManifest = {
    version: 1,
    unit: 'm',
    renderVertexCount: renderCount,
    jointPoints,
    vertexCount,
    mesh: 'synthetic.glb',
    morphs: 'morphs.bin',
    macroVariables: [
      macroVar('gender', GENDER),
      macroVar('muscle', MUSCLE),
      macroVar('weight', WEIGHT),
      macroVar('height', HEIGHT),
    ],
    targets,
    modifiers: mods,
    license: { assets: 'CC0-1.0', source: 'synthetic test fixture' },
  };

  // ---- measures
  const drv = (key: string): string[] => [`measure/measure-${key}-decr|incr`];
  const ring = (t: number): number[] => ringVerts(ringAt(t));
  const measureDefs: MeasureDef[] = [
    { id: 'height', type: 'height', drivers: [] },
    { id: 'neck', type: 'circumference', verts: ring(0.88), drivers: drv('neck-circ') },
    { id: 'chest', type: 'circumference', verts: ring(0.72), drivers: drv('chest-circ') },
    { id: 'waist', type: 'circumference', verts: ring(0.62), drivers: drv('waist-circ') },
    { id: 'hip', type: 'circumference', verts: ring(0.5), drivers: drv('hip-circ') },
    { id: 'thigh', type: 'circumference', verts: ring(0.4), drivers: drv('thigh-circ') },
    {
      id: 'shoulder',
      type: 'distance',
      verts: [ringAt(0.8) * S, ringAt(0.8) * S + S / 2],
      axis: 'x',
      drivers: drv('shoulder-width'),
    },
    { id: 'inseam', type: 'vertexHeight', vert: ringAt(0.46) * S, drivers: drv('inseam') },
  ];
  const measures: MeasuresDef = { version: 1, measures: measureDefs };

  // ---- rig (deliberately not in topological order)
  const j = (name: string): number => renderCount + jointDefs.findIndex((d) => d.name === name);
  const rig: RigDef = {
    version: 1,
    bones: [
      {
        name: 'head',
        parent: 'neck',
        head: { strategy: 'VERTEX', vert: j('neck-base') },
        tail: { strategy: 'VERTEX', vert: topPole },
        roll: 0,
      },
      {
        name: 'foot',
        parent: 'leg',
        head: { strategy: 'MEAN', verts: ringVerts(ringAt(0.04)) },
        tail: { strategy: 'VERTEX', vert: 0 },
        roll: 0,
      },
      {
        name: 'spine',
        parent: 'root',
        head: { strategy: 'VERTEX', vert: j('pelvis') },
        tail: { strategy: 'VERTEX', vert: j('chest') },
        roll: 0,
      },
      {
        name: 'root',
        parent: null,
        head: { strategy: 'FIXED', position: [0, 0, 0] },
        tail: { strategy: 'VERTEX', vert: j('pelvis') },
        roll: 0,
      },
      {
        name: 'neck',
        parent: 'spine',
        head: { strategy: 'VERTEX', vert: j('chest') },
        tail: { strategy: 'VERTEX', vert: j('neck-base') },
        roll: 0,
      },
      {
        name: 'leg',
        parent: 'root',
        head: { strategy: 'VERTEX', vert: j('pelvis') },
        tail: { strategy: 'MEAN', verts: ringVerts(ringAt(0.04)) },
        roll: 0,
      },
    ],
  };

  const data: SolverData = { manifest, base, morphs: buffer, measures, indices };
  return {
    manifest,
    gltfPositions: gltf,
    base,
    morphs: buffer,
    measures,
    rig,
    indices,
    data,
    rings: R,
    segments: S,
    baseHeightM: H0,
    ringVerts,
    ringAt,
    radiusAt: profile,
  };
}
