/** Conventions: meters, right-handed, +Y up, character faces +Z, feet on y=0.
 *  Rest pose = MakeHuman rest pose; "T-pose" is a PoseDef preset. */
export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number]; // x,y,z,w
export type MacroVar =
  'gender' | 'age' | 'muscle' | 'weight' | 'height' | 'proportions' | 'cupsize' | 'firmness';
export interface MacroVariableDef {
  id: MacroVar;
  min: number;
  max: number;
  default: number;
  /** tent/piecewise-linear interpolation points, e.g. weight: minweight@0, averageweight@0.5, maxweight@1 */
  buckets: { name: string; at: number }[];
}
export interface TargetDef {
  id: string;
  group: 'macro' | 'measure' | 'body' | 'face' | 'foot';
  /** byte offset into morphs.bin; each entry = 16 bytes little-endian: uint32 vertexIndex, float32 dx, dy, dz (meters) */
  byteOffset: number;
  count: number;
  /** macro targets only: effective weight = product of tent weights of each condition */
  macroConditions?: { variable: MacroVar; bucket: string }[];
}
export interface ModifierDef {
  id: string;
  min: number;
  max: number;
  default: number;
  /** value<0 -> |value| * decrTarget ; value>0 -> value * incrTarget */
  decrTarget?: string;
  incrTarget?: string;
}
export interface BodyManifest {
  version: 1;
  unit: 'm';
  /** Vertices in base.glb (glTF order, after UV-seam split). Indices [0, renderVertexCount) are render vertices. */
  renderVertexCount: number;
  /**
   * Virtual points appended AFTER the render vertices (index = renderVertexCount + i). They are morphed by targets
   * like any vertex and are used only for the skeleton (and, optionally, measures).
   */
  jointPoints: { name: string; position: Vec3 }[];
  /**
   * renderVertexCount + jointPoints.length. ALL vertex indices in morphs.bin, rig.json and measures.json use this
   * combined index space.
   */
  vertexCount: number;
  mesh: string;
  morphs: string;
  macroVariables: MacroVariableDef[];
  targets: TargetDef[];
  modifiers: ModifierDef[];
  license: { assets: string; source: string };
}
export type JointRef =
  | { strategy: 'MEAN'; verts: number[] }
  | { strategy: 'VERTEX'; vert: number }
  | { strategy: 'FIXED'; position: Vec3 };
export interface BoneDef {
  name: string;
  parent: string | null;
  head: JointRef;
  tail: JointRef;
  /** Always 0: bones are world-aligned (identity rest rotation). Kept for format stability. */
  roll: number;
}
export interface RigDef {
  version: 1;
  bones: BoneDef[];
}
export type MeasureId =
  | 'height'
  | 'neck'
  | 'shoulder'
  | 'chest'
  | 'waist'
  | 'hip'
  | 'thigh'
  | 'upperArm'
  | 'armLength'
  | 'inseam'
  | 'footLength';
/**
 * `drivers` = ids of the modifiers the solver may adjust to hit the measure (e.g. armLength = [upperarm-length,
 * lowerarm-length]). Symmetric l/r modifiers are pre-merged by the pipeline.
 */
export type MeasureDef =
  /** Tape measure: perimeter of the 2-D convex hull of the loop vertices projected on the loop's plane
   *  (plane normal via Newell's method over the ordered loop). */
  | { id: MeasureId; type: 'circumference'; verts: number[]; drivers: string[] }
  /** Euclidean distance, or only along `axis` when given (shoulder: x, footLength: z). */
  | {
      id: MeasureId;
      type: 'distance';
      verts: [number, number];
      axis?: 'x' | 'y' | 'z';
      drivers: string[];
    }
  | { id: MeasureId; type: 'polyline'; verts: number[]; drivers: string[] }
  /** Vertex y minus the floor (min y over render vertices); used for inseam. */
  | { id: MeasureId; type: 'vertexHeight'; vert: number; drivers: string[] }
  /** Bounding-box Y extent over render vertices only (joint points excluded). */
  | { id: 'height'; type: 'height'; drivers: string[] };
export interface MeasuresDef {
  version: 1;
  measures: MeasureDef[];
}
export interface PoseDef {
  id: string;
  label: { tr: string; en: string };
  /** bone-local rotation relative to rest pose */
  bones: Record<string, Quat>;
}
export type ShoeSystem = 'EU' | 'US_M' | 'US_W' | 'UK';
export interface BodyParams {
  gender: number; // 0 female .. 1 male
  heightCm: number;
  weightKg: number;
  shoulderCm?: number;
  neckCm?: number;
  chestCm?: number;
  waistCm?: number;
  hipCm?: number;
  thighCm?: number;
  upperArmCm?: number;
  armLengthCm?: number;
  inseamCm?: number;
  shoe: { system: ShoeSystem; size: number };
}
export interface GarmentDef {
  /* DRAFT — finalized in Wave 4 */
  id: string;
  category: 'top' | 'bottom' | 'shoes';
  license: 'CC0-1.0' | 'CC-BY-4.0' | 'user';
  attribution?: string;
  mesh: string;
  bindings?: string;
}
