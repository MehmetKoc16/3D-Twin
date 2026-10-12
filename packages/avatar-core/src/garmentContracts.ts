import type { Quat, Vec3 } from './contracts';

export type GarmentCategory = 'top' | 'bottom' | 'shoes';
export type GarmentKind =
  | 'tshirt'
  | 'longsleeve'
  | 'sweatshirt'
  | 'pants'
  | 'jeans'
  | 'shorts'
  | 'sneakers'
  | 'shoes'
  | 'boots';
export type GarmentLicense = 'CC0-1.0' | 'CC-BY-3.0' | 'CC-BY-4.0' | 'user';

export function isGarmentLicense(value: unknown): value is GarmentLicense {
  return value === 'CC0-1.0' || value === 'CC-BY-3.0' || value === 'CC-BY-4.0' || value === 'user';
}

/** Validate exact licence identifiers and the credit required by either CC-BY version. */
export function validateGarmentLicense(license: unknown, attribution: unknown): void {
  if (!isGarmentLicense(license)) throw new Error('garment licence: unsupported licence');
  if (
    (license === 'CC-BY-3.0' || license === 'CC-BY-4.0') &&
    (typeof attribution !== 'string' || attribution.trim().length === 0)
  )
    throw new Error('garment licence: CC-BY requires attribution');
}

/** Importer's <id>.rigid.json, in garment GLB render-vertex order and neutral bone-local metres. */
export interface GarmentRigidAttachments {
  version: 1;
  soleHeightMm: number;
  attachments: {
    bone: 'foot_l' | 'ball_l' | 'foot_r' | 'ball_r';
    vertices: number[];
    localPositionsM: Vec3[];
  }[];
  /** Review metadata from the importer; these fields do not disable runtime support. */
  requiresRuntimeChange?: boolean;
  warnings?: string[];
}

/** Size-chart measurement points, always stored as full circumference / length in cm. */
export type GarmentMeasureId =
  'chest' | 'waist' | 'hip' | 'length' | 'sleeve' | 'shoulder' | 'inseam' | 'thigh' | 'footLength';

/**
 * Proxy binding file (`binding`, little-endian, 36 bytes per garment vertex, in garment vertex order):
 * uint32 i1, i2, i3 (body vertices, combined index space) · float32 w1, w2, w3 · float32 ox, oy, oz (meters).
 * Garment rest position = w1·v[i1] + w2·v[i2] + w3·v[i3] + (ox·sx, oy·sy, oz·sz),
 * where s{x,y,z} = |v[a] − v[b]|_axis / refM from `scaleRefs` (MakeHuman MHCLO semantics).
 */
export interface GarmentScaleRefs {
  x: [a: number, b: number, refM: number];
  y: [a: number, b: number, refM: number];
  z: [a: number, b: number, refM: number];
}

/** One entry of `apps/web/public/assets/garments/index.json` (`{ version: 1, garments: GarmentTemplateDef[] }`). */
export interface GarmentTemplateDef {
  id: string;
  kind: GarmentKind;
  category: GarmentCategory;
  label: { tr: string; en: string };
  license: GarmentLicense;
  /** Required for CC-BY assets. */
  attribution?: string;
  /** Upstream asset path + pinned SHA. */
  source?: string;
  /** Unskinned garment mesh (.glb): POSITION (neutral-body rest frame, meters), NORMAL, TEXCOORD_0, one material. */
  mesh: string;
  /** Proxy binding (.bin, format above). Required for mixed rigid soles and barycentric uppers. */
  binding?: string;
  /** Mixed shoe skinning sidecar (.rigid.json). Shoes also probe <id>.rigid.json when absent. */
  rigidAttachments?: string;
  scaleRefs?: GarmentScaleRefs;
  /** Body render vertices hidden while this garment is worn (.bin, uint32[]), to avoid poke-through. */
  deleteVerts?: string;
  /** Rigid items only: attachment to a bone, transform relative to that bone at the neutral body. */
  attach?: { bone: string; position: Vec3; rotation: Quat; innerLengthM: number };
  /** Garment measurements of the template on the neutral body, cm. */
  nativeMeasures: Partial<Record<GarmentMeasureId, number>>;
  /** Default ease (garment − body, cm) of a regular fit, per measure. */
  defaultEase: Partial<Record<GarmentMeasureId, number>>;
  /** Layer order for stacking (e.g. tshirt 1 < sweatshirt 2; pants 1; shoes 1). */
  layer: number;
  baseColor: string;
}

export type BodyPartCategory = 'eyes' | 'eyebrows' | 'eyelashes' | 'hair';

/**
 * Proxy-bound body parts (same binding format as garments). One entry of
 * `apps/web/public/assets/parts/index.json`
 * (`{ version: 1, parts: BodyPartDef[], defaults: Partial<Record<BodyPartCategory, string>> }`).
 */
export interface BodyPartDef {
  id: string;
  category: BodyPartCategory;
  label: { tr: string; en: string };
  license: GarmentLicense;
  attribution?: string;
  source?: string;
  /** Unskinned mesh (.glb), same frame as garment meshes. */
  mesh: string;
  binding: string;
  scaleRefs?: GarmentScaleRefs;
  deleteVerts?: string;
  material: {
    alphaMode: 'OPAQUE' | 'MASK' | 'BLEND';
    alphaCutoff?: number;
    doubleSided: boolean;
    /** Colour can be changed at runtime (hair/eyebrow colour, iris colour). */
    tintable: boolean;
  };
  /** Eyes only: iris region in the eye texture's UV space, for recolouring the iris. */
  irisUv?: { center: [number, number]; radius: number };
}

/** A product the user entered from an online store. */
export interface StoreItemDef {
  id: string;
  name: string;
  storeUrl?: string;
  templateId: string;
  color: string;
  /** Size labels in chart order, e.g. ['S', 'M', 'L']. */
  sizes: string[];
  /** Per measure, one value per size (full circumference / length, cm). */
  chart: Partial<Record<GarmentMeasureId, number[]>>;
  selectedSize: string;
}

export type FitVerdict = 'tight' | 'snug' | 'regular' | 'loose' | 'oversized';

export interface FitRegionResult {
  id: GarmentMeasureId;
  bodyCm: number;
  garmentCm: number;
  /** garmentCm − bodyCm */
  easeCm: number;
  verdict: FitVerdict;
}

export interface FitReport {
  itemId: string;
  size: string;
  regions: FitRegionResult[];
  overall: FitVerdict;
  /** Best size label from the chart for a regular fit, if any. */
  recommendedSize?: string;
}
