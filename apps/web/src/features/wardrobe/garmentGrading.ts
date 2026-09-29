import {
  bodyMeasuresForGarment,
  garmentClearance,
  gradeGarment,
  type GarmentBinding,
  type GarmentMeasureId,
  type GarmentTemplateDef,
  type GradeRing,
  type MeasureId,
  type MeasuresDef,
  type StoreItemDef,
} from '@dt/avatar-core';
import { GIRTH_MEASURES } from './chartModel';

/** Pure garment update helpers (no three.js): planes, grade rings and the leg-aware grade / clearance passes. */

type Measures = Partial<Record<GarmentMeasureId, number>>;

/** Heights (m, grounded body frame) of the measure planes used to place grade rings. */
export interface BodyPlanes {
  planeY: Partial<Record<GarmentMeasureId, number>>;
  /** Height of the crotch: below it the two legs are treated separately. */
  crotchY?: number;
}

const RING_MEASURES = GIRTH_MEASURES;

function meanY(verts: readonly number[], positions: ArrayLike<number>): number | undefined {
  let sum = 0;
  let n = 0;
  for (const v of verts) {
    const y = positions[v * 3 + 1];
    if (y === undefined) continue;
    sum += y;
    n++;
  }
  return n > 0 ? sum / n : undefined;
}

/** Plane heights from the loop vertices of the measure definitions on the solved (grounded) render positions. */
export function bodyPlanes(measures: MeasuresDef, positions: ArrayLike<number>): BodyPlanes {
  const planeY: BodyPlanes['planeY'] = {};
  let crotchY: number | undefined;
  for (const def of measures.measures) {
    if (def.type === 'circumference' && (RING_MEASURES as readonly string[]).includes(def.id)) {
      const y = meanY(def.verts, positions);
      if (y !== undefined) planeY[def.id as GarmentMeasureId] = y;
    } else if (def.type === 'vertexHeight' && def.id === 'inseam') {
      crotchY = positions[def.vert * 3 + 1];
    }
  }
  return crotchY === undefined ? { planeY } : { planeY, crotchY };
}

/**
 * Grade rings for one chart size. The bound garment already follows the body with the template's own ease
 * (`defaultEase`), so the girth still to add is `chart value - (body + defaultEase)`; without a `defaultEase` the
 * template's native measurement is the reference. Only girths that have a plane and a chart value produce a ring.
 */
export function buildGradeRings(
  item: StoreItemDef,
  template: GarmentTemplateDef,
  bodyCm: Measures,
  planes: BodyPlanes,
): GradeRing[] {
  const index = item.sizes.indexOf(item.selectedSize);
  if (index < 0) return [];
  const rings: GradeRing[] = [];
  for (const id of RING_MEASURES) {
    const chartValue = item.chart[id]?.[index];
    const planeY = planes.planeY[id];
    if (chartValue === undefined || planeY === undefined || !Number.isFinite(chartValue)) continue;
    const body = bodyCm[id];
    const ease = template.defaultEase[id];
    const native = template.nativeMeasures[id];
    let currentCm: number | undefined;
    if (body !== undefined && ease !== undefined) currentCm = body + ease;
    else if (native !== undefined) currentCm = native;
    if (currentCm === undefined) continue;
    rings.push({ planeY, deltaCircumferenceM: (chartValue - currentCm) / 100 });
  }
  return rings;
}

/** Body measures (cm) in the chart vocabulary from the solver's achieved values. */
export function bodyCmFromSolve(achievedCm: Partial<Record<MeasureId, number>>): Measures {
  return bodyMeasuresForGarment(achievedCm);
}

const FAR_Y = -1000;

/** Scratch buffers of the leg-aware passes (grown on demand, reused between solves). */
export class SplitScratch {
  private masked = new Float32Array(0);
  private sideOut = new Float32Array(0);
  private choices = new Int8Array(0);

  maskedBody(length: number): Float32Array {
    if (this.masked.length < length) this.masked = new Float32Array(length);
    return this.masked.subarray(0, length);
  }

  choiceBuffer(length: number): Int8Array {
    if (this.choices.length < length) this.choices = new Int8Array(length);
    return this.choices.subarray(0, length);
  }

  outBuffer(length: number): Float32Array {
    if (this.sideOut.length < length) this.sideOut = new Float32Array(length);
    return this.sideOut.subarray(0, length);
  }
}

function sideOfX(x: number): 1 | -1 {
  return x >= 0 ? 1 : -1;
}

/**
 * The section-centroid model of avatar-core pools both legs at one height, so a vertex on the inner side of a thigh
 * would be pushed into the leg. Below the crotch this runs `run` once per leg on a body copy that keeps only that
 * leg's vertices in the height bins, and takes each garment vertex from the run of the leg it is bound to (vertices
 * bound across the midline, or above the crotch, come from the plain run). `stride` is 3 for positions, 1 for scalars.
 */
export function runLegAware(opts: {
  binding: GarmentBinding;
  body: Float32Array;
  crotchY: number | undefined;
  stride: 1 | 3;
  out: Float32Array;
  scratch: SplitScratch;
  run: (body: Float32Array, out: Float32Array) => void;
}): void {
  const { binding, body, crotchY, stride, out, scratch, run } = opts;
  run(body, out);
  if (crotchY === undefined) return;
  const size = binding.count * stride;
  const sides: (1 | -1)[] = [1, -1];
  // decide, per garment vertex, which run to use: 0 = plain, otherwise the leg side
  const choice = scratch.choiceBuffer(binding.count);
  choice.fill(0);
  for (let v = 0; v < binding.count; v++) {
    const o = v * 3;
    let by = 0;
    let side: 1 | -1 | 0 = 0;
    let consistent = true;
    for (let j = 0; j < 3; j++) {
      const p = binding.indices[o + j]! * 3;
      by += binding.weights[o + j]! * body[p + 1]!;
      const s = sideOfX(body[p]!);
      if (side === 0) side = s;
      else if (side !== s) consistent = false;
    }
    if (consistent && side !== 0 && by < crotchY) choice[v] = side;
  }
  for (const side of sides) {
    if (!choice.includes(side)) continue;
    const masked = scratch.maskedBody(body.length);
    masked.set(body);
    for (let i = 1; i < body.length; i += 3) {
      if (body[i]! < crotchY && sideOfX(body[i - 1]!) !== side) masked[i] = FAR_Y;
    }
    const sideOut = scratch.outBuffer(size);
    run(masked, sideOut);
    for (let v = 0; v < binding.count; v++) {
      if (choice[v] !== side) continue;
      for (let k = 0; k < stride; k++) out[v * stride + k] = sideOut[v * stride + k]!;
    }
  }
}

export function gradeGarmentLegAware(
  restPositions: ArrayLike<number>,
  binding: GarmentBinding,
  body: Float32Array,
  rings: GradeRing[],
  crotchY: number | undefined,
  out: Float32Array,
  scratch: SplitScratch,
): void {
  runLegAware({
    binding,
    body,
    crotchY,
    stride: 3,
    out,
    scratch,
    run: (b, o) => gradeGarment(restPositions, binding, b, rings, o),
  });
}

export function clearanceLegAware(
  garmentPositions: ArrayLike<number>,
  binding: GarmentBinding,
  body: Float32Array,
  crotchY: number | undefined,
  out: Float32Array,
  scratch: SplitScratch,
): void {
  runLegAware({
    binding,
    body,
    crotchY,
    stride: 1,
    out,
    scratch,
    run: (b, o) => garmentClearance(garmentPositions, binding, b, o),
  });
}
