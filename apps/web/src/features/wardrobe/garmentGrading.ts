import {
  bodyMeasuresForGarment,
  type GarmentMeasureId,
  type GarmentTemplateDef,
  type GradeRing,
  type MeasureId,
  type MeasuresDef,
  type StoreItemDef,
  type LengthGradeSpec,
  type Vec3,
} from '@dt/avatar-core';
import { GIRTH_MEASURES } from './chartModel';

/** Pure garment update helpers (no three.js): planes and grade rings. */

type Measures = Partial<Record<GarmentMeasureId, number>>;

/** Heights (m, grounded body frame) of the measure planes used to place grade rings. */
export interface BodyPlanes {
  planeY: Partial<Record<GarmentMeasureId, number>>;
  /** Height of the crotch for trouser length grading. */
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

/** Chart-length passes in the solved rest frame. Long sleeves use shoulder and wrist joints. */
export function buildLengthGrades(
  item: StoreItemDef,
  template: GarmentTemplateDef,
  achievedCm: Partial<Record<MeasureId, number>>,
  planes: BodyPlanes | undefined,
  rest: ArrayLike<number>,
  joint: (name: string) => Vec3 | undefined,
): LengthGradeSpec[] {
  const size = item.sizes.indexOf(item.selectedSize);
  if (size < 0 || template.category === 'shoes') return [];
  const specs: LengthGradeSpec[] = [];
  const hemHeights: number[] = [];
  for (let i = 1; i < rest.length; i += 3) {
    if (template.category === 'top' && Math.abs(rest[i - 1]!) > 0.2) continue;
    hemHeights.push(rest[i]!);
  }
  if (hemHeights.length === 0) return specs;
  hemHeights.sort((a, b) => a - b);
  // The lowest vertex can be a single seam tip. Use the lower hem band so the edge moves together.
  const hemY = hemHeights[Math.floor(hemHeights.length * 0.1)]!;
  const waistY = planes?.planeY.waist;
  const lengthChart = item.chart.length?.[size];
  if (template.category === 'top') {
    if (lengthChart !== undefined && waistY !== undefined && hemY < waistY - 0.01) {
      // No torso-length body measure exists; scale the template length with the solved body height.
      const referenceCm =
        (template.nativeMeasures.length ?? lengthChart) * ((achievedCm.height ?? 165.9) / 165.9);
      specs.push({ chartCm: lengthChart, referenceCm, anchor: [0, waistY, 0], hem: [0, hemY, 0] });
    }
    if (template.kind === 'longsleeve' || template.kind === 'sweatshirt') {
      const sleeveChart = item.chart.sleeve?.[size];
      const armLength = achievedCm.armLength;
      if (sleeveChart !== undefined && armLength !== undefined) {
        for (const [suffix, side] of [
          ['l', 1],
          ['r', -1],
        ] as const) {
          const shoulder = joint(`upperarm_${suffix}`);
          const wrist = joint(`hand_${suffix}`);
          if (!shoulder || !wrist) continue;
          specs.push({
            chartCm: sleeveChart,
            referenceCm: armLength + (template.defaultEase.sleeve ?? 0),
            anchor: shoulder,
            hem: wrist,
            side,
            minAbsX: Math.abs(shoulder[0]) * 0.8,
          });
        }
      }
    }
  } else {
    const inseamChart = item.chart.inseam?.[size];
    const inseam = achievedCm.inseam;
    if (inseamChart !== undefined && inseam !== undefined && planes?.crotchY !== undefined) {
      specs.push({
        chartCm: inseamChart,
        referenceCm: inseam + (template.defaultEase.inseam ?? 0),
        anchor: [0, planes.crotchY, 0],
        hem: [0, hemY, 0],
        floorY: 0,
      });
    } else if (lengthChart !== undefined && waistY !== undefined) {
      const referenceCm =
        inseam === undefined
          ? (template.nativeMeasures.length ?? lengthChart)
          : inseam +
            (template.nativeMeasures.length ?? lengthChart) -
            (template.nativeMeasures.inseam ?? inseam);
      specs.push({
        chartCm: lengthChart,
        referenceCm,
        anchor: [0, waistY, 0],
        hem: [0, hemY, 0],
        floorY: 0,
      });
    }
  }
  return specs;
}
