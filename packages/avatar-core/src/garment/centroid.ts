/** A horizontal body section, split into limbs at gaps in the XZ plane. */
export interface SectionComponent {
  x: number;
  z: number;
}

export class GarmentSections {
  private readonly bins = new Map<number, SectionComponent[]>();

  constructor(bodyPositions: ArrayLike<number>) {
    const points = new Map<number, number[]>();
    for (let i = 0; i + 2 < bodyPositions.length; i += 3) {
      const y = bodyPositions[i + 1]!;
      if (!Number.isFinite(y)) continue;
      const key = Math.round(y * 100);
      const bin = points.get(key);
      if (bin) bin.push(i);
      else points.set(key, [i]);
    }
    for (const [key, indices] of points) {
      indices.sort((a, b) => bodyPositions[a]! - bodyPositions[b]!);
      const cuts: number[] = [0];
      for (let i = 1; i < indices.length; i++) {
        const left = bodyPositions[indices[i - 1]!]!;
        const right = bodyPositions[indices[i]!]!;
        // The midline gap can be narrower than sparse samples around a leg ring.
        if ((left < 0 && right > 0 && right - left > 0.025) || right - left > 0.08) cuts.push(i);
      }
      cuts.push(indices.length);
      const components: SectionComponent[] = [];
      for (let c = 1; c < cuts.length; c++) {
        const group = indices.slice(cuts[c - 1]!, cuts[c]!);
        group.sort((a, b) => bodyPositions[a + 2]! - bodyPositions[b + 2]!);
        let start = 0;
        for (let i = 1; i <= group.length; i++) {
          if (
            i < group.length &&
            bodyPositions[group[i]! + 2]! - bodyPositions[group[i - 1]! + 2]! <= 0.1
          )
            continue;
          let x = 0,
            z = 0;
          for (let j = start; j < i; j++) {
            const p = group[j]!;
            x += bodyPositions[p]!;
            z += bodyPositions[p + 2]!;
          }
          components.push({ x: x / (i - start), z: z / (i - start) });
          start = i;
        }
      }
      this.bins.set(key, components);
    }
  }

  /** Select the component containing the closest body section to a vertex's bound point. */
  centroidAt(y: number, x: number, z: number): SectionComponent {
    const key = Math.round(y * 100);
    let components = this.bins.get(key);
    if (!components) {
      for (let distance = 1; distance < 250 && !components; distance++)
        components = this.bins.get(key - distance) ?? this.bins.get(key + distance);
    }
    if (!components?.length) return { x: 0, z: 0 };
    let best = components[0]!;
    let bestD = Infinity;
    for (const candidate of components) {
      const d = (candidate.x - x) ** 2 + (candidate.z - z) ** 2;
      if (d < bestD) {
        bestD = d;
        best = candidate;
      }
    }
    return best;
  }
}

/** Build once per solved body and share across grading and clearance passes. */
export function createGarmentSections(bodyPositions: ArrayLike<number>): GarmentSections {
  return new GarmentSections(bodyPositions);
}
