import { garmentSlot, type GarmentTemplateDef } from '../garmentContracts';

/** Depth tie ordering; an explicit outer slot still sorts above an ordinary top. */
export function garmentLayerOrder(template: GarmentTemplateDef): number {
  return (
    Math.max(template.layer, garmentSlot(template) === 'outer' ? 3 : 0) +
    (template.category === 'top' ? 0.25 : 0)
  );
}

/** Small physical separation in addition to authored binding offsets and collision clearance. */
export function garmentNormalOffset(
  template: GarmentTemplateDef,
  innerOffsets: readonly number[] = [],
): number {
  if (template.category === 'shoes' || innerOffsets.length === 0) return 0;
  return Math.max(0, ...innerOffsets.map((offset) => offset + 0.002));
}

export interface OffsetNormalsOptions {
  /** Optional reference vectors pointing away from the body (3 floats per vertex, e.g. positions - bound). */
  outward?: ArrayLike<number>;
  /** Maximum distance between opposing faces to consider as a thin flap (default 0.006 m = 6 mm). */
  flapThresholdM?: number;
}

/** Applied to a freshly fitted surface on every update; never accumulate offsets across solves. */
export function offsetAlongNormals(
  positions: Float32Array,
  normals: ArrayLike<number>,
  offsetM: number,
  options: OffsetNormalsOptions = {},
): void {
  if (offsetM <= 0) return;
  const n = Math.floor(positions.length / 3);
  if (n === 0) return;

  const outward = options.outward;
  const threshold = options.flapThresholdM ?? 0.006;
  const thresholdSq = threshold * threshold;

  const dirX = new Float32Array(n);
  const dirY = new Float32Array(n);
  const dirZ = new Float32Array(n);

  for (let v = 0; v < n; v++) {
    const nx = normals[v * 3]!;
    const ny = normals[v * 3 + 1]!;
    const nz = normals[v * 3 + 2]!;
    const len = Math.hypot(nx, ny, nz);
    if (len < 0.5) continue;
    let dx = nx / len;
    let dy = ny / len;
    let dz = nz / len;

    if (outward) {
      const ox = outward[v * 3]!;
      const oy = outward[v * 3 + 1]!;
      const oz = outward[v * 3 + 2]!;
      if (dx * ox + dy * oy + dz * oz < -1e-6) {
        dx = -dx;
        dy = -dy;
        dz = -dz;
      }
    }
    dirX[v] = dx;
    dirY[v] = dy;
    dirZ[v] = dz;
  }

  const cell = 0.02;
  const key = (ix: number, iy: number, iz: number): number =>
    ((ix + 1024) * 2048 + (iy + 1024)) * 2048 + (iz + 1024);
  const grid = new Map<number, number[]>();

  for (let v = 0; v < n; v++) {
    if (dirX[v] === 0 && dirY[v] === 0 && dirZ[v] === 0) continue;
    const px = positions[v * 3]!;
    const py = positions[v * 3 + 1]!;
    const pz = positions[v * 3 + 2]!;
    const ix = Math.floor(px / cell);
    const iy = Math.floor(py / cell);
    const iz = Math.floor(pz / cell);

    let paired = false;
    for (let di = -1; di <= 1 && !paired; di++) {
      for (let dj = -1; dj <= 1 && !paired; dj++) {
        for (let dk = -1; dk <= 1 && !paired; dk++) {
          const list = grid.get(key(ix + di, iy + dj, iz + dk));
          if (!list) continue;
          for (const u of list) {
            const distSq =
              (px - positions[u * 3]!) ** 2 +
              (py - positions[u * 3 + 1]!) ** 2 +
              (pz - positions[u * 3 + 2]!) ** 2;
            if (distSq <= thresholdSq) {
              const dot =
                normals[v * 3]! * normals[u * 3]! +
                normals[v * 3 + 1]! * normals[u * 3 + 1]! +
                normals[v * 3 + 2]! * normals[u * 3 + 2]!;
              if (dot < -0.2) {
                dirX[v] = dirX[u]!;
                dirY[v] = dirY[u]!;
                dirZ[v] = dirZ[u]!;
                paired = true;
                break;
              }
            }
          }
        }
      }
    }
    const k = key(ix, iy, iz);
    const cellList = grid.get(k);
    if (cellList) cellList.push(v);
    else grid.set(k, [v]);
  }

  for (let v = 0; v < n; v++) {
    positions[v * 3] = positions[v * 3]! + dirX[v]! * offsetM;
    positions[v * 3 + 1] = positions[v * 3 + 1]! + dirY[v]! * offsetM;
    positions[v * 3 + 2] = positions[v * 3 + 2]! + dirZ[v]! * offsetM;
  }
}
