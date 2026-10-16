/** Hiding body triangles under worn garments (pure, no three.js). */

interface WeldLike {
  groupOf: ArrayLike<number>;
  start: ArrayLike<number>;
  members: ArrayLike<number>;
}

/**
 * Marks the deleted body render vertices (union of all worn items) and every UV-seam copy of them: the pipeline
 * lists MakeHuman vertices, but the render mesh holds one copy per UV island.
 */
export function hiddenVertexMask(
  deleteLists: readonly ArrayLike<number>[],
  vertexCount: number,
  weld?: WeldLike,
): Uint8Array {
  const mask = new Uint8Array(vertexCount);
  for (const list of deleteLists) {
    for (let i = 0; i < list.length; i++) {
      const v = list[i]!;
      if (v >= vertexCount) continue;
      if (weld) {
        const g = weld.groupOf[v]!;
        for (let m = weld.start[g]!; m < weld.start[g + 1]!; m++) mask[weld.members[m]!] = 1;
      } else {
        mask[v] = 1;
      }
    }
  }
  return mask;
}

/** Bone name prefixes whose skin influence protects a body vertex (same list as the importer's `pose_safety.py`). */
const PROTECTED_BONE_PREFIXES = [
  'hand',
  'neck',
  'head',
  'thumb',
  'index',
  'middle',
  'ring',
  'pinky',
] as const;
/** Skin weight above which a bone counts as influencing a vertex. */
const PROTECTED_MIN_WEIGHT = 0.001;
/** Everything at or above this distance below the neck joint (metres) is protected as neck / head. */
const NECK_MARGIN_M = 0.025;

export interface ProtectedBodyInput {
  /** Bone name per skin joint index (the skeleton's bone order). */
  boneNames: readonly string[];
  /** Four skin joint indices per render vertex. */
  skinIndex: ArrayLike<number>;
  /** Four skin weights per render vertex. */
  skinWeight: ArrayLike<number>;
  /** Solved, grounded render positions (3 per vertex). */
  positions: ArrayLike<number>;
  /** Height (y, metres) of the `neck_01` joint head in the same frame as `positions`; omit to skip the height rule. */
  neckY?: number;
  /** Groups the UV-seam copies of a vertex: a vertex is protected if any copy is. */
  weld?: WeldLike;
}

/**
 * Body render vertices that a worn garment must never hide: skin influenced by a hand, finger, neck or head bone, and
 * everything within 2.5 cm below the neck joint (the face, scalp and collar skin), including all UV-seam copies. This
 * is the same definition the garment importer uses for its delete lists (`tools/asset-pipeline/garmentimport/
 * pose_safety.py::protected_vertices`), derived from the body's skin data instead of index lists. One byte per vertex.
 */
export function protectedBodyVertices(input: ProtectedBodyInput): Uint8Array {
  const { boneNames, skinIndex, skinWeight, positions, neckY, weld } = input;
  const count = Math.floor(positions.length / 3);
  const boneProtected = boneNames.map((name) =>
    PROTECTED_BONE_PREFIXES.some((prefix) => name.startsWith(prefix)),
  );
  const out = new Uint8Array(count);
  const limit = neckY === undefined ? Infinity : neckY - NECK_MARGIN_M;
  for (let v = 0; v < count; v++) {
    if (positions[v * 3 + 1]! >= limit) {
      out[v] = 1;
      continue;
    }
    for (let k = 0; k < 4; k++) {
      if (skinWeight[v * 4 + k]! > PROTECTED_MIN_WEIGHT && boneProtected[skinIndex[v * 4 + k]!]) {
        out[v] = 1;
        break;
      }
    }
  }
  if (weld) {
    for (let g = 0; g < weld.start.length - 1; g++) {
      let any = false;
      for (let m = weld.start[g]!; m < weld.start[g + 1]!; m++) {
        if (out[weld.members[m]!]) any = true;
      }
      if (!any) continue;
      for (let m = weld.start[g]!; m < weld.start[g + 1]!; m++) out[weld.members[m]!] = 1;
    }
  }
  return out;
}

/** Vertical tolerance (metres) below a shortened hem that still counts as covered by the hem edge. */
const HEM_MARGIN_M = 0.002;
/** Minimum upward hem displacement (metres) before a static delete list is clipped to the graded lower edge. */
const MIN_HEM_RISE_M = 0.003;
/** Half-width (metres) of the torso / leg column in the rest A-pose; sleeve vertices lie outside it. */
const TORSO_HALF_WIDTH_M = 0.2;

export interface ClipDeleteOptions {
  /** Un-graded bound garment positions (3 floats per garment vertex); when given, only a raised hem clips. */
  rest?: ArrayLike<number>;
  /** Tolerance below the graded hem (metres) still treated as covered; defaults to 2 mm. */
  marginM?: number;
  /** UV-seam weld map; when given, all seam copies of a vertex share the same keep/drop outcome. */
  weld?: WeldLike;
}

interface HemFloors {
  coreFloorY: number;
  armFloorY: number;
}

function minHeights(positions: ArrayLike<number>): {
  all: number;
  core: number;
  arm: number;
} {
  let all = Infinity;
  let core = Infinity;
  let arm = Infinity;
  const count = Math.floor(positions.length / 3);
  for (let v = 0; v < count; v++) {
    const x = positions[v * 3]!;
    const y = positions[v * 3 + 1]!;
    if (y < all) all = y;
    if (Math.abs(x) <= TORSO_HALF_WIDTH_M) {
      if (y < core) core = y;
    } else if (y < arm) {
      arm = y;
    }
  }
  return { all, core, arm };
}

function computeHemFloors(garment: ArrayLike<number>, options: ClipDeleteOptions): HemFloors {
  const margin = options.marginM ?? HEM_MARGIN_M;
  const graded = minHeights(garment);
  if (!Number.isFinite(graded.all)) return { coreFloorY: -Infinity, armFloorY: -Infinity };
  const split = Number.isFinite(graded.core) && Number.isFinite(graded.arm);
  const { rest } = options;
  if (rest && rest.length >= garment.length) {
    const base = minHeights(rest);
    if (!split) {
      const floor = graded.all > base.all + MIN_HEM_RISE_M ? graded.all - margin : -Infinity;
      return { coreFloorY: floor, armFloorY: floor };
    }
    return {
      coreFloorY:
        Number.isFinite(base.core) && graded.core > base.core + MIN_HEM_RISE_M
          ? graded.core - margin
          : -Infinity,
      armFloorY:
        Number.isFinite(base.arm) && graded.arm > base.arm + MIN_HEM_RISE_M
          ? graded.arm - margin
          : -Infinity,
    };
  }
  if (!split) {
    const floor = graded.all - margin;
    return { coreFloorY: floor, armFloorY: floor };
  }
  return {
    coreFloorY: graded.core - margin,
    armFloorY: graded.arm - margin,
  };
}

function isVertexAboveHem(
  v: number,
  body: ArrayLike<number>,
  bodyVertexCount: number,
  floors: HemFloors,
): boolean {
  if (v < 0 || v >= bodyVertexCount) return true;
  const bx = body[v * 3]!;
  const by = body[v * 3 + 1]!;
  const floorY = Math.abs(bx) <= TORSO_HALF_WIDTH_M ? floors.coreFloorY : floors.armFloorY;
  return by >= floorY;
}

function isGroupAboveHem(
  v: number,
  body: ArrayLike<number>,
  bodyVertexCount: number,
  floors: HemFloors,
  weld?: WeldLike,
): boolean {
  if (!isVertexAboveHem(v, body, bodyVertexCount, floors)) return false;
  if (!weld || v < 0 || v >= weld.groupOf.length) return true;
  const g = weld.groupOf[v]!;
  for (let m = weld.start[g]!; m < weld.start[g + 1]!; m++) {
    if (!isVertexAboveHem(weld.members[m]!, body, bodyVertexCount, floors)) return false;
  }
  return true;
}

/**
 * Restricts a garment's static delete list (`deleteVerts`) to the body region still above the graded garment's lower
 * edge (hem / cuff). When `options.rest` is supplied and the hem has not moved up (unchanged or lengthened), the
 * original list is returned untouched so default body hiding stays bit-for-bit identical.
 */
export function clipDeleteVerts(
  deleteVerts: ArrayLike<number>,
  body: ArrayLike<number>,
  garment: ArrayLike<number>,
  options: ClipDeleteOptions = {},
): Uint32Array {
  const floors = computeHemFloors(garment, options);
  if (floors.coreFloorY === -Infinity && floors.armFloorY === -Infinity) {
    return deleteVerts instanceof Uint32Array ? deleteVerts : Uint32Array.from(deleteVerts);
  }
  const bodyVertexCount = Math.floor(body.length / 3);
  const { weld } = options;
  let kept = 0;
  for (let i = 0; i < deleteVerts.length; i++) {
    if (isGroupAboveHem(deleteVerts[i]!, body, bodyVertexCount, floors, weld)) kept++;
  }
  if (kept === deleteVerts.length) {
    return deleteVerts instanceof Uint32Array ? deleteVerts : Uint32Array.from(deleteVerts);
  }
  const out = new Uint32Array(kept);
  let o = 0;
  for (let i = 0; i < deleteVerts.length; i++) {
    const v = deleteVerts[i]!;
    if (isGroupAboveHem(v, body, bodyVertexCount, floors, weld)) out[o++] = v;
  }
  return out;
}

/**
 * Clears footprint-coverage flags below a shortened garment hem so triangle-prism slack (`bandM` / `slack`) cannot
 * hide ankle or waist skin exposed below the new hem. No-op when the hem has not moved up.
 */
export function clipCoverageToHem(
  coverage: Uint8Array,
  body: ArrayLike<number>,
  garment: ArrayLike<number>,
  options: ClipDeleteOptions = {},
): Uint8Array {
  const floors = computeHemFloors(garment, options);
  if (floors.coreFloorY === -Infinity && floors.armFloorY === -Infinity) return coverage;
  const bodyVertexCount = Math.min(coverage.length, Math.floor(body.length / 3));
  const { weld } = options;
  for (let v = 0; v < bodyVertexCount; v++) {
    if (coverage[v] && !isGroupAboveHem(v, body, bodyVertexCount, floors, weld)) {
      coverage[v] = 0;
    }
  }
  return coverage;
}

/**
 * The final hidden mask of the body: the union of the garments' delete lists (with their UV-seam copies) and the
 * app's own footprint coverage (`coveredBodyVertices`, one byte per vertex or null), with the `protectedMask` vertices
 * cleared again. The guard applies to the COMBINED mask, so neither source can hide hands, neck or face.
 */
export function combinedHiddenMask(
  deleteLists: readonly ArrayLike<number>[],
  coverages: readonly (ArrayLike<number> | null)[],
  vertexCount: number,
  weld?: WeldLike,
  protectedMask?: ArrayLike<number>,
): Uint8Array {
  const mask = hiddenVertexMask(deleteLists, vertexCount, weld);
  for (const coverage of coverages) {
    if (!coverage) continue;
    for (let v = 0; v < vertexCount; v++) {
      if (!coverage[v]) continue;
      if (weld && v < weld.groupOf.length) {
        const g = weld.groupOf[v]!;
        for (let m = weld.start[g]!; m < weld.start[g + 1]!; m++) mask[weld.members[m]!] = 1;
      } else {
        mask[v] = 1;
      }
    }
  }
  if (protectedMask) for (let v = 0; v < vertexCount; v++) if (protectedMask[v]) mask[v] = 0;
  return mask;
}

/** Body triangle list without the triangles whose three vertices are all hidden. Returns `indices` itself if none. */
export function filterBodyIndex(indices: Uint32Array, hidden: Uint8Array): Uint32Array {
  let kept = 0;
  for (let t = 0; t < indices.length; t += 3) {
    if (!(hidden[indices[t]!] && hidden[indices[t + 1]!] && hidden[indices[t + 2]!])) kept += 3;
  }
  if (kept === indices.length) return indices;
  const out = new Uint32Array(kept);
  let o = 0;
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t]!;
    const b = indices[t + 1]!;
    const c = indices[t + 2]!;
    if (hidden[a] && hidden[b] && hidden[c]) continue;
    out[o++] = a;
    out[o++] = b;
    out[o++] = c;
  }
  return out;
}

/** Parses a `.delete.bin` (little-endian uint32 list). */
export function parseDeleteVerts(buffer: ArrayBuffer): Uint32Array {
  if (buffer.byteLength % 4 !== 0)
    throw new Error('garment delete list: byte length must be a multiple of 4');
  const view = new DataView(buffer);
  const out = new Uint32Array(buffer.byteLength / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getUint32(i * 4, true);
  return out;
}

/** Barycentric slack of the footprint test (fraction of a triangle). */
const SLACK = 0.3;

export interface CoverOptions {
  /** Largest distance of a body vertex from the garment surface (either side) that counts as covered, metres. */
  bandM?: number;
  /** Grid cell size, metres. */
  cellM?: number;
  /** Projection slack; use zero for garment openings so an open jacket keeps its visible inner top. */
  slack?: number;
}

/**
 * Body vertices that lie under the garment's footprint: the vertex projects inside a garment triangle (along that
 * triangle's normal) and is at most `bandM` away from its plane. MakeHuman delete lists leave small islands (the navel
 * ring) whose triangles poke through the fabric between sparse garment vertices. A body triangle whose three
 * vertices are all covered can be dropped; triangles at the garment edge always keep an uncovered vertex, so no hole
 * opens next to the hem. Only garment surfaces within a centimetre or so count: farther fabric (loose clothes) hides
 * the skin by itself. Returns one byte per body vertex.
 */
export function coveredBodyVertices(
  body: ArrayLike<number>,
  vertexCount: number,
  garment: ArrayLike<number>,
  garmentIndex: ArrayLike<number>,
  options: CoverOptions = {},
): Uint8Array {
  const band = options.bandM ?? 0.02;
  const cell = options.cellM ?? 0.03;
  const slack = options.slack ?? SLACK;
  const key = (ix: number, iy: number, iz: number): number =>
    ((ix + 1024) * 2048 + (iy + 1024)) * 2048 + (iz + 1024);
  const grid = new Map<number, number[]>();
  for (let v = 0; v < vertexCount; v++) {
    const k = key(
      Math.floor(body[v * 3]! / cell),
      Math.floor(body[v * 3 + 1]! / cell),
      Math.floor(body[v * 3 + 2]! / cell),
    );
    const list = grid.get(k);
    if (list) list.push(v);
    else grid.set(k, [v]);
  }
  const covered = new Uint8Array(vertexCount);
  const triangleCount = Math.floor(garmentIndex.length / 3);
  for (let t = 0; t < triangleCount; t++) {
    const i0 = garmentIndex[t * 3]! * 3;
    const i1 = garmentIndex[t * 3 + 1]! * 3;
    const i2 = garmentIndex[t * 3 + 2]! * 3;
    const e1x = garment[i1]! - garment[i0]!;
    const e1y = garment[i1 + 1]! - garment[i0 + 1]!;
    const e1z = garment[i1 + 2]! - garment[i0 + 2]!;
    const e2x = garment[i2]! - garment[i0]!;
    const e2y = garment[i2 + 1]! - garment[i0 + 1]!;
    const e2z = garment[i2 + 2]! - garment[i0 + 2]!;
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;
    const area2 = Math.hypot(nx, ny, nz);
    if (area2 < 1e-12) continue;
    nx /= area2;
    ny /= area2;
    nz /= area2;
    const d11 = e1x * e1x + e1y * e1y + e1z * e1z;
    const d12 = e1x * e2x + e1y * e2y + e1z * e2z;
    const d22 = e2x * e2x + e2y * e2y + e2z * e2z;
    const det = d11 * d22 - d12 * d12;
    if (Math.abs(det) < 1e-18) continue;
    const lo = [
      Math.min(garment[i0]!, garment[i1]!, garment[i2]!) - band,
      Math.min(garment[i0 + 1]!, garment[i1 + 1]!, garment[i2 + 1]!) - band,
      Math.min(garment[i0 + 2]!, garment[i1 + 2]!, garment[i2 + 2]!) - band,
    ];
    const hi = [
      Math.max(garment[i0]!, garment[i1]!, garment[i2]!) + band,
      Math.max(garment[i0 + 1]!, garment[i1 + 1]!, garment[i2 + 1]!) + band,
      Math.max(garment[i0 + 2]!, garment[i1 + 2]!, garment[i2 + 2]!) + band,
    ];
    for (let ix = Math.floor(lo[0]! / cell); ix <= Math.floor(hi[0]! / cell); ix++)
      for (let iy = Math.floor(lo[1]! / cell); iy <= Math.floor(hi[1]! / cell); iy++)
        for (let iz = Math.floor(lo[2]! / cell); iz <= Math.floor(hi[2]! / cell); iz++) {
          const list = grid.get(key(ix, iy, iz));
          if (!list) continue;
          for (const v of list) {
            if (covered[v] === 1) continue;
            const px = body[v * 3]! - garment[i0]!;
            const py = body[v * 3 + 1]! - garment[i0 + 1]!;
            const pz = body[v * 3 + 2]! - garment[i0 + 2]!;
            if (Math.abs(px * nx + py * ny + pz * nz) > band) continue;
            // barycentric coordinates of the projection onto the triangle plane
            const p1 = px * e1x + py * e1y + pz * e1z;
            const p2 = px * e2x + py * e2y + pz * e2z;
            const u = (d22 * p1 - d12 * p2) / det;
            const w = (d11 * p2 - d12 * p1) / det;
            // a little slack closes the wedges between the prisms of neighbouring triangles at folds
            if (u >= -slack && w >= -slack && u + w <= 1 + slack) covered[v] = 1;
          }
        }
  }
  return covered;
}
