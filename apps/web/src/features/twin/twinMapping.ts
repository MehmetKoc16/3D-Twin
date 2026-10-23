import { coveredBodyVertices } from '../wardrobe/bodyHide';
import { TwinFormatError } from './twinDef';

/** A worn garment surface for footprint testing. */
export interface GarmentSurface {
  positions: ArrayLike<number>;
  index: ArrayLike<number>;
}

/** Maximum distance (m) from a worn garment surface to keep mapping-hidden twin vertices, keeping loose torso/sleeves hidden while exposing skin beyond cuffs. */
export const MAPPING_NEAR_GARMENT_M = 0.06;

/**
 * Hiding twin triangles under worn garments (pure, no three.js).
 *
 * `mh2twin.bin` maps every twin vertex to its nearest MakeHuman render vertex (uint32 little endian). The hidden
 * MakeHuman body already loses the triangles under the worn garments (pipeline delete lists plus the wardrobe's
 * footprint pass, see wardrobeRig.updateBodyIndex); a twin triangle is hidden when its three vertices map to body
 * vertices that no remaining body triangle uses any more.
 */

/** Parses a `mh2twin.bin`. Throws `TwinFormatError('mapping')` on a size that is not a whole number of uint32. */
export function parseMapping(buffer: ArrayBuffer): Uint32Array {
  if (buffer.byteLength === 0 || buffer.byteLength % 4 !== 0)
    throw new TwinFormatError(
      'mapping',
      'mh2twin.bin: byte length must be a positive multiple of 4',
    );
  const view = new DataView(buffer);
  const out = new Uint32Array(buffer.byteLength / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getUint32(i * 4, true);
  return out;
}

/** Checks the mapping against the twin mesh and the body it points into. */
export function validateMapping(
  mapping: Uint32Array,
  twinVertexCount: number,
  renderVertexCount: number,
): void {
  if (mapping.length !== twinVertexCount)
    throw new TwinFormatError(
      'mapping',
      `mh2twin.bin has ${mapping.length} entries, the twin mesh has ${twinVertexCount} vertices`,
    );
  for (let i = 0; i < mapping.length; i++)
    if (mapping[i]! >= renderVertexCount)
      throw new TwinFormatError('mapping', `mh2twin.bin: vertex ${i} points outside the body`);
}

/**
 * Body vertices that no triangle of `index` references: the vertices whose every triangle was removed by the
 * wardrobe. Vertices at the border of a hidden area are still referenced by a kept triangle and count as visible.
 */
export function unreferencedVertices(
  index: ArrayLike<number>,
  vertexCount: number,
  out: Uint8Array = new Uint8Array(vertexCount),
): Uint8Array {
  out.fill(1);
  for (let i = 0; i < index.length; i++) out[index[i]!] = 0;
  return out;
}

/** Per twin vertex: 1 when the body vertex it maps to is hidden. */
export function hiddenTwinVertices(
  mapping: Uint32Array,
  hiddenBody: Uint8Array,
  out: Uint8Array = new Uint8Array(mapping.length),
): Uint8Array {
  for (let i = 0; i < mapping.length; i++) out[i] = hiddenBody[mapping[i]!]!;
  return out;
}

/**
 * Writes the triangles of `source` that have at least one visible vertex into `target` (in place, same order) and
 * returns the number of indices written. `target` must be at least `source.length` long.
 */
export function compactTwinIndex(
  source: ArrayLike<number>,
  hidden: Uint8Array,
  target: Uint32Array,
): number {
  let kept = 0;
  for (let t = 0; t + 2 < source.length; t += 3) {
    const a = source[t]!;
    const b = source[t + 1]!;
    const c = source[t + 2]!;
    if (hidden[a] === 1 && hidden[b] === 1 && hidden[c] === 1) continue;
    target[kept++] = a;
    target[kept++] = b;
    target[kept++] = c;
  }
  return kept;
}

/**
 * Keeps mapping-hidden twin vertices only if they also fall within a wide footprint (`bandM`, slack 0)
 * of at least one worn garment surface.
 * When there are no garment surfaces, today's mapping behaviour is preserved.
 */
export function filterMappingNearGarments(
  mappingHidden: Uint8Array,
  positions: ArrayLike<number>,
  vertexCount: number,
  surfaces: readonly GarmentSurface[],
  bandM = MAPPING_NEAR_GARMENT_M,
  out: Uint8Array = mappingHidden,
): Uint8Array {
  if (surfaces.length === 0) {
    if (out !== mappingHidden) out.set(mappingHidden);
    return out;
  }
  let anyHidden = false;
  for (let v = 0; v < vertexCount; v++) {
    if (mappingHidden[v]) {
      anyHidden = true;
      break;
    }
  }
  if (!anyHidden) {
    if (out !== mappingHidden) out.fill(0);
    return out;
  }
  const nearGarment = new Uint8Array(vertexCount);
  for (let s = 0; s < surfaces.length; s++) {
    const surface = surfaces[s]!;
    const covered = coveredBodyVertices(positions, vertexCount, surface.positions, surface.index, {
      bandM,
      slack: 0,
    });
    for (let v = 0; v < vertexCount; v++) {
      if (covered[v]) nearGarment[v] = 1;
    }
  }
  for (let v = 0; v < vertexCount; v++) {
    out[v] = mappingHidden[v]! & nearGarment[v]!;
  }
  return out;
}
