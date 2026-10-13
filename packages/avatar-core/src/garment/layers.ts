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
  if (template.category === 'shoes') return 0; // Preserve rigid soles and the floor contact.
  return Math.max(
    0,
    garmentLayerOrder(template) * 0.001,
    ...innerOffsets.map((offset) => offset + 0.002),
  );
}

/** Applied to a freshly fitted surface on every update; never accumulate offsets across solves. */
export function offsetAlongNormals(
  positions: Float32Array,
  normals: ArrayLike<number>,
  offsetM: number,
): void {
  for (let i = 0; i < positions.length; i++) positions[i] = positions[i]! + normals[i]! * offsetM;
}
