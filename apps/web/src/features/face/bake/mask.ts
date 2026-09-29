/** Pure math for the face alpha mask (feathered oval, softened forehead). */

export type Point = readonly [number, number];

export function smoothstep(edge0: number, edge1: number, x: number): number {
  if (edge1 === edge0) return x < edge0 ? 0 : 1;
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/** Alpha for a point `inside` pixels inside the oval boundary (<= 0 outside): 0 at the edge, 1 after `feather`. */
export function featherAlpha(inside: number, feather: number): number {
  return smoothstep(0, feather, inside);
}

/** Alpha factor that fades the area above the brow line: 0 at `topY`, 1 at `browY` (works for either order). */
export function foreheadFade(y: number, topY: number, browY: number): number {
  return smoothstep(topY, browY, y);
}

export function pointInPolygon(x: number, y: number, polygon: readonly Point[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const [xi, yi] = polygon[i] ?? [0, 0];
    const [xj, yj] = polygon[j] ?? [0, 0];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export function distanceToSegment(px: number, py: number, a: Point, b: Point): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lengthSq = dx * dx + dy * dy;
  const t = lengthSq === 0 ? 0 : Math.max(0, Math.min(1, ((px - a[0]) * dx + (py - a[1]) * dy) / lengthSq));
  return Math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy));
}

/** Signed distance to the polygon boundary: positive inside, negative outside. */
export function insideDistance(x: number, y: number, polygon: readonly Point[]): number {
  let best = Infinity;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    best = Math.min(best, distanceToSegment(x, y, polygon[j] ?? [0, 0], polygon[i] ?? [0, 0]));
  }
  return pointInPolygon(x, y, polygon) ? best : -best;
}

export interface MaskSpec {
  width: number;
  height: number;
  /** Oval polygon in mask pixel coordinates. */
  polygon: readonly Point[];
  /** Feather width in mask pixels. */
  feather: number;
  /**
   * Optional forehead fade along an arbitrary direction (the head UV island is rotated, so "up" is not +y):
   * t = dot(p - origin, axis) (axis is a unit vector pointing toward the top of the head); alpha is 0 at
   * t >= topT and 1 at t <= browT.
   */
  forehead?: { origin: Point; axis: Point; topT: number; browT: number };
  /** Regions that must stay unpainted (eye openings, mouth slit): alpha 0 inside, grown by `grow` px, then feathered. */
  holes?: { polygon: readonly Point[]; grow: number; feather: number }[];
}

/** Orders points by angle around their centroid (for convex-ish contours whose index order is unknown). */
export function sortByAngle(points: readonly Point[]): Point[] {
  if (points.length === 0) return [];
  const cx = points.reduce((sum, p) => sum + p[0], 0) / points.length;
  const cy = points.reduce((sum, p) => sum + p[1], 0) / points.length;
  return [...points].sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
}

export interface MaskResult {
  alpha: Float32Array;
  /** Signed inside distance per pixel (px), used for border colour matching. */
  inside: Float32Array;
}

export function buildMask(spec: MaskSpec): MaskResult {
  const { width, height, polygon, feather, forehead, holes } = spec;
  const alpha = new Float32Array(width * height);
  const inside = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const d = insideDistance(x + 0.5, y + 0.5, polygon);
      let a = featherAlpha(d, feather);
      if (forehead && a > 0) {
        const t = (x + 0.5 - forehead.origin[0]) * forehead.axis[0] + (y + 0.5 - forehead.origin[1]) * forehead.axis[1];
        a *= foreheadFade(t, forehead.topT, forehead.browT);
      }
      if (holes && a > 0) {
        for (const hole of holes) {
          a *= featherAlpha(-insideDistance(x + 0.5, y + 0.5, hole.polygon) - hole.grow, hole.feather);
          if (a <= 0) break;
        }
      }
      alpha[y * width + x] = a;
      inside[y * width + x] = d;
    }
  }
  return { alpha, inside };
}
