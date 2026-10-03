import type * as ThreeNamespace from 'three';
import type { Color, Material, Mesh, MeshPhysicalMaterial, Object3D, Texture } from 'three';

/**
 * Typed surface of the vendored `hair-shader.js` (MIT, creategamecharacters.com). Only the parts the app uses are
 * declared; the JS file stays untouched. See README.md in this folder.
 */

/** What the shader asks of the renderer: the GL context (MSAA detection) or the WebGPU flag. */
export interface HairRendererLike {
  isWebGPURenderer?: boolean;
  getContext?(): unknown;
}

export interface HairShaderOptions {
  /** The three.js namespace (`import * as THREE from 'three'`). */
  THREE: typeof ThreeNamespace;
  renderer?: HairRendererLike;
  /** Strand data atlas: R coverage, G root-to-tip, B per-strand variation. Must be `NoColorSpace` data. */
  atlas: Texture;
  /** Hair colour (the tip colour in `multi` root mode). Any `THREE.Color` input, an sRGB hex string here. */
  color: string;
  /** `mono`: the root end is only darkened; `multi`: the root end takes `rootColor`. */
  rootMode?: 'mono' | 'multi';
  rootColor?: string;
  rootStrength?: number;
  /** Force alpha-to-coverage on or off; by default it is read from the renderer's MSAA sample count. */
  alphaToCoverage?: boolean;
  /** Coverage gain (default 2.5 with MSAA, 1.0 without). */
  density?: number;
  innerThreshold?: number;
  blendOpacity?: number;
  blendRoughness?: number;
  roughness?: number;
  seedVariation?: number;
  alphaChannel?: 'r' | 'g' | 'b' | 'a';
  /** Which meshes under the root receive the passes (default: every mesh not already a pass). */
  filter?: (mesh: Mesh) => boolean;
}

export interface HairShaderHandle {
  /** Every material created, outer passes first. */
  materials: MeshPhysicalMaterial[];
  outer: MeshPhysicalMaterial[];
  /** Alpha-tested core passes (without MSAA only). */
  inner: Material[];
  /** Blended fringe passes (with alpha-to-coverage only). */
  blend: Material[];
  setColor(color: string | number | Color): void;
  setDensity(value: number): void;
  setBlendOpacity(value: number): void;
  setMode(mode: 'a2c' | 'both'): void;
  /** Removes the added pass meshes and disposes the pass materials (not the atlas, not the geometry). */
  dispose(): void;
}

export function applyHairShader(root: Object3D, options: HairShaderOptions): HairShaderHandle;
