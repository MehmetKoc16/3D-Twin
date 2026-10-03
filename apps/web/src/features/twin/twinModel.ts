import {
  DoubleSide,
  MeshStandardMaterial,
  SkinnedMesh,
  SRGBColorSpace,
  type BufferAttribute,
  type InterleavedBufferAttribute,
  type Material,
  type Object3D,
  type Texture,
  type Group,
} from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import {
  headsFromBoneInverses,
  matchBones,
  normalizeSkinWeights,
  remapSkinIndices,
} from './twinBinding';
import { TwinFormatError, type TwinDef } from './twinDef';
import { loadTwinAccessories, disposeAccessory } from './twinAccessories';

/**
 * The parsed `rigged.glb` of a twin, re-indexed to the avatar's bone order. Everything is copied out of the glTF, so
 * the loader's scene can be dropped; only the material (and its texture) is kept and disposed by `dispose`.
 */
export interface TwinModel {
  vertexCount: number;
  /** Rest positions in the twin's own frame (feet on y = 0), 3 floats per vertex. */
  position: Float32Array;
  normal: Float32Array;
  uv: Float32Array | null;
  index: Uint32Array;
  /** Skin joints in the AVATAR's bone order (remapped) and the renormalised weights. */
  skinIndex: Uint16Array;
  skinWeight: Float32Array;
  /** Rest heads of the twin's bones, 3 floats per bone, in the twin's joint order. */
  heads: Float32Array;
  /** Twin bone index -> avatar bone index. */
  remap: Int32Array;
  material: MeshStandardMaterial;
  accessories?: Group[];
  dispose(): void;
}

function skinnedMeshes(root: Object3D): SkinnedMesh[] {
  const found: SkinnedMesh[] = [];
  root.traverse((o) => {
    if (o instanceof SkinnedMesh) found.push(o);
  });
  return found;
}

/** Copies an attribute into a tight Float32Array (safe for interleaved and normalised accessors). */
function copyAttribute(attribute: BufferAttribute | InterleavedBufferAttribute): Float32Array {
  const size = attribute.itemSize;
  const out = new Float32Array(attribute.count * size);
  for (let i = 0; i < attribute.count; i++) {
    out[i * size] = attribute.getX(i);
    if (size > 1) out[i * size + 1] = attribute.getY(i);
    if (size > 2) out[i * size + 2] = attribute.getZ(i);
    if (size > 3) out[i * size + 3] = attribute.getW(i);
  }
  return out;
}

/** glTF `alphaMode: BLEND` is rendered as MASK with this cutoff (the glTF default `alphaCutoff`). */
export const BLEND_AS_MASK_CUTOFF = 0.5;

function firstMaterial(material: Material | Material[]): Material | undefined {
  return Array.isArray(material) ? material[0] : material;
}

/**
 * The twin's baked colour: standard shading with a neutral, matte response (no skin composite, no sheen). The
 * texture is the pipeline's delit albedo, so the app's lights shade it like everything else.
 *
 * Alpha: GLTFLoader turns `alphaMode: MASK` into `alphaTest = alphaCutoff` (default 0.5) and `BLEND` into
 * `transparent = true`. A cut-out material (hair cards of the template-character twin) keeps that `alphaTest`, reading
 * the base-colour texture's alpha channel, and `doubleSided` keeps both faces. `BLEND` is treated as MASK at 0.5:
 * hair cards need no sorted translucency, and a transparent draw would need per-frame sorting against the garments and
 * hands and would break depth writes. The shadow pass needs no custom depth material: three's shadow renderer already
 * applies `alphaTest` with the material's *current* `map` (also after TwinOpeningRepair swaps it), so cut-out hair
 * casts a cut-out shadow instead of a block. Opaque materials are untouched.
 */
export function twinMaterial(source: Material | undefined): MeshStandardMaterial {
  const map: Texture | null = source instanceof MeshStandardMaterial ? source.map : null;
  if (map) map.colorSpace = SRGBColorSpace;
  const material = new MeshStandardMaterial({ map, roughness: 0.88, metalness: 0 });
  if (!map && source instanceof MeshStandardMaterial) material.color.copy(source.color);
  material.name = 'twin';
  if (source instanceof MeshStandardMaterial) {
    if (source.alphaTest > 0) material.alphaTest = source.alphaTest;
    else if (source.transparent) material.alphaTest = BLEND_AS_MASK_CUTOFF;
    if (source.side === DoubleSide) material.side = DoubleSide;
  }
  return material;
}

/**
 * Parses a twin `rigged.glb` and checks it against the avatar's skeleton and `twin.json`.
 * Throws `TwinFormatError` (glb / noSkin / rest / bones / mismatch).
 */
export async function loadTwinModel(
  buffer: ArrayBuffer,
  def: TwinDef,
  appBoneNames: readonly string[],
): Promise<TwinModel> {
  let root: Object3D;
  let accessories: Group[];
  try {
    const loaded = await new GLTFLoader().parseAsync(buffer.slice(0), '');
    root = loaded.scene;
    accessories = await loadTwinAccessories(loaded.parser);
  } catch (error) {
    throw new TwinFormatError(
      'glb',
      error instanceof Error ? error.message : 'rigged.glb could not be parsed',
    );
  }
  const meshes = skinnedMeshes(root);
  if (meshes.length !== 1)
    throw new TwinFormatError(
      'noSkin',
      `rigged.glb must hold exactly one skinned mesh (found ${meshes.length})`,
    );
  const mesh = meshes[0]!;
  const geometry = mesh.geometry;
  const index = geometry.getIndex();
  const position = geometry.getAttribute('position');
  const skinIndex = geometry.getAttribute('skinIndex');
  const skinWeight = geometry.getAttribute('skinWeight');
  if (!index || !skinIndex || !skinWeight)
    throw new TwinFormatError('noSkin', 'the twin mesh needs an index and skin weights');
  const remap = matchBones(
    mesh.skeleton.bones.map((b) => b.name),
    appBoneNames,
    def.boneOrder,
  );
  const heads = headsFromBoneInverses(mesh.skeleton.boneInverses);
  if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
  const normal = geometry.getAttribute('normal');
  const uv = geometry.getAttribute('uv');
  const weights = copyAttribute(skinWeight);
  normalizeSkinWeights(weights);
  const material = twinMaterial(firstMaterial(mesh.material));
  return {
    vertexCount: position.count,
    position: copyAttribute(position),
    normal: copyAttribute(normal),
    uv: uv ? copyAttribute(uv) : null,
    index: Uint32Array.from(index.array as ArrayLike<number>),
    skinIndex: remapSkinIndices(copyAttribute(skinIndex), remap),
    skinWeight: weights,
    heads,
    remap,
    material,
    accessories,
    dispose: () => {
      material.map?.dispose();
      material.dispose();
      geometry.dispose();
      accessories.forEach(disposeAccessory);
    },
  };
}
