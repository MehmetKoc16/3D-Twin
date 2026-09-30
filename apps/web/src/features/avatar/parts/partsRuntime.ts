import { Mesh, MeshStandardMaterial, type BufferAttribute, type Object3D, type Texture } from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { garmentSkinWeights, parseGarmentBinding, type BodyPartDef, type GarmentBinding } from '@dt/avatar-core';
import { buildWeldMap, type WeldMap } from '../meshMath';
import { parseDeleteVerts } from './bodyMask';
import { meanCoveredLinear } from './irisRecolor';

/** Everything loaded once per part: geometry index, UVs, binding, skin weights, glb material. */
export interface PartRuntime {
  def: BodyPartDef;
  vertexCount: number;
  index: Uint32Array;
  uv: Float32Array;
  binding: GarmentBinding;
  skinIndices: Uint16Array;
  skinWeights: Float32Array;
  weld: WeldMap;
  deleteVerts: Uint32Array;
  /** Material of the glb (texture, alpha mode, sidedness); instances clone it. */
  material: MeshStandardMaterial;
  /** Tintable hair / brow textures: mean linear texel value, the divisor that makes a picked colour read true. */
  textureMean: number;
  /** Eyes: the texture's pixels (RGBA, top row first) and size, so the iris can be recoloured from the original. */
  texturePixels?: { data: Uint8ClampedArray; width: number; height: number };
}

/** Skin data of the avatar body the parts are skinned like. */
export interface BodySkinData {
  renderVertexCount: number;
  skinIndex: ArrayLike<number>;
  skinWeight: ArrayLike<number>;
}

async function fetchBuffer(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  return response.arrayBuffer();
}

function firstMesh(root: Object3D): Mesh {
  let found: Mesh | null = null;
  root.traverse((o) => {
    if (!found && o instanceof Mesh) found = o;
  });
  if (!found) throw new Error('part glb contains no mesh');
  return found;
}

/** Draws the texture image on a canvas (at most `maxSide` px) and returns its pixels, or undefined without a DOM. */
function readTexture(texture: Texture | null, maxSide: number): { data: Uint8ClampedArray; width: number; height: number } | undefined {
  const image = texture?.image as (CanvasImageSource & { width: number; height: number }) | undefined;
  if (!image || typeof document === 'undefined') return undefined;
  const scale = Math.min(1, maxSide / Math.max(image.width, image.height));
  const width = Math.max(1, Math.round(image.width * scale));
  const height = Math.max(1, Math.round(image.height * scale));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) return undefined;
  context.drawImage(image, 0, 0, width, height);
  return { data: context.getImageData(0, 0, width, height).data, width, height };
}

export async function loadPartRuntime(def: BodyPartDef, body: BodySkinData, baseUrl: string): Promise<PartRuntime> {
  const [gltf, bindingBuffer, deleteBuffer] = await Promise.all([
    new GLTFLoader().loadAsync(`${baseUrl}${def.mesh}`),
    fetchBuffer(`${baseUrl}${def.binding}`),
    def.deleteVerts ? fetchBuffer(`${baseUrl}${def.deleteVerts}`) : Promise.resolve(null),
  ]);
  const mesh = firstMesh(gltf.scene);
  const position = mesh.geometry.getAttribute('position') as BufferAttribute;
  const uv = mesh.geometry.getAttribute('uv') as BufferAttribute | undefined;
  const vertexCount = position.count;
  const binding = parseGarmentBinding(bindingBuffer, body.renderVertexCount);
  if (binding.count !== vertexCount)
    throw new Error(`part ${def.id}: binding has ${binding.count} vertices, the mesh ${vertexCount}`);
  const indexAttribute = mesh.geometry.getIndex();
  const index = indexAttribute
    ? Uint32Array.from(indexAttribute.array as ArrayLike<number>)
    : Uint32Array.from({ length: vertexCount }, (_, i) => i);
  const { skinIndices, skinWeights } = garmentSkinWeights(binding, body.skinIndex, body.skinWeight);
  const source = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
  const material = source instanceof MeshStandardMaterial ? source : new MeshStandardMaterial({ roughness: 0.8 });

  let textureMean = 1;
  let texturePixels: PartRuntime['texturePixels'];
  if (def.category === 'eyes') {
    texturePixels = readTexture(material.map, 1024);
  } else if (def.material.tintable) {
    const pixels = readTexture(material.map, 128);
    if (pixels) textureMean = meanCoveredLinear(pixels.data, def.material.alphaCutoff ?? 0);
  }
  return {
    def,
    vertexCount,
    index,
    uv: uv ? Float32Array.from(uv.array as ArrayLike<number>) : new Float32Array(vertexCount * 2),
    binding,
    skinIndices,
    skinWeights,
    weld: buildWeldMap(position.array as ArrayLike<number>),
    deleteVerts: deleteBuffer ? parseDeleteVerts(deleteBuffer) : new Uint32Array(0),
    material,
    textureMean,
    texturePixels,
  };
}
