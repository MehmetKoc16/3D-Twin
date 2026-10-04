import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import {
  BufferAttribute,
  BufferGeometry,
  Matrix4,
  MeshBasicMaterial,
  DoubleSide,
  MeshDepthMaterial,
  MeshStandardMaterial,
  SRGBColorSpace,
  MeshPhysicalMaterial,
  NoColorSpace,
  Quaternion,
  SkinnedMesh,
  Vector3,
} from 'three';
import { parseTwinBundle } from './twinBundle';
import { bundleExtras } from './twinBundleTestkit';
import { parseTwinDef, TwinFormatError } from './twinDef';
import { fakeAssets, fakeModel, envelope, mapping, bodyIndex } from './twinTestkit';
import { hairBoneRemap, parseTwinHair, rendererHasMsaa } from './twinHair';
import {
  editGlbJson,
  fakeHairModel,
  fakeShellModel,
  hairExtras,
  hairGlb,
  hairJson,
  shellExtras,
} from './twinHairTestkit';
import { loadTwinModel } from './twinModel';
import { TwinRig } from './twinRig';
import { filterBodyIndex, hiddenVertexMask } from '../wardrobe/bodyHide';

afterEach(() => vi.restoreAllMocks());

describe('parseTwinHair (asset.extras.dtHairNode + material extras.dtHair)', () => {
  it('is null for bundles without dtHairNode: they behave exactly as before', () => {
    expect(parseTwinHair({})).toBeNull();
    expect(parseTwinHair({ asset: { extras: {} } })).toBeNull();
  });

  it('reads the node name, colours and card count', () => {
    expect(parseTwinHair(hairJson())).toEqual({
      nodeName: 'dtHair',
      format: 'rcov-groot-bvar/1',
      colorHex: '#2a1e18',
      rootHex: '#1f1611',
      tipHex: '#3a2a20',
      cardCount: 1,
    });
  });

  it('treats root, tip and card count as optional', () => {
    const info = parseTwinHair(
      hairJson({ extras: { format: hairExtras.format, colorHex: '#102030' } }),
    );
    expect(info).toMatchObject({
      colorHex: '#102030',
      rootHex: null,
      tipHex: null,
      cardCount: null,
    });
  });

  it.each([
    ['non-string node', { node: 7 }],
    ['empty node', { node: '' }],
    ['missing extras', { extras: undefined }],
    ['extras not an object', { extras: 'x' }],
    ['unknown format', { extras: { ...hairExtras, format: 'other/2' } }],
    ['bad colorHex', { extras: { ...hairExtras, colorHex: 'brown' } }],
    ['short colorHex', { extras: { ...hairExtras, colorHex: '#abc' } }],
    ['bad rootHex', { extras: { ...hairExtras, rootHex: 12 } }],
    ['bad tipHex', { extras: { ...hairExtras, tipHex: '#12345g' } }],
    ['negative cardCount', { extras: { ...hairExtras, cardCount: -1 } }],
    ['fractional cardCount', { extras: { ...hairExtras, cardCount: 1.5 } }],
  ])('rejects %s with TwinFormatError', (_name, overrides) => {
    expect(() => parseTwinHair(hairJson(overrides))).toThrow(TwinFormatError);
  });

  it('rejects a dtHairNode that is not a node of the glTF', () => {
    expect(() => parseTwinHair(hairJson({ node: 'ponytail' }))).toThrow(/not a node/);
  });
});

describe('parseTwinBundle with hair', () => {
  const parser = (hair?: ReturnType<typeof hairJson>) => ({
    json: {
      asset: { extras: { dtTwin: bundleExtras(), ...(hair ? { dtHairNode: 'dtHair' } : {}) } },
      bufferViews: [{}],
      ...(hair ? { nodes: hair.nodes, meshes: hair.meshes, materials: hair.materials } : {}),
    },
    getDependency: vi.fn(async () => Uint32Array.from([1, 3]).buffer),
  });

  it('exposes the hair info and stays null without it', async () => {
    expect((await parseTwinBundle(parser()))?.hair).toBeNull();
    expect((await parseTwinBundle(parser(hairJson())))?.hair?.colorHex).toBe('#2a1e18');
  });

  it('rejects malformed hair extras like the other extras', async () => {
    const bad = parser(hairJson({ extras: { ...hairExtras, colorHex: 'nope' } }));
    await expect(parseTwinBundle(bad)).rejects.toThrow(/colorHex/);
  });
});

describe('loadTwinModel with a dtHair node', () => {
  const def = parseTwinDef(bundleExtras().twin);
  const names = ['Root'];

  // The 1x1 atlas PNG needs an image decoder; node has none, so stand in for createImageBitmap (blob: fetch is native).
  beforeAll(() => {
    vi.stubGlobal('self', globalThis);
    vi.stubGlobal('createImageBitmap', async () => ({
      width: 1,
      height: 1,
      close: () => undefined,
    }));
  });

  afterAll(() => vi.unstubAllGlobals());

  it('splits the hair from the body, remaps its skin and loads the atlas as data', async () => {
    const model = await loadTwinModel(hairGlb(), def, names);
    expect(model.vertexCount).toBe(3);
    expect(model.hair).toBeDefined();
    const hair = model.hair!;
    expect(hair.info.nodeName).toBe('dtHair');
    expect(hair.vertexCount).toBe(3);
    expect(hair.position[1]).toBeCloseTo(1.5);
    expect(hair.uv).toHaveLength(6);
    expect(Array.from(hair.skinIndex.slice(0, 4))).toEqual([0, 0, 0, 0]);
    expect(hair.atlas.colorSpace).toBe(NoColorSpace);
    // the body material is its own: no atlas, no cut-out
    expect(model.material.map).toBeNull();
    expect(model.material.alphaTest).toBe(0);
    model.dispose();
  });

  it('keeps a bundle without dtHairNode exactly as before', async () => {
    const model = await loadTwinModel(hairGlb({ hair: false }), def, names);
    expect(model.hair).toBeUndefined();
    model.dispose();
  });

  it('still insists on one body mesh: a second skinned mesh without dtHairNode is an error', async () => {
    const plain = editGlbJson(hairGlb(), (json) => {
      delete (json.asset as { extras?: unknown }).extras;
    });
    await expect(loadTwinModel(plain, def, names)).rejects.toThrow(/exactly one skinned mesh/);
  });

  it('rejects hair extras that are malformed, with TwinFormatError', async () => {
    await expect(
      loadTwinModel(hairGlb({ extras: { format: 'x' } }), def, names),
    ).rejects.toBeInstanceOf(TwinFormatError);
  });

  it('rejects a dtHairNode that names no node', async () => {
    await expect(loadTwinModel(hairGlb({ node: 'ponytail' }), def, names)).rejects.toThrow(
      /not a node/,
    );
  });
});

describe('hairBoneRemap', () => {
  it('maps by name and refuses a hair skin with another rest pose than the body', () => {
    const assets = fakeAssets();
    const hair = fakeAssets().skeleton;
    const appNames = assets.skeleton.bones.map((b) => b.name);
    expect(Array.from(hairBoneRemap(hair, assets.skeleton, appNames))).toEqual([0, 1, 2]);
    const elements = hair.boneInverses[1]!.elements;
    elements[13] = (elements[13] ?? 0) + 0.05;
    expect(() => hairBoneRemap(hair, assets.skeleton, appNames)).toThrow(/another rest pose/);
    elements[13] = (elements[13] ?? 0) - 0.05;
    hair.bones[2]!.name = 'tail';
    expect(() => hairBoneRemap(hair, assets.skeleton, appNames)).toThrow(/does not exist/);
  });
});

describe('rendererHasMsaa', () => {
  const renderer = (samples: unknown) => ({
    getContext: () => ({ SAMPLES: 0x80a9, getParameter: () => samples }),
  });
  it('reads gl.SAMPLES like the shader does, and falls back to no MSAA', () => {
    expect(rendererHasMsaa(renderer(4))).toBe(true);
    expect(rendererHasMsaa(renderer(0))).toBe(false);
    expect(rendererHasMsaa(null)).toBe(false);
    expect(rendererHasMsaa({})).toBe(false);
    expect(
      rendererHasMsaa({
        getContext: () => {
          throw new Error('lost');
        },
      }),
    ).toBe(false);
  });
});

function rigWithHair(options: { hairMsaa?: boolean } = {}) {
  const assets = fakeAssets();
  const model = { ...fakeModel(), hair: fakeHairModel() };
  const rig = new TwinRig(assets, model, mapping, undefined, '#c99a7e', options);
  return { assets, model, rig, hair: model.hair };
}

describe('TwinRig hair', () => {
  it('applies no hair and adds nothing to the scene without a dtHair model', () => {
    const assets = fakeAssets();
    const rig = new TwinRig(assets, fakeModel(), mapping);
    expect(rig.hair).toBeNull();
    expect(assets.scene.children.some((c) => c.name.startsWith('twin:hair'))).toBe(false);
    rig.dispose();
  });

  it('binds every hair mesh to the avatar skeleton, hidden until the first aligned solve', () => {
    const { assets, rig } = rigWithHair({ hairMsaa: true });
    const hair = rig.hair!;
    expect(hair.group.parent).toBe(assets.scene);
    expect(hair.group.visible).toBe(false);
    for (const mesh of hair.meshes) {
      expect(mesh).toBeInstanceOf(SkinnedMesh);
      expect(mesh.skeleton).toBe(assets.skeleton);
    }
    rig.onSolve(envelope(true));
    expect(hair.group.visible).toBe(true);
    rig.dispose();
  });

  it('follows the pose through the shared skeleton', () => {
    const { assets, rig } = rigWithHair({ hairMsaa: true });
    rig.onSolve(envelope(true));
    const probe = new Vector3();
    const at = () => {
      assets.scene.updateMatrixWorld(true);
      assets.skeleton.update();
      rig.hair!.mesh.getVertexPosition(0, probe);
      return probe.clone();
    };
    const rest = at();
    expect(rest.y).toBeCloseTo(1.5, 5);
    assets.skeleton.bones[2]!.quaternion.copy(
      new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), Math.PI / 2),
    );
    const posed = at();
    expect(posed.distanceTo(rest)).toBeGreaterThan(0.1);
    rig.dispose();
  });

  it('translates the hair with the body onto the avatar rest heads', () => {
    const assets = fakeAssets();
    const model = { ...fakeModel(0.013), hair: fakeHairModel() };
    const rig = new TwinRig(assets, model, mapping);
    rig.onSolve(envelope(true));
    expect(rig.hair!.mesh.geometry.getAttribute('position').getY(0)).toBeCloseTo(1.5 + 0.013, 5);
    rig.dispose();
  });

  describe('with MSAA (alpha-to-coverage)', () => {
    it('shades the hair as an opaque coverage pass plus a blended fringe', () => {
      const { hair, rig } = rigWithHair({ hairMsaa: true });
      const shader = rig.hair!.shader!;
      expect(rig.hair!.msaa).toBe(true);
      expect(shader.outer).toHaveLength(1);
      expect(shader.inner).toHaveLength(0);
      expect(shader.blend).toHaveLength(1);
      const outer = shader.outer[0]!;
      expect(outer).toBeInstanceOf(MeshPhysicalMaterial);
      expect(outer.alphaToCoverage).toBe(true);
      expect(outer.transparent).toBe(false);
      expect(outer.alphaMap).toBe(hair.atlas);
      expect(outer.map).toBeNull();
      expect(hair.atlas.colorSpace).toBe(NoColorSpace);
      expect(shader.blend[0]!.transparent).toBe(true);
      expect(rig.hair!.mesh.material).toBe(outer);
      expect(rig.hair!.meshes.map((m) => m.name)).toEqual(['twin:hair', 'twin:hair__blend']);
      rig.dispose();
    });

    it('uses the tip colour as the shader colour, the root colour as the root of a two-colour strand', () => {
      const { rig } = rigWithHair({ hairMsaa: true });
      const outer = rig.hair!.shader!.outer[0]!;
      expect(outer.color.getHexString()).toBe('3a2a20');
      rig.dispose();
    });
  });

  describe('without MSAA (fallback)', () => {
    it('shades the hair as a blended, depth-writing outer pass over an alpha-tested core', () => {
      const { rig } = rigWithHair({ hairMsaa: false });
      const shader = rig.hair!.shader!;
      expect(rig.hair!.msaa).toBe(false);
      expect(shader.blend).toHaveLength(0);
      expect(shader.inner).toHaveLength(1);
      const outer = shader.outer[0]!;
      expect(outer.alphaToCoverage).toBe(false);
      expect(outer.transparent).toBe(true);
      expect(outer.depthWrite).toBe(true);
      expect(shader.inner[0]!.alphaTest).toBeGreaterThan(0);
      expect(rig.hair!.meshes.map((m) => m.name)).toEqual(['twin:hair', 'twin:hair__inner']);
      rig.dispose();
    });
  });

  it('detects the path from the renderer: MSAA samples > 0 selects alpha-to-coverage', () => {
    const gl = { SAMPLES: 0x80a9, getParameter: () => 4 };
    const rig = new TwinRig(
      fakeAssets(),
      { ...fakeModel(), hair: fakeHairModel() },
      mapping,
      undefined,
      '#c99a7e',
      { renderer: { getContext: () => gl } },
    );
    expect(rig.hair!.msaa).toBe(true);
    rig.dispose();
    const plain = new TwinRig(fakeAssets(), { ...fakeModel(), hair: fakeHairModel() }, mapping);
    expect(plain.hair!.msaa).toBe(false); // no renderer: the safe non-MSAA path
    plain.dispose();
  });

  it('casts a cut-out shadow from the main pass only, never a solid block', () => {
    for (const hairMsaa of [true, false]) {
      const { hair, rig } = rigWithHair({ hairMsaa });
      const [main, ...others] = rig.hair!.meshes;
      expect(main!.castShadow).toBe(true);
      const depth = main!.customDepthMaterial as MeshDepthMaterial;
      expect(depth).toBeInstanceOf(MeshDepthMaterial);
      expect(depth.alphaMap).toBe(hair.atlas);
      for (const other of others) {
        expect(other.castShadow).toBe(false);
        expect(other.customDepthMaterial).toBeUndefined();
      }
      // the shadow shader samples coverage (atlas R), not three's alphaMap channel (G = root-to-tip)
      const shader = { fragmentShader: '#include <alphamap_fragment>' };
      depth.onBeforeCompile(shader as never, undefined as never);
      expect(shader.fragmentShader).toContain('alphaMap, vAlphaMapUv ).r <');
      expect(shader.fragmentShader).toContain('discard');
      rig.dispose();
    }
  });

  it('is never hidden by garments: hide-triangles and the footprint pass only touch the twin body', () => {
    const { assets, rig } = rigWithHair({ hairMsaa: true });
    rig.onSolve(envelope(true));
    const hairIndexCount = rig.hair!.mesh.geometry.getIndex()!.count;
    // the wardrobe deletes body triangles ...
    const mask = hiddenVertexMask([Uint32Array.from([0, 1, 2, 3, 4, 5, 6, 7])], 10);
    assets.mesh.geometry.setIndex(new BufferAttribute(filterBodyIndex(bodyIndex, mask), 1));
    // ... and garment surfaces cover every twin vertex, the hair card included (one per plane)
    for (const [i, z] of [0.004, 0.104].entries()) {
      const garment = new BufferGeometry();
      garment.setAttribute(
        'position',
        new BufferAttribute(Float32Array.from([-1, 0.5, z, 3, 0.5, z, 1, 2.5, z]), 3),
      );
      garment.setIndex(new BufferAttribute(Uint32Array.from([0, 1, 2]), 1));
      const surface = new SkinnedMesh(garment, new MeshBasicMaterial());
      surface.name = `garment:test${i}`;
      assets.scene.add(surface);
    }
    assets.mesh.geometry.setIndex(new BufferAttribute(bodyIndex, 1));
    expect(rig.mesh.geometry.drawRange.count).toBe(0); // the scan body is hidden under the garment ...
    expect(rig.hair!.group.visible).toBe(true); // ... the hair is not
    for (const mesh of rig.hair!.meshes) {
      expect(mesh.visible).toBe(true);
      expect(mesh.geometry.drawRange.count).toBe(Infinity);
      expect(mesh.geometry.getIndex()!.count).toBe(hairIndexCount);
    }
    // the coverage passes did not push the hair in or otherwise modify it
    expect(rig.hair!.mesh.geometry.getAttribute('position').getY(0)).toBeCloseTo(1.5, 6);
    rig.dispose();
  });

  it('disposes the shader passes, materials, depth material, geometry and atlas, and leaves the scene clean', () => {
    for (const hairMsaa of [true, false]) {
      const { assets, hair, rig } = rigWithHair({ hairMsaa });
      const h = rig.hair!;
      const materials = h.shader!.materials.slice();
      const depth = h.mesh.customDepthMaterial as MeshDepthMaterial;
      const geometry = h.mesh.geometry;
      const disposed = materials.map((m) => vi.spyOn(m, 'dispose'));
      const depthDispose = vi.spyOn(depth, 'dispose');
      const geometryDispose = vi.spyOn(geometry, 'dispose');
      const atlasDispose = vi.fn();
      hair.atlas.addEventListener('dispose', atlasDispose);
      expect(materials.length).toBeGreaterThan(1);
      rig.dispose();
      for (const spy of disposed) expect(spy).toHaveBeenCalled();
      expect(depthDispose).toHaveBeenCalled();
      expect(geometryDispose).toHaveBeenCalled();
      expect(atlasDispose).toHaveBeenCalled();
      expect(hair.disposed.count).toBeGreaterThanOrEqual(1);
      expect(h.group.parent).toBeNull();
      expect(h.group.children).toHaveLength(1); // the pass clones were removed, only the main mesh is left
      expect(assets.scene.children.some((c) => c.name.startsWith('twin:hair'))).toBe(false);
      expect(() => h.dispose()).not.toThrow(); // idempotent
    }
  });

  it('uses the avatar bind matrix for the hair too', () => {
    const { assets, rig } = rigWithHair({ hairMsaa: false });
    assets.mesh.bindMatrix.makeTranslation(0, 0.25, 0);
    rig.onSolve(envelope(true));
    for (const mesh of rig.hair!.meshes)
      expect(mesh.bindMatrix.equals(new Matrix4().makeTranslation(0, 0.25, 0))).toBe(true);
    rig.dispose();
  });
});

describe('shell hair (format shell/1)', () => {
  const shellJson = (extras: unknown) => hairJson({ extras });

  it('parses both formats and exposes the format; cardCount, rootHex and tipHex are optional for shell', () => {
    expect(parseTwinHair(hairJson())?.format).toBe('rcov-groot-bvar/1');
    expect(parseTwinHair(shellJson(shellExtras))).toMatchObject({
      format: 'shell/1',
      rootHex: null,
      tipHex: null,
      cardCount: null,
    });
    expect(parseTwinHair(shellJson({ ...shellExtras, cardCount: 0 }))?.cardCount).toBe(0);
  });

  it('rejects unknown formats', () => {
    expect(() => parseTwinHair(shellJson({ ...shellExtras, format: 'shell/2' }))).toThrow(
      TwinFormatError,
    );
  });

  describe('loading', () => {
    const def = parseTwinDef(bundleExtras().twin);
    beforeAll(() => {
      vi.stubGlobal('self', globalThis);
      vi.stubGlobal('createImageBitmap', async () => ({
        width: 1,
        height: 1,
        close: () => undefined,
      }));
    });
    afterAll(() => vi.unstubAllGlobals());

    it('keeps the base colour sRGB and honours MASK / doubleSided', async () => {
      const model = await loadTwinModel(hairGlb({ extras: shellExtras }), def, ['Root']);
      const hair = model.hair!;
      expect(hair.info.format).toBe('shell/1');
      expect(hair.atlas.colorSpace).toBe(SRGBColorSpace);
      const material = hair.shellMaterial!;
      expect(material.map).toBe(hair.atlas);
      expect(material.map!.colorSpace).toBe(SRGBColorSpace);
      expect(material.alphaTest).toBe(0.5);
      expect(material.side).toBe(DoubleSide);
      model.dispose();
    });

    it('strand format keeps NoColorSpace and has no shell material', async () => {
      const model = await loadTwinModel(hairGlb(), def, ['Root']);
      expect(model.hair!.atlas.colorSpace).toBe(NoColorSpace);
      expect(model.hair!.shellMaterial).toBeUndefined();
      model.dispose();
    });
  });

  const rigWithShell = (mask = true) => {
    const assets = fakeAssets();
    const hair = fakeShellModel(mask);
    const rig = new TwinRig(assets, { ...fakeModel(), hair }, mapping);
    return { assets, hair, rig };
  };

  it('renders the glTF material (normal map kept) without the strand shader', () => {
    const { hair, rig } = rigWithShell();
    const h = rig.hair!;
    expect(h.shader).toBeNull();
    expect(h.format).toBe('shell/1');
    expect(h.meshes).toHaveLength(1);
    expect(h.mesh.material).toBe(hair.shellMaterial);
    expect((h.mesh.material as MeshStandardMaterial).normalMap).toBe(hair.normalMap);
    expect((h.mesh.material as MeshStandardMaterial).map!.colorSpace).toBe(SRGBColorSpace);
    rig.dispose();
  });

  it('binds, hides until the first solve and follows alignment like strand hair', () => {
    const { assets, rig } = rigWithShell();
    expect(rig.hair!.group.visible).toBe(false);
    expect(rig.hair!.mesh.skeleton).toBe(assets.skeleton);
    rig.onSolve(envelope(true));
    expect(rig.hair!.group.visible).toBe(true);
    rig.dispose();
  });

  it('casts a normal shadow, cut out on the base colour alpha when MASK', () => {
    const masked = rigWithShell(true);
    const depth = masked.rig.hair!.mesh.customDepthMaterial as MeshDepthMaterial;
    expect(masked.rig.hair!.mesh.castShadow).toBe(true);
    expect(depth).toBeInstanceOf(MeshDepthMaterial);
    expect(depth.map).toBe(masked.hair.map);
    expect(depth.alphaTest).toBe(0.5);
    expect(depth.alphaMap).toBeNull(); // not the strand-atlas R cut-out
    masked.rig.dispose();
    const opaque = rigWithShell(false);
    expect(opaque.rig.hair!.mesh.castShadow).toBe(true);
    expect(opaque.rig.hair!.mesh.customDepthMaterial).toBeUndefined();
    opaque.rig.dispose();
  });

  it('is never hidden by garments', () => {
    const { assets, rig } = rigWithShell();
    rig.onSolve(envelope(true));
    const garment = new BufferGeometry();
    garment.setAttribute(
      'position',
      new BufferAttribute(Float32Array.from([-1, 0.5, 0.104, 3, 0.5, 0.104, 1, 2.5, 0.104]), 3),
    );
    garment.setIndex(new BufferAttribute(Uint32Array.from([0, 1, 2]), 1));
    const surface = new SkinnedMesh(garment, new MeshBasicMaterial());
    surface.name = 'garment:test';
    assets.scene.add(surface);
    assets.mesh.geometry.setIndex(new BufferAttribute(bodyIndex, 1));
    expect(rig.hair!.group.visible).toBe(true);
    expect(rig.hair!.mesh.visible).toBe(true);
    expect(rig.hair!.mesh.geometry.drawRange.count).toBe(Infinity);
    rig.dispose();
  });

  it('disposes material, normal map, base colour map, depth material and geometry; idempotent', () => {
    const { assets, hair, rig } = rigWithShell();
    const h = rig.hair!;
    const depth = h.mesh.customDepthMaterial as MeshDepthMaterial;
    const spies = [
      vi.spyOn(hair.shellMaterial!, 'dispose'),
      vi.spyOn(depth, 'dispose'),
      vi.spyOn(h.mesh.geometry, 'dispose'),
      vi.spyOn(hair.map, 'dispose'),
      vi.spyOn(hair.normalMap, 'dispose'),
    ];
    rig.dispose();
    for (const spy of spies) expect(spy).toHaveBeenCalled();
    expect(h.group.parent).toBeNull();
    expect(assets.scene.children.some((c) => c.name.startsWith('twin:hair'))).toBe(false);
    expect(() => h.dispose()).not.toThrow();
  });
});
