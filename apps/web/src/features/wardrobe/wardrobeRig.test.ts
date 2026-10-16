import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  Bone,
  BufferAttribute,
  BufferGeometry,
  Group,
  MeshPhysicalMaterial,
  MeshStandardMaterial,
  Skeleton,
  SkinnedMesh,
} from 'three';
import type { GarmentTemplateDef } from '@dt/avatar-core';
import type { AvatarAssets } from '../avatar/avatarAssets';
import { buildWeldMap } from '../avatar/meshMath';
import type { SolveEnvelope } from '../../workers/avatarClient';
import { useWardrobeStore } from '../../store/wardrobeStore';
import { GarmentInstance } from './garmentInstance';
import { loadTemplateRuntime, type TemplateRuntime } from './templateLoader';
import { WardrobeRig } from './wardrobeRig';

vi.mock('./templateLoader', () => ({ loadTemplateRuntime: vi.fn() }));

const body = Float32Array.from([
  0.2, 1, 0.1, 0.3, 1, 0.1, 0.2, 1.1, 0.1, 0.2, 1.5, 0.1, 0.3, 1.5, 0.1, 0.2, 1.6, 0.1,
]);
const indices = Uint32Array.from([0, 1, 2, 3, 4, 5]);
const definitions: GarmentTemplateDef[] = [
  { id: 'tee', category: 'top', kind: 'tshirt', layer: 2 },
  { id: 'coat', category: 'top', kind: 'longsleeve', layer: 3 },
  { id: 'trousers', category: 'bottom', kind: 'pants', layer: 1 },
  { id: 'shoe', category: 'shoes', kind: 'sneakers', layer: 1 },
].map(
  (def) =>
    ({
      ...def,
      label: { tr: def.id, en: def.id },
      mesh: 'test.glb',
      license: 'CC0-1.0',
      nativeMeasures: {},
      defaultEase: {},
      baseColor: '#888888',
    }) as GarmentTemplateDef,
);

function runtime(def: GarmentTemplateDef): TemplateRuntime {
  return {
    def,
    vertexCount: 3,
    index: Uint32Array.from([0, 1, 2]),
    uv: new Float32Array(6),
    neutralPositions: body.slice(0, 9),
    binding: {
      count: 3,
      indices: Uint32Array.from([0, 0, 0, 1, 1, 1, 2, 2, 2]),
      weights: Float32Array.from([1, 0, 0, 1, 0, 0, 1, 0, 0]),
      offsets: Float32Array.from([0, 0, 0.004, 0, 0, 0.004, 0, 0, 0.004]),
    },
    skinIndices: new Uint16Array(12),
    skinWeights: Float32Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
    weld: buildWeldMap(body.slice(0, 9)),
    deleteVerts: Uint32Array.from([0, 1, 2, 3, 4, 5]),
    rigidAttachments: null,
    material: new MeshStandardMaterial(),
  };
}

function assets(): AvatarAssets {
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(body.slice(), 3));
  geometry.setAttribute(
    'skinIndex',
    new BufferAttribute(
      Uint16Array.from([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0]),
      4,
    ),
  );
  geometry.setAttribute(
    'skinWeight',
    new BufferAttribute(
      Float32Array.from({ length: 24 }, (_, i) => (i % 4 === 0 ? 1 : 0)),
      4,
    ),
  );
  geometry.setIndex(new BufferAttribute(indices, 1));
  const bones = ['spine_01', 'neck_01', 'hand_l', 'head'].map((name) => {
    const bone = new Bone();
    bone.name = name;
    return bone;
  });
  const skeleton = new Skeleton(bones);
  const material = new MeshPhysicalMaterial();
  const mesh = new SkinnedMesh(geometry, material);
  mesh.bind(skeleton);
  const scene = new Group();
  scene.add(mesh, ...bones);
  return {
    scene,
    mesh,
    material,
    skeleton,
    indices,
    weld: buildWeldMap(body),
    parents: new Int32Array(4),
    rigIndexOfBone: Int32Array.from([0, 1, 2, 3]),
    normalScratch: new Float32Array(18),
    client: {} as AvatarAssets['client'],
    onSolve: () => () => undefined,
  };
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  useWardrobeStore.setState({ worn: {}, items: [], templates: [] });
});

describe('WardrobeRig outer slot', () => {
  it.each([
    { fixedShape: false, tucked: false },
    { fixedShape: true, tucked: false },
    { fixedShape: false, tucked: true },
    { fixedShape: true, tucked: true },
  ])(
    'fits and restores layers on the shared rig (fixed twin shape: $fixedShape, tucked: $tucked)',
    async ({ fixedShape, tucked }) => {
      const topId = tucked ? 'tshirt' : 'tee';
      const templates = definitions.map((def) => (def.id === 'tee' ? { ...def, id: topId } : def));
      const runtimes = new Map(templates.map((def) => [def.id, runtime(def)]));
      vi.mocked(loadTemplateRuntime).mockImplementation(async (def) => runtimes.get(def.id)!);
      vi.stubGlobal(
        'fetch',
        vi.fn(async () => ({ ok: true, json: async () => ({ version: 1, measures: [] }) })),
      );
      useWardrobeStore.setState({
        templates,
        catalogStatus: 'ready',
        hydrated: true,
        heatmap: false,
        items: templates.map((def) => ({
          id: def.id,
          name: def.id,
          templateId: def.id,
          color: def.baseColor,
          sizes: ['M'],
          selectedSize: 'M',
          chart: {},
        })),
        worn: { top: topId, outer: 'coat', bottom: 'trousers', shoes: 'shoe' },
      });
      const avatar = assets();
      const rig = new WardrobeRig(avatar);
      const envelope: SolveEnvelope = {
        params: { gender: 0.5, heightCm: 170, weightKg: 70, shoe: { system: 'EU', size: 42 } },
        roundTripMs: 0,
        result: {
          positions: body,
          joints: Float32Array.from([
            0, 1, 0, 0, 1.1, 0, 0, 1.4, 0, 0, 1.5, 0, 0.3, 1.2, 0, 0.3, 1.3, 0, 0, 1.5, 0, 0, 1.6,
            0,
          ]),
          groundOffsetY: 0,
          achievedCm: {},
          residualsCm: {},
          unreachable: [],
          estimatedMassKg: 70,
          solveMs: 0,
          ...(fixedShape ? { fixedShape: true as const } : {}),
        },
      };
      const fit = vi.spyOn(GarmentInstance.prototype, 'fit');
      const commit = vi.spyOn(GarmentInstance.prototype, 'commit');
      try {
        rig.onSolve(envelope);
        await vi.waitFor(() =>
          expect(
            avatar.scene.children.filter((child) => child.name.startsWith('garment:')),
          ).toHaveLength(4),
        );
        fit.mockClear();
        commit.mockClear();
        rig.onSolve(envelope);
        expect(
          fit.mock.contexts.map((instance) => {
            if (!(instance instanceof GarmentInstance))
              throw new Error('Unexpected garment fit context');
            return instance.template.id;
          }),
        ).toEqual(
          tucked ? ['shoe', topId, 'trousers', 'coat'] : ['shoe', 'trousers', topId, 'coat'],
        );
        expect(
          commit.mock.contexts.map((instance) => {
            if (!(instance instanceof GarmentInstance))
              throw new Error('Unexpected garment commit context');
            return instance.template.id;
          }),
        ).toEqual(['shoe', 'trousers', topId, 'coat']);
        // The outer fit must see the final tucked hem, after the top was committed into the trousers.
        expect(fit.mock.invocationCallOrder[3]).toBeGreaterThan(
          commit.mock.invocationCallOrder[2]!,
        );
        const topContext = fit.mock.calls[tucked ? 1 : 2]![0];
        const outerContext = fit.mock.calls[3]![0];
        expect(outerContext.lowers).toHaveLength(2);
        expect(outerContext.normalOffsetM).toBeGreaterThan(topContext.normalOffsetM!);
        // All four delete lists try to hide the protected hand / neck / head patch; the final guard keeps it.
        expect(Array.from(avatar.mesh.geometry.getIndex()!.array)).toEqual([3, 4, 5]);
        const tee = avatar.scene.getObjectByName(`garment:${topId}`) as SkinnedMesh;
        expect(tee.geometry.drawRange.count).toBe(0);
        useWardrobeStore.getState().takeOff('outer');
        expect(avatar.scene.getObjectByName('garment:coat')).toBeUndefined();
        expect(avatar.scene.getObjectByName(`garment:${topId}`)).toBe(tee);
        expect(tee.geometry.drawRange.count).toBe(tucked ? 0 : 3);
        useWardrobeStore.getState().wear('coat');
        await vi.waitFor(() => expect(avatar.scene.getObjectByName('garment:coat')).toBeDefined());
        useWardrobeStore.getState().takeOff('top');
        expect(avatar.scene.getObjectByName('garment:coat')).toBeDefined();
        expect(avatar.scene.getObjectByName(`garment:${topId}`)).toBeUndefined();
        const coatCalls = fit.mock.calls.filter((_, idx) => {
          const inst = fit.mock.contexts[idx];
          return inst instanceof GarmentInstance && inst.template.id === 'coat';
        });
        expect(coatCalls[coatCalls.length - 1]?.[0].normalOffsetM).toBe(0);
      } finally {
        rig.dispose();
        avatar.mesh.geometry.dispose();
        avatar.material.dispose();
        avatar.skeleton.dispose();
        for (const value of runtimes.values()) value.material.dispose();
      }
    },
  );

  it('reveals ankle skin (and UV-seam copies) below an 8 cm shortened trouser hem while hiding the calf and protecting hands/neck/head', async () => {
    // Why: When the size chart inseam is 8 cm shorter than the body inseam, length grading pulls the
    // trouser cuff up by 8 cm; the static delete list must not leave a missing-skin gap at the ankle.
    const legBody = Float32Array.from([
      0.08,
      0.1,
      0.1, // 0: ankle left (y = 0.10 m)
      0.12,
      0.1,
      0.1, // 1: ankle right (y = 0.10 m)
      0.1,
      0.14,
      0.1, // 2: upper ankle (y = 0.14 m, 4 cm below the 0.18 m shortened cuff)
      0.12,
      0.1,
      0.1, // 3: UV-seam copy of vertex 1 at the ankle
      0.08,
      0.25,
      0.1, // 4: calf left (y = 0.25 m, above the 0.18 m cuff)
      0.12,
      0.25,
      0.1, // 5: calf right (y = 0.25 m, above the 0.18 m cuff)
      0.1,
      0.8,
      0.1, // 6: crotch / upper leg (y = 0.80 m, inseam measure vertex)
      0.2,
      1.5,
      0.1, // 7: protected hand_l
      0.3,
      1.5,
      0.1, // 8: protected neck_01
      0.2,
      1.6,
      0.1, // 9: protected head
    ]);
    const legIndices = Uint32Array.from([0, 1, 2, 0, 3, 2, 4, 5, 6, 7, 8, 9]);
    const trouserDef: GarmentTemplateDef = {
      id: 'trousers',
      category: 'bottom',
      kind: 'pants',
      layer: 1,
      label: { tr: 'trousers', en: 'trousers' },
      mesh: 'test.glb',
      license: 'CC0-1.0',
      nativeMeasures: { inseam: 72 },
      defaultEase: { inseam: 0 },
      baseColor: '#888888',
    };
    const trouserPositions = Float32Array.from([
      0.08, 0.1, 0.104, 0.12, 0.1, 0.104, 0.12, 0.8, 0.104, 0.08, 0.8, 0.104,
    ]);
    const trouserRuntime: TemplateRuntime = {
      def: trouserDef,
      vertexCount: 4,
      index: Uint32Array.from([0, 1, 2, 0, 2, 3]),
      uv: new Float32Array(8),
      neutralPositions: trouserPositions,
      binding: {
        count: 4,
        indices: Uint32Array.from([0, 0, 0, 1, 1, 1, 5, 6, 6, 4, 6, 6]),
        weights: Float32Array.from([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]),
        offsets: Float32Array.from([0, 0, 0.004, 0, 0, 0.004, 0.02, 0, 0.004, -0.02, 0, 0.004]),
      },
      skinIndices: new Uint16Array(16),
      skinWeights: Float32Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
      weld: buildWeldMap(trouserPositions),
      // Includes ankle (0, 1, 2), calf (4, 5, 6), and protected (7, 8, 9); 3 is a seam copy of 1
      deleteVerts: Uint32Array.from([0, 1, 2, 4, 5, 6, 7, 8, 9]),
      rigidAttachments: null,
      material: new MeshStandardMaterial(),
    };
    vi.mocked(loadTemplateRuntime).mockResolvedValue(trouserRuntime);
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          version: 1,
          measures: [{ id: 'inseam', type: 'vertexHeight', vert: 6, drivers: [] }],
        }),
      })),
    );
    useWardrobeStore.setState({
      templates: [trouserDef],
      catalogStatus: 'ready',
      hydrated: true,
      heatmap: false,
      items: [
        {
          id: 'trousers',
          name: 'trousers',
          templateId: 'trousers',
          color: '#888888',
          sizes: ['Regular', 'Short'],
          selectedSize: 'Regular',
          chart: { inseam: [72, 64] },
        },
      ],
      worn: { bottom: 'trousers' },
    });

    const avatar = assets();
    avatar.mesh.geometry.setAttribute('position', new BufferAttribute(legBody.slice(), 3));
    avatar.mesh.geometry.setAttribute(
      'skinIndex',
      new BufferAttribute(
        Uint16Array.from([
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
          0, 0, 1, 0, 0, 0, 3, 0, 0, 0,
        ]),
        4,
      ),
    );
    avatar.mesh.geometry.setAttribute(
      'skinWeight',
      new BufferAttribute(
        Float32Array.from({ length: 40 }, (_, i) => (i % 4 === 0 ? 1 : 0)),
        4,
      ),
    );
    avatar.mesh.geometry.setIndex(new BufferAttribute(legIndices, 1));
    avatar.indices = legIndices;
    avatar.weld = buildWeldMap(legBody);

    const rig = new WardrobeRig(avatar);
    const envelope: SolveEnvelope = {
      params: { gender: 0.5, heightCm: 170, weightKg: 70, shoe: { system: 'EU', size: 42 } },
      roundTripMs: 0,
      result: {
        positions: legBody,
        joints: Float32Array.from([
          0, 1, 0, 0, 1.1, 0, 0, 1.4, 0, 0, 1.5, 0, 0.3, 1.2, 0, 0.3, 1.3, 0, 0, 1.5, 0, 0, 1.6, 0,
        ]),
        groundOffsetY: 0,
        achievedCm: { inseam: 72 },
        residualsCm: {},
        unreachable: [],
        estimatedMassKg: 70,
        solveMs: 0,
      },
    };
    try {
      rig.onSolve(envelope);
      await vi.waitFor(() =>
        expect(avatar.scene.getObjectByName('garment:trousers')).toBeDefined(),
      );
      // Regular (72 cm, unchanged length): ankle + seam copy + calf are hidden; only protected [7, 8, 9] remains.
      expect(Array.from(avatar.mesh.geometry.getIndex()!.array)).toEqual([7, 8, 9]);

      // Switch to Short (64 cm, 8 cm shorter): cuff rises to y = 0.18 m; ankle [0, 1, 2] and seam copy [0, 3, 2]
      // are > 1 cm below the new cuff and become visible again, while calf [4, 5, 6] stays hidden.
      useWardrobeStore.setState((state) => ({
        items: state.items.map((item) => ({ ...item, selectedSize: 'Short' })),
      }));
      await vi.waitFor(() =>
        expect(Array.from(avatar.mesh.geometry.getIndex()!.array)).toEqual([
          0, 1, 2, 0, 3, 2, 7, 8, 9,
        ]),
      );
    } finally {
      rig.dispose();
      avatar.mesh.geometry.dispose();
      avatar.material.dispose();
      avatar.skeleton.dispose();
      trouserRuntime.material.dispose();
    }
  });
});
