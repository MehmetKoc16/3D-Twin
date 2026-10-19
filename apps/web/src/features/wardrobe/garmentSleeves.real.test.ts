// @ts-expect-error Node types are supplied by Vitest.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { Bone, Matrix4, MeshStandardMaterial, Skeleton } from 'three';
import {
  bindGarment,
  buildBasePositions,
  computeJoints,
  createGarmentSections,
  gradeGarment,
  parseGarmentBinding,
  type BodyManifest,
  type GarmentTemplateDef,
  type RigDef,
} from '@dt/avatar-core';
import { bytes, parseGlb, primitive, root } from './realAssets.testkit';
import { boundBodyPoints, fitToChart } from './garmentGrading';
import { containInnerSleeves, pushOutside, sleeveWeight } from './garmentCollide';
import { GarmentInstance, type GarmentUpdateContext } from './garmentInstance';
import { buildWeldMap } from '../avatar/meshMath';

describe.skipIf(!root)('real denim and polo sleeve containment', () => {
  it('compresses the real polo sleeves while preserving the jacket lower front', () => {
    const dir = root ?? '';
    const mesh = primitive(parseGlb(readFileSync(`${dir}/body/base.glb`)));
    const manifest = JSON.parse(readFileSync(`${dir}/body/manifest.json`, 'utf8')) as BodyManifest;
    const rig = JSON.parse(readFileSync(`${dir}/body/rig.json`, 'utf8')) as RigDef;
    const body = buildBasePositions(mesh.position, manifest);
    const joints = computeJoints(rig, body);
    const arms = ['l', 'r'].map((side) => ({
      shoulder: joints.get(`upperarm_${side}`)!.head,
      elbow: joints.get(`lowerarm_${side}`)!.head,
    }));
    const catalogue = JSON.parse(readFileSync(`${dir}/garments/index.json`, 'utf8')) as {
      garments: GarmentTemplateDef[];
    };
    const load = (id: string) => {
      const def = catalogue.garments.find((g) => g.id === id)!;
      const binding = parseGarmentBinding(bytes(`${dir}/garments/${def.binding!}`));
      const rest = new Float32Array(binding.count * 3);
      const bound = new Float32Array(rest.length);
      bindGarment(binding, def.scaleRefs, body, rest);
      boundBodyPoints(binding, body, bound);
      const index = primitive(parseGlb(readFileSync(`${dir}/garments/${def.mesh}`))).indices;
      return { def, binding, rest, bound, index };
    };
    const jacket = load('mh-denim-jacket');
    const polo = load('mh-male-polo-shirt');
    const inner = { positions: polo.rest.slice(), bound: polo.bound, index: polo.index };
    const rings = [{ planeY: 1, deltaCircumferenceM: -0.12 }];
    const fit = (contain = false) => {
      const result = new Float32Array(jacket.rest.length);
      fitToChart(
        {
          def: jacket.def,
          item: {
            id: 'jacket',
            name: 'Jacket',
            templateId: jacket.def.id,
            color: '#888888',
            sizes: ['M'],
            selectedSize: 'M',
            chart: { chest: [jacket.def.nativeMeasures.chest! - 12] },
          },
          binding: jacket.binding,
          body,
          sections: createGarmentSections(body),
          achievedCm: {},
          planes: { planeY: { chest: 1 } },
          joint: (name) => joints.get(name)?.head,
          innerSurface: inner,
          containInnerSleeves: contain,
        },
        new Float32Array(result.length),
        result,
      );
      pushOutside(result, jacket.binding.count, [inner], {
        orientation: 'surface',
        marginM: 0.012,
        frontDepthOnly: true,
        vertexWeights: contain
          ? Float32Array.from(
              { length: jacket.binding.count },
              (_, v) => 1 - sleeveWeight(jacket.bound, v * 3, arms),
            )
          : undefined,
      });
      return result;
    };
    const before = fit();
    const independent = new Float32Array(jacket.rest.length);
    gradeGarment(jacket.rest, jacket.binding, body, rings, independent, undefined, { outer: true });
    const moved = containInnerSleeves(
      inner.positions,
      inner.bound,
      { positions: independent, bound: jacket.bound, index: jacket.index },
      arms,
    );
    const after = fit(true);
    let maxChange = 0;
    let frontChange = 0;
    for (let o = 0; o < after.length; o += 3) {
      const change = Math.hypot(...[0, 1, 2].map((k) => after[o + k]! - before[o + k]!));
      maxChange = Math.max(maxChange, change);
      if (Math.abs(jacket.rest[o]!) < 0.1 && jacket.rest[o + 1]! < 1.1) {
        frontChange = Math.max(frontChange, change);
      }
    }
    expect(moved).toBeGreaterThan(100);
    expect(maxChange).toBeGreaterThan(0.005);
    expect(frontChange).toBeLessThan(0.00001);

    const skeleton = new Skeleton(
      rig.bones.map(({ name }) => {
        const bone = new Bone();
        bone.name = name;
        return bone;
      }),
    );
    const material = new MeshStandardMaterial();
    const instance = new GarmentInstance(
      {
        def: polo.def,
        vertexCount: polo.binding.count,
        index: polo.index,
        binding: polo.binding,
        neutralPositions: polo.rest,
        uv: new Float32Array(polo.binding.count * 2),
        skinIndices: new Uint16Array(polo.binding.count * 4),
        skinWeights: new Float32Array(polo.binding.count * 4),
        weld: buildWeldMap(polo.rest),
        deleteVerts: new Uint32Array(0),
        rigidAttachments: null,
        material,
      },
      {
        id: 'polo',
        name: 'Polo',
        templateId: polo.def.id,
        color: '#888888',
        sizes: ['M'],
        selectedSize: 'M',
        chart: {},
      },
      skeleton,
      new Matrix4(),
    );
    const context: GarmentUpdateContext = {
      body,
      joints: Float32Array.from(
        rig.bones.flatMap(({ name }) => {
          const joint = joints.get(name)!;
          return [...joint.head, ...joint.tail];
        }),
      ),
      sections: createGarmentSections(body),
      achievedCm: {},
      planes: undefined,
      heatmap: false,
      lowers: [],
    };
    try {
      instance.update(context);
      const original = Float32Array.from(instance.surface().positions);
      const jacketSurface = { positions: after, bound: jacket.bound, index: jacket.index };
      instance.containSleeves(context, jacketSurface);
      instance.hideUnder([jacketSurface]);
      const geometry = instance.mesh.geometry;
      const visible = Array.from(geometry.getIndex()!.array.slice(0, geometry.drawRange.count));
      const cuff = (v: number) => Math.abs(polo.rest[v * 3]!) > 0.24 && polo.rest[v * 3 + 1]! > 1;
      expect(Array.from(polo.index).filter(cuff).length).toBeGreaterThan(100);
      expect(visible.filter(cuff)).toHaveLength(0);
      // Removing the jacket re-fits the polo and restores both its full sleeve shape and index.
      instance.update(context);
      instance.hideUnder(null);
      expect(geometry.drawRange.count).toBe(polo.index.length);
      expect(Float32Array.from(instance.surface().positions)).toEqual(original);
    } finally {
      instance.dispose();
      material.dispose();
      skeleton.dispose();
    }
  });
});
