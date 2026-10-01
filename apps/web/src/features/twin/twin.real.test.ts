// @ts-expect-error Node types are not in the browser app's tsconfig; Vitest supplies this module.
import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  buildBasePositions,
  computeJoints,
  type BodyManifest,
  type MeasuresDef,
  type RigDef,
} from '@dt/avatar-core';
import { solveFixedShape } from '../../workers/fixedShape';
import { flattenJoints, groundRenderPositions } from '../../workers/ground';
import { headsFromJoints } from '../avatar/meshMath';
import { bytes, parseGlb, primitive, root } from '../wardrobe/realAssets.testkit';
import { restAlignment, translatePositions } from './twinBinding';
import { parseTwinJson, twinMacros } from './twinDef';
import { loadTwinModel } from './twinModel';
import { loadTwinBundle } from './twinBundle';
import { TwinHands } from './twinHands';
import { TwinRig } from './twinRig';
import { rebuildRestSkeleton } from '../avatar/applySolve';
import type { AvatarAssets } from '../avatar/avatarAssets';
import type { SolveEnvelope } from '../../workers/avatarClient';
import { Bone, SkinnedMesh, Vector3 } from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { parseMapping, validateMapping } from './twinMapping';

/**
 * The twin package end to end on the NON-personal stand-in fixture (a re-fitted, re-posed CC0 MakeHuman body, made
 * with tools/twin-lab/rig/make_standin.py --textured + rig_scan.py). Checks that what the Python pipeline exports
 * (fitted shape, measurements, rest heads, mapping) is what the browser reproduces with avatar-core. Skipped when the
 * body assets or the fixture are not there.
 */
const fixtureDir = ['e2e/fixtures/twin-standin', 'apps/web/e2e/fixtures/twin-standin'].find((d) =>
  existsSync(`${d}/twin.json`),
);

/** Drops textures so GLTFLoader can parse in Node (no image decoding needed for the geometry checks). */
function stripImages(buffer: ArrayBuffer): ArrayBuffer {
  const view = new DataView(buffer);
  const jsonLength = view.getUint32(12, true);
  const json = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 20, jsonLength)),
  ) as Record<string, unknown>;
  delete json.images;
  delete json.textures;
  delete json.samplers;
  for (const material of (json.materials as
    { pbrMetallicRoughness?: Record<string, unknown> }[] | undefined) ?? [])
    if (material.pbrMetallicRoughness) delete material.pbrMetallicRoughness.baseColorTexture;
  const text = new TextEncoder().encode(JSON.stringify(json));
  const jsonBytes = new Uint8Array(text.length + ((4 - (text.length % 4)) % 4)).fill(0x20);
  jsonBytes.set(text);
  const rest = new Uint8Array(buffer, 20 + jsonLength);
  const out = new Uint8Array(20 + jsonBytes.length + rest.length);
  const head = new DataView(out.buffer);
  out.set(new TextEncoder().encode('glTF'), 0);
  head.setUint32(4, 2, true);
  head.setUint32(8, out.length, true);
  head.setUint32(12, jsonBytes.length, true);
  head.setUint32(16, 0x4e4f534a, true);
  out.set(jsonBytes, 20);
  out.set(rest, 20 + jsonBytes.length);
  return out.buffer;
}

describe.skipIf(!root || !fixtureDir)('twin package on the stand-in fixture', () => {
  const dir = root ?? '';
  const fixture = fixtureDir ?? '';
  const manifest = JSON.parse(readFileSync(`${dir}/body/manifest.json`, 'utf8')) as BodyManifest;
  const measures = JSON.parse(readFileSync(`${dir}/body/measures.json`, 'utf8')) as MeasuresDef;
  const rig = JSON.parse(readFileSync(`${dir}/body/rig.json`, 'utf8')) as RigDef;
  const baseGlb = primitive(parseGlb(readFileSync(`${dir}/body/base.glb`)));
  const base = buildBasePositions(baseGlb.position, manifest);
  const morphs = bytes(`${dir}/body/morphs.bin`);
  const def = parseTwinJson(readFileSync(`${fixture}/twin.json`, 'utf8'));
  const boneNames = rig.bones.map((b) => b.name);
  const solved = solveFixedShape(
    { manifest, base, morphs, measures, indices: baseGlb.indices },
    { macros: twinMacros(def), modifiers: def.fittedModifiers },
  );

  it('loads the stand-in bundle, replaces its hands and articulates real fingers from the pose JSONs', async () => {
    const bundle = await loadTwinBundle(stripImages(bytes(`${fixture}/twin.glb`)));
    expect(bundle?.def).toEqual(def);
    expect(bundle?.mapping).toEqual(parseMapping(bytes(`${fixture}/mh2twin.bin`)));
    expect(bundle?.skinToneHex).toBe('#d6a489');
    const gltf = await new GLTFLoader().parseAsync(bytes(`${dir}/body/base.glb`), '');
    let body: SkinnedMesh | undefined;
    gltf.scene.traverse((child) => {
      if (child instanceof SkinnedMesh) body = child;
    });
    if (!body) throw new Error('CC0 fixture must have a skinned body');
    const skeleton = body.skeleton;
    const grounded = groundRenderPositions(solved.positions, manifest.renderVertexCount);
    const joints = flattenJoints(boneNames, computeJoints(rig, solved.positions), grounded.offsetY);
    const rigIndexOfBone = Int32Array.from(skeleton.bones.map((b) => boneNames.indexOf(b.name)));
    const assets = {
      scene: gltf.scene,
      mesh: body,
      skeleton,
      indices: Uint32Array.from(body.geometry.getIndex()!.array),
      rigIndexOfBone,
      parents: Int32Array.from(
        skeleton.bones.map((b) =>
          b.parent instanceof Bone ? skeleton.bones.indexOf(b.parent) : -1,
        ),
      ),
    } as AvatarAssets;
    (body.geometry.getAttribute('position').array as Float32Array).set(grounded.positions);
    body.geometry.computeVertexNormals();
    rebuildRestSkeleton(assets, joints);
    const appHeads = headsFromJoints(joints, rigIndexOfBone);
    const hands = new TwinHands(assets, bundle!.skinToneHex);
    hands.update(appHeads);
    expect(hands.mesh.visible).toBe(true);
    expect(hands.mesh.skeleton).toBe(skeleton);
    expect(hands.mesh.geometry.getIndex()!.count).toBeGreaterThan(300);
    expect(hands.mesh.geometry.getAttribute('position').count).toBeLessThan(
      manifest.renderVertexCount / 3,
    );
    const skinIndex = hands.mesh.geometry.getAttribute('skinIndex');
    const skinWeight = hands.mesh.geometry.getAttribute('skinWeight');
    const distal = skeleton.bones.findIndex((b) => b.name === 'index_03_l');
    const selected = Array.from({ length: skinIndex.count }, (_, v) => v).filter(
      (v) => skinIndex.getX(v) === distal && skinWeight.getX(v) >= 0.75,
    );
    expect(selected.length).toBeGreaterThan(0);
    const wrist = skeleton.bones.find((b) => b.name === 'hand_l')!;
    const pose = (id: string): Vector3[] => {
      const json = JSON.parse(readFileSync(`${dir}/poses/${id}.json`, 'utf8')) as {
        bones: Record<string, [number, number, number, number]>;
      };
      for (const bone of skeleton.bones)
        bone.quaternion.fromArray(json.bones[bone.name] ?? [0, 0, 0, 1]);
      gltf.scene.updateMatrixWorld(true);
      skeleton.update();
      return selected.map((v) =>
        wrist.worldToLocal(hands.mesh.localToWorld(hands.mesh.getVertexPosition(v, new Vector3()))),
      );
    };
    const t = pose('t-pose');
    const relaxed = pose('relaxed');
    expect(Math.max(...t.map((p, i) => p.distanceTo(relaxed[i]!)))).toBeGreaterThan(0.001);
    hands.dispose();
    for (const bone of skeleton.bones) bone.quaternion.identity();
    gltf.scene.updateMatrixWorld(true);
    const model = await loadTwinModel(
      stripImages(bytes(`${fixture}/twin.glb`)),
      bundle!.def,
      skeleton.bones.map((b) => b.name),
    );
    const twin = new TwinRig(assets, model, bundle!.mapping, undefined, bundle!.skinToneHex);
    const envelope = { result: { fixedShape: true, joints } } as SolveEnvelope;
    twin.onSolve(envelope);
    expect(twin.mesh.visible).toBe(true);
    expect(body.visible).toBe(false);
    expect(twin.mesh.geometry.drawRange.count).toBeLessThan(model.index.length - 300);
    const sourcePositions = (body.geometry.getAttribute('position').array as Float32Array).slice();
    twin.dispose();
    expect(body.visible).toBe(true);
    expect(body.geometry.getAttribute('position').array).toEqual(sourcePositions);
    expect(assets.scene.children.some((child) => child.name === 'twin:hands')).toBe(false);
    body.geometry.dispose();
  });

  it('the bone order of twin.json is the rig order', () => {
    expect(def.boneOrder).toEqual(boneNames);
  });

  it('reproduces the measurements the pipeline reported (numpy port vs avatar-core)', () => {
    expect(def.measurementsRawCm).toBeDefined();
    for (const [id, raw] of Object.entries(def.measurementsRawCm ?? {})) {
      const got = solved.achievedCm[id as keyof typeof solved.achievedCm];
      expect(got, id).toBeDefined();
      expect(Math.abs((got as number) - raw), `${id}: ${got} vs ${raw}`).toBeLessThan(0.05);
    }
    // the person's measurements are the raw ones minus the documented allowance
    for (const [id, value] of Object.entries(def.measurementsCm)) {
      const raw = def.measurementsRawCm?.[id as keyof typeof def.measurementsRawCm] ?? 0;
      const allowance = def.clothingAllowanceCm?.[id as keyof typeof def.clothingAllowanceCm] ?? 0;
      expect(value, id).toBeCloseTo(raw - allowance, 1);
    }
  });

  it('rebuilds the body the rig was made from: the rest heads agree up to one frame translation', () => {
    const { offsetY } = groundRenderPositions(solved.positions, manifest.renderVertexCount);
    const joints = flattenJoints(boneNames, computeJoints(rig, solved.positions), offsetY);
    const appHeads = headsFromJoints(joints, Int32Array.from(boneNames.map((_, i) => i)));
    const pipeline = boneNames.flatMap((name) => def.restHeadsM?.[name] ?? [Number.NaN, 0, 0]);
    const identity = Int32Array.from(boneNames.map((_, i) => i));
    const { offset, maxResidual } = restAlignment(pipeline, identity, appHeads);
    expect(maxResidual).toBeLessThan(2e-5); // float32 morphs and 6-decimal head export
    expect(Math.abs(offset[0])).toBeLessThan(1e-5);
    expect(Math.abs(offset[2])).toBeLessThan(1e-5);
  });

  it('the fixture glb binds to the avatar rig with no residual and its mapping points onto the body', async () => {
    const glb = stripImages(bytes(`${fixture}/rigged.glb`));
    const model = await loadTwinModel(glb, def, boneNames);
    expect(model.vertexCount).toBeGreaterThan(10000);
    const { positions, offsetY } = groundRenderPositions(
      solved.positions,
      manifest.renderVertexCount,
    );
    const joints = flattenJoints(boneNames, computeJoints(rig, solved.positions), offsetY);
    const appHeads = headsFromJoints(joints, Int32Array.from(boneNames.map((_, i) => i)));
    const alignment = restAlignment(model.heads, model.remap, appHeads);
    expect(alignment.maxResidual).toBeLessThan(1e-4);
    expect(Math.abs(alignment.offset[0])).toBeLessThan(1e-3);
    expect(Math.abs(alignment.offset[1])).toBeLessThan(0.03); // the ground rules differ by a few mm

    const mapping = parseMapping(bytes(`${fixture}/mh2twin.bin`));
    expect(mapping.length).toBe(def.mapping?.twinVertexCount);
    validateMapping(mapping, model.vertexCount, manifest.renderVertexCount);
    // every twin vertex maps onto a body vertex within a few cm once both are in the avatar frame
    const twin = new Float32Array(model.position.length);
    translatePositions(model.position, alignment.offset, twin);
    let worst = 0;
    for (let i = 0; i < model.vertexCount; i++) {
      const b = mapping[i]! * 3;
      worst = Math.max(
        worst,
        Math.hypot(
          twin[i * 3]! - positions[b]!,
          twin[i * 3 + 1]! - positions[b + 1]!,
          twin[i * 3 + 2]! - positions[b + 2]!,
        ),
      );
    }
    expect(worst).toBeLessThan(0.08);
    // the skin was remapped into the avatar's bone order and stays normalised
    for (let v = 0; v < model.vertexCount; v += 997) {
      const sum =
        model.skinWeight[v * 4]! +
        model.skinWeight[v * 4 + 1]! +
        model.skinWeight[v * 4 + 2]! +
        model.skinWeight[v * 4 + 3]!;
      expect(sum).toBeCloseTo(1, 5);
    }
    model.dispose();
  });
});
