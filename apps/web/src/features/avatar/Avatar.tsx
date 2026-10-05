import { useEffect, useState } from 'react';
import { useThree } from '@react-three/fiber';
import type { CanvasTexture } from 'three';
import { useAvatarLoadStore } from '../../store/avatarLoadStore';
import { useAvatarRuntimeStore } from '../../store/avatarRuntimeStore';
import { MANNEQUIN_COLOR, useAppearanceStore } from '../../store/appearanceStore';
import { useFaceStore } from '../../store/faceStore';
import { useBodyStore } from '../../store/bodyStore';
import { useSolveStore } from '../../store/solveStore';
import { useViewerStore } from '../../store/viewerStore';
import { applySolveResult } from './applySolve';
import { loadAvatarAssets, type AvatarAssets } from './avatarAssets';
import { resolveSkinHex } from './skinComposite';
import { SkinMap } from './SkinMap';
import { PartsRig } from './parts/partsRig';
import { WardrobeRig } from '../wardrobe/wardrobeRig';
import { TwinMode } from '../twin/twinMode';
import { setSkinAppearance } from '../viewer/skinMaterial';

function setMap(assets: AvatarAssets, map: CanvasTexture | null): void {
  if (assets.material.map === map) return;
  assets.material.map = map;
  assets.material.needsUpdate = true; // the shader gains / loses the map sampler
}

/**
 * Mannequin: satin grey. Skin: without a baked face a plain colour, with one the composite skin map (skin tone +
 * face overlay) whose base colour is the photo tone or the preset tone.
 */
function applyAppearance(assets: AvatarAssets, skin: { current: SkinMap | null }): void {
  const material = assets.material;
  const { mode, toneIndex, useFaceTone } = useAppearanceStore.getState();
  const face = useFaceStore.getState();
  if (mode === 'mannequin') {
    setMap(assets, null);
    material.color.set(MANNEQUIN_COLOR);
    setSkinAppearance(material, true);
    useAvatarRuntimeStore.getState().setFaceTextureRevision(0);
    return;
  }
  const hex = resolveSkinHex({ mode, toneIndex, useFaceTone, faceToneHex: face.skinToneHex });
  if (face.overlayCanvas) {
    skin.current ??= new SkinMap();
    skin.current.update(hex, face.overlayCanvas);
    setMap(assets, skin.current.texture);
    material.color.set('#ffffff'); // the map carries the colour
    useAvatarRuntimeStore.getState().setFaceTextureRevision(face.revision);
    if (import.meta.env.DEV) (window as unknown as { __dtSkinCanvas?: HTMLCanvasElement }).__dtSkinCanvas = skin.current.canvas;
  } else {
    setMap(assets, null);
    material.color.set(hex);
    useAvatarRuntimeStore.getState().setFaceTextureRevision(0);
  }
  setSkinAppearance(material, false);
}

/** The MakeHuman body: geometry morphed in a worker, skeleton rebuilt after each solve. Renders inside <Canvas>. */
export function Avatar() {
  const [assets, setAssets] = useState<AvatarAssets | null>(null);
  const gl = useThree((state) => state.gl); // the twin's hair shader reads the MSAA sample count from it
  const failure = useAvatarLoadStore((s) => (s.status === 'error' ? s.error : null));
  if (failure !== null) throw new Error(failure);

  useEffect(() => {
    void useAppearanceStore.getState().hydrate(); // restores hair / eye / brow choices, skin tone and mode
    let alive = true;
    loadAvatarAssets().then(
      (loaded) => {
        if (alive) setAssets(loaded);
      },
      () => undefined, // reported through the load store
    );
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!assets) return;
    const rig = new WardrobeRig(assets); // worn garments: mounted on the shared skeleton, re-fitted after each solve
    // Standard mode: body parts (eyes, brows, lashes, hair), created after the wardrobe rig and disposed before it: it
    // hooks the body geometry's setIndex to compose its own hidden triangles with the wardrobe's. Realistic-twin mode:
    // the parts are dropped, the body is solved with the twin's fitted shape but hidden, and the scan is bound to the
    // same skeleton (features/twin). The mode switch owns both, so the two never hook the body index together.
    const twin = new TwinMode(
      assets,
      () => new PartsRig(assets),
      () => assets.client.solve(useBodyStore.getState().params),
      () => gl,
    );
    const off = assets.onSolve((envelope) => {
      applySolveResult(assets, envelope);
      rig.onSolve(envelope);
      twin.onSolve(envelope);
      const runtime = useAvatarRuntimeStore.getState();
      if (runtime.skeleton !== assets.skeleton) runtime.setSkeleton(assets.skeleton);
      runtime.bumpRestVersion();
      const r = envelope.result;
      if (r.faceFit !== undefined) useFaceStore.getState().setFit(r.faceFit);
      useSolveStore.getState().setSolve({
        ...twin.displaySolve(envelope),
        solveMs: r.solveMs,
        roundTripMs: envelope.roundTripMs,
      });
      useAvatarLoadStore.getState().setReady();
    });
    const unsubscribe = useBodyStore.subscribe((state, previous) => {
      if (state.params !== previous.params && !twin.active) assets.client.solve(state.params);
    });
    assets.client.solve(useBodyStore.getState().params);
    return () => {
      off();
      unsubscribe();
      twin.dispose();
      rig.dispose();
    };
  }, [assets, gl]);

  useEffect(() => {
    if (!assets) return;
    const skin: { current: SkinMap | null } = { current: null };
    const apply = (): void => applyAppearance(assets, skin);
    apply();
    let previousOverlay = useFaceStore.getState().overlayCanvas;
    const offAppearance = useAppearanceStore.subscribe(apply);
    const offFace = useFaceStore.subscribe((state, previous) => {
      if (state.revision !== previous.revision || state.overlayCanvas !== previous.overlayCanvas || state.skinToneHex !== previous.skinToneHex) apply();
      // Face shape: fit in the worker when a new bake arrives, clear it when the photo is removed.
      if (state.overlayCanvas === previousOverlay) return;
      previousOverlay = state.overlayCanvas;
      if (state.overlayCanvas && state.landmarks && state.imageSize) {
        void assets.client.setFace(state.landmarks, state.imageSize.width, state.imageSize.height);
      } else {
        void assets.client.clearFace();
      }
    });
    if (previousOverlay) {
      const state = useFaceStore.getState();
      if (state.landmarks && state.imageSize) void assets.client.setFace(state.landmarks, state.imageSize.width, state.imageSize.height);
    }
    return () => {
      offAppearance();
      offFace();
      skin.current?.dispose();
      setMap(assets, null);
    };
  }, [assets]);

  if (!assets) return null;
  return (
    <primitive
      object={assets.scene}
      onDoubleClick={(event: { stopPropagation: () => void; point: { x: number; y: number; z: number } }) => {
        event.stopPropagation();
        useViewerStore.getState().requestPoint([event.point.x, event.point.y, event.point.z]);
      }}
    />
  );
}
