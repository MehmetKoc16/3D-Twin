import { useEffect, useState } from 'react';
import { useAvatarLoadStore } from '../../store/avatarLoadStore';
import { useAvatarRuntimeStore } from '../../store/avatarRuntimeStore';
import { MANNEQUIN_COLOR, skinTones, useAppearanceStore } from '../../store/appearanceStore';
import { useBodyStore } from '../../store/bodyStore';
import { useSolveStore } from '../../store/solveStore';
import { useViewerStore } from '../../store/viewerStore';
import { applySolveResult } from './applySolve';
import { loadAvatarAssets, type AvatarAssets } from './avatarAssets';

function applyAppearance(assets: AvatarAssets, mode: 'skin' | 'mannequin', toneIndex: number): void {
  const material = assets.material;
  if (mode === 'mannequin') {
    material.color.set(MANNEQUIN_COLOR);
    material.roughness = 0.85;
    material.sheen = 0;
  } else {
    material.color.set((skinTones[toneIndex] ?? skinTones[1]!).color);
    material.roughness = 0.6;
    material.sheen = 0.35;
    material.sheenRoughness = 0.5;
    material.sheenColor.set('#ffd9c4');
  }
}

/** The MakeHuman body: geometry morphed in a worker, skeleton rebuilt after each solve. Renders inside <Canvas>. */
export function Avatar() {
  const [assets, setAssets] = useState<AvatarAssets | null>(null);
  const failure = useAvatarLoadStore((s) => (s.status === 'error' ? s.error : null));
  if (failure !== null) throw new Error(failure);

  useEffect(() => {
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
    const off = assets.onSolve((envelope) => {
      applySolveResult(assets, envelope);
      const runtime = useAvatarRuntimeStore.getState();
      if (runtime.skeleton !== assets.skeleton) runtime.setSkeleton(assets.skeleton);
      runtime.bumpRestVersion();
      const r = envelope.result;
      useSolveStore.getState().setSolve({
        achievedCm: r.achievedCm,
        residualsCm: r.residualsCm,
        unreachable: r.unreachable,
        estimatedMassKg: r.estimatedMassKg,
        solveMs: r.solveMs,
        roundTripMs: envelope.roundTripMs,
      });
      useAvatarLoadStore.getState().setReady();
    });
    const unsubscribe = useBodyStore.subscribe((state, previous) => {
      if (state.params !== previous.params) assets.client.solve(state.params);
    });
    assets.client.solve(useBodyStore.getState().params);
    return () => {
      off();
      unsubscribe();
    };
  }, [assets]);

  useEffect(() => {
    if (!assets) return;
    const apply = (): void => {
      const { mode, toneIndex } = useAppearanceStore.getState();
      applyAppearance(assets, mode, toneIndex);
    };
    apply();
    return useAppearanceStore.subscribe(apply);
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
