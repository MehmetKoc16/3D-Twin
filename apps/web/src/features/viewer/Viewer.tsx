import { Suspense, useEffect, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { CameraControls, ContactShadows } from '@react-three/drei';
import { useTranslation } from 'react-i18next';
import { AvatarSlot } from './AvatarSlot';
import { defaultFocusTargetProvider } from './focusTargets';
import type { FocusPreset, FocusTargetProvider } from './focusTargets';
import { useBodyStore } from '../../store/bodyStore';
import { usePoseStore } from '../../store/poseStore';
import { useViewerStore } from '../../store/viewerStore';
import { PoseBar } from '../poses/PoseBar';

const presets: FocusPreset[] = ['full', 'face', 'upper', 'lower', 'feet'];

function CameraRig({ heightM, provider }: { heightM: number; provider: FocusTargetProvider }) {
  const controls = useRef<React.ComponentRef<typeof CameraControls>>(null);
  const { focusPreset, focusRequest, focusPoint, autoRotate } = useViewerStore();

  useEffect(() => {
    const destination = focusPoint
      ? { target: focusPoint, distance: 0.8 }
      : provider.getTargets(heightM)[focusPreset];
    const [x, y, z] = destination.target;
    void controls.current?.setLookAt(x + destination.distance * 0.32, y + destination.distance * 0.12,
      z + destination.distance * 0.95, x, y, z, true);
  }, [heightM, provider, focusPreset, focusRequest, focusPoint]);

  useFrame((_, delta) => {
    if (autoRotate) controls.current?.rotate(0.35 * delta, 0, true);
  });
  return <CameraControls ref={controls} minDistance={0.25} maxDistance={6} smoothTime={0.55} />;
}

function Scene({ provider }: { provider: FocusTargetProvider }) {
  const params = useBodyStore((state) => state.params);
  const poseId = usePoseStore((state) => state.poseId);
  return <>
    <color attach="background" args={['#1b2529']} />
    <hemisphereLight args={['#d5e8ef', '#38434a', 2.2]} />
    <directionalLight position={[2.5, 5, 4]} intensity={2.6} />
    <directionalLight position={[-3, 3, -2]} intensity={1.8} color="#8bc9cf" />
    <mesh position={[0, -0.11, 0]} receiveShadow>
      <cylinderGeometry args={[1.08, 1.13, 0.2, 64]} />
      <meshStandardMaterial color="#2c383c" metalness={0.35} roughness={0.6} />
    </mesh>
    <mesh position={[0, 0, 0]} receiveShadow>
      <cylinderGeometry args={[1.06, 1.06, 0.015, 64]} />
      <meshStandardMaterial color="#3e5052" metalness={0.2} roughness={0.85} />
    </mesh>
    <ContactShadows position={[0, -0.2, 0]} opacity={0.25} scale={1.4} blur={4} far={2} />
    <AvatarSlot params={params} poseId={poseId} />
    <CameraRig heightM={params.heightCm / 100} provider={provider} />
  </>;
}

export function Viewer({ focusTargetProvider = defaultFocusTargetProvider }: { focusTargetProvider?: FocusTargetProvider }) {
  const { t } = useTranslation();
  const { focusPreset, autoRotate, requestFocus, toggleAutoRotate } = useViewerStore();
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.altKey || event.ctrlKey || event.metaKey || /INPUT|TEXTAREA|SELECT/.test((event.target as HTMLElement).tagName)) return;
      const index = Number(event.key) - 1;
      if (index >= 0 && index < presets.length) requestFocus(presets[index]!);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [requestFocus]);

  return <div className="absolute inset-0 overflow-hidden">
    <Canvas gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }} shadows dpr={[1, 2]}
      camera={{ position: [1.15, 1.35, 3.2], fov: 40 }}>
      <Suspense fallback={null}><Scene provider={focusTargetProvider} /></Suspense>
    </Canvas>
    <div className="pointer-events-none absolute left-4 top-4 rounded-xl border border-white/10 bg-slate-950/55 px-3 py-2 text-xs text-slate-300 backdrop-blur">
      {t('viewer.hint')}
    </div>
    <div className="absolute right-4 top-4 flex max-w-[calc(100%-2rem)] flex-wrap justify-end gap-1.5" aria-label={t('viewer.focusGroup')}>
      {presets.map((preset, index) => <button key={preset} type="button" onClick={() => requestFocus(preset)}
        aria-pressed={focusPreset === preset} title={`${index + 1}`}
        className={`rounded-lg border px-2.5 py-1.5 text-xs font-medium shadow-lg transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400 ${focusPreset === preset ? 'border-teal-300 bg-teal-400 text-slate-950' : 'border-white/15 bg-slate-950/70 text-white hover:bg-slate-800'}`}>
        {t(`viewer.focus.${preset}`)}
      </button>)}
      <button type="button" onClick={toggleAutoRotate} aria-pressed={autoRotate}
        className="rounded-lg border border-white/15 bg-slate-950/70 px-2.5 py-1.5 text-xs text-white hover:bg-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400">
        {t('viewer.turntable')}
      </button>
      <button type="button" onClick={() => requestFocus('full')}
        className="rounded-lg border border-white/15 bg-slate-950/70 px-2.5 py-1.5 text-xs text-white hover:bg-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400">
        {t('viewer.reset')}
      </button>
    </div>
    <PoseBar />
  </div>;
}
