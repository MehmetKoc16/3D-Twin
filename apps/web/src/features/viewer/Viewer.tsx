import { Suspense, useEffect, useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { CameraControls } from '@react-three/drei';
import { AgXToneMapping, PCFShadowMap, SRGBColorSpace, Vector3 } from 'three';
import { useTranslation } from 'react-i18next';
import { AvatarSlot } from './AvatarSlot';
import { boneFocusTargetProvider } from './boneFocusTargets';
import type { FocusPreset, FocusTargetProvider } from './focusTargets';
import { useAvatarLoadStore } from '../../store/avatarLoadStore';
import { useAvatarRuntimeStore } from '../../store/avatarRuntimeStore';
import { useBodyStore } from '../../store/bodyStore';
import { usePoseStore } from '../../store/poseStore';
import { useViewerStore } from '../../store/viewerStore';
import { PoseBar } from '../poses/PoseBar';
import { PoseDriver } from '../poses/PoseDriver';
import { StudioLighting } from './StudioLighting';

const presets: FocusPreset[] = ['full', 'face', 'upper', 'lower', 'feet'];

function CameraRig({ heightM, provider }: { heightM: number; provider: FocusTargetProvider }) {
  const controls = useRef<React.ComponentRef<typeof CameraControls>>(null);
  const [dragging, setDragging] = useState(false);
  const lastRequest = useRef(0);
  const framedAvatar = useRef(false);
  const targetHeight = useRef<number | null>(null);
  const { focusPreset, focusRequest, focusPoint, autoRotate } = useViewerStore();
  const restVersion = useAvatarRuntimeStore((state) => state.restVersion);
  const avatarReady = useAvatarLoadStore((state) => state.status === 'ready');

  useEffect(() => {
    const camera = controls.current;
    if (!camera || dragging) return;
    const requested = focusRequest !== lastRequest.current;
    if (!requested && (!avatarReady || !framedAvatar.current)) {
      if (!avatarReady) return;
    }
    const destination = focusPoint
      ? { target: focusPoint, distance: 0.8 }
      : provider.getTargets(heightM)[focusPreset];
    const [x, y, z] = destination.target;
    if (requested || !framedAvatar.current) {
      void camera.setLookAt(x + destination.distance * 0.32, y + destination.distance * 0.12,
        z + destination.distance * 0.95, x, y, z, true);
      lastRequest.current = focusRequest;
      framedAvatar.current = avatarReady;
      targetHeight.current = y;
    } else if (!focusPoint && targetHeight.current !== null && Math.abs(y - targetHeight.current) > 0.02) {
      const deltaY = y - targetHeight.current;
      const position = camera.getPosition(new Vector3());
      const target = camera.getTarget(new Vector3());
      // Move the camera and target together to keep the user's orbit and distance.
      void camera.setLookAt(position.x, position.y + deltaY, position.z,
        target.x, target.y + deltaY, target.z, true);
      targetHeight.current = y;
    }
  }, [avatarReady, dragging, heightM, restVersion, provider, focusPreset, focusRequest, focusPoint]);

  useFrame((_, delta) => {
    if (autoRotate && !dragging) controls.current?.rotate(0.35 * delta, 0, true);
  });
  return <CameraControls ref={controls} minDistance={0.25} maxDistance={6} smoothTime={0.55}
    onControlStart={() => { controls.current?.stop(); setDragging(true); }}
    onControlEnd={() => setDragging(false)} />;
}

function Scene({ provider }: { provider: FocusTargetProvider }) {
  const params = useBodyStore((state) => state.params);
  const poseId = usePoseStore((state) => state.poseId);
  return <>
    <color attach="background" args={['#1b2529']} />
    <StudioLighting />
    <mesh position={[0, -0.11, 0]} receiveShadow>
      <cylinderGeometry args={[1.08, 1.13, 0.2, 64]} />
      <meshStandardMaterial color="#2c383c" metalness={0.35} roughness={0.5} envMapIntensity={0.8} />
    </mesh>
    <mesh position={[0, -0.0075, 0]} receiveShadow>
      <cylinderGeometry args={[1.06, 1.06, 0.015, 64]} />
      <meshStandardMaterial color="#3e5052" metalness={0.15} roughness={0.7} envMapIntensity={0.7} />
    </mesh>
    <AvatarSlot params={params} poseId={poseId} />
    <PoseDriver />
    <CameraRig heightM={params.heightCm / 100} provider={provider} />
  </>;
}

function AvatarStatus() {
  const { t } = useTranslation();
  const { status, progress, error } = useAvatarLoadStore();
  const faceRevision = useAvatarRuntimeStore((s) => s.faceTextureRevision);
  if (status === 'ready') return <div data-testid="avatar-ready" data-face-revision={faceRevision} className="sr-only" />;
  if (status === 'error') {
    return <div role="alert" data-testid="avatar-error" className="absolute inset-x-4 top-16 mx-auto max-w-md rounded-xl border border-amber-400/40 bg-slate-950/85 p-3 text-sm text-amber-200 shadow-xl backdrop-blur">
      <strong className="block">{t('viewer.loadError')}</strong>
      <span className="mt-1 block text-xs text-amber-100/80">{error}</span>
    </div>;
  }
  const percent = Math.round(progress * 100);
  return <div role="status" data-testid="avatar-loading" className="pointer-events-none absolute inset-0 grid place-items-center bg-[#171d21]/70">
    <div className="w-64 rounded-xl border border-white/10 bg-slate-950/80 p-4 text-center shadow-xl backdrop-blur">
      <p className="text-sm font-medium text-slate-100">{t('viewer.loading')}</p>
      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/10" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}>
        <div className="h-full rounded-full bg-teal-400 transition-[width]" style={{ width: `${percent}%` }} />
      </div>
      <p className="mt-2 text-xs text-slate-400">{percent}%</p>
    </div>
  </div>;
}

export function Viewer({ focusTargetProvider = boneFocusTargetProvider }: { focusTargetProvider?: FocusTargetProvider }) {
  const { t } = useTranslation();
  const { focusPreset, autoRotate, requestFocus, toggleAutoRotate, quality, setQuality } = useViewerStore();
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
    <Canvas gl={{ antialias: true, alpha: false, powerPreference: 'high-performance',
      toneMapping: AgXToneMapping, toneMappingExposure: 1, outputColorSpace: SRGBColorSpace }}
      shadows={{ type: PCFShadowMap }} dpr={quality === 'high' ? [1, 1.5] : 1}
      camera={{ position: [1.15, 1.35, 3.2], fov: 40 }}>
      <Suspense fallback={null}><Scene provider={focusTargetProvider} /></Suspense>
    </Canvas>
    <div className="pointer-events-none absolute left-4 top-4 rounded-xl border border-white/10 bg-slate-950/55 px-3 py-2 text-xs text-slate-300 backdrop-blur">
      {t('viewer.hint')}
    </div>
    <details className="absolute left-4 top-16 max-w-56 rounded-xl border border-white/10 bg-slate-950/85 p-3 text-xs text-slate-200 shadow-lg backdrop-blur">
      <summary className="cursor-pointer focus-visible:outline-teal-400">{t('viewer.settings')}</summary>
      <label className="mt-3 block">
        {t('viewer.quality.label')}
        <select data-testid="viewer-quality" value={quality}
          onChange={(event) => setQuality(event.target.value === 'performance' ? 'performance' : 'high')}
          className="mt-1 block w-full rounded-md border border-white/20 bg-slate-800 p-2 text-white">
          <option value="high">{t('viewer.quality.high')}</option>
          <option value="performance">{t('viewer.quality.performance')}</option>
        </select>
      </label>
      <p className="mt-2 text-slate-400">{t('viewer.quality.help')}</p>
    </details>
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
    <AvatarStatus />
    <PoseBar />
  </div>;
}
