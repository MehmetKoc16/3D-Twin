import { useCallback, useEffect, useRef, useState, type DragEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { useFaceStore } from '../../store/faceStore';
import './i18n';
import { drawPreview } from './overlay';

const busyStatuses = ['loading-model', 'detecting', 'baking'];

export function FacePanel() {
  const { t } = useTranslation();
  const status = useFaceStore((s) => s.status);
  const error = useFaceStore((s) => s.error);
  const hints = useFaceStore((s) => s.hints);
  const bitmap = useFaceStore((s) => s.bitmap);
  const landmarks = useFaceStore((s) => s.landmarks);
  const skinToneHex = useFaceStore((s) => s.skinToneHex);
  const fit = useFaceStore((s) => s.fit);
  const loadPhoto = useFaceStore((s) => s.loadPhoto);
  const bake = useFaceStore((s) => s.bake);
  const clear = useFaceStore((s) => s.clear);
  const hydrate = useFaceStore((s) => s.hydrate);

  const inputRef = useRef<HTMLInputElement>(null);
  const previewRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [dragging, setDragging] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState(false);
  const busy = busyStatuses.includes(status);

  useEffect(() => {
    void hydrate();
  }, [hydrate]);

  useEffect(() => {
    if (bitmap && previewRef.current) drawPreview(previewRef.current, bitmap, landmarks);
  }, [bitmap, landmarks]);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraOn(false);
  }, []);
  useEffect(() => stopCamera, [stopCamera]);

  const startCamera = async () => {
    setCameraError(false);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
      streamRef.current = stream;
      setCameraOn(true);
      requestAnimationFrame(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          void videoRef.current.play();
        }
      });
    } catch {
      setCameraError(true);
    }
  };

  const capture = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    // The raw (un-mirrored) frame: MediaPipe and the face map expect the subject's left on the image right.
    // Only the on-screen preview is mirrored (CSS), like a mirror.
    canvas.getContext('2d')?.drawImage(video, 0, 0);
    stopCamera();
    canvas.toBlob(
      (blob) => {
        if (blob) void loadPhoto(blob);
      },
      'image/jpeg',
      0.92,
    );
  };

  const onFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (file) void loadPhoto(file);
  };
  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    onFiles(event.dataTransfer.files);
  };

  return (
    <section data-testid="face-panel" className="flex flex-col gap-4" aria-labelledby="face-title">
      <header>
        <h2 id="face-title" className="text-lg font-semibold">
          {t('face.title')}
        </h2>
        <p className="text-sm text-slate-400">{t('face.description')}</p>
      </header>

      <p
        data-testid="face-privacy"
        className="rounded-lg border border-emerald-400/40 bg-emerald-400/10 p-3 text-sm text-emerald-100"
      >
        {t('face.privacy')}
      </p>

      {cameraOn ? (
        <div className="flex flex-col gap-2">
          <video ref={videoRef} muted playsInline className="w-full rounded-lg bg-black -scale-x-100" />
          <div className="flex gap-2">
            <button
              type="button"
              data-testid="face-capture"
              className="rounded-lg bg-teal-400 px-3 py-2 text-sm font-semibold text-slate-950"
              onClick={capture}
            >
              {t('face.capture')}
            </button>
            <button type="button" className="rounded-lg border border-white/20 px-3 py-2 text-sm" onClick={stopCamera}>
              {t('face.cameraCancel')}
            </button>
          </div>
        </div>
      ) : (
        <div
          data-testid="face-dropzone"
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`flex flex-col items-center gap-3 rounded-lg border-2 border-dashed p-6 text-center ${
            dragging ? 'border-teal-400 bg-teal-400/10' : 'border-white/20'
          }`}
        >
          <p className="font-medium">{t('face.drop')}</p>
          <p className="text-xs text-slate-400">{t('face.formats')}</p>
          <div className="flex flex-wrap justify-center gap-2">
            <button
              type="button"
              className="rounded-lg bg-teal-400 px-3 py-2 text-sm font-semibold text-slate-950"
              disabled={busy}
              onClick={() => inputRef.current?.click()}
            >
              {t('face.choose')}
            </button>
            <button
              type="button"
              data-testid="face-camera"
              className="rounded-lg border border-white/20 px-3 py-2 text-sm"
              disabled={busy}
              onClick={() => void startCamera()}
            >
              {t('face.camera')}
            </button>
          </div>
          <input
            ref={inputRef}
            data-testid="face-file-input"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="sr-only"
            onChange={(event) => {
              onFiles(event.target.files);
              event.target.value = '';
            }}
          />
        </div>
      )}
      {cameraError && (
        <p role="alert" className="text-sm text-red-300">
          {t('face.cameraError')}
        </p>
      )}

      <p data-testid="face-status" data-status={status} role="status" className="text-sm font-medium">
        {t(`face.status.${status}`)}
      </p>
      {error && (
        <p data-testid="face-error" role="alert" className="text-sm text-red-300">
          {t(error)}
        </p>
      )}

      {hints.length > 0 && (
        <ul data-testid="face-hints" className="list-disc pl-5 text-sm text-amber-200">
          {hints.map((hint) => (
            <li key={hint} data-hint={hint}>
              {t(`face.hints.${hint}`)}
            </li>
          ))}
        </ul>
      )}

      {bitmap && (
        <div className="flex flex-col gap-2">
          <canvas
            ref={previewRef}
            data-testid="face-preview"
            role="img"
            aria-label={t('face.previewAlt')}
            className="max-h-96 w-full rounded-lg object-contain"
          />
          {landmarks && (
            <p className="text-xs text-slate-400">
              <span data-testid="face-landmark-count" data-count={landmarks.length / 3}>
                {t('face.landmarks', { count: landmarks.length / 3 })}
              </span>
            </p>
          )}
        </div>
      )}

      {skinToneHex && (
        <p className="flex items-center gap-2 text-sm">
          <span
            data-testid="face-skin-tone"
            data-hex={skinToneHex}
            className="inline-block size-5 rounded-full border border-white/30"
            style={{ backgroundColor: skinToneHex }}
          />
          {t('face.skinTone')}: {skinToneHex}
        </p>
      )}

      {status === 'baked' && (
        <p
          data-testid="face-fit"
          data-rms={fit ? fit.rmsResidual.toFixed(2) : ''}
          data-state={fit === undefined ? 'pending' : fit === null ? 'failed' : 'fitted'}
          className="text-sm text-slate-300"
        >
          {fit === undefined
            ? t('face.fitPending')
            : fit === null
              ? t('face.fitFailed')
              : t('face.fit', { rms: fit.rmsResidual.toFixed(2) })}
        </p>
      )}

      <div className="flex gap-2">
        <button
          type="button"
          data-testid="face-apply"
          className="rounded-lg bg-teal-400 px-3 py-2 text-sm font-semibold text-slate-950 disabled:opacity-50"
          disabled={busy || !bitmap || !landmarks}
          onClick={() => void bake()}
        >
          {t('face.apply')}
        </button>
        <button
          type="button"
          data-testid="face-remove"
          className="rounded-lg border border-white/20 px-3 py-2 text-sm disabled:opacity-50"
          disabled={busy || (!bitmap && status !== 'error')}
          onClick={() => void clear()}
        >
          {t('face.remove')}
        </button>
      </div>
    </section>
  );
}
