import { useRef, useState } from 'react';
import type { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { isTwinActive, useTwinStore } from '../../store/twinStore';
import './i18n';

const buttonClass =
  'rounded-lg border border-white/15 bg-white/5 px-3 py-2 text-xs font-medium hover:bg-white/10 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400';
const primaryClass =
  'rounded-lg bg-teal-400 px-3 py-2 text-xs font-semibold text-slate-950 hover:bg-teal-300 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-200';

function formatBytes(bytes: number): string {
  return bytes >= 1024 * 1024
    ? `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

interface PickedFile {
  name: string;
  size?: number;
}

function FileRow({
  testId,
  label,
  file,
  optional,
}: {
  testId: string;
  label: string;
  file: PickedFile | null;
  optional?: boolean;
}) {
  const { t } = useTranslation();
  return (
    <li
      data-testid={testId}
      data-present={file !== null}
      className="flex items-baseline justify-between gap-3 py-1.5 text-xs"
    >
      <span className="text-slate-300">
        {label}
        {optional && <span className="ml-1 text-slate-500">({t('twin.panel.optional')})</span>}
      </span>
      <span className={`truncate ${file ? 'text-teal-200' : 'text-slate-500'}`}>
        {file
          ? `${file.name}${file.size !== undefined ? ` · ${formatBytes(file.size)}` : ''}`
          : t('twin.panel.notSelected')}
      </span>
    </li>
  );
}

/** Twin tab: pick rigged.glb + twin.json (+ mh2twin.bin), see the status, remove. Nothing leaves the browser. */
export function TwinPanel() {
  const { t, i18n } = useTranslation();
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const { status, error, pack, pending, runtime, loadFiles, remove } = useTwinStore();
  const active = useTwinStore(isTwinActive);
  const onPick = async (event: ChangeEvent<HTMLInputElement>): Promise<void> => {
    const files = Array.from(event.target.files ?? []);
    event.target.value = '';
    if (files.length === 0) return;
    setBusy(true);
    try {
      await loadFiles(files);
    } finally {
      setBusy(false);
    }
  };
  const glb: PickedFile | null = pack
    ? { name: pack.names.glb, size: pack.glb.byteLength }
    : pending.glb
      ? { name: pending.glb.name, size: pending.glb.buffer.byteLength }
      : null;
  const json: PickedFile | null = pack
    ? { name: pack.names.json }
    : pending.json
      ? { name: pending.json.name }
      : null;
  const mapping: PickedFile | null = pack
    ? pack.names.mapping
      ? { name: pack.names.mapping, ...(pack.mapping ? { size: pack.mapping.byteLength } : {}) }
      : null
    : pending.mapping
      ? { name: pending.mapping.name, size: pending.mapping.buffer.byteLength }
      : null;
  return (
    <div data-testid="twin-panel">
      <h2 className="text-lg font-semibold tracking-tight">{t('twin.panel.title')}</h2>
      <p className="mt-2 text-xs leading-relaxed text-slate-400">{t('twin.panel.description')}</p>
      <p
        data-testid="twin-privacy"
        className="mt-3 rounded-lg border border-teal-400/25 bg-teal-400/5 p-2.5 text-[11px] leading-relaxed text-teal-100/90"
      >
        {t('twin.panel.privacy')}
      </p>
      <div className="mt-4 flex flex-wrap items-center gap-2">
        <button
          type="button"
          data-testid="twin-pick"
          className={primaryClass}
          onClick={() => input.current?.click()}
          disabled={busy}
        >
          {t('twin.panel.pick')}
        </button>
        <input
          ref={input}
          type="file"
          multiple
          accept=".glb,.json,.bin,model/gltf-binary,application/json,application/octet-stream"
          data-testid="twin-file-input"
          aria-label={t('twin.panel.pick')}
          className="sr-only"
          onChange={(event) => void onPick(event)}
        />
        {pack && (
          <button
            type="button"
            data-testid="twin-remove"
            className={buttonClass}
            onClick={() => void remove()}
          >
            {t('twin.panel.remove')}
          </button>
        )}
      </div>
      <p className="mt-2 text-[11px] leading-relaxed text-slate-500">{t('twin.panel.pickHint')}</p>
      <ul className="mt-3 divide-y divide-white/5 rounded-lg border border-white/10 px-3">
        <FileRow testId="twin-file-glb" label={t('twin.panel.files.glb')} file={glb} />
        <FileRow testId="twin-file-json" label={t('twin.panel.files.json')} file={json} />
        <FileRow
          testId="twin-file-mapping"
          label={t('twin.panel.files.mapping')}
          file={mapping}
          optional
        />
      </ul>
      <p
        role="status"
        data-testid="twin-status"
        data-status={status}
        data-active={active}
        className={`mt-3 text-sm ${status === 'error' ? 'text-amber-300' : 'text-slate-200'}`}
      >
        {busy ? '…' : t(`twin.panel.status.${status}`)}
      </p>
      {error && (
        <p
          role="alert"
          data-testid="twin-error"
          data-code={error.code}
          className="mt-1 text-xs leading-relaxed text-amber-200"
        >
          {t(`twin.errors.${error.code}`)}
          {error.detail && (
            <span className="mt-0.5 block text-[11px] text-amber-100/60">{error.detail}</span>
          )}
        </p>
      )}
      {pack && !pack.mapping && (
        <p
          data-testid="twin-no-mapping"
          className="mt-2 text-[11px] leading-relaxed text-amber-200/80"
        >
          {t('twin.panel.noMapping')}
        </p>
      )}
      {active && runtime && (
        <div
          data-testid="twin-runtime"
          data-hidden={runtime.hiddenTriangles}
          className="mt-3 text-[11px] leading-relaxed text-slate-400"
        >
          <p>
            {t('twin.panel.stats', {
              vertices: runtime.vertices.toLocaleString(i18n.language),
              triangles: runtime.triangles.toLocaleString(i18n.language),
              hidden: runtime.hiddenTriangles.toLocaleString(i18n.language),
            })}
          </p>
          <p>{t('twin.panel.alignment', { residual: runtime.residualMm.toFixed(2) })}</p>
        </div>
      )}
      <p className="mt-4 text-[11px] leading-relaxed text-slate-500">{t('twin.panel.howTo')}</p>
    </div>
  );
}
