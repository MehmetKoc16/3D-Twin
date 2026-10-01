import { useTranslation } from 'react-i18next';
import { isTwinActive, useTwinStore, type ModelMode } from '../../store/twinStore';
import './i18n';

const chip =
  'flex-1 rounded-lg border px-3 py-2 text-xs font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400';

/**
 * "Model" switch above the panel tabs: standard mannequin or the user's realistic twin. Without loaded twin files the
 * twin button sends the user to the twin tab instead of switching.
 */
export function ModelSwitch({ onNeedFiles }: { onNeedFiles: () => void }) {
  const { t } = useTranslation();
  const mode = useTwinStore((state) => state.mode);
  const hasPack = useTwinStore((state) => state.pack !== null);
  const active = useTwinStore(isTwinActive);
  const setMode = useTwinStore((state) => state.setMode);
  const choose = (next: ModelMode): void => {
    if (next === 'twin' && !hasPack) onNeedFiles();
    else setMode(next);
  };
  return (
    <section className="mb-4" aria-label={t('twin.model.label')} data-testid="model-switch">
      <h3 className="text-xs font-bold uppercase tracking-[0.16em] text-teal-300">
        {t('twin.model.label')}
      </h3>
      <div className="mt-2 flex gap-2" role="group" aria-label={t('twin.model.label')}>
        {(['standard', 'twin'] as const).map((value) => (
          <button
            key={value}
            type="button"
            data-testid={`model-${value}`}
            aria-pressed={mode === value}
            onClick={() => choose(value)}
            className={`${chip} ${mode === value ? 'border-teal-300 bg-teal-400 text-slate-950' : 'border-white/15 bg-white/5 hover:bg-white/10'}`}
          >
            {t(`twin.model.${value}`)}
          </button>
        ))}
      </div>
      {!hasPack && (
        <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
          {t('twin.model.needFiles')}
        </p>
      )}
      {active && (
        <p
          data-testid="model-twin-note"
          className="mt-2 text-[11px] leading-relaxed text-slate-400"
        >
          {t('twin.model.twinActive')}
        </p>
      )}
    </section>
  );
}
