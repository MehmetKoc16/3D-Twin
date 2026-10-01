import { useTranslation } from 'react-i18next';
import { useTwinStore } from '../../store/twinStore';
import { TWIN_MEASURES } from './twinDef';
import './i18n';

/** Read-only measurements of the twin (twin.json), shown instead of the body sliders in twin mode. */
export function TwinMeasurements() {
  const { t } = useTranslation();
  const def = useTwinStore((state) => state.pack?.def);
  if (!def) return null;
  return (
    <section className="mt-5" data-testid="twin-measurements">
      <h3 className="text-xs font-bold uppercase tracking-[0.16em] text-teal-300">
        {t('twin.measures.title')}
      </h3>
      <p
        data-testid="twin-measurements-note"
        className="mt-2 text-xs leading-relaxed text-slate-400"
      >
        {t('twin.measures.note')}
      </p>
      <dl className="mt-3">
        {TWIN_MEASURES.map(({ id, field }) => {
          const value = def.measurementsCm[id];
          if (value === undefined) return null;
          return (
            <div
              key={id}
              className="flex items-baseline justify-between border-b border-white/6 py-2 text-sm last:border-b-0"
            >
              <dt className="font-medium text-slate-200">{t(`measure.${field}.label`)}</dt>
              <dd data-testid={`twin-measure-${id}`} className="tabular-nums text-white">
                {value.toFixed(1)} <span className="text-xs text-slate-400">cm</span>
              </dd>
            </div>
          );
        })}
      </dl>
      {def.measurementsRawCm && (
        <details className="mt-3">
          <summary className="cursor-pointer text-xs text-slate-400 focus-visible:outline-2 focus-visible:outline-teal-400">
            {t('twin.measures.details')}
          </summary>
          <table
            className="mt-2 w-full text-xs text-slate-300"
            data-testid="twin-measurements-details"
          >
            <thead>
              <tr className="text-slate-500">
                <th className="text-left font-normal" />
                <th className="text-right font-normal">{t('twin.measures.person')}</th>
                <th className="text-right font-normal">{t('twin.measures.body')}</th>
                <th className="text-right font-normal">{t('twin.measures.allowance')}</th>
              </tr>
            </thead>
            <tbody>
              {TWIN_MEASURES.map(({ id, field }) => {
                const person = def.measurementsCm[id];
                const raw = def.measurementsRawCm?.[id];
                if (person === undefined || raw === undefined) return null;
                return (
                  <tr key={id}>
                    <td>{t(`measure.${field}.label`)}</td>
                    <td className="text-right tabular-nums">{person.toFixed(1)}</td>
                    <td className="text-right tabular-nums">{raw.toFixed(1)}</td>
                    <td className="text-right tabular-nums">{(raw - person).toFixed(1)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </details>
      )}
    </section>
  );
}
