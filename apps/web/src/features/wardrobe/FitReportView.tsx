import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type {
  FitVerdict,
  GarmentMeasureId,
  GarmentTemplateDef,
  StoreItemDef,
} from '@dt/avatar-core';
import { useSolveStore } from '../../store/solveStore';
import { useWardrobeStore } from '../../store/wardrobeStore';
import { measuresForKind } from './chartModel';
import {
  analyzeItemFit,
  formatEase,
  isInfoOnly,
  isLengthMeasure,
  VERDICT_STYLE,
} from './fitAnalysis';
import './i18n';

const chipBase =
  'inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium';

type Translate = ReturnType<typeof useTranslation>['t'];

function verdictText(t: Translate, id: GarmentMeasureId, verdict: FitVerdict): string {
  return t(`wardrobe.verdict.${isLengthMeasure(id) ? 'length' : 'fit'}.${verdict}`);
}

/** Verdict chips per region, the overall verdict and the recommended size for the selected size of one item. */
export function FitReportView({
  item,
  template,
}: {
  item: StoreItemDef;
  template: GarmentTemplateDef;
}) {
  const { t } = useTranslation();
  const achievedCm = useSolveStore((state) => state.achievedCm);
  const selectSize = useWardrobeStore((state) => state.selectSize);
  const report = useMemo(
    () => analyzeItemFit(item, template, achievedCm),
    [item, template, achievedCm],
  );
  if (!report) return null;

  const order = measuresForKind(template.kind);
  const regions = [...report.regions].sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id));
  const hasReference = regions.some((r) => isLengthMeasure(r.id) && !isInfoOnly(template, r.id));
  const recommended = report.recommendedSize;

  return (
    <div
      data-testid={`fit-report-${template.category}`}
      data-size={report.size}
      className="mt-3 rounded-xl border border-white/10 bg-slate-950/40 p-3"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h4 className="text-xs font-bold uppercase tracking-[0.14em] text-slate-300">
          {t('wardrobe.fit.title')}
        </h4>
        <span
          data-testid="fit-overall"
          data-verdict={report.overall}
          className={`${chipBase} ${VERDICT_STYLE[report.overall]}`}
        >
          {t('wardrobe.fit.overall')}: {t(`wardrobe.verdict.fit.${report.overall}`)}
        </span>
      </div>

      <ul className="mt-3 flex flex-wrap gap-1.5" aria-label={t('wardrobe.fit.title')}>
        {regions.map((region) => {
          const info = isInfoOnly(template, region.id);
          const name = t(`wardrobe.measure.${region.id}`);
          const value = info
            ? `${Math.round(region.garmentCm * 10) / 10} cm`
            : formatEase(region.easeCm);
          return (
            <li
              key={region.id}
              data-testid={`fit-region-${region.id}`}
              data-verdict={info ? 'info' : region.verdict}
              title={
                info
                  ? undefined
                  : t('wardrobe.fit.regionLabel', {
                      measure: name,
                      ease: formatEase(region.easeCm),
                      verdict: verdictText(t, region.id, region.verdict),
                    })
              }
              className={`${chipBase} ${info ? 'border-white/20 bg-white/5 text-slate-300' : VERDICT_STYLE[region.verdict]}`}
            >
              <span>{name}</span>
              <span className="tabular-nums">{value}</span>
              {!info && (
                <span className="font-semibold">{verdictText(t, region.id, region.verdict)}</span>
              )}
            </li>
          );
        })}
      </ul>
      {hasReference && (
        <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
          {t('wardrobe.fit.lengthNote')}
        </p>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
        {recommended ? (
          <>
            <strong data-testid="fit-recommended" data-size={recommended} className="text-teal-200">
              {t('wardrobe.fit.recommended', { size: recommended })}
            </strong>
            {recommended !== item.selectedSize && (
              <button
                type="button"
                data-testid="fit-use-recommended"
                onClick={() => selectSize(item.id, recommended)}
                className="rounded-lg border border-teal-300/50 px-2.5 py-1 text-xs font-medium text-teal-100 hover:bg-teal-400/10 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400"
              >
                {t('wardrobe.fit.useSize')}
              </button>
            )}
          </>
        ) : (
          <span data-testid="fit-no-recommendation" className="text-xs text-slate-400">
            {t('wardrobe.fit.noRecommendation')}
          </span>
        )}
      </div>
    </div>
  );
}

const LEGEND_STOPS = ['#ff0000', '#ffff00', '#00ff00', '#00ff00', '#00ffff', '#0000ff'];

/** Colour legend of the clearance heatmap (red = touching, green = comfortable, blue = very loose). */
export function HeatmapLegend() {
  const { t } = useTranslation();
  return (
    <div data-testid="heatmap-legend" className="mt-2">
      <p className="text-[11px] text-slate-400">{t('wardrobe.heatmap.legendTitle')}</p>
      <div
        aria-hidden="true"
        className="mt-1 h-2.5 rounded-full border border-white/10"
        style={{ background: `linear-gradient(to right, ${LEGEND_STOPS.join(', ')})` }}
      />
      <div className="mt-1 flex justify-between gap-1 text-[10px] text-slate-400">
        <span>0 mm · {t('wardrobe.heatmap.penetrating')}</span>
        <span>{t('wardrobe.heatmap.snug')} 5</span>
        <span>{t('wardrobe.heatmap.regular')} 10–25</span>
        <span>{t('wardrobe.heatmap.loose')} 40</span>
        <span>60+ {t('wardrobe.heatmap.oversized')}</span>
      </div>
    </div>
  );
}
