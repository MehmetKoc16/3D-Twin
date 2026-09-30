import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { GarmentCategory, GarmentTemplateDef, StoreItemDef } from '@dt/avatar-core';
import { useViewerStore } from '../../store/viewerStore';
import { useWardrobeStore } from '../../store/wardrobeStore';
import { FitReportView, HeatmapLegend } from './FitReportView';
import { StoreItemForm } from './StoreItemForm';
import './i18n';

const CATEGORIES: readonly GarmentCategory[] = ['top', 'bottom', 'shoes'];
const buttonClass =
  'rounded-lg border border-white/15 bg-white/5 px-3 py-2 text-xs font-medium hover:bg-white/10 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400';
const primaryClass =
  'rounded-lg bg-teal-400 px-3 py-2 text-xs font-semibold text-slate-950 hover:bg-teal-300 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-200';
const sectionTitle = 'text-xs font-bold uppercase tracking-[0.16em] text-teal-300';

function LicenseBadge({ template }: { template: GarmentTemplateDef }) {
  const { t } = useTranslation();
  const free = template.license === 'CC0-1.0';
  return (
    <span
      data-testid={`license-${template.id}`}
      tabIndex={0}
      title={
        t('wardrobe.catalog.licenseTitle', { license: t(`wardrobe.license.${template.license}`) }) +
        (template.attribution ? `\n${template.attribution}` : '')
      }
      className={`rounded px-1.5 py-0.5 text-[10px] font-bold tracking-wide focus-visible:outline-2 focus-visible:outline-teal-400 ${free ? 'bg-emerald-400/20 text-emerald-200' : 'bg-amber-400/20 text-amber-100'}`}
    >
      {t(`wardrobe.license.${template.license}`)}
    </span>
  );
}

function ColorDot({ color }: { color: string }) {
  return (
    <span
      aria-hidden="true"
      className="inline-block size-3.5 shrink-0 rounded-full border border-white/30"
      style={{ backgroundColor: color }}
    />
  );
}

function WornCard({
  category,
  item,
  template,
}: {
  category: GarmentCategory;
  item: StoreItemDef;
  template: GarmentTemplateDef;
}) {
  const { t, i18n } = useTranslation();
  const language = i18n.language === 'en' ? 'en' : 'tr';
  const takeOff = useWardrobeStore((state) => state.takeOff);
  const selectSize = useWardrobeStore((state) => state.selectSize);
  return (
    <li
      data-testid={`worn-${category}`}
      className="rounded-xl border border-white/10 bg-white/[0.03] p-3"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-[11px] uppercase tracking-wider text-slate-400">
            {t(`wardrobe.categories.${category}`)}
          </p>
          <p className="flex items-center gap-2 truncate text-sm font-medium">
            <ColorDot color={item.color} />
            <span className="truncate">{item.name}</span>
          </p>
          <p className="text-[11px] text-slate-500">{template.label[language]}</p>
        </div>
        <button
          type="button"
          data-testid={`take-off-${category}`}
          onClick={() => takeOff(category)}
          className={buttonClass}
        >
          {t('wardrobe.worn.takeOff')}
        </button>
      </div>
      <div
        className="mt-3 flex flex-wrap items-center gap-1.5"
        role="group"
        aria-label={t('wardrobe.worn.size')}
      >
        <span className="mr-1 text-xs text-slate-400">{t('wardrobe.worn.size')}</span>
        {item.sizes.map((size) => (
          <button
            key={size}
            type="button"
            data-testid={`size-${category}-${size}`}
            aria-pressed={item.selectedSize === size}
            onClick={() => selectSize(item.id, size)}
            className={`min-w-9 rounded-lg border px-2 py-1 text-xs font-semibold focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400 ${item.selectedSize === size ? 'border-teal-300 bg-teal-400 text-slate-950' : 'border-white/15 bg-white/5 hover:bg-white/10'}`}
          >
            {size}
          </button>
        ))}
      </div>
      <FitReportView item={item} template={template} />
    </li>
  );
}

function ItemRow({
  item,
  template,
  worn,
  onEdit,
}: {
  item: StoreItemDef;
  template: GarmentTemplateDef | undefined;
  worn: boolean;
  onEdit: () => void;
}) {
  const { t, i18n } = useTranslation();
  const language = i18n.language === 'en' ? 'en' : 'tr';
  const wear = useWardrobeStore((state) => state.wear);
  const takeOff = useWardrobeStore((state) => state.takeOff);
  const deleteItem = useWardrobeStore((state) => state.deleteItem);
  const [confirming, setConfirming] = useState(false);
  return (
    <li
      data-testid={`item-${item.id}`}
      data-name={item.name}
      className="rounded-xl border border-white/10 bg-white/[0.03] p-3"
    >
      <div className="flex items-center gap-2">
        <ColorDot color={item.color} />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium">{item.name}</p>
          <p className="truncate text-[11px] text-slate-500">
            {template ? `${template.label[language]} · ` : ''}
            {t('wardrobe.items.sizes', { sizes: item.sizes.join(' / ') })}
          </p>
        </div>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        {template &&
          (worn ? (
            <button
              type="button"
              data-testid={`unwear-${item.id}`}
              onClick={() => takeOff(template.category)}
              className={buttonClass}
            >
              {t('wardrobe.items.takeOff')}
            </button>
          ) : (
            <button
              type="button"
              data-testid={`wear-${item.id}`}
              onClick={() => wear(item.id)}
              className={primaryClass}
            >
              {t('wardrobe.items.wear')}
            </button>
          ))}
        <button type="button" onClick={onEdit} className={buttonClass}>
          {t('wardrobe.items.edit')}
        </button>
        {confirming ? (
          <span
            className="flex items-center gap-1.5 text-xs text-slate-300"
            role="group"
            aria-label={t('wardrobe.items.confirmDelete')}
          >
            {t('wardrobe.items.confirmDelete')}
            <button
              type="button"
              data-testid={`delete-confirm-${item.id}`}
              onClick={() => void deleteItem(item.id)}
              className="rounded-lg border border-red-400/60 bg-red-500/20 px-2.5 py-1.5 text-xs font-medium text-red-100 hover:bg-red-500/30 focus-visible:outline-2 focus-visible:outline-red-300"
            >
              {t('wardrobe.items.yes')}
            </button>
            <button type="button" onClick={() => setConfirming(false)} className={buttonClass}>
              {t('wardrobe.items.no')}
            </button>
          </span>
        ) : (
          <button
            type="button"
            data-testid={`delete-${item.id}`}
            onClick={() => setConfirming(true)}
            className={`${buttonClass} text-red-200`}
          >
            {t('wardrobe.items.delete')}
          </button>
        )}
      </div>
    </li>
  );
}

export function WardrobePanel() {
  const { t, i18n } = useTranslation();
  const language = i18n.language === 'en' ? 'en' : 'tr';
  const templates = useWardrobeStore((state) => state.templates);
  const catalogStatus = useWardrobeStore((state) => state.catalogStatus);
  const items = useWardrobeStore((state) => state.items);
  const worn = useWardrobeStore((state) => state.worn);
  const heatmap = useWardrobeStore((state) => state.heatmap);
  const setHeatmap = useWardrobeStore((state) => state.setHeatmap);
  const saveItem = useWardrobeStore((state) => state.saveItem);
  const wear = useWardrobeStore((state) => state.wear);
  const hydrate = useWardrobeStore((state) => state.hydrate);
  const requestFocus = useViewerStore((state) => state.requestFocus);
  const [form, setForm] = useState<{ templateId: string; editing?: StoreItemDef } | null>(null);

  useEffect(() => {
    void hydrate();
  }, [hydrate]);

  const templateOf = (id: string): GarmentTemplateDef | undefined =>
    templates.find((x) => x.id === id);
  const wornEntries = CATEGORIES.flatMap((category) => {
    const item = items.find((i) => i.id === worn[category]);
    const template = item ? templateOf(item.templateId) : undefined;
    return item && template ? [{ category, item, template }] : [];
  });

  const openForm = (templateId: string, editing?: StoreItemDef): void => {
    setForm(editing ? { templateId, editing } : { templateId });
  };
  const handleSave = (item: StoreItemDef, wearNow: boolean): void => {
    void saveItem(item).then(() => {
      if (wearNow) wear(item.id);
    });
    setForm(null);
  };

  return (
    <section
      data-testid="wardrobe-panel"
      className="flex flex-col gap-5"
      aria-labelledby="wardrobe-title"
    >
      <header>
        <h2 id="wardrobe-title" className="text-lg font-semibold">
          {t('wardrobe.title')}
        </h2>
        <p className="text-sm text-slate-400">{t('wardrobe.description')}</p>
      </header>

      {(catalogStatus === 'error' || (catalogStatus === 'ready' && templates.length === 0)) && (
        <p
          role="alert"
          data-testid="wardrobe-load-error"
          className="rounded-lg border border-amber-400/40 bg-amber-400/10 p-3 text-sm text-amber-100"
        >
          {t('wardrobe.loadError')}
        </p>
      )}
      {catalogStatus === 'loading' && (
        <p role="status" className="text-sm text-slate-400">
          {t('wardrobe.loading')}
        </p>
      )}

      <div>
        <h3 className={sectionTitle}>{t('wardrobe.worn.title')}</h3>
        {wornEntries.length === 0 ? (
          <p data-testid="worn-empty" className="mt-2 text-sm text-slate-400">
            {t('wardrobe.worn.empty')}
          </p>
        ) : (
          <>
            <ul className="mt-2 flex flex-col gap-3">
              {wornEntries.map((entry) => (
                <WornCard key={entry.category} {...entry} />
              ))}
            </ul>
            <label className="mt-3 flex items-center gap-2 text-sm text-slate-200">
              <input
                data-testid="heatmap-toggle"
                type="checkbox"
                checked={heatmap}
                onChange={(event) => setHeatmap(event.target.checked)}
                className="accent-teal-400 focus-visible:outline-2 focus-visible:outline-teal-400"
              />
              {t('wardrobe.heatmap.toggle')}
            </label>
            {heatmap && <HeatmapLegend />}
            {worn.shoes && (
              <div
                data-testid="shoes-hint"
                className="mt-3 flex flex-wrap items-center gap-2 rounded-lg border border-sky-400/30 bg-sky-400/10 p-2.5 text-xs text-sky-100"
              >
                <span className="min-w-0 flex-1">{t('wardrobe.shoesHint')}</span>
                <button
                  type="button"
                  data-testid="focus-feet"
                  onClick={() => requestFocus('feet')}
                  className={buttonClass}
                >
                  {t('wardrobe.focusFeet')}
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {form ? (
        <StoreItemForm
          key={form.editing?.id ?? `new-${form.templateId}`}
          templates={templates}
          initialTemplateId={form.templateId}
          {...(form.editing ? { editing: form.editing } : {})}
          onSave={handleSave}
          onCancel={() => setForm(null)}
        />
      ) : (
        <div>
          <div className="flex items-center justify-between gap-2">
            <h3 className={sectionTitle}>{t('wardrobe.items.title')}</h3>
            <button
              type="button"
              data-testid="wardrobe-add"
              disabled={templates.length === 0}
              onClick={() => openForm(templates[0]?.id ?? '')}
              className={`${primaryClass} disabled:opacity-40`}
            >
              {t('wardrobe.items.add')}
            </button>
          </div>
          {items.length === 0 ? (
            <p className="mt-2 text-sm text-slate-400">{t('wardrobe.items.empty')}</p>
          ) : (
            <ul data-testid="item-list" className="mt-2 flex flex-col gap-2">
              {items.map((item) => {
                const template = templateOf(item.templateId);
                return (
                  <ItemRow
                    key={item.id}
                    item={item}
                    template={template}
                    worn={template ? worn[template.category] === item.id : false}
                    onEdit={() => openForm(item.templateId, item)}
                  />
                );
              })}
            </ul>
          )}
        </div>
      )}

      <div>
        <h3 className={sectionTitle}>{t('wardrobe.catalog.title')}</h3>
        <p className="mt-1 text-xs text-slate-500">{t('wardrobe.catalog.description')}</p>
        <ul data-testid="catalog" className="mt-2 grid grid-cols-1 gap-2 min-[420px]:grid-cols-2">
          {templates.map((template) => (
            <li
              key={template.id}
              data-testid={`template-${template.id}`}
              className="flex flex-col gap-2 rounded-xl border border-white/10 bg-white/[0.03] p-3"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-medium">{template.label[language]}</span>
                <LicenseBadge template={template} />
              </div>
              <p className="text-[11px] text-slate-500">
                {t(`wardrobe.categories.${template.category}`)}
              </p>
              <details className="text-[11px] text-slate-400">
                <summary className="cursor-pointer focus-visible:outline-2 focus-visible:outline-teal-400">
                  {t('wardrobe.catalog.attribution')}
                </summary>
                <p className="mt-1 break-words leading-relaxed">
                  {template.attribution ?? t('wardrobe.catalog.noAttribution')}
                </p>
              </details>
              <button
                type="button"
                data-testid={`use-template-${template.id}`}
                onClick={() => openForm(template.id)}
                className={`${buttonClass} mt-auto`}
              >
                {t('wardrobe.catalog.use')}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
