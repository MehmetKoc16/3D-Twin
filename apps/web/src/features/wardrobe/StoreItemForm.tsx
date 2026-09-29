import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { GarmentMeasureId, GarmentTemplateDef, StoreItemDef } from '@dt/avatar-core';
import {
  addSize,
  createDraft,
  draftToItem,
  euRow,
  isGirth,
  itemToDraft,
  MAX_SIZES,
  measuresForKind,
  removeSize,
  renameSize,
  requiredMeasures,
  setCell,
  type ChartDraft,
  type ShoeMode,
  type ValidationIssue,
} from './chartModel';
import { swatchesFromImageFile } from './colorExtract';
import './i18n';

const inputClass =
  'w-full rounded-md border border-white/15 bg-slate-900 px-2 py-1.5 text-sm text-white placeholder:text-slate-600 focus-visible:outline-2 focus-visible:outline-teal-400 aria-[invalid=true]:border-red-400';
const buttonClass =
  'rounded-lg border border-white/15 bg-white/5 px-3 py-2 text-xs font-medium hover:bg-white/10 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400';
const primaryClass =
  'rounded-lg bg-teal-400 px-3 py-2 text-xs font-semibold text-slate-950 hover:bg-teal-300 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-200';

interface Props {
  templates: GarmentTemplateDef[];
  /** Template preselected for a new item. */
  initialTemplateId: string;
  /** The item being edited (undefined: new item). */
  editing?: StoreItemDef;
  onSave: (item: StoreItemDef, wear: boolean) => void;
  onCancel: () => void;
}

export function StoreItemForm({ templates, initialTemplateId, editing, onSave, onCancel }: Props) {
  const { t, i18n } = useTranslation();
  const language = i18n.language === 'en' ? 'en' : 'tr';
  const fileInput = useRef<HTMLInputElement>(null);
  const [draft, setDraft] = useState<ChartDraft>(() => {
    const template = templates.find((x) => x.id === (editing?.templateId ?? initialTemplateId)) ?? templates[0];
    if (!template) throw new Error('wardrobe form opened without templates');
    return editing ? itemToDraft(editing, template) : createDraft(template);
  });
  const [issues, setIssues] = useState<ValidationIssue[]>([]);
  const [swatches, setSwatches] = useState<string[]>([]);
  const [colorFailed, setColorFailed] = useState(false);
  const [newSize, setNewSize] = useState('');

  const template = templates.find((x) => x.id === draft.templateId);
  const measures = template ? measuresForKind(template.kind) : [];
  const required = template ? requiredMeasures(template.kind) : [];
  const isShoe = template?.category === 'shoes';
  const hasGirth = measures.some(isGirth);
  const euMode = isShoe && draft.shoeMode === 'eu';
  const invalidCells = new Set(
    issues.filter((i) => i.measure && i.size !== undefined).map((i) => `${i.measure}|${i.size}`),
  );
  const invalidRows = new Set(issues.filter((i) => i.measure && i.size === undefined).map((i) => i.measure));

  const update = (next: ChartDraft): void => {
    setDraft(next);
    if (issues.length > 0) setIssues([]);
  };

  const changeTemplate = (id: string): void => {
    const next = templates.find((x) => x.id === id);
    if (!next) return;
    // a different kind has different rows: start a fresh chart but keep name, link and colour choices
    const fresh = createDraft(next);
    const keepChart = template && next.kind === template.kind;
    update(keepChart ? { ...draft, templateId: id } : { ...fresh, name: draft.name, storeUrl: draft.storeUrl, color: next.baseColor });
  };

  const pickPhoto = async (file: File | undefined): Promise<void> => {
    if (!file) return;
    setColorFailed(false);
    try {
      const found = await swatchesFromImageFile(file);
      if (found.length === 0) throw new Error('no colour');
      setSwatches(found);
      update({ ...draft, color: found[0]! });
    } catch {
      setSwatches([]);
      setColorFailed(true);
    }
  };

  const submit = (wear: boolean): void => {
    const result = draftToItem(draft, template, editing?.id);
    if (!result.ok) {
      setIssues(result.issues);
      return;
    }
    onSave(result.item, wear);
  };

  const issueText = (issue: ValidationIssue): string =>
    t(`wardrobe.errors.${issue.code}`, {
      ...issue.params,
      measure: issue.measure ? t(`wardrobe.measure.${issue.measure}`) : '',
      size: issue.size ?? '',
    });

  const rowLabel = (id: GarmentMeasureId): string =>
    isShoe && id === 'footLength' ? t('wardrobe.measure.footLength') : t(`wardrobe.measure.${id}`);
  const rowValues = (id: GarmentMeasureId): string[] =>
    id === 'footLength' && euMode ? euRow(draft.sizes) : (draft.chart[id] ?? []);

  return (
    <form
      data-testid="wardrobe-form"
      noValidate
      onSubmit={(event) => {
        event.preventDefault();
        submit(false);
      }}
      className="rounded-xl border border-teal-400/30 bg-slate-900/60 p-4"
    >
      <h3 className="text-sm font-semibold text-teal-200">{t(editing ? 'wardrobe.form.editTitle' : 'wardrobe.form.addTitle')}</h3>

      <label className="mt-3 block text-xs text-slate-300">
        {t('wardrobe.form.name')}
        <input
          data-testid="form-name"
          type="text"
          value={draft.name}
          maxLength={80}
          placeholder={t('wardrobe.form.namePlaceholder')}
          aria-invalid={issues.some((i) => i.code === 'nameRequired' || i.code === 'nameTooLong')}
          onChange={(event) => update({ ...draft, name: event.target.value })}
          className={`${inputClass} mt-1`}
        />
      </label>

      <label className="mt-3 block text-xs text-slate-300">
        {t('wardrobe.form.storeUrl')}
        <input
          data-testid="form-url"
          type="text"
          inputMode="url"
          value={draft.storeUrl}
          aria-invalid={issues.some((i) => i.code === 'urlInvalid')}
          onChange={(event) => update({ ...draft, storeUrl: event.target.value })}
          className={`${inputClass} mt-1`}
        />
        <span className="mt-1 block text-[11px] text-slate-500">{t('wardrobe.form.storeUrlHelp')}</span>
      </label>

      <label className="mt-3 block text-xs text-slate-300">
        {t('wardrobe.form.template')}
        <select
          data-testid="form-template"
          value={draft.templateId}
          onChange={(event) => changeTemplate(event.target.value)}
          className={`${inputClass} mt-1`}
        >
          {templates.map((x) => (
            <option key={x.id} value={x.id}>
              {x.label[language]} · {t(`wardrobe.categories.${x.category}`)}
            </option>
          ))}
        </select>
      </label>

      <fieldset className="mt-3">
        <legend className="text-xs text-slate-300">{t('wardrobe.form.color')}</legend>
        <div className="mt-1 flex flex-wrap items-center gap-2">
          <input
            data-testid="form-color"
            type="color"
            value={/^#[0-9a-fA-F]{6}$/.test(draft.color) ? draft.color : '#808080'}
            aria-label={t('wardrobe.form.color')}
            onChange={(event) => update({ ...draft, color: event.target.value })}
            className="h-9 w-12 cursor-pointer rounded border border-white/20 bg-transparent p-0.5"
          />
          <input
            data-testid="form-color-hex"
            type="text"
            value={draft.color}
            maxLength={7}
            aria-label={t('wardrobe.form.colorHex')}
            aria-invalid={issues.some((i) => i.code === 'colorInvalid')}
            onChange={(event) => update({ ...draft, color: event.target.value })}
            className={`${inputClass} w-24 font-mono`}
          />
          <button type="button" data-testid="form-color-photo" className={buttonClass} onClick={() => fileInput.current?.click()}>
            {t('wardrobe.form.pickFromPhoto')}
          </button>
          <input
            ref={fileInput}
            data-testid="form-color-file"
            type="file"
            accept="image/*"
            className="sr-only"
            tabIndex={-1}
            aria-label={t('wardrobe.form.pickFromPhoto')}
            onChange={(event) => {
              void pickPhoto(event.target.files?.[0]);
              event.target.value = '';
            }}
          />
        </div>
        <p className="mt-1 text-[11px] text-slate-500">{t('wardrobe.form.photoPrivacy')}</p>
        {swatches.length > 0 && (
          <div className="mt-2 flex items-center gap-2" role="group" aria-label={t('wardrobe.form.swatches')}>
            {swatches.map((hex) => (
              <button
                key={hex}
                type="button"
                title={hex}
                aria-label={hex}
                aria-pressed={draft.color.toLowerCase() === hex}
                onClick={() => update({ ...draft, color: hex })}
                style={{ backgroundColor: hex }}
                className={`size-7 rounded-full border-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400 ${draft.color.toLowerCase() === hex ? 'border-white' : 'border-white/20'}`}
              />
            ))}
          </div>
        )}
        {colorFailed && (
          <p role="alert" className="mt-1 text-xs text-amber-200">
            {t('wardrobe.form.colorFailed')}
          </p>
        )}
      </fieldset>

      <fieldset className="mt-4">
        <legend className="text-xs font-bold uppercase tracking-[0.14em] text-teal-300">{t('wardrobe.form.chart')}</legend>
        <p className="mt-1 text-[11px] text-slate-500">{t('wardrobe.form.chartHelp')}</p>

        {isShoe && (
          <div className="mt-2 text-xs text-slate-300" role="radiogroup" aria-label={t('wardrobe.form.shoeMode')}>
            {(['eu', 'cm'] as ShoeMode[]).map((mode) => (
              <label key={mode} className="mb-1 flex items-center gap-2">
                <input
                  type="radio"
                  name="shoe-mode"
                  data-testid={`shoe-mode-${mode}`}
                  checked={draft.shoeMode === mode}
                  onChange={() => update({ ...draft, shoeMode: mode })}
                  className="accent-teal-400"
                />
                {t(mode === 'eu' ? 'wardrobe.form.shoeEu' : 'wardrobe.form.shoeCm')}
              </label>
            ))}
            <p className="text-[11px] text-slate-500">{t('wardrobe.form.shoeHelp')}</p>
          </div>
        )}

        <div className="mt-2 overflow-x-auto">
          <table className="w-full min-w-[320px] border-separate border-spacing-x-1 border-spacing-y-1 text-xs">
            <thead>
              <tr>
                <th scope="col" className="w-20 text-left font-medium text-slate-400" />
                {draft.sizes.map((size, index) => (
                  <th key={index} scope="col" className="min-w-14 align-bottom">
                    <div className="flex items-center gap-1">
                      <input
                        data-testid={`size-label-${index}`}
                        type="text"
                        value={size}
                        maxLength={8}
                        aria-label={t('wardrobe.form.sizeLabel', { index: index + 1 })}
                        onChange={(event) => update(renameSize(draft, index, event.target.value))}
                        className={`${inputClass} px-1.5 text-center font-semibold`}
                      />
                      {draft.sizes.length > 1 && (
                        <button
                          type="button"
                          aria-label={t('wardrobe.form.removeSize', { size })}
                          onClick={() => update(removeSize(draft, index))}
                          className="grid size-5 shrink-0 place-items-center rounded-full text-slate-400 hover:bg-white/10 hover:text-white focus-visible:outline-2 focus-visible:outline-teal-400"
                        >
                          ×
                        </button>
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {measures.map((id) => (
                <tr key={id}>
                  <th scope="row" className="text-left font-medium text-slate-300">
                    {rowLabel(id)}
                    {required.includes(id) && <span aria-hidden="true" className="text-teal-300"> *</span>}
                  </th>
                  {draft.sizes.map((size, index) => {
                    const native = template?.nativeMeasures[id];
                    const readonlyCell = id === 'footLength' && euMode;
                    return (
                      <td key={index}>
                        <input
                          data-testid={`cell-${id}-${index}`}
                          type="text"
                          inputMode="decimal"
                          value={rowValues(id)[index] ?? ''}
                          readOnly={readonlyCell}
                          placeholder={native === undefined ? '' : String(Math.round(native))}
                          aria-label={t('wardrobe.form.cell', { measure: rowLabel(id), size })}
                          aria-invalid={invalidCells.has(`${id}|${size.trim()}`) || invalidRows.has(id)}
                          onChange={(event) => update(setCell(draft, id, index, event.target.value))}
                          className={`${inputClass} px-1.5 text-center ${readonlyCell ? 'opacity-70' : ''}`}
                        />
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-2 flex items-center gap-2">
          <input
            data-testid="form-new-size"
            type="text"
            value={newSize}
            maxLength={8}
            placeholder={t('wardrobe.form.newSizePlaceholder')}
            aria-label={t('wardrobe.form.addSize')}
            onChange={(event) => setNewSize(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault();
                if (newSize.trim()) {
                  update(addSize(draft, newSize.trim()));
                  setNewSize('');
                }
              }
            }}
            className={`${inputClass} w-24`}
          />
          <button
            type="button"
            data-testid="form-add-size"
            disabled={draft.sizes.length >= MAX_SIZES || newSize.trim().length === 0}
            onClick={() => {
              update(addSize(draft, newSize.trim()));
              setNewSize('');
            }}
            className={`${buttonClass} disabled:opacity-40`}
          >
            {t('wardrobe.form.addSize')}
          </button>
        </div>

        {hasGirth && (
          <label className="mt-3 flex items-start gap-2 text-xs text-slate-300">
            <input
              data-testid="form-flat"
              type="checkbox"
              checked={draft.flat}
              onChange={(event) => update({ ...draft, flat: event.target.checked })}
              className="mt-0.5 accent-teal-400"
            />
            <span>{t('wardrobe.form.flat')}</span>
          </label>
        )}

        <label className="mt-3 block text-xs text-slate-300">
          {t('wardrobe.form.selectedSize')}
          <select
            data-testid="form-selected-size"
            value={draft.selectedSize}
            onChange={(event) => update({ ...draft, selectedSize: event.target.value })}
            className={`${inputClass} mt-1 w-32`}
          >
            {draft.sizes.map((size, index) => (
              <option key={index} value={size}>
                {size}
              </option>
            ))}
          </select>
        </label>
      </fieldset>

      {issues.length > 0 && (
        <div role="alert" data-testid="form-errors" className="mt-4 rounded-lg border border-red-400/40 bg-red-500/10 p-3 text-xs text-red-100">
          <p className="font-semibold">{t('wardrobe.form.problems')}</p>
          <ul className="mt-1 list-disc space-y-0.5 pl-4">
            {issues.map((issue, index) => (
              <li key={index}>{issueText(issue)}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="mt-4 flex flex-wrap gap-2">
        <button type="submit" data-testid="form-save" className={primaryClass}>
          {t('wardrobe.form.save')}
        </button>
        <button type="button" data-testid="form-save-wear" className={buttonClass} onClick={() => submit(true)}>
          {t('wardrobe.form.saveAndWear')}
        </button>
        <button type="button" data-testid="form-cancel" className={buttonClass} onClick={onCancel}>
          {t('wardrobe.form.cancel')}
        </button>
      </div>
    </form>
  );
}
