import { useRef, useState } from 'react';
import type { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { convertShoeSize, footLengthCmFromShoe, roundShoeSize } from '@dt/avatar-core';
import type { BodyParams, MeasureId, ShoeSystem } from '@dt/avatar-core';
import { skinTones, useAppearanceStore } from '../../store/appearanceStore';
import { useBodyStore, isBodyParams } from '../../store/bodyStore';
import { useSolveStore } from '../../store/solveStore';

type NumericField = Exclude<keyof BodyParams, 'shoe'>;
interface FieldDef { key: NumericField; min: number; max: number; step?: number; unit: string; optional?: boolean }
const basics: FieldDef[] = [
  { key: 'gender', min: 0, max: 1, step: 0.01, unit: '' },
  { key: 'heightCm', min: 140, max: 210, unit: 'cm' },
  { key: 'weightKg', min: 40, max: 150, unit: 'kg' },
];
const upper: FieldDef[] = [
  { key: 'shoulderCm', min: 35, max: 60, unit: 'cm' },
  { key: 'neckCm', min: 30, max: 50, unit: 'cm' },
  { key: 'chestCm', min: 70, max: 140, unit: 'cm' },
];
const lower: FieldDef[] = [
  { key: 'waistCm', min: 55, max: 140, unit: 'cm' },
  { key: 'hipCm', min: 70, max: 150, unit: 'cm' },
];
const advanced: FieldDef[] = [
  { key: 'thighCm', min: 35, max: 85, unit: 'cm', optional: true },
  { key: 'upperArmCm', min: 20, max: 55, unit: 'cm', optional: true },
  { key: 'armLengthCm', min: 45, max: 85, unit: 'cm', optional: true },
  { key: 'inseamCm', min: 55, max: 105, unit: 'cm', optional: true },
];
/** Panel field -> solver measure id (gender and weight have no measure; weight shows the estimated mass). */
const measureOf: Partial<Record<NumericField, MeasureId>> = {
  heightCm: 'height', shoulderCm: 'shoulder', neckCm: 'neck', chestCm: 'chest', waistCm: 'waist', hipCm: 'hip',
  thighCm: 'thigh', upperArmCm: 'upperArm', armLengthCm: 'armLength', inseamCm: 'inseam',
};
const systems: ShoeSystem[] = ['EU', 'US_M', 'US_W', 'UK'];
const buttonClass = 'rounded-lg border border-white/15 bg-white/5 px-3 py-2 text-xs font-medium hover:bg-white/10 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400';

function WarningIcon() {
  return <svg viewBox="0 0 20 20" className="size-4 fill-amber-400" aria-hidden="true">
    <path d="M10 2 1 18h18L10 2Zm-1 6h2v5H9V8Zm0 6h2v2H9v-2Z" />
  </svg>;
}

/** "≈ 99.8 cm" as reported by the solver, plus a warning when the target could not be reached. */
function Achieved({ field }: { field: FieldDef }) {
  const { t } = useTranslation();
  const id = measureOf[field.key];
  const achieved = useSolveStore((state) => (id ? state.achievedCm[id] : undefined));
  const unreachable = useSolveStore((state) => (id ? state.unreachable.includes(id) : false));
  const mass = useSolveStore((state) => state.estimatedMassKg);
  const isWeight = field.key === 'weightKg';
  const value = isWeight ? mass : achieved;
  if (value === null || value === undefined) return null;
  const text = `≈ ${value.toFixed(1)} ${isWeight ? 'kg' : 'cm'}`;
  return <div className="mt-1 flex items-center justify-end gap-1.5 text-xs text-slate-400" data-testid={`achieved-${field.key}`}>
    <span className={unreachable ? 'text-amber-300' : ''}>{isWeight ? `${t('measure.estimatedMass')} ${text}` : `${t('measure.achieved')} ${text}`}</span>
    {unreachable && <span className="group relative inline-flex">
      <button type="button" aria-label={t('measure.unreachable.label')} aria-describedby={`warn-${field.key}`}
        className="grid size-5 place-items-center rounded-full focus-visible:outline-2 focus-visible:outline-teal-400"><WarningIcon /></button>
      <span id={`warn-${field.key}`} role="tooltip" className="pointer-events-none absolute bottom-full right-0 z-20 mb-2 hidden w-56 rounded-lg border border-amber-400/40 bg-slate-800 p-2 text-left text-xs leading-relaxed text-slate-100 shadow-xl group-hover:block group-focus-within:block">
        {t('measure.unreachable.tip', { value: value.toFixed(1) })}
      </span>
    </span>}
  </div>;
}

function MeasurementField({ field }: { field: FieldDef }) {
  const { t } = useTranslation();
  const value = useBodyStore((state) => state.params[field.key]);
  const setField = useBodyStore((state) => state.setField);
  const manual = value !== undefined;
  const display = value ?? Math.round((field.min + field.max) / 2);
  const update = (next: number) => { if (Number.isFinite(next)) setField(field.key, Math.min(field.max, Math.max(field.min, next))); };
  return <div className="border-b border-white/6 py-3 last:border-b-0">
    <div className="mb-2 flex items-center gap-1.5 text-sm">
      <label htmlFor={`number-${field.key}`} className="font-medium text-slate-200">{t(`measure.${field.key}.label`)}</label>
      <span className="group relative inline-flex">
        <button type="button" aria-label={t('measure.howToMeasure', { name: t(`measure.${field.key}.label`) })}
          aria-describedby={`tip-${field.key}`}
          className="grid size-4 place-items-center rounded-full border border-slate-500 text-[10px] text-slate-400 focus-visible:outline-2 focus-visible:outline-teal-400">?</button>
        <span id={`tip-${field.key}`} role="tooltip" className="pointer-events-none absolute bottom-full left-0 z-20 mb-2 hidden w-56 rounded-lg border border-white/15 bg-slate-800 p-2 text-xs leading-relaxed text-slate-100 shadow-xl group-hover:block group-focus-within:block">
          {t(`measure.${field.key}.help`)}
        </span>
      </span>
      {field.optional && <label className="ml-auto flex items-center gap-1.5 text-xs text-slate-400">
        <input type="checkbox" checked={!manual} onChange={(event) => setField(field.key, event.target.checked ? undefined : display)}
          className="accent-teal-400 focus-visible:outline-2 focus-visible:outline-teal-400" />
        {t('measure.auto')}
      </label>}
    </div>
    {field.key === 'gender' && <div className="mb-1 flex justify-between text-[11px] text-slate-400"><span>{t('measure.female')}</span><span>{t('measure.male')}</span></div>}
    <div className="flex items-center gap-3">
      <input id={`slider-${field.key}`} type="range" min={field.min} max={field.max} step={field.step ?? 1} value={display}
        disabled={field.optional && !manual} onChange={(event) => update(Number(event.target.value))}
        aria-label={t(`measure.${field.key}.label`)} className="h-1.5 min-w-0 flex-1 cursor-pointer accent-teal-400 disabled:opacity-40" />
      <div className="flex items-center gap-1">
        <input id={`number-${field.key}`} type="number" min={field.min} max={field.max} step={field.step ?? 1}
          value={manual ? value : ''} placeholder={field.optional ? t('measure.auto') : undefined}
          disabled={field.optional && !manual} onChange={(event) => update(Number(event.target.value))}
          className="w-16 rounded-md border border-white/15 bg-slate-900 px-1.5 py-1 text-right text-sm text-white focus-visible:outline-2 focus-visible:outline-teal-400 disabled:opacity-40" />
        <span className="w-5 text-xs text-slate-400">{field.unit}</span>
      </div>
    </div>
    <Achieved field={field} />
  </div>;
}

function Section({ title, fields }: { title: string; fields: FieldDef[] }) {
  return <section className="mt-5"><h3 className="text-xs font-bold uppercase tracking-[0.16em] text-teal-300">{title}</h3>
    <div className="mt-1">{fields.map((field) => <MeasurementField key={field.key} field={field} />)}</div>
  </section>;
}

/** Selectable sizes of the current system: EU 35..48 converted through avatar-core, plus the current value. */
function shoeSizeOptions(shoe: BodyParams['shoe']): number[] {
  const sizes = new Set<number>([shoe.size]);
  for (let eu = 35; eu <= 48; eu++) sizes.add(roundShoeSize(convertShoeSize({ system: 'EU', size: eu }, shoe.system)));
  return [...sizes].sort((a, b) => a - b);
}

function AppearanceSection() {
  const { t } = useTranslation();
  const { mode, toneIndex, setMode, setTone } = useAppearanceStore();
  const chip = 'rounded-lg border px-3 py-2 text-xs font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400';
  return <section className="mt-5"><h3 className="text-xs font-bold uppercase tracking-[0.16em] text-teal-300">{t('panel.appearance')}</h3>
    <div className="mt-3 flex gap-2" role="group" aria-label={t('panel.appearance')}>
      {(['skin', 'mannequin'] as const).map((value) => <button key={value} type="button" aria-pressed={mode === value} onClick={() => setMode(value)}
        className={`${chip} ${mode === value ? 'border-teal-300 bg-teal-400 text-slate-950' : 'border-white/15 bg-white/5 hover:bg-white/10'}`}>{t(`appearance.${value}`)}</button>)}
    </div>
    <div className="mt-3 flex gap-2" role="group" aria-label={t('appearance.tone.label')}>
      {skinTones.map((tone, index) => <button key={tone.id} type="button" onClick={() => setTone(index)}
        aria-label={t(`appearance.tone.${tone.id}`)} aria-pressed={mode === 'skin' && toneIndex === index} title={t(`appearance.tone.${tone.id}`)}
        style={{ backgroundColor: tone.color }}
        className={`size-8 rounded-full border-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400 ${mode === 'skin' && toneIndex === index ? 'border-white' : 'border-transparent'}`} />)}
    </div>
  </section>;
}

export function BodyPanel() {
  const { t } = useTranslation();
  const { params, setShoe, reset, save, replace } = useBodyStore();
  const fileInput = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState('');
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const handleImport = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed: unknown = JSON.parse(await file.text());
      if (!isBodyParams(parsed)) throw new Error('Invalid profile');
      replace(parsed);
      setStatus(t('panel.imported'));
    } catch { setStatus(t('panel.importError')); }
    event.target.value = '';
  };
  const exportJson = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(params, null, 2)], { type: 'application/json' }));
    const anchor = document.createElement('a');
    anchor.href = url; anchor.download = 'digital-twin-profile.json'; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const shoeOptions = shoeSizeOptions(params.shoe);
  return <div className="px-5 pb-10 pt-6">
    <div className="mb-5"><p className="text-[11px] font-bold uppercase tracking-[0.2em] text-teal-300">{t('panel.subtitle')}</p>
      <h2 className="mt-1 text-2xl font-semibold tracking-tight">{t('panel.title')}</h2>
      <p className="mt-2 text-xs leading-relaxed text-slate-400">{t('panel.description')}</p></div>
    <div className="flex border-b border-white/10" role="tablist" aria-label={t('panel.tabs')}>
      <button type="button" role="tab" aria-selected="true" className="border-b-2 border-teal-400 px-2 py-2 text-sm font-semibold text-teal-300 focus-visible:outline-2 focus-visible:outline-teal-400">{t('panel.measurements')}</button>
      {(['face', 'wardrobe'] as const).map((key) => <button key={key} type="button" role="tab" aria-selected="false" disabled
        className="px-2 py-2 text-xs text-slate-500">{t(`panel.${key}`)} <span className="rounded bg-white/10 px-1 py-0.5 text-[10px]">{t('panel.soon')}</span></button>)}
    </div>
    <Section title={t('panel.basics')} fields={basics} />
    <Section title={t('panel.upper')} fields={upper} />
    <Section title={t('panel.lower')} fields={lower} />
    <details open={advancedOpen} onToggle={(event) => setAdvancedOpen(event.currentTarget.open)} className="mt-5">
      <summary className="cursor-pointer text-xs font-bold uppercase tracking-[0.16em] text-teal-300 focus-visible:outline-2 focus-visible:outline-teal-400">{t('panel.advanced')}</summary>
      <div className="mt-1">{advanced.map((field) => <MeasurementField key={field.key} field={field} />)}</div>
    </details>
    <section className="mt-5"><h3 className="text-xs font-bold uppercase tracking-[0.16em] text-teal-300">{t('panel.feet')}</h3>
      <div className="mt-3 flex gap-2">
        <label className="flex-1 text-xs text-slate-400">{t('measure.shoeSize.label')}
          <select value={params.shoe.size} onChange={(event) => setShoe({ ...params.shoe, size: Number(event.target.value) })}
            className="mt-1 block w-full rounded-lg border border-white/15 bg-slate-900 p-2 text-sm text-white focus-visible:outline-2 focus-visible:outline-teal-400">
            {shoeOptions.map((size) => <option key={size} value={size}>{size}</option>)}
          </select></label>
        <label className="flex-1 text-xs text-slate-400">{t('measure.shoeSystem.label')}
          <select value={params.shoe.system} onChange={(event) => { const system = event.target.value as ShoeSystem; setShoe({ system, size: roundShoeSize(convertShoeSize(params.shoe, system)) }); }}
            className="mt-1 block w-full rounded-lg border border-white/15 bg-slate-900 p-2 text-sm text-white focus-visible:outline-2 focus-visible:outline-teal-400">
            {systems.map((system) => <option key={system} value={system}>{t(`measure.system.${system}`)}</option>)}
          </select></label>
      </div>
      <p className="mt-3 text-sm text-slate-300">{t('measure.footLength')}: <strong className="text-white">{footLengthCmFromShoe(params.shoe).toFixed(1)} cm</strong></p>
      <p className="mt-1 text-xs text-slate-500">{t('measure.shoeSize.help')}</p>
    </section>
    <AppearanceSection />
    <section className="mt-7 border-t border-white/10 pt-5"><h3 className="text-xs font-bold uppercase tracking-[0.16em] text-teal-300">{t('panel.profile')}</h3>
      <div className="mt-3 flex flex-wrap gap-2">
        <button type="button" className={`${buttonClass} border-teal-400/50 text-teal-200`} onClick={() => void save().then((ok) => setStatus(t(ok ? 'panel.saved' : 'panel.saveError')))}>{t('panel.save')}</button>
        <button type="button" className={buttonClass} onClick={() => { reset(); setStatus(t('panel.resetDone')); }}>{t('panel.reset')}</button>
        <button type="button" className={buttonClass} onClick={exportJson}>{t('panel.export')}</button>
        <button type="button" className={buttonClass} onClick={() => fileInput.current?.click()}>{t('panel.import')}</button>
        <input ref={fileInput} type="file" accept="application/json,.json" onChange={(event) => void handleImport(event)} className="sr-only" aria-label={t('panel.import')} />
      </div>
      <p role="status" className="mt-3 min-h-4 text-xs text-slate-400">{status}</p>
    </section>
  </div>;
}

