import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { BodyPartCategory, BodyPartDef } from '@dt/avatar-core';
import {
  effectiveEyebrowColor,
  eyeColors,
  hairColors,
  useAppearanceStore,
  type ColorPreset,
} from '../../store/appearanceStore';
import { useFaceStore } from '../../store/faceStore';
import { loadPartsIndex, partsOf, type PartsIndex } from '../avatar/parts/partsIndex';
import type { PhotoColors } from '../face/photoColors';

const chip =
  'rounded-lg border px-3 py-2 text-xs font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400';
const chipOn = 'border-teal-300 bg-teal-400 text-slate-950';
const chipOff = 'border-white/15 bg-white/5 hover:bg-white/10';

function sameColor(a: string | null, b: string): boolean {
  return a !== null && a.toLowerCase() === b.toLowerCase();
}

interface ColorRowProps {
  testId: string;
  label: string;
  presets: readonly ColorPreset[];
  presetLabel: (id: string) => string;
  value: string;
  onChange: (hex: string) => void;
  customLabel: string;
  /** Marks no preset as selected (e.g. while the eyebrows follow the hair colour). */
  dimmed?: boolean;
}

function ColorRow({ testId, label, presets, presetLabel, value, onChange, customLabel, dimmed }: ColorRowProps) {
  const custom = !presets.some((p) => sameColor(value, p.color));
  return (
    <div className="mt-2" role="group" aria-label={label}>
      <div className="flex flex-wrap items-center gap-2">
        {presets.map((preset) => {
          const selected = !dimmed && sameColor(value, preset.color);
          return (
            <button
              key={preset.id}
              type="button"
              data-testid={`${testId}-${preset.id}`}
              aria-label={presetLabel(preset.id)}
              aria-pressed={selected}
              title={presetLabel(preset.id)}
              onClick={() => onChange(preset.color)}
              style={{ backgroundColor: preset.color }}
              className={`size-7 rounded-full border-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400 ${selected ? 'border-white' : 'border-transparent'}`}
            />
          );
        })}
        <label
          title={customLabel}
          className={`relative grid size-7 cursor-pointer place-items-center overflow-hidden rounded-full border-2 text-xs text-white focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-teal-400 ${!dimmed && custom ? 'border-white' : 'border-white/30'}`}
          style={{ backgroundColor: !dimmed && custom ? value : '#2b3238' }}
        >
          <span aria-hidden="true">{!dimmed && custom ? '' : '+'}</span>
          <input
            type="color"
            data-testid={`${testId}-custom`}
            aria-label={customLabel}
            value={value}
            onChange={(event) => onChange(event.target.value)}
            className="absolute inset-0 cursor-pointer opacity-0"
          />
        </label>
      </div>
    </div>
  );
}

function StyleChips({
  testId,
  parts,
  selected,
  onSelect,
  language,
  noneLabel,
}: {
  testId: string;
  parts: BodyPartDef[];
  selected: string | null;
  onSelect: (id: string | null) => void;
  language: 'tr' | 'en';
  noneLabel?: string;
}) {
  return (
    <div className="mt-2 flex flex-wrap gap-2">
      {noneLabel !== undefined && (
        <button
          type="button"
          data-testid={`${testId}-none`}
          aria-pressed={selected === null}
          onClick={() => onSelect(null)}
          className={`${chip} ${selected === null ? chipOn : chipOff}`}
        >
          {noneLabel}
        </button>
      )}
      {parts.map((part) => (
        <button
          key={part.id}
          type="button"
          data-testid={`${testId}-${part.id}`}
          aria-pressed={selected === part.id}
          onClick={() => onSelect(part.id)}
          className={`${chip} ${selected === part.id ? chipOn : chipOff}`}
        >
          {part.label[language]}
        </button>
      ))}
    </div>
  );
}

function Heading({ children }: { children: string }) {
  return <h4 className="mt-4 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">{children}</h4>;
}

/** Photo-derived colours of the stored selfie (undefined without a photo or before the sampling ran). */
function usePhotoColors(): PhotoColors | undefined {
  const bitmap = useFaceStore((state) => state.bitmap);
  const landmarks = useFaceStore((state) => state.landmarks);
  const [colors, setColors] = useState<{ bitmap: ImageBitmap; landmarks: Float32Array; value: PhotoColors } | null>(null);
  useEffect(() => {
    if (!bitmap || !landmarks) return;
    let alive = true;
    void import('../face/photoColors').then(({ samplePhotoColors }) => {
      if (!alive) return;
      try {
        setColors({ bitmap, landmarks, value: samplePhotoColors(bitmap, landmarks) });
      } catch {
        setColors({ bitmap, landmarks, value: {} });
      }
    });
    return () => {
      alive = false;
    };
  }, [bitmap, landmarks]);
  return bitmap && landmarks && colors?.bitmap === bitmap && colors.landmarks === landmarks ? colors.value : undefined;
}

/** Hair, eyebrow and eye controls of the appearance section. */
export function PartsAppearance() {
  const { t, i18n } = useTranslation();
  const language = i18n.language === 'en' ? 'en' : 'tr';
  const [index, setIndex] = useState<PartsIndex | null>(null);
  const store = useAppearanceStore();
  const photo = usePhotoColors();
  const hasPhoto = useFaceStore((state) => state.bitmap !== undefined && state.landmarks !== undefined);
  useEffect(() => {
    void loadPartsIndex().then(setIndex, () => undefined);
  }, []);
  const of = (category: BodyPartCategory): BodyPartDef[] => (index ? partsOf(index, category) : []);
  const browColor = effectiveEyebrowColor(store);
  return (
    <div data-testid="parts-appearance">
      <Heading>{t('appearance.hair.label')}</Heading>
      <StyleChips
        testId="hair-style"
        parts={of('hair')}
        selected={store.hairId}
        onSelect={store.setHair}
        language={language}
        noneLabel={t('appearance.hair.none')}
      />
      <Heading>{t('appearance.hair.color')}</Heading>
      <ColorRow
        testId="hair-color"
        label={t('appearance.hair.color')}
        presets={hairColors}
        presetLabel={(id) => t(`appearance.hair.colors.${id}`)}
        value={store.hairColor}
        onChange={store.setHairColor}
        customLabel={t('appearance.hair.custom')}
      />
      <Heading>{t('appearance.eyebrows.label')}</Heading>
      <StyleChips
        testId="brow-style"
        parts={of('eyebrows')}
        selected={store.eyebrowId}
        onSelect={(id) => id && store.setEyebrow(id)}
        language={language}
      />
      <Heading>{t('appearance.eyebrows.color')}</Heading>
      <ColorRow
        testId="brow-color"
        label={t('appearance.eyebrows.color')}
        presets={hairColors}
        presetLabel={(id) => t(`appearance.hair.colors.${id}`)}
        value={browColor}
        onChange={store.setEyebrowColor}
        customLabel={t('appearance.hair.custom')}
        dimmed={store.eyebrowColor === null}
      />
      <label className="mt-2 flex items-center gap-2 text-xs text-slate-300">
        <input
          type="checkbox"
          data-testid="brow-follow-hair"
          checked={store.eyebrowColor === null}
          onChange={(event) => store.setEyebrowColor(event.target.checked ? null : store.hairColor)}
          className="accent-teal-400 focus-visible:outline-2 focus-visible:outline-teal-400"
        />
        {t('appearance.eyebrows.followHair')}
      </label>
      <Heading>{t('appearance.eyes.label')}</Heading>
      <ColorRow
        testId="eye-color"
        label={t('appearance.eyes.label')}
        presets={eyeColors}
        presetLabel={(id) => t(`appearance.eyes.colors.${id}`)}
        value={store.eyeColor}
        onChange={store.setEyeColor}
        customLabel={t('appearance.eyes.custom')}
      />
      {hasPhoto && (
        <div className="mt-3 flex flex-wrap gap-2" data-testid="photo-colors" data-ready={photo !== undefined}>
          <button
            type="button"
            data-testid="eye-from-photo"
            title={t('appearance.eyes.fromPhotoHint')}
            disabled={!photo?.iris}
            onClick={() => photo?.iris && store.setEyeColor(photo.iris)}
            className={`${chip} ${chipOff} flex items-center gap-2 disabled:opacity-40`}
          >
            <span
              className="inline-block size-4 rounded-full border border-white/30"
              style={{ backgroundColor: photo?.iris ?? 'transparent' }}
              aria-hidden="true"
            />
            {t('appearance.eyes.fromPhoto')}
          </button>
          <button
            type="button"
            data-testid="hair-from-photo"
            disabled={!photo?.brow}
            onClick={() => {
              if (!photo?.brow) return;
              store.setHairColor(photo.brow);
              store.setEyebrowColor(null);
            }}
            className={`${chip} ${chipOff} flex items-center gap-2 disabled:opacity-40`}
          >
            <span
              className="inline-block size-4 rounded-full border border-white/30"
              style={{ backgroundColor: photo?.brow ?? 'transparent' }}
              aria-hidden="true"
            />
            {t('appearance.eyes.photoBrow')}
          </button>
          {photo && !photo.iris && !photo.brow && (
            <p className="w-full text-xs text-amber-300">{t('appearance.eyes.photoUnavailable')}</p>
          )}
        </div>
      )}
    </div>
  );
}
