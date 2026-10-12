import { useTranslation } from 'react-i18next';
import type { GarmentTemplateDef } from '@dt/avatar-core';

export function LicenseBadge({ template }: { template: GarmentTemplateDef }) {
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

export function GarmentAttribution({ template }: { template: GarmentTemplateDef }) {
  const { t } = useTranslation();
  return (
    <details className="text-[11px] text-slate-400">
      <summary className="cursor-pointer focus-visible:outline-2 focus-visible:outline-teal-400">
        {t('wardrobe.catalog.attribution')}
      </summary>
      <p className="mt-1 break-words leading-relaxed">
        {template.attribution ?? t('wardrobe.catalog.noAttribution')}
      </p>
    </details>
  );
}
