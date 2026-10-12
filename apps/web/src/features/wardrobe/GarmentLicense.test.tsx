import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { createInstance } from 'i18next';
import { I18nextProvider } from 'react-i18next';
import type { GarmentTemplateDef } from '@dt/avatar-core';
import { GarmentAttribution, LicenseBadge } from './GarmentLicense';
import tr from './locales/tr.json';
import en from './locales/en.json';

describe('garment licence UI', () => {
  it.each(['tr', 'en'])(
    'renders exact CC-BY versions and full attribution in %s',
    async (language) => {
      const i18n = createInstance();
      await i18n.init({
        lng: language,
        resources: { tr: { translation: { wardrobe: tr } }, en: { translation: { wardrobe: en } } },
      });
      for (const license of ['CC-BY-3.0', 'CC-BY-4.0'] as const) {
        const template: GarmentTemplateDef = {
          id: 'synthetic-jacket',
          kind: 'longsleeve',
          category: 'top',
          label: { tr: 'Test', en: 'Test' },
          license,
          attribution: `Synthetic jacket by Test Author, https://example.test/jacket, ${license}. Changes: converted.`,
          mesh: 'test.glb',
          binding: 'test.bind.bin',
          nativeMeasures: {},
          defaultEase: {},
          layer: 1,
          baseColor: '#888888',
        };
        const html = renderToStaticMarkup(
          <I18nextProvider i18n={i18n}>
            <LicenseBadge template={template} />
            <GarmentAttribution template={template} />
          </I18nextProvider>,
        );
        expect(html).toContain(license === 'CC-BY-3.0' ? 'CC BY 3.0' : 'CC BY 4.0');
        expect(html).toContain(template.attribution);
        expect(html).toContain(language === 'tr' ? 'Kaynak ve atıf' : en.catalog.attribution);
        expect(html).not.toContain('wardrobe.license.');
        expect(html).not.toContain(tr.catalog.noAttribution);
        expect(html).not.toContain(en.catalog.noAttribution);
      }
    },
  );
});
