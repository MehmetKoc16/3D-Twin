import i18n from 'i18next';
import tr from './locales/tr.json';
import en from './locales/en.json';

// Registered here (not in shared/i18n locales) so the wardrobe feature owns its own strings.
i18n.addResourceBundle('tr', 'translation', { wardrobe: tr }, true, true);
i18n.addResourceBundle('en', 'translation', { wardrobe: en }, true, true);

export default i18n;
