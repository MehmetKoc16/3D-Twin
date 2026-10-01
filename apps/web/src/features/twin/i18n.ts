import i18n from 'i18next';
import tr from './locales/tr.json';
import en from './locales/en.json';

// Registered here (not in shared/i18n locales) so the twin feature owns its own strings.
i18n.addResourceBundle('tr', 'translation', { twin: tr }, true, true);
i18n.addResourceBundle('en', 'translation', { twin: en }, true, true);

export default i18n;
