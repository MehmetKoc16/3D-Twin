import { lazy, Suspense, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { useBodyStore } from '../store/bodyStore';
import { useFaceStore } from '../store/faceStore';
import { useTwinStore } from '../store/twinStore';
import { useWardrobeStore } from '../store/wardrobeStore';

const Viewer = lazy(() => import('../features/viewer/Viewer').then((module) => ({ default: module.Viewer })));
const BodyPanel = lazy(() => import('../features/body-panel/BodyPanel').then((module) => ({ default: module.BodyPanel })));

export function App() {
  const { t, i18n } = useTranslation();
  useEffect(() => {
    void useBodyStore.getState().hydrate();
    void useFaceStore.getState().ensureBaked(); // restores a saved selfie and bakes it
    void useWardrobeStore.getState().hydrate(); // restores saved store items and what was worn
    void useTwinStore.getState().hydrate(); // restores the user's own twin files (IndexedDB) and the chosen model
  }, []);

  return (
    <div className="flex min-h-dvh flex-col bg-[#101316] text-slate-100 min-[900px]:h-dvh min-[900px]:overflow-hidden">
      <header className="flex h-16 shrink-0 items-center justify-between border-b border-white/10 px-5 sm:px-8">
        <div className="flex items-center gap-3">
          <div aria-hidden="true" className="grid size-9 place-items-center rounded-xl bg-teal-400 font-black text-slate-950">D</div>
          <h1 className="text-lg font-semibold tracking-tight">{t('app.title')}</h1>
        </div>
        <button type="button" onClick={() => void i18n.changeLanguage(i18n.language === 'tr' ? 'en' : 'tr')}
          aria-label={t('app.changeLanguage')}
          className="rounded-lg border border-white/20 px-3 py-1.5 text-sm font-semibold transition hover:bg-white/10 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400">
          {t('app.languageToggle')}
        </button>
      </header>
      <div className="flex min-h-0 flex-1 flex-col min-[900px]:flex-row">
        <main className="relative min-h-[540px] min-w-0 flex-1 bg-[#171d21] max-[899px]:h-[62dvh] max-[899px]:min-h-[450px]">
          <Suspense fallback={null}><Viewer /></Suspense>
        </main>
        <aside className="w-full shrink-0 border-t border-white/10 bg-[#15191d] min-[900px]:min-h-0 min-[900px]:w-[380px] min-[900px]:overflow-y-auto min-[900px]:border-l min-[900px]:border-t-0" aria-label={t('panel.title')}>
          <Suspense fallback={null}><BodyPanel /></Suspense>
        </aside>
      </div>
    </div>
  );
}
