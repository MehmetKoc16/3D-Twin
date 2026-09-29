import { useTranslation } from 'react-i18next';
import { usePoseStore } from '../../store/poseStore';
import type { PoseId } from '../../store/poseStore';

const poses: PoseId[] = ['t-pose', 'a-pose', 'relaxed', 'hands-on-hips', 'walk', 'side'];
export function PoseBar() {
  const { t } = useTranslation();
  const { poseId, setPose } = usePoseStore();
  return <div className="absolute inset-x-0 bottom-0 p-4">
    <div className="mx-auto max-w-max rounded-xl border border-white/15 bg-slate-950/80 p-2 shadow-xl backdrop-blur">
      <div className="mb-1.5 px-1 text-[10px] font-bold uppercase tracking-[0.16em] text-slate-400">{t('pose.title')}</div>
      <div className="flex max-w-[calc(100vw-2rem)] gap-1 overflow-x-auto" role="group" aria-label={t('pose.title')}>
        {poses.map((id) => <button key={id} type="button" onClick={() => setPose(id)} aria-pressed={poseId === id}
          className={`shrink-0 rounded-lg px-3 py-2 text-xs font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400 ${poseId === id ? 'bg-teal-400 text-slate-950' : 'text-slate-200 hover:bg-white/10'}`}>
          {t(`pose.${id}`)}
        </button>)}
      </div>
    </div>
  </div>;
}
