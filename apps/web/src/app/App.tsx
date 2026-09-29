import { Canvas } from '@react-three/fiber';
import { useTranslation } from 'react-i18next';

function Placeholder() {
  return (
    <>
      <ambientLight intensity={0.8} />
      <directionalLight position={[3, 5, 4]} intensity={1.5} />
      <mesh position={[0, 0.9, 0]}>
        <capsuleGeometry args={[0.2, 1.0, 8, 16]} />
        <meshStandardMaterial color="#9ca3af" />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[1, 48]} />
        <meshStandardMaterial color="#4b5563" />
      </mesh>
    </>
  );
}

export function App() {
  const { t, i18n } = useTranslation();
  const toggleLang = () => void i18n.changeLanguage(i18n.language === 'tr' ? 'en' : 'tr');

  return (
    <div className="flex h-full bg-neutral-900 text-neutral-100">
      <main className="relative min-w-0 flex-1">
        <Canvas camera={{ position: [0, 1.3, 3.5], fov: 40 }}>
          <Placeholder />
        </Canvas>
      </main>
      <aside className="w-[360px] shrink-0 border-l border-neutral-700 p-4">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold">{t('app.title')}</h1>
          <button
            type="button"
            onClick={toggleLang}
            className="rounded border border-neutral-500 px-2 py-1 text-sm"
          >
            {t('lang.toggle')}
          </button>
        </div>
        <h2 className="mt-6 text-sm uppercase tracking-wide text-neutral-400">
          {t('panel.measurements')}
        </h2>
      </aside>
    </div>
  );
}
