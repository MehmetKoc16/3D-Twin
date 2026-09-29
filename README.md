# Dijital İkiz

Vücut ölçülerinden kendi 3D dijital ikizinizi oluşturduğunuz, poz verdirip yakınlaştırarak inceleyebildiğiniz ve
ileride selfie ile yüzünüzü ekleyip kıyafet/ayakkabı deneyebildiğiniz bir **web sanal kabin** projesi.
Vücut modeli [MakeHuman](https://github.com/makehumancommunity) (CC0 asset'ler) tabanlıdır. Arayüz Türkçe (varsayılan)
ve İngilizce'dir. Yalnızca ücretsiz servisler ve serbest lisanslı asset'ler kullanılır.

## Yol haritası (dalgalar)

| Dalga | Kapsam                                                                                     |
| ----- | ------------------------------------------------------------------------------------------ |
| 0     | Repo iskeleti, veri sözleşmeleri, dokümantasyon                                            |
| 1     | Asset pipeline (MakeHuman -> web formatı), UI kabuğu, `avatar-core` (morph, ölçüm, solver) |
| 2     | Entegrasyon: `<Avatar/>`, worker, pozlar (T-Pose, A-Pose, ...), yakınlaştırma (**MVP**)    |
| 3     | Yüz: selfie -> MediaPipe landmark -> doku (tamamen tarayıcıda)                             |
| 4     | Gardırop: şablon kıyafetler, ayakkabı, beden tablosu, oturma analizi (fit heatmap)         |

## Teknoloji

- Vite, React 19, TypeScript (strict), Tailwind CSS 4
- three.js, @react-three/fiber 9, @react-three/drei 10
- zustand, i18next, comlink (Web Worker), idb-keyval
- Vitest, Playwright
- Asset pipeline: Python 3.12 (numpy, pygltflib)
- npm workspaces, Git LFS (büyük 3D dosyalar)

## Kurulum

```bash
npm install
git lfs pull
npm run dev        # http://localhost:5173
```

Diğer komutlar: `npm run build`, `npm run test`, `npm run test:e2e`, `npm run lint`, `npm run typecheck`,
`npm run format`, `npm run assets:build` (Python pipeline).

## Klasör yapısı

```
apps/web/            Vite + React uygulaması (@dt/web)
packages/avatar-core/ Saf TypeScript çekirdek: morph, ölçüm, solver, iskelet (@dt/avatar-core)
tools/asset-pipeline/ Python: ham MakeHuman verisi -> web formatı
docs/                ARCHITECTURE.md, adr/, research/
.claude/agents/      Proje alt-agent tanımları
```

Ayrıntılar için [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ve [AGENTS.md](AGENTS.md).

## Gizlilik

Selfie ve kişisel fotoğraflar **cihazınızdan çıkmaz**; yüz işleme tamamen tarayıcıda yapılır. Kişisel veriler
depoya eklenmez (`.gitignore` ile korunur).

## Lisans ve atıflar

Kod: [MIT](LICENSE). Üçüncü taraf asset'ler ve atıflar için bkz. [CREDITS.md](CREDITS.md).
