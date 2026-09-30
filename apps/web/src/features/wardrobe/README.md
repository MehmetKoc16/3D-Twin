Wardrobe: try-on of store items on the avatar (garment templates, size charts, fit report, heatmap). Owner: Wave 4c.

- `WardrobePanel` / `StoreItemForm` / `FitReportView`: UI (feature-local i18n in `locales/`).
- `chartModel`, `colorExtract`, `fitAnalysis`: pure, unit-tested logic (chart editor, k-means colour, fit verdicts).
- `wardrobeRig` + `garmentInstance` + `templateLoader`: R3F-side mounting, created by `features/avatar/Avatar.tsx`.
- `garmentGrading`, `garmentCollide`, `bodyHide`: pure geometry helpers (grade rings and length specs, layering, body hiding). Limb sections and longitudinal grading run in avatar-core.
- State: `src/store/wardrobeStore.ts` (IndexedDB keys `wardrobe:*`). See the "Wardrobe" section of `docs/ARCHITECTURE.md`.
