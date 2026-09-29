export type FocusPreset = 'full' | 'face' | 'upper' | 'lower' | 'feet';
export interface FocusTarget { target: [number, number, number]; distance: number }
export interface FocusTargetProvider { getTargets: (heightM: number) => Record<FocusPreset, FocusTarget> }

export function getFocusTargets(heightM: number): Record<FocusPreset, FocusTarget> {
  const h = Math.max(1.4, Math.min(2.1, heightM));
  return {
    full: { target: [0, h * 0.5, 0], distance: h * 2.25 },
    face: { target: [0, h * 0.93, 0], distance: 0.55 },
    upper: { target: [0, h * 0.72, 0], distance: 1.35 },
    lower: { target: [0, h * 0.30, 0], distance: 1.35 },
    feet: { target: [0, h * 0.04, 0], distance: 0.65 },
  };
}

export const defaultFocusTargetProvider: FocusTargetProvider = { getTargets: getFocusTargets };
