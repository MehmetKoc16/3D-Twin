import { create } from 'zustand';
import type { MeasureId } from '@dt/avatar-core';

/** Latest solver output shown in the body panel. Writer: features/avatar. */
interface SolveState {
  achievedCm: Partial<Record<MeasureId, number>>;
  residualsCm: Partial<Record<MeasureId, number>>;
  unreachable: readonly MeasureId[];
  estimatedMassKg: number | null;
  solveMs: number | null;
  roundTripMs: number | null;
  solveCount: number;
  setSolve: (value: {
    achievedCm: Partial<Record<MeasureId, number>>;
    residualsCm: Partial<Record<MeasureId, number>>;
    unreachable: readonly MeasureId[];
    estimatedMassKg: number;
    solveMs: number;
    roundTripMs: number;
  }) => void;
}

export const useSolveStore = create<SolveState>((set) => ({
  achievedCm: {},
  residualsCm: {},
  unreachable: [],
  estimatedMassKg: null,
  solveMs: null,
  roundTripMs: null,
  solveCount: 0,
  setSolve: (value) =>
    set((state) => ({
      ...value,
      estimatedMassKg: Number.isFinite(value.estimatedMassKg) ? value.estimatedMassKg : null,
      solveCount: state.solveCount + 1,
    })),
}));
