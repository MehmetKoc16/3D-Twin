import type { SolveEnvelope } from '../../workers/avatarClient';
import type { AvatarAssets } from '../avatar/avatarAssets';
import { isTwinActive, useTwinStore, type TwinError, type TwinPack } from '../../store/twinStore';
import { MAX_HEAD_RESIDUAL_M, TwinRig } from './twinRig';
import { loadTwinModel } from './twinModel';
import { TwinFormatError, twinMacros } from './twinDef';

/** Body parts (eyes, brows, hair) are a standard-mode feature: the controller mounts them only in that mode. */
export interface ModeParts {
  onSolve(envelope: SolveEnvelope): void;
  dispose(): void;
}

/** Solve values shown in the body panel. */
export interface DisplaySolve {
  achievedCm: SolveEnvelope['result']['achievedCm'];
  residualsCm: SolveEnvelope['result']['residualsCm'];
  unreachable: SolveEnvelope['result']['unreachable'];
  estimatedMassKg: number;
}

/**
 * Switches the avatar between the standard mannequin and the realistic twin (created by the Avatar next to the
 * wardrobe rig; the avatar forwards every solve result).
 *
 * Twin mode: the worker returns the twin's fitted MakeHuman body (`setFixedShape`), that body is hidden but stays the
 * source of the skeleton and of the garment fitting, the body parts are removed and a `TwinRig` shows the scan bound
 * to the same skeleton. Standard mode is the unchanged pre-twin behaviour.
 */
export class TwinMode {
  private parts: ModeParts | null = null;
  private twinRig: TwinRig | null = null;
  private last: SolveEnvelope | null = null;
  private wantsTwin = false;
  /**
   * A `setFixedShape(shape)` was posted to the worker and not yet followed by `setFixedShape(null)`. Messages reach the
   * worker in order, so posting the reset whenever this is set always ends in the right state, even when a newer
   * `apply` overtakes one that is still waiting for the worker.
   */
  private fixedRequested = false;
  private generation = 0;
  private appliedRevision = -1;
  private readonly unsubscribe: () => void;
  private disposed = false;

  constructor(
    private readonly assets: AvatarAssets,
    private readonly createParts: () => ModeParts,
    private readonly requestSolve: () => void,
  ) {
    this.unsubscribe = useTwinStore.subscribe((state, previous) => {
      if (
        isTwinActive(state) !== isTwinActive(previous) ||
        (isTwinActive(state) && state.revision !== previous.revision)
      )
        void this.apply();
    });
    void this.apply();
  }

  /** True while the twin's fixed body is what the worker returns; body-parameter solves are pointless then. */
  get active(): boolean {
    return this.wantsTwin;
  }

  private async apply(): Promise<void> {
    if (this.disposed) return;
    const token = ++this.generation;
    const state = useTwinStore.getState();
    const wantsTwin = isTwinActive(state);
    if (!wantsTwin) {
      this.wantsTwin = false;
      this.appliedRevision = -1;
      this.twinRig?.dispose();
      this.twinRig = null;
      useTwinStore.getState().setRuntime(null);
      if (this.fixedRequested) {
        this.leaveFixedShape();
        this.requestSolve();
      }
      this.parts ??= this.createParts();
      if (this.last) this.parts.onSolve(this.last);
      return;
    }
    const pack = state.pack as TwinPack;
    this.wantsTwin = true;
    if (this.appliedRevision === state.revision && this.twinRig) return;
    this.parts?.dispose();
    this.parts = null;
    try {
      const model = await loadTwinModel(
        pack.glb,
        pack.def,
        this.assets.skeleton.bones.map((b) => b.name),
      );
      if (token !== this.generation || this.disposed) {
        model.dispose();
        return;
      }
      this.fixedRequested = true;
      await this.assets.client.setFixedShape({
        macros: twinMacros(pack.def),
        modifiers: pack.def.fittedModifiers,
      });
      if (token !== this.generation || this.disposed) {
        model.dispose();
        return;
      }
      this.twinRig?.dispose();
      this.twinRig = new TwinRig(this.assets, model, pack.mapping, (info) =>
        useTwinStore.getState().setRuntime(info),
      );
      this.appliedRevision = state.revision;
      this.requestSolve();
    } catch (error) {
      if (token !== this.generation || this.disposed) return;
      console.error('Could not show the twin', error);
      this.twinRig?.dispose();
      this.twinRig = null;
      this.wantsTwin = false;
      const failure: TwinError =
        error instanceof TwinFormatError
          ? { code: error.code, detail: error.message }
          : { code: 'shape', detail: error instanceof Error ? error.message : String(error) };
      const undo = this.fixedRequested;
      if (undo) this.leaveFixedShape();
      useTwinStore.getState().fail(failure);
      if (undo) this.requestSolve();
    }
  }

  private leaveFixedShape(): void {
    this.fixedRequested = false;
    this.assets.client.setFixedShape(null).catch((error: unknown) => {
      console.error('Could not leave the twin shape', error);
    });
  }

  /** Forward every solve result after the body geometry and skeleton were updated. */
  onSolve(envelope: SolveEnvelope): void {
    this.last = envelope;
    if (this.twinRig) {
      const alignment = this.twinRig.onSolve(envelope);
      if (alignment && alignment.maxResidual > MAX_HEAD_RESIDUAL_M) {
        // twin.json and rigged.glb (or the avatar's rig) do not belong together
        const detail = `rest heads differ by ${(alignment.maxResidual * 1000).toFixed(1)} mm`;
        this.twinRig.dispose();
        this.twinRig = null;
        this.wantsTwin = false;
        this.generation++;
        useTwinStore.getState().fail({ code: 'mismatch', detail });
        return;
      }
    }
    this.parts?.onSolve(envelope);
  }

  /**
   * What the body panel shows for a solve: in twin mode the twin's own measurements (fitted body minus the clothing
   * allowance) replace the hidden body's geometric ones - the wardrobe rig keeps using the geometric values, so the
   * garments fit the shape that is actually displayed, while sizes are judged against the person.
   */
  displaySolve(envelope: SolveEnvelope): DisplaySolve {
    const r = envelope.result;
    const pack = useTwinStore.getState().pack;
    if (r.fixedShape === true && pack) {
      return {
        achievedCm: { ...r.achievedCm, ...pack.def.measurementsCm },
        residualsCm: {},
        unreachable: [],
        estimatedMassKg: r.estimatedMassKg,
      };
    }
    return {
      achievedCm: r.achievedCm,
      residualsCm: r.residualsCm,
      unreachable: r.unreachable,
      estimatedMassKg: r.estimatedMassKg,
    };
  }

  dispose(): void {
    this.disposed = true;
    this.generation++;
    this.unsubscribe();
    this.parts?.dispose();
    this.parts = null;
    this.twinRig?.dispose();
    this.twinRig = null;
    if (this.fixedRequested) this.leaveFixedShape();
    this.wantsTwin = false;
  }
}
