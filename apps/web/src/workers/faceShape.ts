import type { FaceMapDef } from '@dt/avatar-core';
import type { AvatarFaceFit } from './avatarProtocol';

/** Inputs of one face-shape fit; the caller adds the solver data and the base weights. */
export interface FaceFitRequest {
  faceMap: FaceMapDef;
  landmarks: Float32Array;
  imageWidth: number;
  imageHeight: number;
}

export type FaceFitter = (request: FaceFitRequest) => AvatarFaceFit;

export interface FaceApplyResult {
  /** Modifier values to merge into the solve weights; null = no face shape active. */
  values: Record<string, number> | null;
  /** Set only when a (re)fit ran during this call: the fit, or null if it failed. */
  report?: AvatarFaceFit | null;
}

/**
 * Face-shape state of the avatar worker: the photo landmarks are kept, fitted lazily against the next solved body
 * (the fit needs the solved weights) and the resulting modifier values are merged into every later solve.
 */
export class FaceShapeState {
  private request: FaceFitRequest | null = null;
  private values: Record<string, number> | null = null;
  private dirty = false;

  constructor(private readonly fit: FaceFitter) {}

  get active(): boolean {
    return this.request !== null;
  }

  set(request: FaceFitRequest): void {
    this.request = request;
    this.values = null;
    this.dirty = true;
  }

  clear(): void {
    this.request = null;
    this.values = null;
    this.dirty = false;
  }

  /** Call once per solve, after the body weights of that solve are known to the fitter. */
  apply(): FaceApplyResult {
    if (!this.request) return { values: null };
    if (!this.dirty) return { values: this.values };
    this.dirty = false;
    try {
      const result = this.fit(this.request);
      this.values = result.modifierValues;
      return { values: this.values, report: result };
    } catch (error) {
      console.warn('[face] shape fit failed', error);
      this.values = null;
      return { values: null, report: null };
    }
  }
}
