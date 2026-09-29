/** Error thrown for malformed data (manifest, morphs.bin, rig, measures) or invalid arguments. */
export class AvatarCoreError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AvatarCoreError';
  }
}
