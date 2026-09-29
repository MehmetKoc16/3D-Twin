/** Minimal ambient globals used by tests (the package tsconfig has no DOM / node types). */
declare const console: { log(...args: unknown[]): void };
declare const performance: { now(): number };
declare function structuredClone<T>(value: T): T;
declare const process: { cwd(): string };
declare module 'node:fs' {
  export function existsSync(path: string): boolean;
  export function readFileSync(path: string): { buffer: ArrayBuffer; byteOffset: number; byteLength: number };
}
declare class TextDecoder {
  decode(input?: ArrayBufferView | ArrayBuffer): string;
}
