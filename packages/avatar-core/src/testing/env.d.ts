/** Minimal ambient globals used by tests (the package tsconfig has no DOM / node types). */
declare const console: { log(...args: unknown[]): void };
declare const performance: { now(): number };
declare function structuredClone<T>(value: T): T;
