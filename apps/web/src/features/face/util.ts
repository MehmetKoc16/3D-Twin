/** Index read that satisfies noUncheckedIndexedAccess (out-of-range reads yield 0). */
export function at(values: ArrayLike<number>, index: number): number {
  return values[index] ?? 0;
}
