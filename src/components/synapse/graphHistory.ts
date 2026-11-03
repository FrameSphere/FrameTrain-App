/** Undo/Redo-Stapel für den Canvas. Hält Schnappschüsse, kennt kein React. */

export interface HistoryState<T> {
  past: T[];
  future: T[];
}

const LIMIT = 100;

export function emptyHistory<T>(): HistoryState<T> {
  return { past: [], future: [] };
}

/** Vor einer Änderung den alten Stand merken; Redo-Zweig verfällt. */
export function record<T>(h: HistoryState<T>, before: T): HistoryState<T> {
  return { past: [...h.past, before].slice(-LIMIT), future: [] };
}

export function undo<T>(h: HistoryState<T>, current: T): { history: HistoryState<T>; value: T } | null {
  const value = h.past[h.past.length - 1];
  if (value === undefined) return null;
  return { history: { past: h.past.slice(0, -1), future: [current, ...h.future] }, value };
}

export function redo<T>(h: HistoryState<T>, current: T): { history: HistoryState<T>; value: T } | null {
  const value = h.future[0];
  if (value === undefined) return null;
  return { history: { past: [...h.past, current], future: h.future.slice(1) }, value };
}
