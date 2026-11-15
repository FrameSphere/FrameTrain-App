// Kurve "Vorhersage gegen echte Werte" aus forecast_curve der Test-Engine.
// Reine Funktionen, damit die Geometrie ohne DOM testbar ist.

export interface ForecastCurve {
  series_id?: string;
  channel?: string;
  step?: number;
  /** Index (in actual), ab dem die Vorhersage beginnt. */
  forecast_start: number;
  actual: Array<number | null>;
  forecast: Array<number | null>;
  naive?: Array<number | null>;
}

export function isForecastCurve(v: unknown): v is ForecastCurve {
  if (!v || typeof v !== 'object') return false;
  const c = v as Record<string, unknown>;
  return Array.isArray(c.actual) && Array.isArray(c.forecast) && c.actual.length > 1;
}

/** SVG-Pfad; null-Werte unterbrechen die Linie. */
export function linePath(
  values: Array<number | null>, width: number, height: number, min: number, max: number,
): string {
  const n = values.length;
  if (n < 2) return '';
  const span = max - min || 1;
  let d = '';
  let pen = false;
  values.forEach((v, i) => {
    if (v == null || !Number.isFinite(v)) { pen = false; return; }
    const x = (i / (n - 1)) * width;
    const y = height - ((v - min) / span) * height;
    d += `${pen ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
    pen = true;
  });
  return d;
}

export function valueRange(...lists: Array<Array<number | null> | undefined>): [number, number] {
  const all = lists.flatMap(l => (l ?? []).filter((v): v is number => v != null && Number.isFinite(v)));
  if (all.length === 0) return [0, 1];
  const min = Math.min(...all);
  const max = Math.max(...all);
  const pad = (max - min) * 0.05 || 1;
  return [min - pad, max + pad];
}
