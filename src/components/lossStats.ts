// Kennzahlen zum Loss-Verlauf.
//
// Der erste aufgezeichnete Punkt ist nicht immer brauchbar: enthält das erste
// Log noch keinen Loss, steht dort 0 — und die Verbesserung wurde als
// "↑ Infinity%" angezeigt.

export interface LossPointLike {
  train_loss?: number | null;
}

/** Erster Punkt mit echtem Loss (> 0), sonst undefined. */
export function firstUsableLoss(points: LossPointLike[]): number | undefined {
  return points.find(p => typeof p.train_loss === 'number' && p.train_loss > 0)?.train_loss ?? undefined;
}

/**
 * Verbesserung in Prozent gegenüber dem ersten brauchbaren Wert.
 * Positiv = Loss gefallen. `null`, wenn sich nichts sinnvoll berechnen lässt.
 */
export function lossImprovementPct(points: LossPointLike[]): number | null {
  const first = firstUsableLoss(points);
  const last = points[points.length - 1]?.train_loss;
  if (first == null || last == null || first <= 0 || first === last) return null;
  return ((first - last) / first) * 100;
}
