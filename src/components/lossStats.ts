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

/**
 * Haengt einen Loss-Punkt an — oder fuehrt ihn mit dem letzten zusammen, wenn
 * beide denselben `step` haben.
 *
 * Eval-Events (die finale Evaluation bei max_steps, oder step-basierte Eval, die
 * mit einem Log-Step zusammenfaellt) tragen denselben `step` wie der letzte
 * Trainingspunkt. Als eigener Punkt wuerden sie den Loss-Graphen ueber die echte
 * Schrittzahl hinaus verlaengern ("Step 60/60, aber 9 Punkte" und scheinbares
 * Weiterlaufen ueber 100%). Ein vorhandenes val_loss bleibt erhalten, wenn das
 * neue Event keins mitbringt.
 */
export function appendLossPoint<T extends { step: number; val_loss?: number | null }>(
  points: T[],
  point: T,
): T[] {
  const last = points[points.length - 1];
  if (last && last.step === point.step) {
    const merged = { ...last, ...point, val_loss: point.val_loss ?? last.val_loss };
    return [...points.slice(0, -1), merged];
  }
  return [...points, point];
}
