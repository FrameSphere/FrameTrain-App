// Zahleneingabe für die Trainings-Konfiguration.
//
// Hintergrund: Ein `<input type="number">`, dessen Wert direkt an eine Zahl
// gebunden ist, laesst sich nicht mit Dezimalstellen befuellen. Tippt man
// "0.001", liefert der Browser nach dem Punkt einen LEEREN Wert (ungueltiger
// Zwischenzustand). parseFloat("") ist NaN, `|| 0` macht daraus 0 — und das
// Rendern loescht das gerade getippte Zeichen wieder. Im Feld landete "0001".
// Betroffen waren Learning Rate, Warmup Ratio, Weight Decay, Dropout,
// Label Smoothing und die Adam-Betas.

/** Zwischenstaende beim Tippen, die noch keine Zahl ergeben, aber gueltig sind. */
export function isIncompleteNumber(raw: string): boolean {
  return /^-?$|^-?[.,]$|^-?\d+[.,]$|^-?[.,]\d*$/.test(raw.trim());
}

/**
 * Wandelt die Eingabe in eine Zahl um. Akzeptiert Punkt und Komma als
 * Dezimaltrennzeichen — im Deutschen zeigt die App "0,06" an, also muss man
 * das auch eintippen koennen.
 *
 * @returns die Zahl, oder null wenn (noch) keine gueltige Zahl vorliegt.
 */
export function parseNumberInput(raw: string): number | null {
  const cleaned = raw.trim().replace(',', '.');
  if (cleaned === '' || isIncompleteNumber(raw)) return null;
  const n = Number(cleaned);
  return Number.isFinite(n) ? n : null;
}

/** Begrenzt einen Wert auf das erlaubte Intervall. */
export function clampNumber(n: number, min?: number, max?: number): number {
  if (min !== undefined && n < min) return min;
  if (max !== undefined && n > max) return max;
  return n;
}
