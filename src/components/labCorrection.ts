// Korrekturen im Labor: "so haette es aussehen muessen".
//
// Das Labor kennt bisher nur das Urteil richtig/falsch. Was daran falsch war,
// blieb im Kopf des Nutzers. Eine Korrektur haelt es fest — und je nach
// Modalitaet sieht sie anders aus: eine Box im Bild, eine Klasse aus der
// Liste, ein umgeschriebener Text.
//
// Verfuegbar ist eine Loesung nicht immer: gibt es Dataset-Labels, wird der
// Editor damit vorbelegt und man korrigiert nur; gibt es keine, legt der
// Nutzer sie hier ueberhaupt erst an.

import type { TruthBox } from './labGroundTruth';

export type Correction =
  | { kind: 'boxes'; boxes: TruthBox[] }
  | { kind: 'label'; label: string }
  | { kind: 'text';  text: string };

/** Welchen Editor eine Modalitaet braucht. */
export type CorrectionKind = 'boxes' | 'label' | 'text';

/**
 * Editor-Form aus der Modalitaet, die der Lab-Server beim Laden meldet.
 *
 * Bewusst ueber die Modalitaet und nicht ueber das Plugin: das Labor kennt
 * den Server, nicht die Plugin-Registry — und der Dev-Script-Modus hat gar
 * kein Plugin.
 */
export function correctionKindFor(
  modality: string | null | undefined,
  fileKind?: 'image' | 'audio' | null,
): CorrectionKind {
  if (modality === 'detect') return 'boxes';
  if (modality === 'seq2seq') return 'text';
  if (modality === 'text' || modality === 'image' || modality === 'audio' || modality === 'canvas') {
    return 'label';
  }
  // Unbekannte Modalitaet (Dev-Script): bei Dateien hilft nur ein Label,
  // bei Text ist der freie Text die ehrlichere Annahme.
  return fileKind ? 'label' : 'text';
}

/**
 * Vorbelegung des Editors.
 *
 * Reihenfolge mit Absicht: das Soll aus dem Dataset zuerst — wer eine
 * Labeldatei hat, will sie nicht abtippen. Sonst die Vorhersage des Modells:
 * eine fast richtige Box zu verschieben ist schneller als sie neu zu ziehen.
 */
export function initialCorrection(
  kind: CorrectionKind,
  opts: {
    truthBoxes?: TruthBox[];
    predictedBoxes?: TruthBox[];
    expectedLabel?: string;
    predictedText?: string;
  },
): Correction {
  if (kind === 'boxes') {
    const boxes = (opts.truthBoxes?.length ? opts.truthBoxes : opts.predictedBoxes) ?? [];
    return { kind: 'boxes', boxes: boxes.map(b => ({ ...b })) };
  }
  if (kind === 'label') {
    return { kind: 'label', label: opts.expectedLabel ?? opts.predictedText ?? '' };
  }
  return { kind: 'text', text: opts.expectedLabel ?? opts.predictedText ?? '' };
}

/** Ist an der Korrektur ueberhaupt etwas dran? */
export function isCorrectionEmpty(c: Correction | undefined | null): boolean {
  if (!c) return true;
  if (c.kind === 'boxes') return c.boxes.length === 0;
  if (c.kind === 'label') return c.label.trim() === '';
  return c.text.trim() === '';
}

/** Einzeiler fuer Ergebnisliste und CSV-Export. */
export function describeCorrection(c: Correction | undefined | null): string {
  if (!c || isCorrectionEmpty(c)) return '';
  if (c.kind === 'label') return c.label.trim();
  if (c.kind === 'text') return c.text.trim();
  const labels: string[] = [];
  for (const b of c.boxes) if (!labels.includes(b.label)) labels.push(b.label);
  return `${c.boxes.length}x ${labels.slice(0, 3).join(', ')}${labels.length > 3 ? ` +${labels.length - 3}` : ''}`;
}

/**
 * Bildschirm-Punkt in Bildkoordinaten umrechnen.
 *
 * Das Bild wird mit object-contain eingepasst, das Overlay mit demselben
 * preserveAspectRatio. Beim Zeichnen einer Box muss der Mauszeiger denselben
 * Weg zurueck: Randstreifen abziehen, durch den Massstab teilen. Ohne das
 * landen Boxen bei nicht-quadratischen Bildern verschoben.
 */
export function clientToImagePoint(
  rect: { left: number; top: number; width: number; height: number },
  imageWidth: number,
  imageHeight: number,
  clientX: number,
  clientY: number,
): { x: number; y: number } {
  if (imageWidth <= 0 || imageHeight <= 0 || rect.width <= 0 || rect.height <= 0) {
    return { x: 0, y: 0 };
  }
  const scale = Math.min(rect.width / imageWidth, rect.height / imageHeight);
  const offsetX = (rect.width - imageWidth * scale) / 2;
  const offsetY = (rect.height - imageHeight * scale) / 2;
  const x = (clientX - rect.left - offsetX) / scale;
  const y = (clientY - rect.top - offsetY) / scale;
  return {
    x: Math.min(Math.max(x, 0), imageWidth),
    y: Math.min(Math.max(y, 0), imageHeight),
  };
}

/** Box aus zwei Punkten – egal in welche Richtung gezogen wurde. */
export function boxFromPoints(
  a: { x: number; y: number },
  b: { x: number; y: number },
  label: string,
): TruthBox {
  return {
    label,
    x1: Math.min(a.x, b.x), y1: Math.min(a.y, b.y),
    x2: Math.max(a.x, b.x), y2: Math.max(a.y, b.y),
  };
}

/** Zu kleine Rechtecke sind Fehlklicks, keine Boxen. */
export function isUsableBox(box: TruthBox, imageWidth: number, imageHeight: number): boolean {
  const minSide = Math.max(4, Math.min(imageWidth, imageHeight) * 0.01);
  return (box.x2 - box.x1) >= minSide && (box.y2 - box.y1) >= minSide;
}
