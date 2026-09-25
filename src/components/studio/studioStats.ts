// Stand der Zahlen rechts in der Werkbank, ohne nach jedem Tastendruck neu
// zu zaehlen.
//
// Bisher wurde nach dem Speichern nur der Status verschoben (offen ->
// bestaetigt). Boxen und die Anzahl je Klasse blieben stehen, bis man das
// Projekt neu oeffnete — "Boxen 0" neben einem Bild mit Box. Hier steht, was
// eine einzelne Aenderung an allen Zahlen bewirkt, genau so gezaehlt wie in
// studio_stats im Backend:
//   - Boxen zaehlen immer, auch vorgeschlagene.
//   - Ein Label (Text, Audio) zaehlt nur, wenn das Sample bestaetigt ist —
//     ein Modellvorschlag ist noch keine Aussage ueber die Verteilung.
//   - "Bestaetigt ohne Box" gibt es nur bei Bildern.

import type { SampleStatus, StudioStats } from './studioTypes';

export interface SampleStand {
  status: SampleStatus;
  boxes: { cls: number }[];
  label?: string | null;
}

/** Dieselbe Zuordnung wie class_index_for im Backend. */
export function classIndexFor(label: string | null | undefined, classes: string[]): number {
  if (!label) return -1;
  const needle = label.trim().toLowerCase();
  return classes.findIndex(c => c.trim().toLowerCase() === needle);
}

function beitrag(stand: SampleStand, classes: string[], bild: boolean) {
  const perClass = new Array<number>(classes.length).fill(0);
  for (const b of stand.boxes) if (b.cls >= 0 && b.cls < perClass.length) perClass[b.cls] += 1;
  if (stand.status === 'confirmed') {
    const i = classIndexFor(stand.label, classes);
    if (i >= 0) perClass[i] += 1;
  }
  return {
    boxes: stand.boxes.length,
    perClass,
    leer: bild && stand.status === 'confirmed' && stand.boxes.length === 0 ? 1 : 0,
  };
}

/** Zahlen nach einer Aenderung eines Samples von `vorher` nach `nachher`. */
export function statsNachAenderung(
  stats: StudioStats, classes: string[], vorher: SampleStand, nachher: SampleStand, bild: boolean,
): StudioStats {
  const a = beitrag(vorher, classes, bild);
  const b = beitrag(nachher, classes, bild);
  const perClass = classes.map((_, i) =>
    Math.max(0, (stats.per_class?.[i] ?? 0) - a.perClass[i] + b.perClass[i]));
  const next: StudioStats = {
    ...stats,
    boxes_total: Math.max(0, stats.boxes_total - a.boxes + b.boxes),
    per_class: perClass,
    empty_confirmed: Math.max(0, stats.empty_confirmed - a.leer + b.leer),
  };
  if (vorher.status !== nachher.status) {
    next[vorher.status] = Math.max(0, stats[vorher.status] - 1);
    next[nachher.status] = stats[nachher.status] + 1;
  }
  return next;
}
