// Boxen im Dataset Studio: Speicherformat, Auswahl, Verschieben, Groesse.
//
// Gespeichert wird normalisiert (Mittelpunkt + Groesse, 0..1) — das ist die
// YOLO-Konvention und macht den Export zur reinen Textausgabe. Gearbeitet wird
// in Bildpixeln, weil dort gezeichnet und getroffen wird. Die Umrechnung
// passiert genau an dieser Grenze und nirgends sonst.
//
// Das Zeichnen selbst (Bildschirmpunkt -> Bildpunkt, Box aus zwei Punkten)
// liegt bereits in labCorrection.ts und wird von dort benutzt, nicht kopiert.

/** Wie es auf der Platte liegt. */
export interface StudioBox {
  cls: number;
  x: number; y: number; w: number; h: number;
}

/** Wie damit gearbeitet wird: Kanten in Bildpixeln. */
export interface PixelBox {
  cls: number;
  x1: number; y1: number; x2: number; y2: number;
}

export type Handle = 'nw' | 'ne' | 'sw' | 'se';

/** Kleinste Kantenlaenge in Bildanteil — alles darunter ist ein Fehlklick. */
export const MIN_SIDE = 0.005;

export function toPixel(b: StudioBox, width: number, height: number): PixelBox {
  return {
    cls: b.cls,
    x1: (b.x - b.w / 2) * width,
    y1: (b.y - b.h / 2) * height,
    x2: (b.x + b.w / 2) * width,
    y2: (b.y + b.h / 2) * height,
  };
}

export function toNormalized(p: PixelBox, width: number, height: number): StudioBox {
  if (width <= 0 || height <= 0) return { cls: p.cls, x: 0, y: 0, w: 0, h: 0 };
  const x1 = Math.min(p.x1, p.x2), x2 = Math.max(p.x1, p.x2);
  const y1 = Math.min(p.y1, p.y2), y2 = Math.max(p.y1, p.y2);
  const clamp = (v: number) => Math.min(Math.max(v, 0), 1);
  return {
    cls: p.cls,
    x: clamp(((x1 + x2) / 2) / width),
    y: clamp(((y1 + y2) / 2) / height),
    w: clamp((x2 - x1) / width),
    h: clamp((y2 - y1) / height),
  };
}

/** Zu kleine Boxen entstehen durch Klicken statt Ziehen. */
export function isUsable(b: StudioBox): boolean {
  return b.w >= MIN_SIDE && b.h >= MIN_SIDE;
}

/**
 * Welche Box liegt unter dem Punkt?
 *
 * Bei verschachtelten Boxen gewinnt die kleinere. Andernfalls waere eine Box,
 * die innerhalb einer grossen liegt, nie anklickbar.
 */
export function hitTest(boxes: PixelBox[], x: number, y: number): number {
  let best = -1;
  let bestArea = Infinity;
  boxes.forEach((b, i) => {
    if (x < Math.min(b.x1, b.x2) || x > Math.max(b.x1, b.x2)) return;
    if (y < Math.min(b.y1, b.y2) || y > Math.max(b.y1, b.y2)) return;
    const area = Math.abs(b.x2 - b.x1) * Math.abs(b.y2 - b.y1);
    if (area < bestArea) { bestArea = area; best = i; }
  });
  return best;
}

/** Ecke unter dem Punkt, oder null. `tol` in Bildpixeln. */
export function handleAt(b: PixelBox, x: number, y: number, tol: number): Handle | null {
  const corners: [Handle, number, number][] = [
    ['nw', b.x1, b.y1], ['ne', b.x2, b.y1],
    ['sw', b.x1, b.y2], ['se', b.x2, b.y2],
  ];
  for (const [handle, cx, cy] of corners) {
    if (Math.abs(x - cx) <= tol && Math.abs(y - cy) <= tol) return handle;
  }
  return null;
}

/** Ecke auf den Punkt ziehen. Ueber die gegenueberliegende hinaus klappt die Box um. */
export function resizeTo(b: PixelBox, handle: Handle, x: number, y: number): PixelBox {
  const out = { ...b };
  if (handle === 'nw' || handle === 'sw') out.x1 = x; else out.x2 = x;
  if (handle === 'nw' || handle === 'ne') out.y1 = y; else out.y2 = y;
  return {
    cls: out.cls,
    x1: Math.min(out.x1, out.x2), y1: Math.min(out.y1, out.y2),
    x2: Math.max(out.x1, out.x2), y2: Math.max(out.y1, out.y2),
  };
}

/** Verschieben, ohne das Bild zu verlassen — eine Box ausserhalb waere im Export weg. */
export function movedBy(b: PixelBox, dx: number, dy: number, width: number, height: number): PixelBox {
  const w = b.x2 - b.x1;
  const h = b.y2 - b.y1;
  const x1 = Math.min(Math.max(b.x1 + dx, 0), Math.max(width - w, 0));
  const y1 = Math.min(Math.max(b.y1 + dy, 0), Math.max(height - h, 0));
  return { cls: b.cls, x1, y1, x2: x1 + w, y2: y1 + h };
}

/** Klassenname zu einer ID. Namen kommen immer aus dem Projekt, nie aus einer festen Liste. */
export function classLabel(cls: number, classes: string[]): string {
  return classes[cls] ?? `#${cls}`;
}

/** Kurzfassung der Boxen eines Bildes, z.B. "2x Lift, Sky". */
export function summarize(boxes: { cls: number }[], classes: string[]): string {
  if (boxes.length === 0) return '';
  const counts = new Map<number, number>();
  boxes.forEach(b => counts.set(b.cls, (counts.get(b.cls) ?? 0) + 1));
  return [...counts.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([cls, n]) => (n > 1 ? `${n}x ${classLabel(cls, classes)}` : classLabel(cls, classes)))
    .join(', ');
}

/**
 * Naechstes Bild, das noch Arbeit braucht.
 *
 * Nach dem Bestaetigen soll der Blick nicht auf etwas Fertigem landen. Gesucht
 * wird ab `from` vorwaerts, danach von vorn — sonst bleibt die Lücke am Anfang
 * einer langen Liste für immer liegen.
 */
export function nextOpenIndex(statuses: string[], from: number): number {
  const open = (s: string) => s === 'new' || s === 'suggested';
  for (let i = from + 1; i < statuses.length; i++) if (open(statuses[i])) return i;
  for (let i = 0; i <= from && i < statuses.length; i++) if (open(statuses[i])) return i;
  return -1;
}

// ── Boxen per Tastatur ─────────────────────────────────────────────────────
//
// Zeichnen geht nur mit der Maus; wer schnell labelt, hat die Hand aber an
// der Tastatur. Diese Funktionen stehen hinter B (neue Box), den Pfeiltasten
// (verschieben, mit ⌥ Groesse aendern), ⌘D (duplizieren) und Tab (naechste
// Box). Alle bleiben im Bild — eine Box ausserhalb waere im Export weg.

/** Neue Box mittig, ein Viertel so gross wie das Bild. */
export function boxInDerMitte(cls: number, width: number, height: number): PixelBox {
  const w = width / 4;
  const h = height / 4;
  return { cls, x1: (width - w) / 2, y1: (height - h) / 2, x2: (width + w) / 2, y2: (height + h) / 2 };
}

/**
 * Groesse um die Mitte aendern. `dw`/`dh` sind Pixel je Seite zusammen;
 * kleiner als MIN_SIDE wird eine Box nie, groesser als das Bild auch nicht.
 */
export function resizedBy(b: PixelBox, dw: number, dh: number, width: number, height: number): PixelBox {
  const minW = Math.max(1, width * MIN_SIDE * 2);
  const minH = Math.max(1, height * MIN_SIDE * 2);
  const w = Math.min(Math.max(b.x2 - b.x1 + dw, minW), width);
  const h = Math.min(Math.max(b.y2 - b.y1 + dh, minH), height);
  const cx = (b.x1 + b.x2) / 2;
  const cy = (b.y1 + b.y2) / 2;
  const x1 = Math.min(Math.max(cx - w / 2, 0), width - w);
  const y1 = Math.min(Math.max(cy - h / 2, 0), height - h);
  return { cls: b.cls, x1, y1, x2: x1 + w, y2: y1 + h };
}

/** Kopie leicht versetzt, damit man sie sieht und gleich verschieben kann. */
export function duplicated(b: PixelBox, width: number, height: number): PixelBox {
  const d = Math.min(width, height) * 0.03;
  const moved = movedBy(b, d, d, width, height);
  // Stand die Box schon am Rand, rutscht die Kopie in die andere Richtung.
  if (moved.x1 === b.x1 && moved.y1 === b.y1) return movedBy(b, -d, -d, width, height);
  return moved;
}

/** Naechste (dir=1) oder vorige (dir=-1) Box, im Kreis. Ohne Auswahl die erste bzw. letzte. */
export function cycleSelection(count: number, selected: number, dir: 1 | -1): number {
  if (count === 0) return -1;
  if (selected < 0) return dir === 1 ? 0 : count - 1;
  return (selected + dir + count) % count;
}

/** Schrittweite der Pfeiltasten: 1 % der kuerzeren Bildseite, mit ⇧ 5 %. */
export function nudgeStep(width: number, height: number, weit: boolean): number {
  return Math.max(1, Math.min(width, height) * (weit ? 0.05 : 0.01));
}

// ── Boxen ueber die Zwischenablage ─────────────────────────────────────────
//
// ⌘C legt die Boxen als Text in die System-Zwischenablage, ⌘V holt sie wieder.
// Ueber die Systemablage statt einer internen: dann gilt, was zuletzt kopiert
// wurde — ein danach kopierter Screenshot wird als Bild eingefuegt, nicht
// von alten Boxen verdrängt. Normalisiert, damit sie auf Bildern anderer
// Groesse an derselben Stelle landen.

const ABLAGE_KENNUNG = 'frametrain-boxes';

export function boxesToClipboardText(boxes: StudioBox[]): string {
  return JSON.stringify({ [ABLAGE_KENNUNG]: boxes });
}

export function boxesFromClipboardText(text: string | null | undefined): StudioBox[] | null {
  if (!text || !text.includes(ABLAGE_KENNUNG)) return null;
  try {
    const parsed = JSON.parse(text) as Record<string, unknown>;
    const liste = parsed[ABLAGE_KENNUNG];
    if (!Array.isArray(liste)) return null;
    const boxes = liste.filter((b): b is StudioBox =>
      !!b && typeof b === 'object'
      && ['cls', 'x', 'y', 'w', 'h'].every(k => typeof (b as Record<string, unknown>)[k] === 'number'));
    return boxes.length > 0 ? boxes : null;
  } catch {
    return null;
  }
}
