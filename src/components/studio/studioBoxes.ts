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
