// Soll-Werte fuer Objekterkennung im Labor.
//
// Bei Bild-Datasets nimmt das Labor den uebergeordneten Ordnernamen als
// erwartetes Label (ImageFolder-Konvention). Bei einem YOLO-Dataset heisst
// dieser Ordner "images" — als Erwartung ist das wertlos. Die Wahrheit liegt
// dort in einer .txt neben dem Bild, und die Klassennamen stehen in der
// data.yaml des Datasets.
//
// Nichts davon ist fest verdrahtet: Klassennamen kommen aus dem Dataset oder
// vom Modell, nie aus einer Liste in FrameTrain.

/** Eine Box in Pixelkoordinaten des Originalbildes. */
export interface TruthBox {
  label: string;
  x1: number; y1: number; x2: number; y2: number;
}

/**
 * Pfad der YOLO-Labeldatei zu einem Bild.
 *
 * Konvention von Ultralytics: der letzte Ordner "images" im Pfad wird zu
 * "labels", die Endung zu .txt. Es gibt Datasets, die beides nebeneinander
 * legen (bild.jpg + bild.txt) — auch das wird zurueckgegeben.
 */
export function labelPathsForImage(imagePath: string): string[] {
  const norm = imagePath.replace(/\\/g, '/');
  const dot = norm.lastIndexOf('.');
  if (dot <= 0) return [];
  const withoutExt = norm.slice(0, dot);

  const candidates: string[] = [];
  // Letztes /images/-Segment ersetzen – nicht das erste: ein Pfad kann
  // "/images/" auch weiter oben enthalten.
  const idx = withoutExt.toLowerCase().lastIndexOf('/images/');
  if (idx >= 0) {
    candidates.push(
      withoutExt.slice(0, idx) + '/labels/' + withoutExt.slice(idx + '/images/'.length) + '.txt',
    );
  }
  candidates.push(withoutExt + '.txt');
  return candidates;
}

/**
 * Klassennamen aus einer data.yaml / dataset.yaml.
 *
 * Ultralytics erlaubt zwei Schreibweisen, beide kommen in echten Datasets vor:
 *   names: ['Tree', 'Stone']        bzw.  names:\n  - Tree\n  - Stone
 *   names:\n  0: Tree\n  1: Stone
 * Ein vollstaendiger YAML-Parser waere fuer diesen einen Schluessel zu viel.
 */
export function classNamesFromYaml(yamlText: string): Record<number, string> {
  const out: Record<number, string> = {};
  const lines = yamlText.split(/\r?\n/);

  const strip = (raw: string) => raw.trim().replace(/^['"]|['"]$/g, '').trim();

  for (let i = 0; i < lines.length; i++) {
    const m = /^\s*names\s*:\s*(.*)$/.exec(lines[i]);
    if (!m) continue;

    const inline = m[1].trim();
    // Einzeiler: names: ['a', 'b'] oder names: [a, b]
    if (inline.startsWith('[')) {
      const inner = inline.slice(1, inline.lastIndexOf(']') > 0 ? inline.lastIndexOf(']') : undefined);
      inner.split(',').map(strip).filter(Boolean).forEach((name, idx) => { out[idx] = name; });
      return out;
    }

    // Block darunter einlesen, bis die Einrueckung endet.
    let autoIndex = 0;
    for (let j = i + 1; j < lines.length; j++) {
      const line = lines[j];
      if (!line.trim()) continue;
      if (!/^\s/.test(line)) break;            // neuer Top-Level-Schluessel
      if (/^\s*#/.test(line)) continue;        // Kommentar

      const listItem = /^\s*-\s*(.+)$/.exec(line);
      if (listItem) {
        const name = strip(listItem[1]);
        if (name) out[autoIndex++] = name;
        continue;
      }
      const mapItem = /^\s*(\d+)\s*:\s*(.+)$/.exec(line);
      if (mapItem) {
        out[Number(mapItem[1])] = strip(mapItem[2]);
        continue;
      }
      break;                                    // etwas anderes – Block zu Ende
    }
    return out;
  }
  return out;
}

/** Klassenname zu einer Id – Dataset zuerst, dann Modell, sonst die nackte Id. */
export function labelForClassId(
  id: number,
  fromDataset: Record<number, string>,
  fromModel: string[],
): string {
  return fromDataset[id] ?? fromModel[id] ?? `Klasse ${id}`;
}

/**
 * Liest eine YOLO-Labeldatei.
 *
 * Zeilenformat `cls cx cy w h`, alle Werte auf 0–1 normiert. Segmentierungs-
 * Datasets schreiben stattdessen ein Polygon (`cls x1 y1 x2 y2 …`); daraus
 * wird die umschliessende Box berechnet, damit auch diese Datasets im Labor
 * eine Erwartung haben.
 */
export function parseYoloLabelFile(
  text: string,
  imageWidth: number,
  imageHeight: number,
  fromDataset: Record<number, string> = {},
  fromModel: string[] = [],
): TruthBox[] {
  if (imageWidth <= 0 || imageHeight <= 0) return [];
  const boxes: TruthBox[] = [];

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;

    const parts = line.split(/\s+/);
    const nums = parts.map(Number);
    if (nums.some(n => !Number.isFinite(n))) continue;

    const id = Math.trunc(nums[0]);
    const label = labelForClassId(id, fromDataset, fromModel);

    if (nums.length === 5) {
      const [, cx, cy, w, h] = nums;
      boxes.push({
        label,
        x1: (cx - w / 2) * imageWidth,
        y1: (cy - h / 2) * imageHeight,
        x2: (cx + w / 2) * imageWidth,
        y2: (cy + h / 2) * imageHeight,
      });
      continue;
    }

    if (nums.length >= 7 && (nums.length - 1) % 2 === 0) {
      const xs: number[] = [];
      const ys: number[] = [];
      for (let i = 1; i < nums.length; i += 2) { xs.push(nums[i]); ys.push(nums[i + 1]); }
      boxes.push({
        label,
        x1: Math.min(...xs) * imageWidth,
        y1: Math.min(...ys) * imageHeight,
        x2: Math.max(...xs) * imageWidth,
        y2: Math.max(...ys) * imageHeight,
      });
    }
  }
  return boxes;
}

/**
 * Kurzfassung einer Boxliste — dieselbe Form, die der YOLO-Server fuer die
 * Vorhersage liefert ("3x Sky, Tree"), damit Soll und Ist vergleichbar
 * nebeneinander stehen.
 */
export function summarizeBoxes(boxes: { label: string }[]): string {
  if (boxes.length === 0) return 'Keine Objekte';
  const seen: string[] = [];
  for (const b of boxes) if (!seen.includes(b.label)) seen.push(b.label);
  const head = seen.slice(0, 3).join(', ');
  return `${boxes.length}x ${head}${seen.length > 3 ? ` +${seen.length - 3}` : ''}`;
}

/**
 * Vergleicht die Klassen aus der Labeldatei mit den erkannten.
 *
 * Bei Objekterkennung ist ein Textvergleich sinnlos: "6x Sky, Tree" gleicht
 * nie einem Ordnernamen, und zwei Boxen mehr machen die Erkennung nicht
 * falsch. Interessant ist, welche Soll-Klasse fehlt und welche Klasse dazukam
 * — das entscheidet am Ende trotzdem der Mensch per Bewertung.
 */
export function compareClassSets(
  truth: { label: string }[],
  predicted: { label: string }[],
): { missing: string[]; extra: string[] } {
  const truthSet = new Set(truth.map(b => b.label));
  const predSet = new Set(predicted.map(b => b.label));
  return {
    missing: [...truthSet].filter(l => !predSet.has(l)),
    extra: [...predSet].filter(l => !truthSet.has(l)),
  };
}

/**
 * Farbpalette fuer Klassen im Bild-Overlay.
 *
 * Helle Toene, weil das Overlay auf dunklen Fotos ebenso lesbar sein muss wie
 * auf hellen. Reihenfolge so gewaehlt, dass benachbarte Eintraege sich deutlich
 * unterscheiden — bei einem Modell mit vielen Klassen liegen sonst zwei
 * aehnliche Toene nebeneinander im selben Bild.
 */
export const CLASS_COLORS = [
  '#f472b6', '#38bdf8', '#34d399', '#fbbf24', '#a78bfa', '#fb923c',
  '#22d3ee', '#f87171', '#4ade80', '#c084fc', '#facc15', '#2dd4bf',
] as const;

/**
 * Feste Farbe je Klasse.
 *
 * Erste Wahl ist die Position in der Klassenliste des Modells: die ist pro
 * Modell fest, und benachbarte Klassen bekommen dadurch garantiert
 * verschiedene Farben. Ein reiner Namens-Hash tut das nicht — bei den 13
 * Ski-Klassen fielen darueber mehrere auf denselben Ton.
 *
 * Der Hash bleibt als Rueckfall fuer Klassen, die nicht in der Liste stehen
 * (etwa aus einer Labeldatei mit eigenen Ids). Wichtig ist in beiden Faellen:
 * dieselbe Klasse behaelt beim Durchklicken ihre Farbe, und Soll und
 * Erkennung derselben Klasse gehoeren sichtbar zusammen.
 */
export function classColor(label: string, classes: readonly string[] = []): string {
  const index = classes.indexOf(label);
  if (index >= 0) return CLASS_COLORS[index % CLASS_COLORS.length];

  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = (hash * 31 + label.charCodeAt(i)) | 0;
  }
  return CLASS_COLORS[Math.abs(hash) % CLASS_COLORS.length];
}

/** Klassen im Bild – Soll und Erkennung zusammen, fuer die Legende. */
export function legendEntries(
  truth: { label: string }[],
  predicted: { label: string }[],
  classes: readonly string[] = [],
): { label: string; color: string }[] {
  const seen: string[] = [];
  for (const b of [...truth, ...predicted]) {
    if (!seen.includes(b.label)) seen.push(b.label);
  }
  return seen.sort().map(label => ({ label, color: classColor(label, classes) }));
}
