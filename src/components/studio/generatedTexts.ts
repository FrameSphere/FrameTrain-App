// Was ein Sprachmodell zurueckschickt, in Zeilen fuer den Datensatz verwandeln.
//
// Gefragt wird nach JSON, geliefert wird alles Moegliche: ein ```json-Block,
// eine nummerierte Liste, Anfuehrungszeichen um jede Zeile, ein Vorspann
// ("Hier sind 10 Beispiele:"). Das ist keine Ausnahme, sondern der Normalfall —
// und ein abgelehnter Lauf kostet den Nutzer echtes Geld. Deshalb steht hier
// eine Kette von Strategien statt eines JSON.parse.

export interface GeneratedItem {
  text: string;
  label?: string | null;
  target?: string | null;
}

/** Entfernt ```json-Zaeune, Nummerierung, Aufzaehlungszeichen und Klammern. */
function zeileSaeubern(z: string): string {
  let s = z.trim();
  s = s.replace(/^[-*•]\s+/, '');
  s = s.replace(/^\d+[.)]\s+/, '');
  s = s.replace(/,\s*$/, '');
  // Eine Zeile, die komplett in Anfuehrungszeichen steht, verliert sie.
  const m = s.match(/^"([\s\S]*)"$/) ?? s.match(/^'([\s\S]*)'$/);
  if (m) s = m[1].replace(/\\"/g, '"').replace(/\\n/g, '\n');
  return s.trim();
}

function ausObjekt(o: Record<string, unknown>): GeneratedItem | null {
  const first = (...keys: string[]): string | null => {
    for (const k of keys) {
      const v = o[k];
      if (typeof v === 'string' && v.trim()) return v.trim();
    }
    return null;
  };
  const text = first('text', 'input', 'source', 'prompt', 'frage', 'satz');
  if (!text) return null;
  return {
    text,
    label: first('label', 'class', 'klasse', 'kategorie'),
    target: first('target', 'output', 'answer', 'antwort', 'response'),
  };
}

function ausArray(arr: unknown[]): GeneratedItem[] {
  const out: GeneratedItem[] = [];
  for (const el of arr) {
    if (typeof el === 'string') {
      const t = el.trim();
      if (t) out.push({ text: t });
    } else if (el && typeof el === 'object' && !Array.isArray(el)) {
      const item = ausObjekt(el as Record<string, unknown>);
      if (item) out.push(item);
    }
  }
  return out;
}

/**
 * Liest die erzeugten Zeilen aus einer Modellantwort.
 *
 * Reihenfolge: erst als JSON-Array (mit und ohne Code-Zaun), dann zeilenweise.
 * Die zeilenweise Lesart ist die wichtige — kleinere Modelle liefern fast nie
 * gueltiges JSON, und eine Liste ist trotzdem brauchbar.
 */
export function parseGenerated(antwort: string): GeneratedItem[] {
  const text = (antwort ?? '').trim();
  if (!text) return [];

  const kandidaten: string[] = [];
  for (const m of text.matchAll(/```(?:json)?\s*([\s\S]*?)```/gi)) kandidaten.push(m[1]);
  const start = text.indexOf('[');
  const ende = text.lastIndexOf(']');
  if (start >= 0 && ende > start) kandidaten.push(text.slice(start, ende + 1));
  kandidaten.push(text);

  for (const k of kandidaten) {
    try {
      const geparst = JSON.parse(k.trim());
      if (Array.isArray(geparst)) {
        const items = ausArray(geparst);
        if (items.length > 0) return items;
      }
    } catch { /* naechste Strategie */ }
  }

  // Zeilenweise: JSONL zuerst, sonst der gesaeuberte Rohtext. Eine Zeile mit
  // Doppelpunkt am Ende ist eine Ueberschrift ("Hier sind 10 Beispiele:") und
  // gehoert nicht in den Datensatz.
  const out: GeneratedItem[] = [];
  for (const roh of text.split('\n')) {
    const zeile = roh.trim();
    if (!zeile || zeile === '[' || zeile === ']' || zeile.startsWith('```')) continue;
    if (zeile.startsWith('{')) {
      try {
        const o = JSON.parse(zeile.replace(/,\s*$/, ''));
        if (o && typeof o === 'object' && !Array.isArray(o)) {
          const item = ausObjekt(o as Record<string, unknown>);
          if (item) { out.push(item); continue; }
        }
      } catch { /* dann eben als Text */ }
    }
    const sauber = zeileSaeubern(zeile);
    if (!sauber || sauber.endsWith(':')) continue;
    out.push({ text: sauber });
  }
  return out;
}

/** Was schon im Projekt steht, soll nicht noch einmal erzeugt werden. */
export function ohneDubletten(items: GeneratedItem[], vorhanden: string[]): GeneratedItem[] {
  const gesehen = new Set(vorhanden.map(t => t.trim().toLowerCase()));
  const out: GeneratedItem[] = [];
  for (const i of items) {
    const key = i.text.trim().toLowerCase();
    if (!key || gesehen.has(key)) continue;
    gesehen.add(key);
    out.push(i);
  }
  return out;
}
