// CSV/TSV-Zerlegung mit Anfuehrungszeichen.
//
// Regression aus dem E2E-Test vom 19.08.2026: Das Labor zerlegte CSV-Zeilen mit
// line.split(','). Bei summary_demo steht das Ziel aber in Anfuehrungszeichen
// ("Bingen: sonnig, 0 Grad"), also wurde es am Komma abgeschnitten — die
// korrekte Vorhersage des Modells galt dadurch als "Abweichend".

/** Zerlegt eine Zeile am Trennzeichen; in Anfuehrungszeichen bleibt es Text. */
export function splitDelimitedLine(line: string, sep: string): string[] {
  const fields: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        // "" innerhalb eines Feldes ist ein echtes Anfuehrungszeichen
        if (line[i + 1] === '"') { current += '"'; i++; }
        else inQuotes = false;
      } else {
        current += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === sep) {
      fields.push(current);
      current = '';
    } else {
      current += ch;
    }
  }
  fields.push(current);
  return fields.map(f => f.trim());
}

/** True, wenn die Zeile ein offenes Anfuehrungszeichen hat (Feld geht weiter). */
function hasOpenQuote(line: string): boolean {
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    if (line[i] !== '"') continue;
    if (inQuotes && line[i + 1] === '"') { i++; continue; }
    inQuotes = !inQuotes;
  }
  return inQuotes;
}

/**
 * Zeilen eines CSV/TSV-Textes, wobei Zeilenumbrueche innerhalb von
 * Anfuehrungszeichen zum selben Datensatz gehoeren.
 */
export function joinQuotedLines(content: string): string[] {
  const out: string[] = [];
  let buffer = '';

  for (const raw of content.split('\n')) {
    const line = raw.replace(/\r$/, '');
    buffer = buffer === '' ? line : `${buffer}\n${line}`;
    if (!hasOpenQuote(buffer)) {
      if (buffer.trim()) out.push(buffer);
      buffer = '';
    }
  }
  if (buffer.trim()) out.push(buffer);
  return out;
}

/**
 * Zerlegt einen CSV/TSV-Text in Objekte (erste Zeile = Kopfzeile).
 * Bei nur einer Spalte kommen die reinen Werte zurueck.
 */
export function parseDelimitedRows(content: string, sep: string): Array<Record<string, string> | string> {
  const lines = joinQuotedLines(content);
  if (lines.length === 0) return [];

  const headers = splitDelimitedLine(lines[0], sep);
  const rows: Array<Record<string, string> | string> = [];

  for (const line of lines.slice(1)) {
    const values = splitDelimitedLine(line, sep);
    if (headers.length > 1) {
      const obj: Record<string, string> = {};
      headers.forEach((h, i) => { obj[h] = values[i] ?? ''; });
      rows.push(obj);
    } else {
      rows.push(values[0]);
    }
  }
  return rows;
}
