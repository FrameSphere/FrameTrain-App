// Zweimal aufgetreten: ein Uebersetzungstext enthaelt {platzhalter}, der Code
// setzt die Werte aber nicht ein — in der Oberflaeche stand dann woertlich
// "Epoche {epoch}/{total} 2/2" bzw. "{tested}/{total}".
// Dieser Test findet solche Stellen, bevor sie jemand in der App sieht.

import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import de from '../../locales/de.json';

function flatten(obj: unknown, prefix = ''): Record<string, string> {
  const out: Record<string, string> = {};
  if (obj && typeof obj === 'object') {
    for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
      const key = prefix ? `${prefix}.${k}` : k;
      if (typeof v === 'string') out[key] = v;
      else Object.assign(out, flatten(v, key));
    }
  }
  return out;
}

function sourceFiles(dir: string, acc: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    if (entry === 'node_modules' || entry === '__tests__') continue;
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) sourceFiles(full, acc);
    else if (/\.tsx?$/.test(entry)) acc.push(full);
  }
  return acc;
}

describe('Uebersetzungs-Platzhalter', () => {
  const strings = flatten(de);
  const withPlaceholder = Object.entries(strings)
    .filter(([, v]) => /\{[a-zA-Z][a-zA-Z0-9_]*\}/.test(v));
  const code = sourceFiles('src').map(f => readFileSync(f, 'utf8')).join('\n');

  it('findet ueberhaupt Texte mit Platzhaltern', () => {
    expect(withPlaceholder.length).toBeGreaterThan(20);
  });

  it('jeder benutzte Text mit Platzhalter wird auch befuellt', () => {
    const broken: string[] = [];
    for (const [key, value] of withPlaceholder) {
      // Wird der Schluessel ueberhaupt im Code verwendet?
      const usages = [...code.matchAll(new RegExp(`t\\(\\s*['"\`]${key.replace(/\./g, '\\.')}['"\`]\\s*[,)]`, 'g'))];
      if (usages.length === 0) continue;
      const placeholders = [...value.matchAll(/\{([a-zA-Z][a-zA-Z0-9_]*)\}/g)].map(m => m[1]);
      for (const m of usages) {
        // Umkreis um die Verwendung: hier muss jeder Platzhalter befuellt werden.
        // Beruecksichtigt auch (bedingung ? t(a) : t(b)).replace(...) und
        // Aufrufe mit Argument-Objekt.
        const around = code.slice(m.index! - 200, m.index! + m[0].length + 300);
        const calledWithArgs = m[0].trimEnd().endsWith(',');
        const missing = placeholders.filter(ph =>
          !around.includes(`'{${ph}}'`) && !around.includes(`"{${ph}}"`) && !around.includes(`${ph}:`)
        );
        if (!calledWithArgs && missing.length > 0) {
          broken.push(`${key} → "${value}" (nicht befuellt: ${missing.join(', ')})`);
          break;
        }
      }
    }
    expect(broken, `Platzhalter bleiben in der Oberflaeche stehen:\n${broken.join('\n')}`)
      .toEqual([]);
  });
});
