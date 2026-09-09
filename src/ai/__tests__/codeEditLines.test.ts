// Welche Zeilen ein Edit im Editor markiert.
//
// Die neuen Zeilen bekamen frueher eine eigene gruene Flaeche — ueber Zeilen,
// die im Skript noch gar nicht existieren. Im Editor stand dann ein grosser
// gruener Block ohne jeden Text, der eine Code-Vorschau vortaeuschte.
//
// Ausfuehren: npx vitest run src/ai/__tests__/codeEditLines.test.ts --config vitest.config.ts

import { describe, it, expect } from 'vitest';
import { calculateAffectedLines } from '../codeEdits';

const script = ['import os', 'x = 1', 'y = 2', 'print(x)'].join('\n');

const edit = (find: string, replace: string) => ({ id: 'e1', find, replace }) as any;

describe('calculateAffectedLines', () => {
  it('markiert die ersetzten Zeilen rot', () => {
    const lines = calculateAffectedLines(script, edit('x = 1', 'x = 42'));
    expect(lines.filter(l => l.type === 'removed').map(l => l.lineNum)).toEqual([2]);
  });

  it('markiert die Einfuegestelle als EINE Marke mit Zeilenzahl', () => {
    const lines = calculateAffectedLines(script, edit('x = 1', 'a = 1\nb = 2\nc = 3'));
    const insertions = lines.filter(l => l.type === 'insertion');
    expect(insertions).toHaveLength(1);
    expect(insertions[0]).toMatchObject({ lineNum: 3, count: 3 });
    // Keine Flaechen-Markierung fuer noch nicht existierende Zeilen.
    expect(lines.some(l => (l.type as string) === 'added')).toBe(false);
  });

  it('setzt keine Marke, wenn nur geloescht wird', () => {
    const lines = calculateAffectedLines(script, edit('x = 1', ''));
    expect(lines.some(l => l.type === 'insertion')).toBe(false);
  });

  it('markiert nichts, wenn der Fundtext fehlt', () => {
    expect(calculateAffectedLines(script, edit('gibt es nicht', 'neu'))).toEqual([]);
  });
});
