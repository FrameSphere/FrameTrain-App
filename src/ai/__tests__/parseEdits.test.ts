// Der Edit-Parser der Code-Assistenten.
//
// Praxisfall aus dem Groq-Test: das Modell liess ##EDIT_END## weg und schrieb
// danach weiter. Der Fallback nahm daraufhin ALLES bis zum Textende als neuen
// Code — nach dem Uebernehmen standen "```", "FIND:" und der Rest der Antwort
// als Code-Zeilen im Skript und machten es unbrauchbar.
//
// Ausfuehren: npx vitest run src/ai/__tests__/parseEdits.test.ts --config vitest.config.ts

import { describe, it, expect } from 'vitest';
import { parseEdits, isUsableEdit, applyAllEdits } from '../codeEdits';

describe('parseEdits', () => {
  it('liest einen vollstaendigen Block', () => {
    const edits = parseEdits('##EDIT_START##\nFIND:\nbatch=64\nREPLACE:\nbatch=8\n##EDIT_END##');
    expect(edits).toHaveLength(1);
    expect(edits[0]).toMatchObject({ find: 'batch=64', replace: 'batch=8' });
  });

  it('entfernt Code-Fences mit und ohne Sprachangabe', () => {
    const edits = parseEdits(
      '##EDIT_START##\nFIND:\n```python\nbatch=64\n```\nREPLACE:\n```\nbatch=8\n```\n##EDIT_END##',
    );
    expect(edits[0]).toMatchObject({ find: 'batch=64', replace: 'batch=8' });
  });

  // Genau der Fall, der das Skript zerstoert hat.
  it('nimmt bei fehlendem ##EDIT_END## nicht den Rest der Antwort als Code', () => {
    const answer = [
      '##EDIT_START##',
      'FIND:',
      'batch=64',
      'REPLACE:',
      'batch=8',
      '',
      'FIND:',
      '```python',
      'model.train(',
      '    data=DATASET_PATH,',
    ].join('\n');
    const edits = parseEdits(answer);
    expect(edits).toHaveLength(1);
    expect(edits[0].replace).toBe('batch=8');
    expect(edits[0].replace).not.toMatch(/FIND:|```/);
  });

  it('verwirft einen Vorschlag, der Protokoll-Reste enthaelt', () => {
    const answer = '##EDIT_START##\nFIND:\nbatch=64\nREPLACE:\nbatch=8\n```\nFIND:\nweiter\n##EDIT_END##';
    for (const e of parseEdits(answer)) {
      expect(e.replace).not.toMatch(/FIND:|```/);
    }
  });

  it('erlaubt einen leeren Ersetzungsteil (Zeile loeschen)', () => {
    const edits = parseEdits('##EDIT_START##\nFIND:\nprint(x)\nREPLACE:\n##EDIT_END##');
    expect(edits).toHaveLength(1);
    expect(edits[0].replace).toBe('');
  });
});

describe('isUsableEdit', () => {
  it('lehnt Markdown- und Protokoll-Reste ab', () => {
    expect(isUsableEdit('x = 1', 'x = 2')).toBe(true);
    expect(isUsableEdit('x = 1', 'x = 2\n```')).toBe(false);
    expect(isUsableEdit('x = 1', 'x = 2\nFIND:\ny')).toBe(false);
    expect(isUsableEdit('x = 1', '##EDIT_END##')).toBe(false);
    expect(isUsableEdit('', 'x = 2')).toBe(false);
  });

  it('laesst Rauten und Backticks INNERHALB einer Zeile zu', () => {
    // "# Kommentar" ist normaler Python-Code, kein Protokoll-Rest.
    expect(isUsableEdit('x = 1', '# Kommentar\nx = 2')).toBe(true);
  });
});

// Die Oberflaeche meldete "uebernommen", auch wenn KEIN Eingriff gepasst hat —
// das Skript blieb unveraendert, der Rueckgaengig-Button erschien trotzdem.
describe('applyAllEdits – Erfolg vs. Fehlschlag', () => {
  const script = 'x = 1\ny = 2\n';

  it('meldet Erfolg pro Eingriff getrennt', () => {
    const { result, results } = applyAllEdits(script, [
      { id: 'a', find: 'x = 1', replace: 'x = 42' },
      { id: 'b', find: 'gibt es nicht', replace: 'egal' },
    ]);
    expect(results[0].success).toBe(true);
    expect(results[1].success).toBe(false);
    expect(result).toContain('x = 42');
  });

  it('laesst das Skript unveraendert, wenn nichts passt', () => {
    const { result, results } = applyAllEdits(script, [
      { id: 'a', find: 'fehlt', replace: 'neu' },
    ]);
    expect(results.every(r => !r.success)).toBe(true);
    expect(result).toBe(script);
  });
});
