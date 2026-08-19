// Regression aus dem E2E-Test vom 19.08.2026:
// Das Labor schnitt "Bingen: sonnig, 0 Grad" am Komma ab und meldete die
// korrekte Vorhersage des Seq2Seq-Modells deshalb als "Abweichend".

import { describe, it, expect } from 'vitest';
import { splitDelimitedLine, joinQuotedLines, parseDelimitedRows } from '../csvRows';

describe('splitDelimitedLine', () => {
  it('laesst Kommas innerhalb von Anfuehrungszeichen stehen', () => {
    expect(splitDelimitedLine('Das Wetter ist sonnig.,"Bingen: sonnig, 0 Grad"', ','))
      .toEqual(['Das Wetter ist sonnig.', 'Bingen: sonnig, 0 Grad']);
  });

  it('zerlegt Zeilen ohne Anfuehrungszeichen wie bisher', () => {
    expect(splitDelimitedLine('a,b,c', ',')).toEqual(['a', 'b', 'c']);
  });

  it('versteht doppelte Anfuehrungszeichen als Zeichen', () => {
    expect(splitDelimitedLine('"er sagte ""hallo""",x', ',')).toEqual(['er sagte "hallo"', 'x']);
  });

  it('funktioniert mit Tabulator als Trennzeichen', () => {
    expect(splitDelimitedLine('a\t"b\tc"', '\t')).toEqual(['a', 'b\tc']);
  });

  it('liefert leere Felder statt sie zu verschlucken', () => {
    expect(splitDelimitedLine('a,,c', ',')).toEqual(['a', '', 'c']);
  });
});

describe('joinQuotedLines', () => {
  it('haelt Zeilenumbrueche innerhalb eines Feldes zusammen', () => {
    const csv = 'source,target\n"Zeile eins\nZeile zwei",kurz\n';
    expect(joinQuotedLines(csv)).toEqual(['source,target', '"Zeile eins\nZeile zwei",kurz']);
  });

  it('ueberspringt leere Zeilen', () => {
    expect(joinQuotedLines('a\n\nb\n')).toEqual(['a', 'b']);
  });
});

describe('parseDelimitedRows', () => {
  it('baut Objekte aus der Kopfzeile', () => {
    const csv = 'source,target\nDas Wetter in Bingen ist sonnig.,"Bingen: sonnig, 0 Grad"\n';
    expect(parseDelimitedRows(csv, ',')).toEqual([
      { source: 'Das Wetter in Bingen ist sonnig.', target: 'Bingen: sonnig, 0 Grad' },
    ]);
  });

  it('liefert bei einer Spalte die reinen Werte', () => {
    expect(parseDelimitedRows('text\nhallo\nwelt\n', ',')).toEqual(['hallo', 'welt']);
  });

  it('fuellt fehlende Spalten mit leerem String', () => {
    expect(parseDelimitedRows('a,b\n1\n', ',')).toEqual([{ a: '1', b: '' }]);
  });

  it('gibt bei leerem Inhalt nichts zurueck', () => {
    expect(parseDelimitedRows('', ',')).toEqual([]);
  });
});
