// Soll-Werte fuer Objekterkennung im Labor.
//
// Kern der Sache: kein Klassenname darf aus FrameTrain kommen. Die Namen
// stehen im Dataset oder im Modell – hier wird geprueft, dass genau diese
// Quellen benutzt werden, in dieser Reihenfolge.

import { describe, it, expect } from 'vitest';
import {
  labelPathsForImage, classNamesFromYaml, labelForClassId,
  parseYoloLabelFile, summarizeBoxes, compareClassSets,
  classColor, legendEntries, CLASS_COLORS,
} from '../labGroundTruth';

describe('labelPathsForImage', () => {
  it('ersetzt den images-Ordner durch labels', () => {
    expect(labelPathsForImage('/data/train/images/ski_0001.jpg')[0])
      .toBe('/data/train/labels/ski_0001.txt');
  });

  it('nimmt das letzte images-Segment, nicht das erste', () => {
    // Ein Dataset kann "images" auch weiter oben im Pfad haben.
    expect(labelPathsForImage('/images/projekt/train/images/a.png')[0])
      .toBe('/images/projekt/train/labels/a.txt');
  });

  it('kennt auch die flache Ablage neben dem Bild', () => {
    expect(labelPathsForImage('/data/a.jpg')).toContain('/data/a.txt');
  });

  it('kommt mit Windows-Pfaden klar', () => {
    expect(labelPathsForImage('C:\\d\\images\\a.jpg')[0]).toBe('C:/d/labels/a.txt');
  });
});

describe('classNamesFromYaml', () => {
  it('liest die Listen-Schreibweise', () => {
    expect(classNamesFromYaml("path: .\nnames:\n  - Tree\n  - 'Stone'\n  - Person\n"))
      .toEqual({ 0: 'Tree', 1: 'Stone', 2: 'Person' });
  });

  it('liest die Mapping-Schreibweise mit eigenen Ids', () => {
    expect(classNamesFromYaml('names:\n  0: Tree\n  1: Stone\n  7: Lift\n'))
      .toEqual({ 0: 'Tree', 1: 'Stone', 7: 'Lift' });
  });

  it('liest die einzeilige Schreibweise', () => {
    expect(classNamesFromYaml("nc: 2\nnames: ['Tree', \"Stone\"]\n"))
      .toEqual({ 0: 'Tree', 1: 'Stone' });
  });

  it('ueberspringt Kommentare und endet am naechsten Schluessel', () => {
    const yaml = "names:\n  # Klassen hier eintragen:\n  - Tree\n  - Stone\nnc: 2\ntrain: x\n";
    expect(classNamesFromYaml(yaml)).toEqual({ 0: 'Tree', 1: 'Stone' });
  });

  it('liefert leer, wenn es keine names gibt', () => {
    expect(classNamesFromYaml('path: .\ntrain: train/images\n')).toEqual({});
  });
});

describe('labelForClassId', () => {
  it('Dataset schlaegt Modell, Modell schlaegt die nackte Id', () => {
    expect(labelForClassId(0, { 0: 'Baum' }, ['Tree'])).toBe('Baum');
    expect(labelForClassId(0, {}, ['Tree'])).toBe('Tree');
    expect(labelForClassId(5, {}, [])).toBe('Klasse 5');
  });
});

describe('parseYoloLabelFile', () => {
  const names = { 0: 'Tree', 2: 'Person' };

  it('rechnet normierte Werte in Bildpixel um', () => {
    const [box] = parseYoloLabelFile('0 0.5 0.5 0.25 0.5\n', 512, 512, names);
    expect(box).toEqual({ label: 'Tree', x1: 192, y1: 128, x2: 320, y2: 384 });
  });

  it('nimmt bei Segmentierungs-Polygonen die umschliessende Box', () => {
    const [box] = parseYoloLabelFile('2 0.1 0.2 0.4 0.2 0.4 0.6 0.1 0.6\n', 100, 100, names);
    expect(box).toEqual({ label: 'Person', x1: 10, y1: 20, x2: 40, y2: 60 });
  });

  it('ignoriert leere Zeilen, Kommentare und kaputte Zeilen', () => {
    const text = '\n# Kommentar\n0 0.5 0.5 0.2 0.2\nmuell hier\n0 0.5\n';
    expect(parseYoloLabelFile(text, 100, 100, names)).toHaveLength(1);
  });

  it('braucht Bildmasse – ohne sie keine Boxen', () => {
    expect(parseYoloLabelFile('0 0.5 0.5 0.2 0.2', 0, 0, names)).toEqual([]);
  });

  it('faellt fuer unbekannte Ids auf Modellnamen zurueck', () => {
    const [box] = parseYoloLabelFile('1 0.5 0.5 0.2 0.2', 10, 10, {}, ['A', 'B']);
    expect(box.label).toBe('B');
  });
});

describe('summarizeBoxes', () => {
  it('zaehlt Boxen und nennt die Klassen ohne Wiederholung', () => {
    expect(summarizeBoxes([{ label: 'Sky' }, { label: 'Sky' }, { label: 'Tree' }]))
      .toBe('3x Sky, Tree');
  });

  it('kuerzt lange Klassenlisten', () => {
    expect(summarizeBoxes(['A', 'B', 'C', 'D'].map(label => ({ label })))).toBe('4x A, B, C +1');
  });

  it('sagt es, wenn nichts da ist', () => {
    expect(summarizeBoxes([])).toBe('Keine Objekte');
  });
});

describe('compareClassSets', () => {
  const b = (...labels: string[]) => labels.map(label => ({ label }));

  it('nennt fehlende und zusaetzliche Klassen', () => {
    expect(compareClassSets(b('Tree', 'Person'), b('Tree', 'Sky')))
      .toEqual({ missing: ['Person'], extra: ['Sky'] });
  });

  it('wertet mehr Boxen derselben Klasse nicht als Fehler', () => {
    // Drei Baeume statt zwei ist keine falsche Erkennung.
    expect(compareClassSets(b('Tree', 'Tree'), b('Tree', 'Tree', 'Tree')))
      .toEqual({ missing: [], extra: [] });
  });

  it('meldet alles als fehlend, wenn nichts erkannt wurde', () => {
    expect(compareClassSets(b('Tree'), [])).toEqual({ missing: ['Tree'], extra: [] });
  });
});

describe('classColor', () => {
  it('gibt derselben Klasse immer dieselbe Farbe', () => {
    // Beim Durchklicken muss "Tree" in jedem Bild gleich aussehen.
    expect(classColor('Tree')).toBe(classColor('Tree'));
  });

  it('liefert nur Farben aus der Palette', () => {
    for (const label of ['Tree', 'Sky', 'Lift', 'Klasse 7', 'ä ö ü', '']) {
      expect(CLASS_COLORS).toContain(classColor(label));
    }
  });

  it('nutzt die Klassenliste des Modells und trennt damit sauber', () => {
    const ski = ['Tree','Stone','Person','Hole','Building','Stick','Emptyspace',
                 'Lift','Slopesign','Slopeborder','Sky','Generallobstacle','Offroad'];
    const used = new Set(ski.map(l => classColor(l, ski)));
    // 13 Klassen auf 12 Farben: genau eine Wiederholung, keine zufaelligen Dopplungen.
    expect(used.size).toBe(CLASS_COLORS.length);
    expect(classColor('Tree', ski)).toBe(CLASS_COLORS[0]);
    expect(classColor('Stone', ski)).toBe(CLASS_COLORS[1]);
  });

  it('faellt fuer unbekannte Klassen auf den Namens-Hash zurueck', () => {
    const color = classColor('Klasse 42', ['Tree', 'Sky']);
    expect(CLASS_COLORS).toContain(color);
    expect(color).toBe(classColor('Klasse 42'));
  });
});

describe('legendEntries', () => {
  it('fuehrt Soll und Erkennung zusammen, ohne Wiederholung', () => {
    const entries = legendEntries([{ label: 'Sky' }, { label: 'Sky' }], [{ label: 'Tree' }]);
    expect(entries.map(e => e.label)).toEqual(['Sky', 'Tree']);
    expect(entries[0].color).toBe(classColor('Sky'));
  });

  it('bleibt leer, wenn es nichts zu zeigen gibt', () => {
    expect(legendEntries([], [])).toEqual([]);
  });
});
