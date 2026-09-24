// Statuszeile unter dem HuggingFace-Fortschrittsbalken. Der Balken stand im
// ersten Anlauf die ganze Zeit auf "Wird vorbereitet" — hier ist festgehalten,
// welcher Zustand welchen Text bekommt, und dass jeder Text in de und en existiert.

import { describe, it, expect } from 'vitest';
import { hfStatusText, formatElapsed, type HfProgress } from '../VersionManager';
import de from '../../locales/de.json';
import en from '../../locales/en.json';

function tFrom(dict: any) {
  return (key: string) => {
    const v = key.split('.').reduce((o, k) => (o == null ? o : o[k]), dict);
    if (typeof v !== 'string') throw new Error(`Fehlender Text: ${key}`);
    return v;
  };
}

const p = (over: Partial<HfProgress> = {}): HfProgress => ({
  percent: 40, phase: 'uploading', message: '', filesDone: null, filesTotal: null,
  bytesDone: null, bytesTotal: null, speed: null, ...over,
});

describe('hfStatusText', () => {
  const t = tFrom(de);

  it('zeigt "vorbereitet", solange noch keine Prozentzahl da ist', () => {
    expect(hfStatusText(null, t)).toBe('Wird vorbereitet …');
    expect(hfStatusText(p({ percent: -1, phase: 'starting' }), t)).toBe('Wird vorbereitet …');
  });

  it('nennt die Phase, solange es noch keine Prozentzahl gibt', () => {
    expect(hfStatusText(p({ percent: -1, phase: 'connect' }), t)).toBe('Wird vorbereitet …');
    expect(hfStatusText(p({ percent: -1, phase: 'create_repo' }), t)).toBe('Repository wird vorbereitet …');
    expect(hfStatusText(p({ percent: -1, phase: 'upload' }), t)).toBe('Daten werden verarbeitet …');
    expect(hfStatusText(p({ percent: -1, phase: 'upload' }), tFrom(en))).toBe('Processing data …');
  });

  it('zeigt bei pipelined_upload Bytes, Tempo und Restzeit', () => {
    const MB = 1024 * 1024;
    // 7.5 von 15 MB bei 0.5 MB/s -> noch 15 s
    expect(hfStatusText(p({ percent: 50, bytesDone: 7.5 * MB, bytesTotal: 15 * MB, speed: 0.5 * MB }), t))
      .toBe('7.5 MB von 15 MB · 512 KB/s · noch 0:15');
    expect(hfStatusText(p({ percent: 50, bytesDone: 7.5 * MB, bytesTotal: 15 * MB, speed: 0.5 * MB }), tFrom(en)))
      .toBe('7.5 MB of 15 MB · 512 KB/s · 0:15 left');
  });

  it('laesst Tempo und Restzeit weg, solange noch kein Tempo gemessen ist', () => {
    const MB = 1024 * 1024;
    expect(hfStatusText(p({ percent: 0, bytesDone: 0, bytesTotal: 15 * MB, speed: 0 }), t)).toBe('0 B von 15 MB');
    expect(hfStatusText(p({ percent: 3, bytesDone: MB / 2, bytesTotal: 15 * MB }), t)).toBe('512 KB von 15 MB');
  });

  it('zeigt beim Xet-Upload die Dateizahl', () => {
    expect(hfStatusText(p({ filesDone: 1, filesTotal: 3 }), t)).toBe('Dateien 1 / 3');
    expect(hfStatusText(p({ filesDone: null, filesTotal: 3 }), t)).toBe('Dateien 0 / 3');
  });

  it('zeigt beim klassischen LFS-Upload den Dateinamen', () => {
    expect(hfStatusText(p({ message: 'model.safetensors' }), t)).toBe('model.safetensors');
  });

  it('faellt ohne Dateiinfo auf "wird hochgeladen" zurueck', () => {
    expect(hfStatusText(p(), t)).toBe('Wird hochgeladen …');
  });

  it('zeigt bei 100 % "wird abgeschlossen" statt eines scheinbar haengenden Balkens', () => {
    expect(hfStatusText(p({ percent: 100, filesDone: 3, filesTotal: 3 }), t)).toBe('Wird abgeschlossen …');
  });

  it('hat alle Texte auch auf Englisch', () => {
    const tEn = tFrom(en);
    expect(hfStatusText(null, tEn)).toBe('Preparing …');
    expect(hfStatusText(p({ filesDone: 2, filesTotal: 5 }), tEn)).toBe('Files 2 / 5');
    expect(hfStatusText(p({ percent: 100 }), tEn)).toBe('Finishing …');
    expect(hfStatusText(p(), tEn)).toBe('Uploading …');
  });
});

describe('formatElapsed', () => {
  it('zeigt Minuten und zweistellige Sekunden', () => {
    expect(formatElapsed(0)).toBe('0:00');
    expect(formatElapsed(42)).toBe('0:42');
    expect(formatElapsed(125)).toBe('2:05');
    expect(formatElapsed(-3)).toBe('0:00');
  });
});
