// Bausteine der Qualitaetspruefung und des Labor-Uebergangs: WAV, Bild-Hash,
// Balance-Warnung, Auswahl der Labor-Ergebnisse. Alles rein und ohne Tauri.

import { describe, it, expect, vi } from 'vitest';

vi.mock('@tauri-apps/api/core', () => ({ invoke: vi.fn(), convertFileSrc: (p: string) => `asset://${p}` }));

import { encodeWav, brauchtWav } from '../studio/wavEncode';
import { dhashFromGray, hamming } from '../studio/dhash';
import { balanceWarnungen } from '../studio/studioStats';
import { labAuswahl, labItems, labModalitaet, type LabErgebnis } from '../studio/labToStudio';
import { hashesNachrechnen } from '../studio/NearDupDialog';
import { invoke } from '@tauri-apps/api/core';

describe('encodeWav', () => {
  it('schreibt einen gueltigen 16-Bit-Mono-Kopf', () => {
    const wav = encodeWav(new Float32Array([0, 1, -1, 0.5]), 16000);
    const v = new DataView(wav.buffer);
    const text = (o: number, n: number) => String.fromCharCode(...wav.slice(o, o + n));
    expect(text(0, 4)).toBe('RIFF');
    expect(text(8, 4)).toBe('WAVE');
    expect(v.getUint16(22, true)).toBe(1);          // Mono
    expect(v.getUint32(24, true)).toBe(16000);
    expect(v.getUint32(40, true)).toBe(8);          // 4 Samples * 2 Bytes
    expect(v.getInt16(46, true)).toBe(32767);
    expect(v.getInt16(48, true)).toBe(-32768);
    expect(wav.length).toBe(44 + 8);
  });

  it('wandelt nur Webview-Aufnahmen um', () => {
    expect(brauchtWav('ab/x.m4a')).toBe(true);
    expect(brauchtWav('ab/x.webm')).toBe(true);
    expect(brauchtWav('ab/x.wav')).toBe(false);
    expect(brauchtWav('ab/x.mp3')).toBe(false);
  });
});

describe('dHash', () => {
  it('ist unempfindlich gegen Helligkeit, aber nicht gegen den Inhalt', () => {
    const verlauf = Array.from({ length: 72 }, (_, i) => (i % 9) * 20);
    const heller = verlauf.map(x => x + 30);
    const anders = verlauf.map((x, i) => (Math.floor(i / 9) % 2 ? x : 200 - x));
    const a = dhashFromGray(verlauf);
    expect(a).toHaveLength(16);
    expect(hamming(a, dhashFromGray(heller))).toBe(0);
    expect(hamming(a, dhashFromGray(anders))).toBeGreaterThan(20);
  });

  it('rechnet fehlende Hashes und legt sie ab', async () => {
    const samples = [{ media: 'ab/a.jpg', abs_path: '/m/a.jpg' }, { media: 'cd/b.jpg', abs_path: '/m/b.jpg' }];
    const hash = vi.fn(async (url: string) => {
      if (url.endsWith('b.jpg')) throw new Error('kaputt');
      return '00ff00ff00ff00ff';
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const n = await hashesNachrechnen('sp', samples as any, hash);
    expect(n).toBe(1);
    expect(invoke).toHaveBeenCalledWith('studio_save_hashes', { projectId: 'sp', hashes: { 'ab/a.jpg': '00ff00ff00ff00ff' } });
  });
});

describe('balanceWarnungen', () => {
  it('schweigt im leeren Projekt und warnt bei Schieflage', () => {
    expect(balanceWarnungen([0, 0], ['a', 'b'])).toEqual([]);
    const w = balanceWarnungen([60, 4, 0], ['ja', 'nein', 'vielleicht']);
    expect(w).toEqual([
      { art: 'wenig', klasse: 'nein', anzahl: 4 },
      { art: 'leer', klasse: 'vielleicht' },
      { art: 'schief', gross: 'ja', grossN: 60, klein: 'nein', kleinN: 4 },
    ]);
    expect(balanceWarnungen([20, 30], ['a', 'b'])).toEqual([]);
  });
});

describe('Labor -> Werkstatt', () => {
  const r = (x: Partial<LabErgebnis>): LabErgebnis => ({
    inputText: 'x', predicted: 'ja', userRating: 'skipped', ...x,
  });

  it('erkennt die Art an der Datei', () => {
    expect(labModalitaet([r({})])).toBe('text');
    expect(labModalitaet([r({ filePath: '/a/b.MP4' })])).toBe('video');
    expect(labModalitaet([r({ filePath: '/a/b.wav' })])).toBe('audio');
    expect(labModalitaet([r({ filePath: '/a/b.jpg' })])).toBe('image');
  });

  it('waehlt falsche und unsichere aus', () => {
    const alle = [
      r({ userRating: 'wrong', confidence: 0.9 }),
      r({ correction: { kind: 'label', label: 'nein' }, confidence: 0.95 }),
      r({ confidence: 0.3 }),
      r({ userRating: 'correct', confidence: 0.99 }),
    ];
    expect(labAuswahl(alle, 'wrong')).toHaveLength(2);
    expect(labAuswahl(alle, 'uncertain')).toHaveLength(1);
    expect(labAuswahl(alle, 'all')).toHaveLength(4);
  });

  it('Korrektur und Bewertung werden zum Label, sonst bleibt es ein Vorschlag', () => {
    const items = labItems([
      r({ correction: { kind: 'label', label: 'nein' } }),
      r({ userRating: 'correct' }),
      r({ confidence: 0.4 }),
      r({ filePath: '/b.jpg', correction: { kind: 'boxes', boxes: [{ label: 'Lift', x1: 1, y1: 2, x2: 3, y2: 4 }] } }),
    ]);
    expect(items[0]).toMatchObject({ label: 'nein', text: 'x' });
    expect(items[1]).toMatchObject({ label: 'ja' });
    expect(items[2]).toMatchObject({ label: null, predicted: 'ja', confidence: 0.4 });
    expect(items[3]).toMatchObject({ filePath: '/b.jpg', text: null, boxes: [{ label: 'Lift' }] });
  });
});
