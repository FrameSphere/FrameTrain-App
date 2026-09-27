// Was die Pruefung von 1.3.3 in der installierten App gefunden hat.
//
// Keiner dieser Fehler fiel in den Tests auf, weil jsdom keine
// Sicherheitsrichtlinie kennt und die Meldungen nie angesehen wurden.

import { describe, it, expect } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import fs from 'node:fs';
import path from 'node:path';
import { NotificationProvider, useNotification } from '../../contexts/NotificationContext';
import { ThemeProvider } from '../../contexts/ThemeContext';
import { empfohlenesModell } from '../studio/studioModels';

describe('Sicherheitsrichtlinie', () => {
  it('erlaubt fetch auf Mediendateien der Werkstatt', () => {
    // Ohne asset: in connect-src scheiterten das Umwandeln alter Aufnahmen
    // nach WAV und jeder Bild-Hash still — die Dubletten-Suche meldete
    // "keine unter 0 Samples".
    const conf = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../../../src-tauri/tauri.conf.json'), 'utf8'));
    const connect = (conf.app.security.csp as string).split(';').map(s => s.trim())
      .find(s => s.startsWith('connect-src')) ?? '';
    expect(connect).toContain('asset:');
    expect(connect).toContain('http://asset.localhost');
  });
});

describe('Meldungen', () => {
  it('zwei Argumente sind Titel und Text, eins ist der Text', () => {
    let api: ReturnType<typeof useNotification> | null = null;
    function Probe() { api = useNotification(); return null; }
    render(<ThemeProvider><NotificationProvider><Probe /></NotificationProvider></ThemeProvider>);
    act(() => { api!.success('Abschnitt geteilt', '4,0–6,0 s'); });
    const titel = screen.getByText('Abschnitt geteilt');
    const text = screen.getByText('4,0–6,0 s');
    // Der Titel steht vor dem Text.
    expect(titel.compareDocumentPosition(text) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    act(() => { api!.info('Nur ein Text'); });
    expect(screen.getByText('Nur ein Text')).toBeInTheDocument();
  });
});

describe('Modell-Tipp', () => {
  it('nennt je Projektart ein frei verfuegbares Modell', () => {
    expect(empfohlenesModell({ modality: 'video', task: 'classify' })).toBe('MCG-NJU/videomae-base');
    expect(empfohlenesModell({ modality: 'audio', task: 'transcript' })).toBe('openai/whisper-small');
    expect(empfohlenesModell({ modality: 'image', task: 'classify' })).toBe('google/vit-base-patch16-224');
    expect(empfohlenesModell({ modality: 'image', task: 'bbox' })).toBe('yolo11n');
  });
});
