// Rohdaten erzeugen lassen.
//
// Zwei Dinge entscheiden hier ueber brauchbar oder nicht: dass die Antwort des
// Modells auch dann zu Zeilen wird, wenn sie kein gueltiges JSON ist (der
// Normalfall bei kleineren Modellen), und dass nichts ungesehen ins Projekt
// rutscht — samt Vermerk, dass ein Modell es geschrieben hat.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

import { parseGenerated, ohneDubletten } from '../studio/generatedTexts';

const callAIMock = vi.fn();
vi.mock('../../ai/aiClient', () => ({ callAI: (...a: unknown[]) => callAIMock(...a) }));

const settingsRef = {
  current: {
    enabled: true, provider: 'anthropic', apiKey: 'sk-ant-api-x',
    selectedModel: 'claude-haiku-4-5', ollamaModel: '', tokenBudget: 'balanced',
  },
};
vi.mock('../../contexts/AISettingsContext', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../contexts/AISettingsContext')>()),
  useAISettings: () => ({ settings: settingsRef.current }),
}));
const error = vi.fn();
vi.mock('../../contexts/NotificationContext', () => ({
  useNotification: () => ({ success: vi.fn(), error, warning: vi.fn(), info: vi.fn() }),
}));

import GenerateDialog from '../studio/GenerateDialog';

const projekt = {
  id: 'sp_text', name: 'Meldungen', modality: 'text', task: 'classification',
  target_format: 'flat_file', classes: ['beschwerde', 'lob'],
  created_at: '', updated_at: '',
};

describe('parseGenerated', () => {
  it('liest ein JSON-Array', () => {
    expect(parseGenerated('["Lift kaputt", "Piste top"]'))
      .toEqual([{ text: 'Lift kaputt' }, { text: 'Piste top' }]);
  });

  it('liest ein Array aus einem Code-Block samt Vorwort', () => {
    const antwort = 'Hier sind die Beispiele:\n```json\n["a", "b"]\n```\nViel Erfolg!';
    expect(parseGenerated(antwort).map(i => i.text)).toEqual(['a', 'b']);
  });

  it('liest Objekte mit Eingabe und Ziel', () => {
    const antwort = '[{"text": "Wo ist mein Paket?", "target": "Sendungsnummer erfragen"}]';
    expect(parseGenerated(antwort)).toEqual([
      { text: 'Wo ist mein Paket?', label: null, target: 'Sendungsnummer erfragen' },
    ]);
  });

  it('rettet eine nummerierte Liste ohne JSON', () => {
    // Genau das liefern kleinere Modelle, und ein Lauf kostet echtes Geld.
    const antwort = 'Hier sind 3 Beispiele:\n1. Der Lift steht wieder\n2. "Kasse war zu"\n- Schnee top\n';
    expect(parseGenerated(antwort).map(i => i.text))
      .toEqual(['Der Lift steht wieder', 'Kasse war zu', 'Schnee top']);
  });

  it('liest JSONL zeilenweise', () => {
    const antwort = '{"text":"a","label":"lob"}\n{"text":"b"}';
    expect(parseGenerated(antwort)).toEqual([
      { text: 'a', label: 'lob', target: null },
      { text: 'b', label: null, target: null },
    ]);
  });

  it('gibt bei leerer Antwort nichts zurueck', () => {
    expect(parseGenerated('')).toEqual([]);
    expect(parseGenerated('   ')).toEqual([]);
  });
});

describe('ohneDubletten', () => {
  it('wirft raus, was schon im Projekt steht — auch mit anderer Schreibweise', () => {
    const items = [{ text: 'Lift kaputt' }, { text: 'lift KAPUTT' }, { text: 'Neu' }];
    expect(ohneDubletten(items, ['Lift kaputt'])).toEqual([{ text: 'Neu' }]);
  });
});

describe('GenerateDialog', () => {
  beforeEach(() => {
    callAIMock.mockReset();
    error.mockReset();
  });

  const props = (onAdd = vi.fn()) => ({
    project: projekt, paare: false, vorhanden: [] as string[],
    onCancel: vi.fn(), onAdd,
  });

  it('fragt je Klasse einmal und haengt die Klasse an jede Zeile', async () => {
    // Eine Anfrage fuer alle Klassen zugleich verteilt ungleich — je Klasse
    // eine Anfrage ist der Grund, warum die Zuordnung ueberhaupt stimmt.
    callAIMock.mockImplementation(async (_s: unknown, o: { messages: { content: string }[] }) =>
      o.messages[0].content.includes('beschwerde') ? '["Lift kaputt"]' : '["Piste top"]');
    const onAdd = vi.fn();

    render(<GenerateDialog {...props(onAdd)} />);
    fireEvent.click(screen.getByRole('button', { name: /Erzeugen$/ }));

    await screen.findByText('Lift kaputt');
    expect(callAIMock).toHaveBeenCalledTimes(2);
    fireEvent.click(screen.getByRole('button', { name: /^[0-9]+ übernehmen$/ }));

    expect(onAdd).toHaveBeenCalledTimes(1);
    const [items, origin] = onAdd.mock.calls[0];
    expect(items).toEqual([
      { text: 'Lift kaputt', label: 'beschwerde' },
      { text: 'Piste top', label: 'lob' },
    ]);
    // Ohne diesen Vermerk sieht erzeugter Text spaeter aus wie gesammelter.
    expect(String(origin)).toMatch(/claude-haiku-4-5/);
  });

  it('uebernimmt nur, was stehen geblieben ist', async () => {
    callAIMock.mockResolvedValue('["a", "b"]');
    const onAdd = vi.fn();

    render(<GenerateDialog {...props(onAdd)} project={{ ...projekt, classes: [] }} />);
    fireEvent.click(screen.getByRole('button', { name: /Erzeugen$/ }));

    await screen.findByText('a');
    fireEvent.click(screen.getAllByRole('button', { name: 'Nicht übernehmen' })[0]);
    fireEvent.click(screen.getByRole('button', { name: /^[0-9]+ übernehmen$/ }));

    expect(onAdd.mock.calls[0][0]).toEqual([{ text: 'b' }]);
  });

  it('sagt es, wenn nichts Brauchbares zurueckkam', async () => {
    callAIMock.mockResolvedValue('   ');
    render(<GenerateDialog {...props()} />);
    fireEvent.click(screen.getByRole('button', { name: /Erzeugen$/ }));

    await waitFor(() => expect(error).toHaveBeenCalled());
  });

  it('bietet das Erzeugen ohne eingerichteten KI-Assistenten nicht an', () => {
    settingsRef.current = { ...settingsRef.current, apiKey: '' };
    render(<GenerateDialog {...props()} />);

    expect(screen.getByRole('button', { name: /Erzeugen$/ })).toBeDisabled();
    expect(screen.getByText(/Einstellungen/)).toBeInTheDocument();
    settingsRef.current = { ...settingsRef.current, apiKey: 'sk-ant-api-x' };
  });
});
