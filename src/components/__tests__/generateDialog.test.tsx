// Rohdaten erzeugen lassen.
//
// Zwei Dinge entscheiden hier ueber brauchbar oder nicht: dass die Antwort des
// Modells auch dann zu Zeilen wird, wenn sie kein gueltiges JSON ist (der
// Normalfall bei kleineren Modellen), und dass nichts ungesehen ins Projekt
// rutscht — samt Vermerk, dass ein Modell es geschrieben hat.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

import { parseGenerated, ohneDubletten, beispieleFuer } from '../studio/generatedTexts';

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

  it('rettet die vollstaendigen Eintraege einer abgeschnittenen Antwort', () => {
    expect(parseGenerated('["eins", "zwei \\"x\\"", "dr').map(i => i.text)).toEqual(['eins', 'zwei "x"']);
    expect(parseGenerated('```json\n[\n  {\n    "text": "a",\n    "target": "b"\n  },\n  {\n    "text": "c",')).toEqual([
      { text: 'a', label: null, target: 'b' },
    ]);
    expect(parseGenerated('1. Ganz\n2. Auch ganz\n3. Übe', true).map(i => i.text)).toEqual(['Ganz', 'Auch ganz']);
    // Eckige Klammern in normalem Text sind kein Array.
    expect(parseGenerated('1. Hallo [Name]\n2. Bis bald').map(i => i.text)).toEqual(['Hallo [Name]', 'Bis bald']);
  });

  it('macht aus kaputtem JSON keine Beispiele aus Klammern und Schluesseln', () => {
    // So antwortete ein echtes kleines Modell (SmolLM2-135M) im Test.
    const antwort = '{\n  "action": {\n    "type": "submit",\n    "text": "Jetzt gewinnen"\n  },\n}\nRuf an!';
    expect(parseGenerated(antwort).map(i => i.text)).toEqual(['Jetzt gewinnen', 'Ruf an!']);
    // Ein Satz mit Doppelpunkt bleibt ein Satz.
    expect(parseGenerated('Achtung: Lift steht\nZweite Zeile').map(i => i.text)).toEqual(['Achtung: Lift steht', 'Zweite Zeile']);
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

describe('beispieleFuer', () => {
  const VORHANDEN = [
    { text: 'Passwort zuruecksetzen bitte', label: 'Request', bestaetigt: true },
    { text: 'Rechnung schicken', label: 'Request', bestaetigt: true },
    { text: 'Seite zeigt Fehler 500', label: 'Error', bestaetigt: false },
    { text: 'App stuerzt ab', label: 'error', bestaetigt: true },
  ];

  it('gibt einer Klasse nur ihre eigenen Beispiele, Bestaetigtes zuerst', () => {
    // Frueher bekam jede Klasse die ersten Texte des Projekts — fast nur Request.
    expect(beispieleFuer(VORHANDEN, 'Error')).toEqual(['App stuerzt ab', 'Seite zeigt Fehler 500']);
  });

  it('gibt lieber keine Beispiele als die einer anderen Klasse', () => {
    expect(beispieleFuer(VORHANDEN, 'Question')).toEqual([]);
  });

  it('zeigt bei Paaren Eingabe und Antwort', () => {
    expect(beispieleFuer([{ text: 'Wo ist mein Paket?', target: 'Nummer erfragen' }], null))
      .toEqual(['Wo ist mein Paket? → Nummer erfragen']);
  });
});

describe('GenerateDialog', () => {
  beforeEach(() => {
    callAIMock.mockReset();
    error.mockReset();
  });

  const props = (onAdd = vi.fn()) => ({
    project: projekt, paare: false,
    vorhanden: [] as { text: string; label?: string | null; bestaetigt?: boolean }[],
    onCancel: vi.fn(), onAdd,
  });

  it('fragt je Klasse einmal und haengt die Klasse an jede Zeile', async () => {
    // Eine Anfrage fuer alle Klassen zugleich verteilt ungleich — je Klasse
    // eine Anfrage ist der Grund, warum die Zuordnung ueberhaupt stimmt.
    callAIMock.mockImplementation(async (_s: unknown, o: { messages: { content: string }[] }) =>
      o.messages[0].content.includes('Klasse "beschwerde"') ? '["Lift kaputt"]' : '["Piste top"]');
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

  it('schickt jeder Klasse nur ihre eigenen Stilbeispiele mit', async () => {
    callAIMock.mockResolvedValue('["x"]');
    render(<GenerateDialog {...props()} vorhanden={[
      { text: 'Passwort zuruecksetzen bitte', label: 'beschwerde', bestaetigt: true },
      { text: 'Tolle Piste', label: 'lob', bestaetigt: true },
    ]} />);
    fireEvent.click(screen.getByRole('button', { name: /Erzeugen$/ }));
    await waitFor(() => expect(callAIMock).toHaveBeenCalledTimes(2));

    const auftraege = callAIMock.mock.calls.map(c => (c[1] as { messages: { content: string }[] }).messages[0].content);
    const fuerLob = auftraege.find(a => a.includes('"lob"'))!;
    expect(fuerLob).toContain('Tolle Piste');
    expect(fuerLob).not.toContain('Passwort');
  });

  it('uebernimmt einen in der Vorschau korrigierten Text', async () => {
    callAIMock.mockResolvedValue('["Der Lift steht stil"]');
    const onAdd = vi.fn();
    render(<GenerateDialog {...props(onAdd)} project={{ ...projekt, classes: [] }} />);
    fireEvent.click(screen.getByRole('button', { name: /Erzeugen$/ }));

    const feld = await screen.findByLabelText('Vorschlag 1 bearbeiten');
    fireEvent.change(feld, { target: { value: 'Der Lift steht still' } });
    fireEvent.click(screen.getByRole('button', { name: /^[0-9]+ übernehmen$/ }));

    expect(onAdd.mock.calls[0][0]).toEqual([{ text: 'Der Lift steht still' }]);
  });

  // Der Auftrag an das Modell: Abgrenzung und Sprache.
  it('nennt die anderen Klassen und laesst die Sprache der Beispiele gelten', async () => {
    callAIMock.mockResolvedValue('["x"]');
    render(<GenerateDialog {...props()} vorhanden={[{ text: 'Great slope', label: 'lob' }]} />);
    fireEvent.click(screen.getByRole('button', { name: /Erzeugen$/ }));
    await waitFor(() => expect(callAIMock).toHaveBeenCalledTimes(2));
    const [beschwerde, lob] = callAIMock.mock.calls.map(c => (c[1] as { messages: { content: string }[] }).messages[0].content);
    expect(lob).toContain('Die anderen Klassen sind: beschwerde');
    expect(lob).toContain('in derselben Sprache wie die vorhandenen Beispiele');
    expect(beschwerde).toContain('Schreibe auf Deutsch, außer');
    // Die Oberflaechensprache darf nicht als Antwortsprache mitgehen.
    expect((callAIMock.mock.calls[0][1] as Record<string, unknown>).responseLanguage).toBeUndefined();
  });

  it('behaelt, was vor einem Fehler schon da war', async () => {
    callAIMock.mockResolvedValueOnce('["Lift kaputt"]').mockRejectedValueOnce(new Error('HTTP 529'));
    render(<GenerateDialog {...props()} />);
    fireEvent.click(screen.getByRole('button', { name: /Erzeugen$/ }));
    await screen.findByText('Lift kaputt');
    expect(screen.getByText(/HTTP 529/)).toBeInTheDocument();
    expect(error).not.toHaveBeenCalled();
  });

  it('laesst sich stoppen und zeigt, was bis dahin kam', async () => {
    callAIMock.mockResolvedValueOnce('["Lift kaputt"]').mockReturnValueOnce(new Promise(() => {}));
    render(<GenerateDialog {...props()} />);
    fireEvent.click(screen.getByRole('button', { name: /Erzeugen$/ }));
    await waitFor(() => expect(callAIMock).toHaveBeenCalledTimes(2));
    fireEvent.click(screen.getByRole('button', { name: 'Stoppen' }));
    await screen.findByText('Lift kaputt');
    expect(screen.getByText(/Gestoppt/)).toBeInTheDocument();
  });

  it('waehlt Paare ohne Antwort ab und speichert sie nicht', async () => {
    callAIMock.mockResolvedValue('[{"text":"a","target":"b"},{"text":"c"}]');
    const onAdd = vi.fn();
    render(<GenerateDialog {...props(onAdd)} paare project={{ ...projekt, classes: [] }} />);
    fireEvent.click(screen.getByRole('button', { name: /Erzeugen$/ }));
    await screen.findByText('a');
    expect(screen.getByText(/keine Antwort/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '1 übernehmen' }));
    expect(onAdd.mock.calls[0][0]).toEqual([{ text: 'a', label: null, target: 'b' }]);
  });

  it('verwirft Vorschlaege nicht durch einen Klick daneben', async () => {
    callAIMock.mockResolvedValue('["a"]');
    const p = props();
    render(<GenerateDialog {...p} project={{ ...projekt, classes: [] }} />);
    fireEvent.click(screen.getByRole('button', { name: /Erzeugen$/ }));
    await screen.findByText('a');
    fireEvent.click(document.querySelector('.fixed.inset-0')!);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(p.onCancel).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Abbrechen' }));
    expect(p.onCancel).toHaveBeenCalled();
  });

  it('laesst die Anzahl leeren und neu tippen', () => {
    render(<GenerateDialog {...props()} />);
    const feld = screen.getByLabelText(/Wie viele/);
    fireEvent.change(feld, { target: { value: '' } });
    expect(feld).toHaveValue(null);
    fireEvent.change(feld, { target: { value: '5' } });
    expect(screen.getByText(/bis zu 10 Texte/)).toBeInTheDocument();
    fireEvent.change(feld, { target: { value: '900' } });
    fireEvent.blur(feld);
    expect(feld).toHaveValue(50);
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
