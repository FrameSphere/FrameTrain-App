// Der Steuerblock am Anfang jeder Code-Assistenten-Antwort.
//
// Blieb er unerkannt, stand das rohe JSON sichtbar im Chat — genau so
// passiert mit groq/compound-mini, das den ```ft_action-Block nicht
// geschlossen hat.
//
// Ausfuehren: npx vitest run src/ai/__tests__/autoModeProtocol.test.ts --config vitest.config.ts

import { describe, it, expect } from 'vitest';
import { parseAutoAction, buildAutoSystemPrompt } from '../autoModeProtocol';

const ACTION = '{"mode":"edit","rationale":"Tippfehler behoben","title":"Fix"}';

describe('parseAutoAction', () => {
  it('liest den regulaeren ft_action-Block und schneidet ihn heraus', () => {
    const { action, cleaned } = parseAutoAction('```ft_action\n' + ACTION + '\n```\n\nHier die Erklaerung.');
    expect(action?.mode).toBe('edit');
    expect(action?.rationale).toBe('Tippfehler behoben');
    expect(cleaned).toBe('Hier die Erklaerung.');
  });

  // Der Praxisfall aus dem Groq-Test.
  it('kommt mit einem NICHT geschlossenen Block zurecht', () => {
    const { action, cleaned } = parseAutoAction('```ft_action\n' + ACTION + '\n\nHier die Erklaerung.');
    expect(action?.mode).toBe('edit');
    expect(cleaned).toBe('Hier die Erklaerung.');
    expect(cleaned).not.toMatch(/mode|rationale/);
  });

  it('akzeptiert den Block auch als json-Fence', () => {
    const { action, cleaned } = parseAutoAction('```json\n' + ACTION + '\n```\nText.');
    expect(action?.mode).toBe('edit');
    expect(cleaned).toBe('Text.');
  });

  // Der Praxisfall aus 1.2.52: groq/compound-mini setzt gar keine Backticks,
  // sondern schreibt die Zeile "ft_action" und darunter das JSON.
  it('erkennt den Block auch ganz ohne Fence', () => {
    const { action, cleaned } = parseAutoAction('ft_action\n' + ACTION + '\n\nKurz: Tippfehler behoben.');
    expect(action?.mode).toBe('edit');
    expect(cleaned).toBe('Kurz: Tippfehler behoben.');
    expect(cleaned).not.toMatch(/ft_action|rationale/);
  });

  it('erkennt ein nacktes Steuer-JSON am Anfang', () => {
    const { action, cleaned } = parseAutoAction(ACTION + '\nDie Erklaerung.');
    expect(action?.mode).toBe('edit');
    expect(cleaned).toBe('Die Erklaerung.');
  });

  // Ein JSON MITTEN in der Antwort ist kein Steuerblock, sondern Inhalt.
  it('greift nur am Anfang der Antwort', () => {
    const text = 'Nutze diese Werte:\n' + ACTION;
    expect(parseAutoAction(text)).toEqual({ action: null, cleaned: text });
  });

  it('laesst Text ohne Steuerblock unveraendert', () => {
    const text = 'Nur eine normale Antwort ohne Steuerblock.';
    expect(parseAutoAction(text)).toEqual({ action: null, cleaned: text });
  });

  // Ein Antwort-JSON, das kein Steuerblock ist, darf nicht verschluckt werden.
  it('ignoriert einen json-Block ohne gueltigen mode', () => {
    const text = '```json\n{"epochs":5}\n```';
    expect(parseAutoAction(text)).toEqual({ action: null, cleaned: text });
  });

  it('stolpert nicht ueber Klammern in Strings', () => {
    const withBrace = '{"mode":"chat","rationale":"nutze dict {a: 1} statt Liste"}';
    const { action, cleaned } = parseAutoAction('```ft_action\n' + withBrace + '\n```\nFertig.');
    expect(action?.mode).toBe('chat');
    expect(cleaned).toBe('Fertig.');
  });
});

// Bei knappem Budget schrieb groq/compound-mini die komplette Erklaerung in
// die rationale und liess den Fliesstext leer — die Antwortblase im Chat
// blieb dadurch komplett leer.
describe('buildAutoSystemPrompt', () => {
  it('stellt klar, dass die Erklaerung nicht in die rationale gehoert', () => {
    const prompt = buildAutoSystemPrompt('BASIS');
    expect(prompt).toContain('BASIS');
    expect(prompt).toMatch(/rationale/i);
    expect(prompt).toMatch(/NICHT angezeigt|niemals ausschliesslich/i);
  });
});
