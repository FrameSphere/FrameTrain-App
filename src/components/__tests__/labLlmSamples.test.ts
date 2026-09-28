// Live-Test 1.4.0: Im Labor bekam ein LLM bei Chat-Daten die ganze JSON-Zeile
// als Eingabe — samt richtiger Antwort — und der Kopf zeigte
// "[object Object], [object Object]".
import { describe, it, expect } from 'vitest';
import { parseSamples } from '../LaboratoryPanel';

describe('Labor: LLM-Datenformate', () => {
  it('Chat (messages): letzte Nutzerfrage ist die Eingabe, Assistent die Erwartung', () => {
    const row = { messages: [
      { role: 'system', content: 'Sei knapp.' },
      { role: 'user', content: 'Wort: Zange' },
      { role: 'assistant', content: 'Werkzeug' },
    ] };
    const [s] = parseSamples(JSON.stringify(row) + '\n', 'test.jsonl');
    expect(s).toMatchObject({ text: 'Wort: Zange', label: 'Werkzeug' });
  });

  it('ShareGPT (conversations/from/value)', () => {
    const row = { conversations: [{ from: 'human', value: 'Hallo?' }, { from: 'gpt', value: 'Hi!' }] };
    expect(parseSamples(JSON.stringify(row), 'x.jsonl')[0]).toMatchObject({ text: 'Hallo?', label: 'Hi!' });
  });

  it('Alpaca mit input, prompt/completion und DPO', () => {
    const rows = [
      { instruction: 'Übersetze', input: 'Haus', output: 'house' },
      { prompt: '2+2=', completion: '4' },
      { prompt: 'Farbe?', chosen: 'Blau', rejected: 'Weiß nicht' },
    ].map(r => JSON.stringify(r)).join('\n');
    expect(parseSamples(rows, 'x.jsonl').map(s => [s.text, s.label])).toEqual([
      ['Übersetze\n\nHaus', 'house'], ['2+2=', '4'], ['Farbe?', 'Blau'],
    ]);
  });

  it('normale Textzeilen bleiben unveraendert', () => {
    expect(parseSamples('{"text": "gut", "label": "pos"}', 'x.jsonl')[0]).toMatchObject({ text: 'gut', label: 'pos' });
  });
});
