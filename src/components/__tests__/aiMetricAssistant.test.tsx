// Der KI-Metrik-Assistent im Training-Panel.
// Erwartung des Nutzers: kurzer, lesbarer Text PLUS uebernehmbare Metriken.
// Ausfuehren: npx vitest run src/components/__tests__/aiMetricAssistant.test.tsx --config vitest.config.ts

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

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

vi.mock('../../contexts/LanguageContext', () => ({
  useLanguage: () => ({ language: 'de' as const, t: (key: string) => key }),
}));

import { AIMetricAssistant, DEFAULT_CONFIG } from '../TrainingPanel';
import { TOKEN_BUDGET_CONFIG } from '../../contexts/AISettingsContext';

/** Genau die Form, die in der Praxis ankam: Markdown + am Limit gekapptes JSON. */
const ANTWORT_ABGESCHNITTEN = [
  '## Analyse der aktuellen Konfiguration',
  '',
  'Die Konfiguration ist auf ein NLP-Setup zugeschnitten und passt nicht zu YOLOv8.',
  '',
  '## Empfehlungen',
  '',
  '- Mehr Epochen',
  '- SGD als Optimizer',
  '',
  '```json',
  '{"epochs":100,"batch_size":16,"learning_rate":0.001,"optimizer":"sgd","scheduler":"cosine","fp16":true,"logging_steps":20,',
].join('\n');

function renderAssistant(onApply = vi.fn()) {
  render(
    <AIMetricAssistant
      config={DEFAULT_CONFIG}
      datasetName="SkiTrain"
      datasetSize={1161}
      modelName="YOLOv8"
      onApply={onApply}
      onClose={vi.fn()}
      onSaveAsTemplate={async () => true}
    />,
  );
  return onApply;
}

async function fragen() {
  await userEvent.click(screen.getByText('trainingPanel.aiAssistant.startButton'));
  await waitFor(() => expect(callAIMock).toHaveBeenCalled());
}

describe('AIMetricAssistant', () => {
  beforeEach(() => {
    callAIMock.mockReset();
    settingsRef.current = { ...settingsRef.current, tokenBudget: 'balanced' };
  });

  // Vorher stand die Markdown-Auszeichnung woertlich im Dialog ("## Analyse …"),
  // weil der Text in ein whitespace-pre-wrap-Div gerendert wurde.
  it('rendert die Antwort als Markdown, nicht als Rohtext', async () => {
    callAIMock.mockResolvedValue(ANTWORT_ABGESCHNITTEN);
    renderAssistant();
    await fragen();

    await waitFor(() => expect(screen.getByText('Analyse der aktuellen Konfiguration')).toBeTruthy());
    expect(screen.queryByText(/## Analyse/)).toBeNull();
  });

  // Der Kern: Text UND uebernehmbare Metriken. Der alte Parser brauchte eine
  // schliessende Klammer — bei abgeschnittenem JSON gab es gar keine Liste.
  it('bietet die Metriken zum Uebernehmen an, auch wenn das JSON abgeschnitten ist', async () => {
    callAIMock.mockResolvedValue(ANTWORT_ABGESCHNITTEN);
    const onApply = renderAssistant();
    await fragen();

    await waitFor(() => expect(screen.getByText('epochs')).toBeTruthy());
    for (const key of ['epochs', 'batch_size', 'learning_rate', 'optimizer', 'scheduler', 'fp16', 'logging_steps']) {
      expect(screen.getByText(key), key).toBeTruthy();
    }

    await userEvent.click(screen.getByText('trainingPanel.aiAssistant.applyButton'));
    expect(onApply).toHaveBeenCalledWith(expect.objectContaining({
      epochs: 100, batch_size: 16, learning_rate: 0.001,
      optimizer: 'sgd', scheduler: 'cosine', fp16: true, logging_steps: 20,
    }));
  });

  it('zeigt den JSON-Block nicht zusaetzlich im Textteil', async () => {
    callAIMock.mockResolvedValue(ANTWORT_ABGESCHNITTEN);
    renderAssistant();
    await fragen();

    await waitFor(() => expect(screen.getByText('Empfehlungen')).toBeTruthy());
    expect(screen.queryByText(/"learning_rate":0\.001/)).toBeNull();
  });

  // Die Laengen-Vorgabe steht nicht mehr im User-Prompt, sondern kommt
  // zentral aus dem Token-Budget (aiClient, style: 'chat'). Hier wird nur
  // geprueft, dass Budget und Stil korrekt durchgereicht werden.
  it('nutzt das eingestellte Token-Budget', async () => {
    settingsRef.current = { ...settingsRef.current, tokenBudget: 'max' };
    callAIMock.mockResolvedValue(ANTWORT_ABGESCHNITTEN);
    renderAssistant();
    await fragen();

    // aiClient.callAI(settings, { system, messages, maxTokens, ... })
    const opts = callAIMock.mock.calls[0][1] as { maxTokens: number; style: string };
    expect(opts.maxTokens).toBe(TOKEN_BUDGET_CONFIG.max.maxTokens);
    expect(opts.style).toBe('chat');
  });

  it('haelt sich bei kleinem Budget kurz', async () => {
    settingsRef.current = { ...settingsRef.current, tokenBudget: 'minimal' };
    callAIMock.mockResolvedValue(ANTWORT_ABGESCHNITTEN);
    renderAssistant();
    await fragen();

    const opts = callAIMock.mock.calls[0][1] as { maxTokens: number; style: string };
    expect(opts.maxTokens).toBe(TOKEN_BUDGET_CONFIG.minimal.maxTokens);
    expect(opts.style).toBe('chat');
  });

  // Der Praxisfall: bei YOLOv8 stand die komplette NLP-Feldliste im Prompt,
  // woraufhin die KI drei Absaetze darauf verwendete zu erklaeren, dass LoRA
  // und max_seq_length fuer einen CNN-Detektor nicht gelten.
  it('laesst NLP-Felder bei einem Detektor aus dem Prompt', async () => {
    callAIMock.mockResolvedValue(ANTWORT_ABGESCHNITTEN);
    render(
      <AIMetricAssistant
        config={{ ...DEFAULT_CONFIG, task_type: 'detect' }}
        datasetName="SkiTrain" datasetSize={580} modelName="YOLOv8"
        onApply={vi.fn()} onClose={vi.fn()} onSaveAsTemplate={async () => true}
      />,
    );
    await fragen();

    const prompt = (callAIMock.mock.calls[0][1] as { messages: { content: string }[] }).messages[0].content;
    // Weder in der Feldliste noch im Dump der aktuellen Konfiguration.
    for (const feld of ['max_seq_length', 'lora_r', 'lora_target_modules', 'group_by_length', 'load_in_4bit']) {
      expect(prompt, feld).not.toContain(`- ${feld}:`);
    }
    expect(prompt).toContain('- epochs:');
    expect(prompt).toContain('Task: detect');
    // Der leer gewordene LoRA-Abschnitt darf nicht als Ueberschrift stehenbleiben.
    expect(prompt).not.toContain('LORA / QLORA');
    // Der Vorspann der Feldliste bleibt erhalten.
    expect(prompt).toContain('ALLE VERFÜGBAREN METRIKEN');
    // plugin_config als Objekt hat im Prompt nichts verloren.
    expect(prompt).not.toContain('[object Object]');
  });

  it('behaelt die NLP-Felder bei einem Text-Modell', async () => {
    callAIMock.mockResolvedValue(ANTWORT_ABGESCHNITTEN);
    render(
      <AIMetricAssistant
        config={{ ...DEFAULT_CONFIG, task_type: 'seq_classification' }}
        datasetName="Texte" datasetSize={500} modelName="distilbert-base-uncased"
        onApply={vi.fn()} onClose={vi.fn()} onSaveAsTemplate={async () => true}
      />,
    );
    await fragen();

    const prompt = (callAIMock.mock.calls[0][1] as { messages: { content: string }[] }).messages[0].content;
    expect(prompt).toContain('- max_seq_length:');
    expect(prompt).toContain('- lora_r:');
  });

  // Nach dem Uebernehmen aktualisiert sich die Config-Prop; ohne Schnappschuss
  // stand in der Diff-Liste dann "50 -> 50".
  it('zeigt nach dem Uebernehmen weiter den Ausgangswert', async () => {
    callAIMock.mockResolvedValue(ANTWORT_ABGESCHNITTEN);
    renderAssistant();
    await fragen();

    await waitFor(() => expect(screen.getByText('epochs')).toBeTruthy());
    expect(screen.getByText(String(DEFAULT_CONFIG.epochs))).toBeTruthy();
    await userEvent.click(screen.getByText('trainingPanel.aiAssistant.applyButton'));
    // Der durchgestrichene Ausgangswert bleibt stehen.
    expect(screen.getByText(String(DEFAULT_CONFIG.epochs))).toBeTruthy();
  });
});
