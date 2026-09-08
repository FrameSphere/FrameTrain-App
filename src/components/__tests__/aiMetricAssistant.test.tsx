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

  it('nutzt das eingestellte Token-Budget', async () => {
    settingsRef.current = { ...settingsRef.current, tokenBudget: 'max' };
    callAIMock.mockResolvedValue(ANTWORT_ABGESCHNITTEN);
    renderAssistant();
    await fragen();

    // aiClient.callAI(settings, { system, messages, maxTokens, ... })
    const opts = callAIMock.mock.calls[0][1] as { maxTokens: number; messages: { content: string }[] };
    expect(opts.maxTokens).toBe(TOKEN_BUDGET_CONFIG.max.maxTokens);
    // Bei grossem Budget faellt die feste "3-4 Saetze"-Vorgabe weg.
    expect(opts.messages[0].content).not.toMatch(/3-4 Sätze/);
    expect(opts.messages[0].content).toMatch(/ausführlich/);
  });

  it('haelt sich bei kleinem Budget kurz', async () => {
    settingsRef.current = { ...settingsRef.current, tokenBudget: 'minimal' };
    callAIMock.mockResolvedValue(ANTWORT_ABGESCHNITTEN);
    renderAssistant();
    await fragen();

    const opts = callAIMock.mock.calls[0][1] as { maxTokens: number; messages: { content: string }[] };
    expect(opts.maxTokens).toBe(TOKEN_BUDGET_CONFIG.minimal.maxTokens);
    expect(opts.messages[0].content).toMatch(/2-3 Sätze/);
  });
});
