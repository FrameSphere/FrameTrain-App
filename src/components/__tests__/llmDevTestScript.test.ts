// Dev-Test-Vorlage fuer LLMs.
// Regression aus dem Dev-Train/Dev-Test-Durchgang vom 26.10.2026: SmolLM2
// bekam im Dev Test die Klassifikations-Vorlage und brach mit
// "Text-Spalte nicht gefunden. Vorhanden: ['messages']" ab.

import { describe, it, expect } from 'vitest';
import { generateCausalLmTestScript } from '../llmDevScript';

const script = generateCausalLmTestScript({ modelPath: '/m/llm', datasetPath: '/ds/chat', outputPath: '/out/dev_test' });

describe('generateCausalLmTestScript', () => {
  it('nutzt ein Decoder-LLM statt eines Klassifikators', () => {
    expect(script).toContain('AutoModelForCausalLM');
    expect(script).not.toContain('AutoModelForSequenceClassification');
  });

  it('liest Chat-Daten: letzte Assistenten-Antwort ist die Referenz', () => {
    expect(script).toContain('row.get("messages")');
    expect(script).toContain('add_generation_prompt=True');
  });

  it('nimmt die Pfade aus den Umgebungsvariablen mit Vorgabewerten', () => {
    expect(script).toContain('os.environ.get("MODEL_PATH",   "/m/llm")');
    expect(script).toContain('os.environ.get("DATASET_PATH", "/ds/chat")');
    expect(script).toContain('os.environ.get("OUTPUT_PATH",  "/out/dev_test")');
  });

  it('schreibt results.json mit Treffer- oder Perplexity-Metrik', () => {
    expect(script).toContain('results.json');
    expect(script).toContain('exact_match');
    expect(script).toContain('perplexity');
  });
});
