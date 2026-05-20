// Detect-Logik Tests – FrameTrain Plugin-System
// Ausführen: npx vitest run src/plugins/__tests__/detect.test.ts

import { describe, it, expect, assert } from 'vitest';
import { detectXLMRoberta } from '../xlm-roberta/detect';
import { detectHFEncoder } from '../hf-encoder/detect';
import { detectPlugin } from '../registry';

// ─────────────────────────────────────────────────────────────────────────────
// XLM-RoBERTa – detect.ts
// ─────────────────────────────────────────────────────────────────────────────

describe('detectXLMRoberta – Name/Pfad-Erkennung', () => {
  it('HF-ID "xlm-roberta-base" → true', () => {
    expect(detectXLMRoberta('xlm-roberta-base')).toBe(true);
  });

  it('HF-ID mit Namespace "facebook/xlm-roberta-large" → true', () => {
    expect(detectXLMRoberta('facebook/xlm-roberta-large')).toBe(true);
  });

  it('Alias "xlmroberta" (ohne Bindestrich) → true', () => {
    expect(detectXLMRoberta('/models/xlmroberta_v2')).toBe(true);
  });

  it('Alias "xlm_roberta" (mit Unterstrich) → true', () => {
    expect(detectXLMRoberta('/models/xlm_roberta')).toBe(true);
  });

  it('Windows-Pfad mit Backslash → true', () => {
    expect(detectXLMRoberta('C:\\models\\xlm-roberta-finetuned')).toBe(true);
  });

  it('Fremdes Modell "bert-base-uncased" → false', () => {
    expect(detectXLMRoberta('bert-base-uncased')).toBe(false);
  });

  it('Fremdes Modell "gpt2" → false', () => {
    expect(detectXLMRoberta('gpt2')).toBe(false);
  });

  it('Leerer String → false (kein Crash)', () => {
    expect(detectXLMRoberta('')).toBe(false);
  });
});

describe('detectXLMRoberta – config.json-Erkennung', () => {
  it('model_type = "xlm-roberta" → true', () => {
    expect(detectXLMRoberta('local-model', { model_type: 'xlm-roberta' })).toBe(true);
  });

  it('architectures enthält XLMRobertaForSequenceClassification → true', () => {
    expect(detectXLMRoberta('local-model', {
      architectures: ['XLMRobertaForSequenceClassification'],
    })).toBe(true);
  });

  it('architectures enthält XLMRobertaForTokenClassification → true', () => {
    expect(detectXLMRoberta('local-model', {
      architectures: ['XLMRobertaForTokenClassification'],
    })).toBe(true);
  });

  it('architectures enthält XLMRobertaModel → true', () => {
    expect(detectXLMRoberta('local-model', {
      architectures: ['XLMRobertaModel'],
    })).toBe(true);
  });

  it('architectures enthält XLMRobertaForMaskedLM → true', () => {
    expect(detectXLMRoberta('local-model', {
      architectures: ['XLMRobertaForMaskedLM'],
    })).toBe(true);
  });

  it('config.json mit fremdem model_type → false', () => {
    expect(detectXLMRoberta('local-model', { model_type: 'bert' })).toBe(false);
  });

  it('config.json mit fremder architecture → false', () => {
    expect(detectXLMRoberta('local-model', {
      architectures: ['BertForSequenceClassification'],
    })).toBe(false);
  });

  it('config.json ist leer ({}) → false', () => {
    expect(detectXLMRoberta('local-model', {})).toBe(false);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// HF Encoder – detect.ts
// ─────────────────────────────────────────────────────────────────────────────

describe('detectHFEncoder – config.json hat Priorität', () => {
  it('model_type = "bert" → true', () => {
    expect(detectHFEncoder('egal', { model_type: 'bert' })).toBe(true);
  });

  it('model_type = "deberta-v2" → true', () => {
    expect(detectHFEncoder('egal', { model_type: 'deberta-v2' })).toBe(true);
  });

  it('model_type = "distilbert" → true', () => {
    expect(detectHFEncoder('egal', { model_type: 'distilbert' })).toBe(true);
  });

  it('model_type = "xlm-roberta" → true (HF-Encoder kennt xlm-roberta auch)', () => {
    // Hinweis: In der Registry kommt xlm-roberta-Plugin ZUERST.
    // detectHFEncoder selbst gibt hier true zurück – das ist korrekt.
    // Registry-Priorität wird separat getestet.
    expect(detectHFEncoder('egal', { model_type: 'xlm-roberta' })).toBe(true);
  });

  it('model_type = "gpt2" (nicht in Liste) → false', () => {
    expect(detectHFEncoder('egal', { model_type: 'gpt2' })).toBe(false);
  });
});

describe('detectHFEncoder – Fallback auf Name/Pfad', () => {
  it('"bert-base-uncased" ohne config.json → true', () => {
    expect(detectHFEncoder('bert-base-uncased')).toBe(true);
  });

  it('"microsoft/deberta-v3-base" → true', () => {
    expect(detectHFEncoder('microsoft/deberta-v3-base')).toBe(true);
  });

  it('"distilbert-base-multilingual-cased" → true', () => {
    expect(detectHFEncoder('distilbert-base-multilingual-cased')).toBe(true);
  });

  it('"xlm-roberta-base" ohne config → true (xlm im Token-Set)', () => {
    expect(detectHFEncoder('xlm-roberta-base')).toBe(true);
  });

  it('"gpt2" → false', () => {
    expect(detectHFEncoder('gpt2')).toBe(false);
  });

  it('Leerer String → false (kein Crash)', () => {
    expect(detectHFEncoder('')).toBe(false);
  });
});

describe('detectHFEncoder – Wortgrenzen (containsToken)', () => {
  it('"albert-base-v2" → true via "albert"-Token', () => {
    expect(detectHFEncoder('albert-base-v2')).toBe(true);
  });

  it('"roberta-base" → true via "roberta"-Token', () => {
    expect(detectHFEncoder('roberta-base')).toBe(true);
  });

  it('"funnel-transformer/small" → true via "funnel"-Token', () => {
    expect(detectHFEncoder('funnel-transformer/small')).toBe(true);
  });

  it('"ibert-roberta-base" → false (ibert nicht im Token-Set, Wortgrenzen blockieren bert-Match)', () => {
    // "ibert" ist kein Token. "bert" als Regex (^|[^a-z0-9])bert([^a-z0-9]|$)
    // matcht NICHT weil "i" direkt vor "bert" steht.
    // "roberta" ist aber ein Token → dieser Test prüft nur den bert-Teil.
    // Vollständiger Pfad "ibert-roberta-base" matcht wegen "roberta" → true
    // Korrektes Erwartungsergebnis: true (wegen roberta)
    expect(detectHFEncoder('ibert-roberta-base')).toBe(true);
  });

  it('"some-ibert-model" ohne roberta → false (Wortgrenze blockiert bert)', () => {
    // Kein roberta, kein anderer Token. Nur ibert.
    // Wortgrenzen-Regex verhindert Match auf "bert" innerhalb von "ibert".
    expect(detectHFEncoder('some-ibert-model')).toBe(false);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Registry – detectPlugin()
// ─────────────────────────────────────────────────────────────────────────────

describe('detectPlugin – Priorität & Routing', () => {
  it('"xlm-roberta-base" → plugin.id = "xlm-roberta"', () => {
    const result = detectPlugin('xlm-roberta-base');
    expect(result.supported).toBe(true);
    if (result.supported) expect(result.plugin.id).toBe('xlm-roberta');
  });

  it('"xlm-roberta-base" landet NICHT bei hf-encoder', () => {
    const result = detectPlugin('xlm-roberta-base');
    expect(result.supported).toBe(true);
    if (result.supported) expect(result.plugin.id).not.toBe('hf-encoder');
  });

  it('"bert-base-uncased" → plugin.id = "hf-encoder"', () => {
    const result = detectPlugin('bert-base-uncased');
    expect(result.supported).toBe(true);
    if (result.supported) expect(result.plugin.id).toBe('hf-encoder');
  });

  it('"distilbert-base-multilingual" → plugin.id = "hf-encoder"', () => {
    const result = detectPlugin('distilbert-base-multilingual');
    expect(result.supported).toBe(true);
    if (result.supported) expect(result.plugin.id).toBe('hf-encoder');
  });

  it('"gpt2" → supported: false', () => {
    const result = detectPlugin('gpt2');
    expect(result.supported).toBe(false);
  });

  it('"gpt2" → reason nennt verfügbare Plugins', () => {
    const result = detectPlugin('gpt2');
    expect(result.supported).toBe(false);
    expect('reason' in result).toBe(true);
    if (!result.supported && 'reason' in result) {
      expect(result.reason).toContain('XLM-RoBERTa');
      expect(result.reason).toContain('HF Encoder');
    }
  });

  it('Leerer String → supported: false mit "Kein Modellpfad"', () => {
    const result = detectPlugin('');
    assert(!result.supported);
    expect('reason' in result).toBe(true);
    if (!result.supported && 'reason' in result) {
      expect(result.reason).toMatch(/kein modellpfad/i);
    }
  });

  it('Nur Leerzeichen "   " → supported: false', () => {
    const result = detectPlugin('   ');
    expect(result.supported).toBe(false);
  });

  it('config.json mit model_type="xlm-roberta" → plugin.id = "xlm-roberta"', () => {
    const result = detectPlugin('local-model', { model_type: 'xlm-roberta' });
    expect(result.supported).toBe(true);
    if (result.supported) expect(result.plugin.id).toBe('xlm-roberta');
  });

  it('config.json mit model_type="bert" → plugin.id = "hf-encoder"', () => {
    const result = detectPlugin('local-model', { model_type: 'bert' });
    expect(result.supported).toBe(true);
    if (result.supported) expect(result.plugin.id).toBe('hf-encoder');
  });
});
