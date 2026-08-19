// Integration Tests – FrameTrain Plugin-System (Bereich 4)
// Ausführen: npx vitest run src/plugins/__tests__/integration.test.ts --config vitest.config.ts

import { describe, it, expect } from 'vitest';
import type { ModelPlugin } from '../types';
import xlmRobertaPlugin from '../xlm-roberta';
import hfEncoderPlugin from '../hf-encoder';
import { PLUGINS, detectPlugin } from '../registry';
import { checkDatasetCompat } from '../datasetCompat';
import { xlmRobertaCompatPlugin } from '../xlm-roberta/datasetCompat';
import { hfEncoderCompatPlugin } from '../hf-encoder/datasetCompat';

// ─────────────────────────────────────────────────────────────────────────────
// Typen-Kontrakt: Plugin-Objekte erfüllen ModelPlugin-Interface zur Laufzeit
// ─────────────────────────────────────────────────────────────────────────────

const REQUIRED_FIELDS: (keyof ModelPlugin)[] = [
  'id', 'name', 'description', 'taskType', 'detect', 'TrainComponent', 'TestComponent',
];

describe('Typen-Kontrakt – xlmRobertaPlugin', () => {
  it.each(REQUIRED_FIELDS)('hat Pflichtfeld "%s"', (field) => {
    expect(xlmRobertaPlugin[field]).toBeDefined();
  });

  it('id ist ein nicht-leerer String', () => {
    expect(typeof xlmRobertaPlugin.id).toBe('string');
    expect(xlmRobertaPlugin.id.length).toBeGreaterThan(0);
  });

  it('name ist ein nicht-leerer String', () => {
    expect(typeof xlmRobertaPlugin.name).toBe('string');
    expect(xlmRobertaPlugin.name.length).toBeGreaterThan(0);
  });

  it('description ist ein nicht-leerer String', () => {
    expect(typeof xlmRobertaPlugin.description).toBe('string');
    expect(xlmRobertaPlugin.description.length).toBeGreaterThan(0);
  });

  it('taskType ist ein nicht-leerer String', () => {
    expect(typeof xlmRobertaPlugin.taskType).toBe('string');
    expect(xlmRobertaPlugin.taskType.length).toBeGreaterThan(0);
  });

  it('detect ist eine Funktion', () => {
    expect(typeof xlmRobertaPlugin.detect).toBe('function');
  });

  it('TrainComponent ist eine Funktion (React-Komponente)', () => {
    expect(typeof xlmRobertaPlugin.TrainComponent).toBe('function');
  });

  it('TestComponent ist eine Funktion (React-Komponente)', () => {
    expect(typeof xlmRobertaPlugin.TestComponent).toBe('function');
  });

  it('id ist exakt "xlm-roberta"', () => {
    expect(xlmRobertaPlugin.id).toBe('xlm-roberta');
  });

  it('taskType ist "seq_classification"', () => {
    expect(xlmRobertaPlugin.taskType).toBe('seq_classification');
  });
});

describe('Typen-Kontrakt – hfEncoderPlugin', () => {
  it.each(REQUIRED_FIELDS)('hat Pflichtfeld "%s"', (field) => {
    expect(hfEncoderPlugin[field]).toBeDefined();
  });

  it('id ist exakt "hf-encoder"', () => {
    expect(hfEncoderPlugin.id).toBe('hf-encoder');
  });

  it('taskType ist "seq_classification"', () => {
    expect(hfEncoderPlugin.taskType).toBe('seq_classification');
  });

  it('id unterscheidet sich von xlmRobertaPlugin.id', () => {
    expect(hfEncoderPlugin.id).not.toBe(xlmRobertaPlugin.id);
  });

  it('TrainComponent ist NICHT dieselbe Funktion wie bei xlm-roberta', () => {
    expect(hfEncoderPlugin.TrainComponent).not.toBe(xlmRobertaPlugin.TrainComponent);
  });

  it('TestComponent ist NICHT dieselbe Funktion wie bei xlm-roberta', () => {
    expect(hfEncoderPlugin.TestComponent).not.toBe(xlmRobertaPlugin.TestComponent);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Registry-Priorität & Fallthrough
// ─────────────────────────────────────────────────────────────────────────────

describe('PLUGINS[] – Reihenfolge & Vollständigkeit', () => {
  it('enthält alle registrierten Basis-Plugins', () => {
    const ids = PLUGINS.map(p => p.id);
    expect(ids).toEqual(
      expect.arrayContaining(['canvas', 'yolo', 'image-classification', 'xlm-roberta', 'hf-encoder'])
    );
  });

  it('canvas liegt an Index 0 (höchste Priorität, vor generischen Plugins)', () => {
    expect(PLUGINS[0].id).toBe('canvas');
  });

  it('xlm-roberta liegt vor hf-encoder (spezifisch vor Fallback)', () => {
    const ids = PLUGINS.map(p => p.id);
    expect(ids.indexOf('xlm-roberta')).toBeLessThan(ids.indexOf('hf-encoder'));
  });

  it('keine doppelten IDs', () => {
    const ids = PLUGINS.map(p => p.id);
    const unique = new Set(ids);
    expect(unique.size).toBe(ids.length);
  });

  it('alle Plugins haben detect-Funktionen die aufrufbar sind', () => {
    for (const plugin of PLUGINS) {
      expect(() => plugin.detect('test-model')).not.toThrow();
    }
  });
});

describe('Registry-Priorität – Überschneidung xlm-roberta / hf-encoder', () => {
  it('xlm-roberta-large landet bei xlm-roberta, nicht hf-encoder', () => {
    // hf-encoder würde "xlm" auch als Token kennen – aber xlm-roberta gewinnt durch Priorität
    const result = detectPlugin('xlm-roberta-large');
    expect(result.supported).toBe(true);
    if (result.supported) expect(result.plugin.id).toBe('xlm-roberta');
  });

  it('bert-large-uncased landet bei hf-encoder', () => {
    const result = detectPlugin('bert-large-uncased');
    expect(result.supported).toBe(true);
    if (result.supported) expect(result.plugin.id).toBe('hf-encoder');
  });

  it('microsoft/deberta-v3-large landet bei hf-encoder', () => {
    const result = detectPlugin('microsoft/deberta-v3-large');
    expect(result.supported).toBe(true);
    if (result.supported) expect(result.plugin.id).toBe('hf-encoder');
  });

  it('Pfad schlägt config.json: Pfad "xlm-roberta" + model_type="bert" → xlm-roberta gewinnt', () => {
    // detectXLMRoberta prüft den Pfad ZUERST – matcht der Pfad, wird config.json
    // nicht mehr abgewartet. Das ist gewollt: ein fine-getuntes XLM-RoBERTa kann
    // im config.json noch "bert" als model_type haben (HF-Artifact).
    // Pfad-Information ist in diesem Fall verlässlicher als config.json.
    const result = detectPlugin('/models/xlm-roberta', { model_type: 'bert' });
    expect(result.supported).toBe(true);
    if (result.supported) expect(result.plugin.id).toBe('xlm-roberta');
  });

  it('config.json model_type="xlm-roberta" bei "bert"-Pfad → xlm-roberta gewinnt', () => {
    const result = detectPlugin('/models/bert-classifier', { model_type: 'xlm-roberta' });
    expect(result.supported).toBe(true);
    if (result.supported) expect(result.plugin.id).toBe('xlm-roberta');
  });

  it('reason bei unsupported enthält ALLE Plugin-Namen', () => {
    const result = detectPlugin('llama-3-70b');
    expect(result.supported).toBe(false);
    if (!result.supported) {
      for (const plugin of PLUGINS) {
        expect((result as any).reason).toContain(plugin.name);
      }
    }
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// ID-Kontrakt: plugin.id === CompatPlugin.modelPluginId
// ─────────────────────────────────────────────────────────────────────────────

describe('ID-Kontrakt zwischen ModelPlugin und DatasetCompatPlugin', () => {
  it('xlmRobertaPlugin.id === xlmRobertaCompatPlugin.modelPluginId', () => {
    expect(xlmRobertaPlugin.id).toBe(xlmRobertaCompatPlugin.modelPluginId);
  });

  it('hfEncoderPlugin.id === hfEncoderCompatPlugin.modelPluginId', () => {
    expect(hfEncoderPlugin.id).toBe(hfEncoderCompatPlugin.modelPluginId);
  });

  it('detectPlugin → plugin.id → checkDatasetCompat: xlm-roberta kein Fallback', () => {
    const detection = detectPlugin('xlm-roberta-base');
    expect(detection.supported).toBe(true);
    if (!detection.supported) return;

    const compat = checkDatasetCompat(detection.plugin.id, ['.csv']);
    // Wäre compat.overallLevel 'ok', hätte der ID-Kontrakt versagt (Fallback)
    expect(compat.overallLevel).toBe('perfect');
  });

  it('detectPlugin → plugin.id → checkDatasetCompat: hf-encoder kein Fallback', () => {
    const detection = detectPlugin('bert-base-uncased');
    expect(detection.supported).toBe(true);
    if (!detection.supported) return;

    const compat = checkDatasetCompat(detection.plugin.id, ['.jsonl']);
    expect(compat.overallLevel).toBe('perfect');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Smoke-Test: neues Plugin (minimal, ohne Registrierung)
// ─────────────────────────────────────────────────────────────────────────────

describe('Smoke-Test – minimales neues Plugin erfüllt Interface', () => {
  // Simuliert was ein Entwickler tun würde, wenn er ein neues Plugin anlegt.
  // Das Plugin wird NICHT in PLUGINS[] eingetragen – wir testen nur den Kontrakt.

  // Bewusst ein Modelltyp, fuer den es KEIN Plugin gibt. Frueher stand hier
  // Whisper — seit dem Audio-Plugin ist das unterstuetzt und der Test pruefte
  // nicht mehr, was er pruefen sollte.
  const mockDetect = (modelPath: string) => modelPath.includes('bark');

  const unsupportedPlugin: ModelPlugin = {
    id: 'text-to-speech-demo',
    name: 'Text-to-Speech (Demo)',
    description: 'Beispiel-Plugin fuer Sprachsynthese – nicht registriert',
    taskType: 'text_to_speech',
    detect: mockDetect,
    TrainComponent: () => null,
    TestComponent: () => null,
  };

  it('minimales Plugin hat alle Pflichtfelder', () => {
    for (const field of REQUIRED_FIELDS) {
      expect(unsupportedPlugin[field]).toBeDefined();
    }
  });

  it('detect-Funktion funktioniert wie erwartet', () => {
    expect(unsupportedPlugin.detect('suno/bark-small')).toBe(true);
    expect(unsupportedPlugin.detect('bert-base-uncased')).toBe(false);
  });

  it('id ist einzigartig gegenüber registrierten Plugins', () => {
    const existingIds = PLUGINS.map(p => p.id);
    expect(existingIds).not.toContain(unsupportedPlugin.id);
  });

  it('NOCH NICHT in PLUGINS[] – detectPlugin findet es nicht', () => {
    const result = detectPlugin('suno/bark-small');
    // Sprachsynthese hat kein Plugin → supported: false
    expect(result.supported).toBe(false);
  });

  it('checkDatasetCompat mit unbekannter Plugin-ID → Fallback, kein Crash', () => {
    const result = checkDatasetCompat(unsupportedPlugin.id, ['.wav', '.mp3']);
    expect(result).toBeDefined();
    expect(result.overallLevel).toBe('ok'); // Fallback-Level
  });

  it('nach Registrierung (simuliert via PLUGINS-Push) → detectPlugin findet es', () => {
    // Temporär eintragen, nach Test entfernen
    PLUGINS.push(unsupportedPlugin);
    try {
      const result = detectPlugin('suno/bark-small');
      expect(result.supported).toBe(true);
      if (result.supported) expect(result.plugin.id).toBe('text-to-speech-demo');
    } finally {
      PLUGINS.pop(); // immer aufräumen
    }
  });

  it('nach Entfernen aus PLUGINS[] → detectPlugin findet es wieder nicht', () => {
    // Sicherstellen dass der finally-Block oben funktioniert hat (whisper wieder raus)
    expect(PLUGINS.map(p => p.id)).not.toContain('whisper');
    const result = detectPlugin('suno/bark-small');
    expect(result.supported).toBe(false);
  });
});
