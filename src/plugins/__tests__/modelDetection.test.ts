// Erkennung eines importierten Modells – nicht nur einer Modell-ID.
//
// Ausloeser: ein lokal importiertes YOLO ("yolo8n", Ordner mit best.pt) galt im
// Labor und im Tests-Bereich als "wird noch nicht unterstuetzt". Grund war der
// source_path: bei lokalen Importen ist das der interne Speicherordner
// (.../models/local_abc123), in dem kein Architekturname mehr steht.

import { describe, it, expect } from 'vitest';
import { detectPluginForModel, isModelSupported, getPluginById } from '../registry';

const IMPORT_DIR = '/Users/x/Library/Application Support/com.frametrain.desktop/models/local_cfc998f409d4';

describe('detectPluginForModel', () => {
  it('erkennt ein lokal importiertes YOLO am Modellnamen', () => {
    const r = detectPluginForModel({
      id: 'local_cfc998f409d4', name: 'yolo8n',
      source_path: IMPORT_DIR, model_type: 'pytorch',
    });
    expect(r.supported && r.plugin.id).toBe('yolo');
  });

  it('erkennt ein trainiertes YOLO am model_type aus dem Checkpoint', () => {
    // best.pt verraet im Namen nichts; das Backend liest "yolo" aus der Datei.
    const r = detectPluginForModel({
      id: 'local_1', name: 'Ski_Model_v0.0.6',
      source_path: IMPORT_DIR, model_type: 'yolo',
    });
    expect(r.supported && r.plugin.id).toBe('yolo');
  });

  it('nimmt die Zuordnung des Nutzers vor jeder Heuristik', () => {
    const r = detectPluginForModel({
      id: 'local_2', name: 'mein-experiment',
      source_path: IMPORT_DIR, model_type: 'pytorch',
      plugin_override: 'audio-classification',
    });
    expect(r.supported && r.plugin.id).toBe('audio-classification');
  });

  it('ignoriert eine Zuordnung auf ein Plugin, das es nicht gibt', () => {
    const r = detectPluginForModel({
      id: 'local_3', name: 'mein-experiment',
      source_path: IMPORT_DIR, model_type: 'pytorch',
      plugin_override: 'gibts-nicht',
    });
    expect(r.supported).toBe(false);
  });

  it('meldet weiterhin nicht unterstuetzt, wenn weder Pfad noch Name etwas hergeben', () => {
    const r = detectPluginForModel({
      id: 'local_4', name: 'mein-experiment',
      source_path: IMPORT_DIR, model_type: 'pytorch',
    });
    expect(r.supported).toBe(false);
    expect(r.supported === false && r.reason).toContain('pytorch');
  });

  it('haelt HuggingFace-Modelle unveraendert: die Repo-ID entscheidet', () => {
    const r = detectPluginForModel({
      id: 'hf_1', name: 'mein-eigener-name',
      source_path: 'xlm-roberta-base', model_type: 'xlm-roberta',
    });
    expect(r.supported && r.plugin.id).toBe('xlm-roberta');
  });

  it('isModelSupported folgt derselben Reihenfolge', () => {
    expect(isModelSupported({ id: 'a', name: 'yolo8n', source_path: IMPORT_DIR })).toBe(true);
    expect(isModelSupported({ id: 'b', name: 'irgendwas', source_path: IMPORT_DIR })).toBe(false);
  });

  it('getPluginById liefert nur registrierte Plugins', () => {
    expect(getPluginById('yolo')?.taskType).toBe('detect');
    expect(getPluginById('gibts-nicht')).toBeUndefined();
    expect(getPluginById(null)).toBeUndefined();
  });

  // "pytorch" ist FrameTrains eigener Sammelwert fuer jede .pt-Datei ohne
  // config.json. Als model_type behandelt, blockierte er in jedem Plugin die
  // Namensheuristik — importierte Modelle galten reihenweise als unbekannt.
  it.each([
    ['resnet18-transfer', 'hf-image-classification'],
    ['mein-bert-v2', 'hf-encoder'],
    ['whisper-small-de', 'audio-classification'],
    ['flan-t5-spellcheck', 'seq2seq'],
    ['yolo11n-custom', 'yolo'],
  ])('erkennt %s trotz model_type "pytorch"', (name, pluginId) => {
    const r = detectPluginForModel({ id: 'x', name, source_path: IMPORT_DIR, model_type: 'pytorch' });
    expect(r.supported && r.plugin.id).toBe(pluginId);
  });

  it('laesst einen echten model_type weiter entscheiden', () => {
    // Nicht jede Architektur ist trainierbar – das darf der Name nicht aushebeln.
    const r = detectPluginForModel({
      id: 'x', name: 'mein-bert-projekt', source_path: IMPORT_DIR, model_type: 'gpt2',
    });
    expect(r.supported).toBe(false);
  });

  it('nennt den erkannten Typ in der Begruendung', () => {
    const r = detectPluginForModel({
      id: 'x', name: 'mein-experiment', source_path: IMPORT_DIR, model_type: 'pytorch',
    });
    expect(r.supported === false && r.reason).toContain('pytorch');
  });
});
