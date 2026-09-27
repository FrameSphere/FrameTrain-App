// LLM-Plugin (causal_lm): Erkennung und Dataset-Pruefung.
// Die Architekturliste muss mit dem Manifest der Train-Engine uebereinstimmen,
// sonst gilt ein Modell im Frontend als trainierbar und scheitert erst im Python.

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { detectCausalLM, CAUSAL_LM_MODEL_TYPES } from '../causal-lm/detect';
import causalLm from '../causal-lm';
import { checkDatasetCompat } from '../datasetCompat';
import type { DatasetAnalysis } from '../datasetCompatHelpers';

const manifest = JSON.parse(readFileSync(
  resolve(__dirname, '../../../src-tauri/python/train_engine/plugins/causal_lm/manifest.json'), 'utf-8'));

const analysis = (detected_type: DatasetAnalysis['detected_type'], extensions: string[]): DatasetAnalysis => ({
  detected_type, confidence: 80, pairing_status: null, warnings: [], file_count: 3, dir_count: 0, extensions, schema_hint: null,
});

describe('causal-lm: Erkennung', () => {
  it('Architekturen stimmen mit dem Manifest ueberein', () => {
    expect([...CAUSAL_LM_MODEL_TYPES].sort()).toEqual([...manifest.supported_architectures].sort());
    expect(causalLm.taskType).toBe(manifest.task_type);
  });

  it('config.json entscheidet', () => {
    expect(detectCausalLM('egal', { model_type: 'qwen2' })).toBe(true);
    expect(detectCausalLM('llama-irgendwas', { model_type: 'bert' })).toBe(false);
  });

  it('Name ohne config.json', () => {
    expect(detectCausalLM('meta-llama/Llama-3.2-3B-Instruct')).toBe(true);
    expect(detectCausalLM('/Users/x/models/SmolLM2-135M-Instruct/versions/ver_abc123')).toBe(true);
    expect(detectCausalLM('Qwen/Qwen2.5-VL-3B-Instruct')).toBe(false);
    expect(detectCausalLM('BAAI/bge-small-en')).toBe(false);
    expect(detectCausalLM('bert-base-uncased')).toBe(false);
  });

  it('LoRA ist die Voreinstellung', () => {
    expect(causalLm.defaultTrainingConfig?.use_lora).toBe(true);
    expect(causalLm.defaultPluginConfig?.backend).toBe('auto');
  });
});

describe('causal-lm: Dataset-Pruefung', () => {
  it('Chat-JSONL und Fliesstext sind geeignet', () => {
    expect(checkDatasetCompat('causal-lm', [], analysis('flat_file', ['.jsonl']), causalLm).overallLevel).not.toBe('bad');
    expect(checkDatasetCompat('causal-lm', ['.txt'], analysis('unknown', ['.txt']), causalLm).overallLevel).not.toBe('bad');
  });

  it('Ein Bilderordner ist nicht geeignet', () => {
    expect(checkDatasetCompat('causal-lm', [], analysis('folder_class', ['.jpg']), causalLm).overallLevel).toBe('bad');
  });
});
