// Aus dem YOLO-Test: Das Trainingsformular zeigte fuer jedes Modell dieselben
// Felder. Bei YOLO waren "Max Seq Length", Warmup und LoRA ohne Wirkung,
// waehrend imgsz/augment/patience gar nicht einstellbar waren — sie kamen
// ausschliesslich aus defaultPluginConfig.

import { describe, it, expect } from 'vitest';
import yoloPlugin from '../yolo';
import seq2seqPlugin from '../seq2seq';
import { PLUGINS } from '../registry';

describe('hiddenTrainingFields', () => {
  it('YOLO blendet die Felder aus, die Ultralytics nicht kennt', () => {
    const hidden = yoloPlugin.hiddenTrainingFields ?? [];
    for (const key of ['max_seq_length', 'warmup_ratio', 'lora', 'gradient_checkpointing']) {
      expect(hidden).toContain(key);
    }
  });

  it('Textmodelle blenden nichts aus — dort wirken alle Felder', () => {
    expect(seq2seqPlugin.hiddenTrainingFields ?? []).toHaveLength(0);
  });

  it('YOLO bringt die eigenen Parameter als Vorgabe mit', () => {
    const cfg = yoloPlugin.defaultPluginConfig ?? {};
    expect(cfg).toHaveProperty('imgsz');
    expect(cfg).toHaveProperty('augment');
    expect(cfg).toHaveProperty('patience');
  });

  it('kein Plugin blendet Epochen oder Batch-Groesse aus', () => {
    for (const plugin of PLUGINS) {
      const hidden = plugin.hiddenTrainingFields ?? [];
      expect(hidden).not.toContain('epochs');
      expect(hidden).not.toContain('batch_size');
      expect(hidden).not.toContain('learning_rate');
    }
  });
});
