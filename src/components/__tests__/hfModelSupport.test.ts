// Regression aus dem YOLO-Test vom 20.08.2026:
// Beim Import von Ultralytics/YOLO11 stand "Für Training nicht geeignet ...
// dafür ein YOLO-Modell nutzen", obwohl das Badge daneben "YOLO Object
// Detection" meldete. Der pipeline_tag hat die Plugin-Erkennung ueberstimmt.

import { describe, it, expect } from 'vitest';
import { checkHfModelSupport } from '../ModelManager';

describe('checkHfModelSupport', () => {
  it('erlaubt YOLO-Modelle trotz pipeline_tag object-detection', () => {
    expect(checkHfModelSupport('Ultralytics/YOLO11', 'object-detection'))
      .toEqual({ supported: true });
    expect(checkHfModelSupport('keremberke/yolov8n-table-extraction', 'object-detection'))
      .toEqual({ supported: true });
  });

  // Bis 1.8.3 abgelehnt — seit dem Plugin hf-object-detection trainierbar.
  it('laesst HF-Detektoren durch (DETR, RT-DETR, YOLOS)', () => {
    for (const id of ['facebook/detr-resnet-50', 'PekingU/rtdetr_r18vd', 'hustvl/yolos-tiny']) {
      expect(checkHfModelSupport(id, 'object-detection').supported, id).toBe(true);
    }
  });

  it('warnt weiter bei Detektoren ohne Plugin', () => {
    const r = checkHfModelSupport('google/owlv2-base-patch16-ensemble', 'object-detection');
    expect(r.supported).toBe(false);
    expect(r.reason).toMatch(/DETR/);
  });

  // Bis 1.3.3 abgelehnt — seit dem causal-lm-Plugin trainierbar.
  it('laesst Decoder-LLMs durch, auch mit unbekanntem Namen', () => {
    expect(checkHfModelSupport('meta-llama/Llama-3-8B', 'text-generation').supported).toBe(true);
    expect(checkHfModelSupport('irgendwer/hauseigenes-sprachmodell', 'text-generation').supported).toBe(true);
  });

  it('warnt weiter bei Aufgaben ohne Plugin', () => {
    const r = checkHfModelSupport('irgendwer/tts-modell', 'text-to-speech');
    expect(r.supported).toBe(false);
    expect(r.reason).toContain('Sprachsynthese');
  });

  it('laesst unterstuetzte Encoder-Modelle durch', () => {
    expect(checkHfModelSupport('bert-base-uncased', 'fill-mask'))
      .toEqual({ supported: true });
  });
});
