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

  it('warnt weiter bei Objekterkennung ohne YOLO-Plugin', () => {
    const r = checkHfModelSupport('facebook/detr-resnet-50', 'object-detection');
    expect(r.supported).toBe(false);
    expect(r.reason).toBeTruthy();
  });

  it('warnt weiter bei Decoder-Modellen', () => {
    const r = checkHfModelSupport('meta-llama/Llama-3-8B', 'text-generation');
    expect(r.supported).toBe(false);
    expect(r.reason).toContain('Textgenerierung');
  });

  it('laesst unterstuetzte Encoder-Modelle durch', () => {
    expect(checkHfModelSupport('bert-base-uncased', 'fill-mask'))
      .toEqual({ supported: true });
  });
});
