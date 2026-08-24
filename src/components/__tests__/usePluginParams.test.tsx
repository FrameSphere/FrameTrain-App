// Regression aus dem Test von 1.2.30: Die neuen Plugin-Parameterfelder liessen
// sich nicht aendern — jeder Tastendruck sprang auf die Vorgabe zurueck.
// Ursache: detectPlugin() liefert bei jedem Render ein neues Objekt; haengt das
// Zuruecksetzen daran, verwirft jeder Render die Eingabe.

import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { usePluginParams, pluginParamDefaultsFrom } from '../usePluginParams';

const YOLO_CONFIG = {
  task_type: 'detect', imgsz: 640, epochs: 100, batch: 16,
  lr0: 0.01, optimizer: 'SGD', augment: true, patience: 50,
};

describe('pluginParamDefaultsFrom', () => {
  it('laesst weg, was schon im allgemeinen Formular steht', () => {
    const d = pluginParamDefaultsFrom(YOLO_CONFIG);
    expect(d).not.toHaveProperty('task_type');
    expect(d).not.toHaveProperty('epochs');
    expect(d).not.toHaveProperty('batch');
    expect(Object.keys(d).sort()).toEqual(['augment', 'imgsz', 'lr0', 'optimizer', 'patience']);
  });

  it('kommt ohne Konfiguration zurecht', () => {
    expect(pluginParamDefaultsFrom(undefined)).toEqual({});
  });
});

describe('usePluginParams', () => {
  it('haelt die Eingabe fest, auch wenn das Config-Objekt neu erzeugt wird', () => {
    const { result, rerender } = renderHook(
      // Bei jedem Render ein frisches Objekt — genau wie detectPlugin es liefert.
      ({ id }: { id: string }) => usePluginParams(id, { ...YOLO_CONFIG }),
      { initialProps: { id: 'yolo' } },
    );

    act(() => result.current.setParams(p => ({ ...p, imgsz: 320 })));
    expect(result.current.params.imgsz).toBe(320);

    rerender({ id: 'yolo' });
    rerender({ id: 'yolo' });
    expect(result.current.params.imgsz, 'Eingabe darf kein Render ueberschreiben').toBe(320);
  });

  it('setzt beim Wechsel des Plugins auf dessen Vorgaben', () => {
    const { result, rerender } = renderHook(
      ({ id, cfg }: { id: string; cfg: Record<string, unknown> }) => usePluginParams(id, cfg),
      { initialProps: { id: 'yolo', cfg: YOLO_CONFIG } },
    );

    act(() => result.current.setParams(p => ({ ...p, imgsz: 320 })));
    rerender({ id: 'seq2seq', cfg: {} });
    expect(result.current.params).toEqual({});
  });
});
