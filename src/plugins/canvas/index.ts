/**
 * Canvas Neural Net Plugin
 * Erkennt Modelle die im Synapse Builder erstellt wurden (model_type === "canvas" oder id startet mit "canvas_").
 * Training läuft über das canvas plugin.py im Python-Backend.
 */

import React from 'react';
import type { ModelPlugin, ModelConfig, TestPluginProps } from '../types';

// Minimale Test-Stub-Komponente — das Training laeuft ueber das TrainingPanel.
const CanvasTestStub: React.FC<TestPluginProps> = () =>
  React.createElement('div', { style: { color: '#94a3b8', fontSize: 12, padding: 16 } },
    'Test-Interface für Canvas-Modelle — bald verfügbar.');

const canvasPlugin: ModelPlugin = {
  id: 'canvas',
  name: 'Canvas Neural Net',
  description: 'Im Synapse Builder erstelltes neuronales Netz — trainierbar mit beliebigen Datensätzen.',
  taskType: 'canvas',
  defaultPluginConfig: {},

  detect(modelPathOrId: string, configJson?: ModelConfig): boolean {
    // Erkennung via model_type im config.json (gesetzt von detect_model_type in Rust)
    if (configJson?.model_type === 'canvas') return true;
    // Erkennung via Modell-ID-Prefix (Fallback)
    const id = modelPathOrId.toLowerCase();
    if (id.startsWith('canvas_') || id.includes('/canvas_')) return true;
    return false;
  },
  TestComponent:  CanvasTestStub,
  // Phase 7: Canvas-Modelle sind flexibel, akzeptieren alle Typen.
  // Der User bestimmt selbst was er im Synapse Builder verdrahtet.
  // Kein preferredDatasetType gesetzt – zeigt stattdessen "flexibel" im UI.
  supportedDatasetTypes: [
    'flat_file', 'folder_class', 'yolo_bbox', 'pascal_voc', 'coco_json',
    'audio_transcript', 'common_voice', 'pre_split', 'multi_shard',
  ],
};

export default canvasPlugin;
