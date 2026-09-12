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
  // Nur was die Canvas-Loader wirklich lesen koennen: image_loader (Ordner pro
  // Klasse, auch train/val/test oder HF-Parquet mit Bildspalte), csv_loader und
  // parquet_loader. YOLO, Pascal VOC, COCO und Audio-Transkripte standen hier
  // vorher auch — geladen werden konnten sie nie.
  // Kein preferredDatasetType gesetzt – zeigt stattdessen "flexibel" im UI.
  supportedDatasetTypes: ['flat_file', 'folder_class', 'pre_split', 'multi_shard'],
};

export default canvasPlugin;
