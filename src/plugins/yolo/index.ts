// YOLO Plugin – index.ts

import type { ModelPlugin } from '../types';
import { detectYOLO } from './detect';
import YOLOTestPlugin from './TestPlugin';

const yoloPlugin: ModelPlugin = {
  id: 'yolo',
  name: 'YOLO Object Detection',
  description: 'YOLOv5 / YOLOv8 / YOLOv9 / YOLO11 – Bounding-Box-Erkennung via Ultralytics',
  taskType: 'detect',
  defaultPluginConfig: {
    task_type: 'detect',
    imgsz: 640,
    epochs: 100,
    batch: 16,
    lr0: 0.01,
    lrf: 0.01,
    optimizer: 'SGD',
    augment: true,
    patience: 50,
  },
  detect: detectYOLO,
  TestComponent: YOLOTestPlugin,
  // Ultralytics kennt weder Sequenzlaenge noch LoRA; Warmup und Scheduler
  // steuert es selbst ueber lr0/lrf.
  hiddenTrainingFields: [
    'max_seq_length', 'warmup_ratio', 'warmup_steps', 'lora', 'gradient_checkpointing',
    'dropout', 'label_smoothing', 'group_by_length', 'max_grad_norm', 'scheduler',
  ],
  // Phase 7: Dataset-Kompatibilität
  supportedDatasetTypes: ['yolo_bbox', 'pre_split', 'pascal_voc'],
  preferredDatasetType: 'yolo_bbox',
};

export default yoloPlugin;
