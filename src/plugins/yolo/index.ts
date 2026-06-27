// YOLO Plugin – index.ts

import type { ModelPlugin } from '../types';
import { detectYOLO } from './detect';
import YOLOTrainPlugin from './TrainPlugin';
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
  TrainComponent: YOLOTrainPlugin,
  TestComponent: YOLOTestPlugin,
  // Phase 7: Dataset-Kompatibilität
  supportedDatasetTypes: ['yolo_bbox', 'pre_split', 'pascal_voc'],
  preferredDatasetType: 'yolo_bbox',
};

export default yoloPlugin;
