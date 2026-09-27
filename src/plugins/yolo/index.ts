// YOLO Plugin – index.ts

import type { ModelPlugin } from '../types';
import { detectYOLO } from './detect';
import YOLOTestPlugin from './TestPlugin';

const yoloPlugin: ModelPlugin = {
  id: 'yolo',
  name: 'YOLO Object Detection',
  description: 'YOLOv5 / YOLOv8 / YOLOv9 / YOLO11 – Erkennung, Segmentierung, Keypoints, gedrehte Boxen und Klassifikation via Ultralytics',
  // Ein task_type fuer alle YOLO-Aufgaben: das Python-Plugin liest die
  // Aufgabe aus den Gewichten (yolo11n-seg.pt, -pose, -obb, -cls).
  taskType: 'detect',
  defaultPluginConfig: {
    task_type: 'detect',
    // auto | detect | segment | pose | obb | classify
    task: 'auto',
    imgsz: 640,
    epochs: 100,
    batch: 16,
    lr0: 0.01,
    lrf: 0.01,
    optimizer: 'SGD',
    augment: true,
    patience: 50,
    // Leer = neu starten. "auto" sucht last.pt in der gewaehlten Version, sonst
    // der Pfad zum Job-Ordner (oder last.pt) eines abgebrochenen Laufs.
    resume: '',
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
  // folder_class: Ordner pro Klasse fuer YOLO-cls (yolo11n-cls.pt).
  supportedDatasetTypes: ['yolo_bbox', 'pre_split', 'pascal_voc', 'folder_class'],
  preferredDatasetType: 'yolo_bbox',
};

export default yoloPlugin;
