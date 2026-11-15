import type { ModelPlugin } from '../types';
import { detectHFObjectDetection } from './detect';
import HFObjectDetectionTestPlugin from './TestPlugin';

const hfObjectDetectionPlugin: ModelPlugin = {
  id: 'hf-object-detection',
  name: 'Object Detection (HuggingFace)',
  description: 'Trainiert HuggingFace-Detektoren (DETR, Conditional/Deformable DETR, RT-DETR, YOLOS) auf YOLO- oder COCO-Boxen.',
  taskType: 'hf_object_detection',
  // DETR-Rezept: AdamW 1e-4 fuer Kopf/Decoder, Backbone ein Zehntel davon.
  defaultTrainingConfig: { learning_rate: 1e-4, batch_size: 4, epochs: 20, weight_decay: 1e-4, max_grad_norm: 0.1 },
  // imgsz: kuerzeste Kante (DETR, YOLOS) bzw. Seitenlaenge (RT-DETR); conf: Schwelle fuer
  // Precision/Recall und die Boxen je Bild (mAP nutzt alle Vorhersagen).
  defaultPluginConfig: { imgsz: 512, augment: true, conf: 0.25 },
  detect: detectHFObjectDetection,
  TestComponent: HFObjectDetectionTestPlugin,
  hiddenTrainingFields: [
    'max_seq_length', 'lora', 'gradient_checkpointing', 'group_by_length', 'label_smoothing', 'dropout',
  ],
  supportedDatasetTypes: ['yolo_bbox', 'coco_json', 'pre_split'],
  preferredDatasetType: 'yolo_bbox',
};

export default hfObjectDetectionPlugin;
