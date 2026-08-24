import { ComponentType } from 'react';
import type { DatasetType, PairingStatus } from './datasetCompatHelpers';

export interface DatasetInfo {
  id:              string;
  name:            string;
  model_id:        string;
  status:          'unused' | 'split';
  file_count:      number;
  size_bytes:      number;
  extensions?:     string[];
  storage_path?:   string;
  // v2: Typ-System
  dataset_type?:   DatasetType;
  pairing_status?: PairingStatus | null;
  warnings?:       string[];
  // v2: Schema-Hint (z.B. dataset.yaml Inhalt bei YOLO)
  schema_hint?:    Record<string, unknown> | null;
}

export interface TestPluginProps {
  modelPath:   string;
  versionId:   string;
  modelId:     string;
  modelName:   string;
  versionName: string;
  datasets:    DatasetInfo[];
}

export interface ModelConfig {
  model_type?:    string;
  architectures?: string[];
  [key: string]: unknown;
}

export interface ModelPlugin {
  /** Eindeutige ID des Plugins, z.B. "xlm-roberta" */
  id: string;
  /** Anzeigename */
  name: string;
  /** Kurzbeschreibung, wofür das Plugin gedacht ist */
  description: string;
  /** Backend task_type für Python-Plugin-Routing (z.B. "seq_classification") */
  taskType: string;
  /** Optional: Default plugin_config, wird 1:1 an Python weitergereicht */
  defaultPluginConfig?: Record<string, unknown>;
  /**
   * Sinnvolle Startwerte der Trainings-Konfiguration für diesen Modelltyp.
   *
   * Ein einziger Standardwert passt nicht mehr: 2e-5 ist für Transformer
   * richtig, für Bild-Transfer-Learning mit eingefrorenem Backbone aber viel
   * zu klein (im direkten Vergleich 50 % statt 100 % Accuracy) und für T5
   * ebenfalls zu niedrig.
   */
  defaultTrainingConfig?: Record<string, number | string | boolean>;
  /**
   * Generische Trainings-Felder, die dieses Plugin gar nicht auswertet.
   *
   * Das Formular zeigte fuer jedes Modell dieselben Felder. Bei YOLO waren
   * "Max Seq Length", "Warmup Ratio" und LoRA ohne jede Wirkung, waehrend die
   * Parameter, auf die es ankommt (imgsz, augment, patience), gar nicht
   * einstellbar waren.
   */
  hiddenTrainingFields?: string[];
  /**
   * Erkennt ob ein Modell von diesem Plugin unterstützt wird.
   * @param modelPathOrId  Lokaler Pfad oder HuggingFace Model-ID
   * @param configJson     Optional: bereits geladenes config.json des Modells
   */
  detect: (modelPathOrId: string, configJson?: ModelConfig) => boolean;
  /** Test-Oberfläche */
  TestComponent: ComponentType<TestPluginProps>;
  /**
   * Phase 7: Welche DatasetTypes kann dieses Modell trainieren?
   * Wird im Import-Modal zur Kompatibilitätsprüfung genutzt.
   */
  supportedDatasetTypes?: DatasetType[];
  /**
   * Phase 7: Bevorzugter Import-Typ – wird im Import-Modal als "Empfohlen" angezeigt.
   */
  preferredDatasetType?: DatasetType;
}
