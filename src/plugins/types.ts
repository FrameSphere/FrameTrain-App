import { ComponentType } from 'react';
import type { DatasetType, PairingStatus } from './datasetCompatHelpers';

export interface TrainPluginProps {
  modelPath: string;
  onNavigateToAnalysis: (versionId: string) => void;
}

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
   * Erkennt ob ein Modell von diesem Plugin unterstützt wird.
   * @param modelPathOrId  Lokaler Pfad oder HuggingFace Model-ID
   * @param configJson     Optional: bereits geladenes config.json des Modells
   */
  detect: (modelPathOrId: string, configJson?: ModelConfig) => boolean;
  /** Training-Oberfläche */
  TrainComponent: ComponentType<TrainPluginProps>;
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
