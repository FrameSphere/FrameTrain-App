// Plugin-System Typen für FrameTrain
// Jedes neue Modell-Training wird als Plugin implementiert

import { ComponentType } from 'react';

export interface TrainPluginProps {
  modelPath: string;
  onNavigateToAnalysis: (versionId: string) => void;
}

export interface DatasetInfo {
  id: string;
  name: string;
  model_id: string;
  status: 'unused' | 'split';
  file_count: number;
  size_bytes: number;
  extensions?: string[];
  storage_path?: string;
}

export interface TestPluginProps {
  modelPath: string;
  versionId: string;
  modelId: string;
  modelName: string;
  versionName: string;
  datasets: DatasetInfo[];
}

export interface ModelConfig {
  model_type?: string;
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
}
