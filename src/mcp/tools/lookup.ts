// Gemeinsame Nachschlage-Helfer: Modelle, Datensaetze, Versionen per ID.
// Mehrere Commands verlangen Name UND ID (start_training, start_test) —
// die Werkzeuge nehmen nur IDs entgegen und loesen den Rest hier auf.

import { call, ToolFailure } from '../util';

export interface ModelInfo {
  id: string;
  name: string;
  source: string;
  source_path?: string | null;
  local_path: string;
  size_bytes: number;
  file_count: number;
  created_at: string;
  model_type?: string | null;
  plugin_override?: string | null;
}

export interface DatasetInfo {
  id: string;
  name: string;
  model_id: string;
  source: string;
  source_path?: string | null;
  storage_path: string;
  size_bytes: number;
  file_count: number;
  created_at: string;
  status: string;
  split_info?: unknown;
  training_count?: number;
  extensions?: string[];
  dataset_type?: unknown;
  warnings?: string[];
  dataset_yaml_path?: string | null;
}

export interface ModelVersion {
  id: string;
  model_id: string;
  version_name: string;
  version_number: number;
  path: string;
  size_bytes: number;
  file_count: number;
  created_at: string;
  is_root: boolean;
  parent_version_id?: string | null;
  training_metrics?: unknown;
}

export async function listModels(): Promise<ModelInfo[]> {
  return call<ModelInfo[]>('list_models');
}

export async function findModel(modelId: string): Promise<ModelInfo> {
  const m = (await listModels()).find((x) => x.id === modelId);
  if (!m) throw new ToolFailure(`Model "${modelId}" not found. Use models_list to see valid IDs.`);
  return m;
}

export async function listDatasets(): Promise<DatasetInfo[]> {
  return call<DatasetInfo[]>('list_all_datasets');
}

export async function findDataset(datasetId: string): Promise<DatasetInfo> {
  const d = (await listDatasets()).find((x) => x.id === datasetId);
  if (!d) throw new ToolFailure(`Dataset "${datasetId}" not found. Use datasets_list to see valid IDs.`);
  return d;
}

export async function listVersions(modelId: string): Promise<ModelVersion[]> {
  return call<ModelVersion[]>('list_model_versions', { modelId });
}

/** Sucht eine Version ueber alle Modelle. */
export async function findVersion(versionId: string): Promise<{ version: ModelVersion; modelName: string }> {
  const models = await call<{ id: string; name: string }[]>('list_models_with_versions');
  for (const m of models) {
    let versions: ModelVersion[] = [];
    try { versions = await listVersions(m.id); } catch { continue; }
    const v = versions.find((x) => x.id === versionId);
    if (v) return { version: v, modelName: m.name };
  }
  throw new ToolFailure(`Model version "${versionId}" not found. Use models_info to list the versions of a model.`);
}

/** Kurzform fuer Listen-Ergebnisse. */
export function modelBrief(m: ModelInfo) {
  return {
    id: m.id, name: m.name, source: m.source, model_type: m.model_type ?? null,
    plugin: m.plugin_override ?? null, size_bytes: m.size_bytes, created_at: m.created_at,
  };
}

export function datasetBrief(d: DatasetInfo) {
  return {
    id: d.id, name: d.name, model_id: d.model_id, source: d.source, status: d.status,
    dataset_type: d.dataset_type ?? null, file_count: d.file_count, size_bytes: d.size_bytes,
    split_info: d.split_info ?? null, warnings: d.warnings ?? [],
  };
}
