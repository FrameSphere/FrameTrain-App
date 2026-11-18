// Tauri-Aufrufe fuer den Team-Katalog und den Team-Speicher.
import { invoke } from '@tauri-apps/api/core';
import {
  autoShareEnabled, loadAutoShared, pendingAutoShares, saveAutoShared, storageRelPath, versionPayload, versionTitle,
  datasetPayload,
  type CatalogEntry, type CatalogResponse, type EntryKind, type LocalDataset, type LocalModel, type LocalVersion,
  type StorageStatus,
} from './catalog';
import type { TeamInfo } from './team';

export function fetchCatalog(apiKey: string, since?: string | null) {
  return invoke<CatalogResponse>('team_request', { apiKey, path: 'catalog', body: since ? { since } : {} });
}

export async function shareEntry(apiKey: string, entry: {
  kind: EntryKind; localRef: string; title: string; payload: Record<string, unknown>;
  storagePath?: string | null; sizeBytes?: number | null; auto?: boolean;
}): Promise<CatalogEntry | null> {
  const res = await invoke<{ entry: CatalogEntry | null }>('team_request', { apiKey, path: 'catalog/share', body: entry });
  return res.entry;
}

export function removeEntry(apiKey: string, id: string) {
  return invoke('team_request', { apiKey, path: 'catalog/remove', body: { id } });
}

export function reportFetched(apiKey: string, id: string) {
  return invoke('team_request', { apiKey, path: 'catalog/fetched', body: { id } }).catch(() => null);
}

export function saveStorageHint(apiKey: string, storageHint: string | null) {
  return invoke('team_request', { apiKey, path: 'settings', body: { storageHint } });
}

export function storageStatus(teamId: string) {
  return invoke<StorageStatus>('team_storage_get', { teamId });
}

export function setStoragePath(team: TeamInfo, path: string | null) {
  return invoke<StorageStatus>('team_storage_set', { teamId: team.id, teamName: team.name, path });
}

/** Alle lokalen Modelle mit ihren Versionen. */
export async function loadLocalVersions(): Promise<{ model: LocalModel; version: LocalVersion }[]> {
  const models = await invoke<LocalModel[]>('list_models').catch(() => [] as LocalModel[]);
  const out: { model: LocalModel; version: LocalVersion }[] = [];
  for (const model of models) {
    const versions = await invoke<LocalVersion[]>('list_model_versions', { modelId: model.id }).catch(() => [] as LocalVersion[]);
    for (const version of versions) out.push({ model, version });
  }
  return out;
}

export function loadLocalDatasets() {
  return invoke<LocalDataset[]>('list_all_datasets').catch(() => [] as LocalDataset[]);
}

/**
 * Fertige Trainingslaeufe automatisch im Team zeigen (nur Kennzahlen).
 * Laeuft beim Lebenszeichen und beim Oeffnen der Team-Seite.
 */
export async function runAutoShare(apiKey: string, team: TeamInfo): Promise<number> {
  if (team.role === 'viewer' || !autoShareEnabled(team.id)) return 0;
  const done = loadAutoShared(team.id);
  const pending = pendingAutoShares(await loadLocalVersions(), done);
  let shared = 0;
  for (const { model, version } of pending) {
    try {
      await shareEntry(apiKey, {
        kind: 'version', localRef: version.id, title: versionTitle(model, version),
        payload: versionPayload(model, version), sizeBytes: version.size_bytes, auto: true,
      });
      done.add(version.id);
      shared++;
    } catch {
      break; // offline o. Ae.: beim naechsten Mal weiter
    }
  }
  saveAutoShared(team.id, done);
  return shared;
}

/** Lokale Quelle eines eigenen Eintrags (fuer "In Team-Speicher legen"). */
export async function localSourcePath(entry: CatalogEntry): Promise<string> {
  if (entry.kind === 'version') return invoke<string>('get_version_path_for_ui', { versionId: entry.localRef });
  return invoke<string>('get_dataset_path', { datasetId: entry.localRef });
}

/** Dateien in den Team-Speicher legen und den Eintrag damit aktualisieren. */
export async function storeEntryFiles(apiKey: string, team: TeamInfo, entry: {
  kind: EntryKind; localRef: string; title: string; payload: Record<string, unknown>; sourcePath: string; authorKey: string;
}): Promise<CatalogEntry | null> {
  const rel = storageRelPath(entry.kind, entry.authorKey, entry.localRef);
  const size = await invoke<number>('team_storage_put', {
    teamId: team.id, sourcePath: entry.sourcePath, relPath: rel,
    manifest: { kind: entry.kind, title: entry.title, payload: entry.payload, app: 'FrameTrain' },
  });
  return shareEntry(apiKey, { kind: entry.kind, localRef: entry.localRef, title: entry.title, payload: entry.payload, storagePath: rel, sizeBytes: size });
}

/** Eintrag aus dem Team-Speicher in die eigene App uebernehmen. */
export async function fetchEntry(apiKey: string, team: TeamInfo, entry: CatalogEntry, targetModelId?: string) {
  if (!entry.storagePath) throw new Error('no_files');
  const source = await invoke<string>('team_storage_resolve', { teamId: team.id, relPath: entry.storagePath });
  if (entry.kind === 'version') {
    await invoke('import_local_model', { sourcePath: source, modelName: entry.title });
  } else {
    if (!targetModelId) throw new Error('no_target');
    await invoke('import_local_dataset', { sourcePath: source, datasetName: entry.title, modelId: targetModelId });
  }
  void reportFetched(apiKey, entry.id);
}

export { versionPayload, versionTitle, datasetPayload };
