// FrameTrain Business Stufe 2: Team-Katalog (geteilte Modellversionen und
// Datasets). Reine Logik ohne Tauri, damit testbar.

export type EntryKind = 'version' | 'dataset';

export interface CatalogEntry {
  id: string;
  kind: EntryKind;
  localRef: string;
  authorKey: string;
  authorName: string;
  title: string;
  payload: Record<string, unknown> | null;
  storagePath: string | null;
  sizeBytes: number | null;
  auto: boolean;
  deleted: boolean;
  mine: boolean;
  canEdit: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface CatalogActivity {
  id: string;
  actorName: string;
  action: 'shared' | 'updated' | 'stored' | 'removed' | 'fetched';
  entryId: string | null;
  title: string;
  createdAt: string;
}

export interface CatalogResponse {
  serverTime: string;
  full: boolean;
  storageHint: string | null;
  you: { key: string; name: string; role: 'admin' | 'member' | 'viewer' };
  entries: CatalogEntry[];
  activity: CatalogActivity[];
}

export interface StorageStatus {
  path: string | null;
  available: boolean;
  foreign_team: string | null;
}

/** Lokale Modellversion, wie list_model_versions sie liefert. */
export interface LocalVersion {
  id: string;
  model_id: string;
  version_name: string;
  version_number: number;
  path: string;
  size_bytes: number;
  created_at: string;
  is_root: boolean;
  training_metrics: {
    final_train_loss: number;
    final_val_loss: number | null;
    total_epochs: number;
    total_steps: number;
    best_epoch: number | null;
    training_duration_seconds: number | null;
  } | null;
}

export interface LocalModel {
  id: string;
  name: string;
  source: string;
  model_type: string | null;
}

export interface LocalDataset {
  id: string;
  name: string;
  model_id: string;
  storage_path: string;
  size_bytes: number;
  file_count: number;
  dataset_type?: string;
  split_info?: { train_count: number; val_count: number; test_count: number } | null;
  created_at: string;
}

/**
 * Abgleich zusammenfuehren: volle Antwort ersetzt, Teilantwort aktualisiert
 * nach ID und wirft entfernte Eintraege raus. Neueste zuerst.
 */
export function mergeEntries(prev: CatalogEntry[], incoming: CatalogEntry[], full: boolean): CatalogEntry[] {
  const map = new Map<string, CatalogEntry>(full ? [] : prev.map(e => [e.id, e]));
  for (const e of incoming) {
    if (e.deleted) map.delete(e.id);
    else map.set(e.id, e);
  }
  return [...map.values()].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
}

/** Ablage im Team-Speicher: models/<id> bzw. datasets/<id> des Autors. */
export function storageRelPath(kind: EntryKind, authorKey: string, localRef: string): string {
  const safe = (s: string) => s.replace(/[^A-Za-z0-9_.-]/g, '_');
  return `${kind === 'version' ? 'models' : 'datasets'}/${safe(authorKey)}-${safe(localRef)}`;
}

export function versionTitle(model: LocalModel, v: LocalVersion): string {
  const label = v.version_name?.trim() || `v${v.version_number}`;
  return `${model.name} · ${label}`.slice(0, 120);
}

export function versionPayload(model: LocalModel, v: LocalVersion): Record<string, unknown> {
  return {
    modelId: model.id,
    modelName: model.name,
    modelType: model.model_type,
    source: model.source,
    versionId: v.id,
    versionName: v.version_name,
    versionNumber: v.version_number,
    createdAt: v.created_at,
    metrics: v.training_metrics,
  };
}

export function datasetPayload(d: LocalDataset): Record<string, unknown> {
  return {
    datasetId: d.id,
    name: d.name,
    type: d.dataset_type ?? null,
    fileCount: d.file_count,
    split: d.split_info ? { train: d.split_info.train_count, val: d.split_info.val_count, test: d.split_info.test_count } : null,
    createdAt: d.created_at,
  };
}

/** Fertig trainierte Versionen (keine Basis), die noch nicht automatisch geteilt wurden. */
export function pendingAutoShares(
  items: { model: LocalModel; version: LocalVersion }[],
  alreadyShared: Set<string>,
): { model: LocalModel; version: LocalVersion }[] {
  return items.filter(({ version }) => !version.is_root && !!version.training_metrics && !alreadyShared.has(version.id));
}

export type EntryState =
  | 'stored'        // Dateien im Team-Speicher, holbar
  | 'author_only'   // nur Kennzahlen, Dateien beim Autor
  | 'mine_local';   // eigener Eintrag, Dateien nur auf diesem Rechner

export function entryState(e: CatalogEntry): EntryState {
  if (e.storagePath) return 'stored';
  return e.mine ? 'mine_local' : 'author_only';
}

/** Kurzbeschreibung eines Eintrags aus seinen Kennzahlen. */
export function entrySummary(e: CatalogEntry): { task: string | null; loss: number | null; epochs: number | null; files: number | null } {
  const p = e.payload ?? {};
  const metrics = (p.metrics ?? null) as LocalVersion['training_metrics'];
  return {
    task: typeof p.modelType === 'string' ? p.modelType : typeof p.type === 'string' ? p.type : null,
    loss: metrics ? (metrics.final_val_loss ?? metrics.final_train_loss) : null,
    epochs: metrics ? metrics.total_epochs : null,
    files: typeof p.fileCount === 'number' ? p.fileCount : null,
  };
}

export function formatBytes(n: number | null): string {
  if (!n || n <= 0) return '';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v >= 10 || i === 0 ? Math.round(v) : v.toFixed(1)} ${units[i]}`;
}

// ── Pro Geraet gemerkt (Komfort, kein verlaesslicher Zustand) ──────────────

const autoKey = (teamId: string) => `ft_team_autoshare_${teamId}`;
const sharedKey = (teamId: string) => `ft_team_autoshared_${teamId}`;

export function autoShareEnabled(teamId: string): boolean {
  try { return localStorage.getItem(autoKey(teamId)) !== '0'; } catch { return true; }
}

export function setAutoShareEnabled(teamId: string, on: boolean) {
  try { localStorage.setItem(autoKey(teamId), on ? '1' : '0'); } catch { /* egal */ }
}

export function loadAutoShared(teamId: string): Set<string> {
  try { return new Set(JSON.parse(localStorage.getItem(sharedKey(teamId)) ?? '[]')); } catch { return new Set(); }
}

export function saveAutoShared(teamId: string, ids: Set<string>) {
  try { localStorage.setItem(sharedKey(teamId), JSON.stringify([...ids].slice(-2000))); } catch { /* egal */ }
}
