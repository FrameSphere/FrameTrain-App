// FrameTrain Business Stufe 2: Team-Katalog — geteilte Modellversionen und
// Datasets, Team-Speicher der Firma, Holen und Teilen.

import { useCallback, useEffect, useRef, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import { invoke } from '@tauri-apps/api/core';
import { open as openDialog } from '@tauri-apps/plugin-dialog';
import { Database, Download, FolderOpen, Layers, Loader2, Server, Share2, Trash2, Upload } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useTheme } from '../../contexts/ThemeContext';
import {
  autoShareEnabled, entryState, entrySummary, formatBytes, mergeEntries, setAutoShareEnabled, storageRelPath,
  type CatalogActivity, type CatalogEntry, type CatalogResponse, type LocalModel, type StorageStatus,
} from '../../team/catalog';
import type { TeamInfo } from '../../team/team';
import {
  fetchCatalog, fetchEntry, localSourcePath, removeEntry, runAutoShare, saveStorageHint, setStoragePath,
  storageStatus, storeEntryFiles,
} from '../../team/teamApi';
import ShareDialog from './ShareDialog';

const POLL_MS = 60 * 1000;

interface Props {
  apiKey: string;
  team: TeamInfo;
  /** Aktivitaeten an die Team-Seite melden (eigener Reiter) */
  onActivity: (a: CatalogActivity[]) => void;
}

export default function TeamCatalog({ apiKey, team, onActivity }: Props) {
  const { t, language } = useLanguage();
  const { currentTheme } = useTheme();
  const [entries, setEntries] = useState<CatalogEntry[]>([]);
  const [you, setYou] = useState<CatalogResponse['you'] | null>(null);
  const [hint, setHint] = useState<string | null>(null);
  const [storage, setStorage] = useState<StorageStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState<Record<string, string>>({});
  const [progress, setProgress] = useState<Record<string, number>>({});
  const [shareOpen, setShareOpen] = useState(false);
  const [autoShare, setAutoShare] = useState(() => autoShareEnabled(team.id));
  const [models, setModels] = useState<LocalModel[]>([]);
  const [target, setTarget] = useState<Record<string, string>>({});
  const [editHint, setEditHint] = useState<string | null>(null);
  const since = useRef<string | null>(null);

  const isViewer = (you?.role ?? team.role) === 'viewer';
  const isAdmin = (you?.role ?? team.role) === 'admin';
  const storageReady = !!storage?.available && !storage.foreign_team;

  const sync = useCallback(async (full = false) => {
    try {
      const res = await fetchCatalog(apiKey, full ? null : since.current);
      since.current = res.serverTime;
      setEntries(prev => mergeEntries(prev, res.entries, res.full));
      setYou(res.you);
      setHint(res.storageHint);
      onActivity(res.activity);
      setError('');
    } catch (e) {
      setError(typeof e === 'string' ? e : t('team.catalog.loadError'));
    } finally {
      setLoading(false);
    }
  }, [apiKey, onActivity, t]);

  useEffect(() => {
    void storageStatus(team.id).then(setStorage).catch(() => setStorage(null));
    void invoke<LocalModel[]>('list_models').then(setModels).catch(() => setModels([]));
    // Erst eigene fertige Laeufe nachtragen, dann den Katalog holen
    void runAutoShare(apiKey, team).finally(() => { void sync(true); });
    const id = window.setInterval(() => { void sync(); }, POLL_MS);
    return () => window.clearInterval(id);
  }, [apiKey, team, sync]);

  // Fortschritt beim Kopieren in den Team-Speicher
  useEffect(() => {
    let un: (() => void) | undefined;
    let disposed = false;
    listen<{ rel_path: string; bytes_done: number; bytes_total: number }>('team-storage-progress', e => {
      const { rel_path, bytes_done, bytes_total } = e.payload;
      setProgress(p => ({ ...p, [rel_path]: bytes_total > 0 ? Math.round((bytes_done / bytes_total) * 100) : 0 }));
    }).then(fn => { if (disposed) fn(); else un = fn; }).catch(() => {});
    return () => { disposed = true; un?.(); };
  }, []);

  const withBusy = async (id: string, label: string, fn: () => Promise<void>) => {
    setBusy(b => ({ ...b, [id]: label }));
    setError('');
    try {
      await fn();
    } catch (e) {
      setError(typeof e === 'string' ? e : e instanceof Error ? e.message : t('team.catalog.genericError'));
    } finally {
      setBusy(b => { const n = { ...b }; delete n[id]; return n; });
    }
  };

  const chooseStorage = async () => {
    const picked = await openDialog({ directory: true, multiple: false, title: t('team.storage.pickTitle') });
    if (typeof picked !== 'string') return;
    await withBusy('storage', 'set', async () => { setStorage(await setStoragePath(team, picked)); });
  };

  const store = (e: CatalogEntry) => withBusy(e.id, 'store', async () => {
    const sourcePath = await localSourcePath(e);
    const updated = await storeEntryFiles(apiKey, team, {
      kind: e.kind, localRef: e.localRef, title: e.title, payload: e.payload ?? {}, sourcePath, authorKey: e.authorKey,
    });
    if (updated) setEntries(prev => mergeEntries(prev, [updated], false));
  });

  const fetchOne = (e: CatalogEntry) => withBusy(e.id, 'fetch', async () => {
    const modelId = target[e.id] ?? models[0]?.id;
    if (e.kind === 'dataset' && !modelId) throw new Error(t('team.catalog.needModel'));
    await fetchEntry(apiKey, team, e, modelId);
    if (e.kind === 'version') setModels(await invoke<LocalModel[]>('list_models').catch(() => models));
    setBusy(b => ({ ...b, [`${e.id}:done`]: 'done' }));
  });

  const remove = (e: CatalogEntry) => {
    const withFiles = !!e.storagePath && storageReady;
    if (!confirm(withFiles ? t('team.catalog.confirmRemoveFiles', { title: e.title }) : t('team.catalog.confirmRemove', { title: e.title }))) return;
    void withBusy(e.id, 'remove', async () => {
      await removeEntry(apiKey, e.id);
      if (withFiles && e.storagePath) await invoke('team_storage_remove', { teamId: team.id, relPath: e.storagePath }).catch(() => null);
      setEntries(prev => prev.filter(x => x.id !== e.id));
    });
  };

  const toggleAuto = (on: boolean) => {
    setAutoShare(on);
    setAutoShareEnabled(team.id, on);
    if (on) void runAutoShare(apiKey, team).then(n => { if (n) void sync(); });
  };

  return (
    <div className="space-y-4">
      {/* Team-Speicher dieses Rechners */}
      <div className="rounded-xl border border-white/10 px-4 py-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2 min-w-0 text-sm">
            <Server className="w-4 h-4 flex-shrink-0 text-gray-400" />
            <span className="text-gray-400">{t('team.storage.label')}</span>
            {storage?.path
              ? <span className="font-mono text-[12.5px] text-white truncate" title={storage.path}>{storage.path}</span>
              : <span className="text-gray-500">{t('team.storage.none')}</span>}
            {storage?.path && (
              <span className={`text-[12.5px] ${storage.foreign_team ? 'text-red-300' : storage.available ? 'text-emerald-300' : 'text-amber-300'}`}>
                {storage.foreign_team ? t('team.storage.foreign') : storage.available ? `● ${t('team.storage.connected')}` : `⚠ ${t('team.storage.offline')}`}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => { void chooseStorage(); }} disabled={!!busy.storage}
              className="px-3 py-1.5 rounded-lg text-sm text-white border border-white/10 hover:bg-white/5 flex items-center gap-1.5">
              <FolderOpen className="w-4 h-4" />{storage?.path ? t('team.storage.change') : t('team.storage.choose')}
            </button>
            {storage?.path && (
              <button onClick={() => { void withBusy('storage', 'unset', async () => { setStorage(await setStoragePath(team, null)); }); }}
                className="px-3 py-1.5 rounded-lg text-sm text-gray-400 hover:text-white">
                {t('team.storage.disconnect')}
              </button>
            )}
          </div>
        </div>
        <div className="mt-2 text-xs text-gray-500 flex flex-wrap items-center gap-x-2">
          {editHint !== null ? (
            <form className="flex items-center gap-2 w-full" onSubmit={ev => {
              ev.preventDefault();
              void withBusy('hint', 'save', async () => { await saveStorageHint(apiKey, editHint.trim() || null); setHint(editHint.trim() || null); setEditHint(null); });
            }}>
              <input value={editHint} onChange={ev => setEditHint(ev.target.value)} maxLength={300} autoFocus
                placeholder={t('team.storage.hintPlaceholder')}
                className="flex-1 px-2 py-1 rounded-md bg-white/5 border border-white/10 text-gray-200 text-xs" />
              <button type="submit" className="text-gray-300 hover:text-white">{t('team.storage.hintSave')}</button>
              <button type="button" onClick={() => setEditHint(null)} className="text-gray-500 hover:text-white">{t('team.storage.hintCancel')}</button>
            </form>
          ) : (
            <>
              <span>{hint ? t('team.storage.hint', { hint }) : t('team.storage.explain')}</span>
              {isAdmin && <button onClick={() => setEditHint(hint ?? '')} className="text-gray-400 hover:text-white underline-offset-2 hover:underline">{t('team.storage.hintEdit')}</button>}
            </>
          )}
        </div>
      </div>

      {/* Kopf des Katalogs */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <label className={`flex items-center gap-2 text-sm text-gray-400 ${isViewer ? 'opacity-50' : 'cursor-pointer'}`}>
          <input type="checkbox" checked={autoShare && !isViewer} disabled={isViewer} onChange={e => toggleAuto(e.target.checked)} />
          {t('team.catalog.autoShare')}
        </label>
        {!isViewer && (
          <button onClick={() => setShareOpen(true)}
            className={`px-4 py-2 rounded-xl text-sm font-medium text-white flex items-center gap-2 bg-gradient-to-r ${currentTheme.colors.gradient}`}>
            <Share2 className="w-4 h-4" />{t('team.catalog.share')}
          </button>
        )}
      </div>

      {error && <div role="alert" className="rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-300">{error}</div>}

      <section aria-label={t('team.catalog.title')} className="rounded-xl border border-white/10">
        {loading && entries.length === 0 && (
          <div className="px-4 py-6 text-sm text-gray-400 flex items-center gap-2"><Loader2 className="w-4 h-4 animate-spin" />{t('team.loading')}</div>
        )}
        {!loading && entries.length === 0 && (
          <div className="px-4 py-8 text-center">
            <p className="text-sm text-white">{t('team.catalog.emptyTitle')}</p>
            <p className="text-sm text-gray-500 mt-1">{t('team.catalog.emptyText')}</p>
          </div>
        )}
        {entries.map((e, i) => {
          const state = entryState(e);
          const sum = entrySummary(e);
          const b = busy[e.id];
          const pct = progress[storageRelPath(e.kind, e.authorKey, e.localRef)];
          const meta = [
            e.mine ? t('team.you') : e.authorName,
            e.kind === 'dataset' ? t('team.catalog.kindDataset') : sum.task,
            sum.loss !== null ? `Loss ${sum.loss.toFixed(3)}` : null,
            sum.epochs ? t('team.catalog.epochs', { n: sum.epochs }) : null,
            sum.files ? t('team.catalog.files', { n: sum.files.toLocaleString(language === 'de' ? 'de-DE' : 'en-US') }) : null,
            formatBytes(e.sizeBytes),
            e.auto ? t('team.catalog.auto') : null,
          ].filter(Boolean).join(' · ');
          return (
            <div key={e.id} className={`grid grid-cols-[20px_minmax(0,1fr)_auto] items-center gap-3 px-4 py-3 ${i ? 'border-t border-white/10' : ''}`}>
              {e.kind === 'dataset' ? <Database className="w-4 h-4 text-gray-400" /> : <Layers className="w-4 h-4 text-gray-400" />}
              <div className="min-w-0">
                <div className="text-sm text-white truncate">{e.title}</div>
                <div className="text-xs text-gray-500 truncate">{meta}</div>
              </div>
              <div className="flex items-center gap-2 justify-end">
                {state === 'stored' && (
                  <>
                    {e.kind === 'dataset' && models.length > 0 && (
                      <select value={target[e.id] ?? models[0]?.id} onChange={ev => setTarget(x => ({ ...x, [e.id]: ev.target.value }))}
                        aria-label={t('team.catalog.targetModel')}
                        className="max-w-[160px] px-2 py-1 rounded-lg bg-white/5 border border-white/10 text-xs text-gray-200">
                        {models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
                      </select>
                    )}
                    {busy[`${e.id}:done`] ? (
                      <span className="text-xs text-emerald-300">✓ {t('team.catalog.fetched')}</span>
                    ) : (
                      <button onClick={() => { void fetchOne(e); }} disabled={!!b || !storageReady}
                        title={storageReady ? undefined : t('team.catalog.needStorage')}
                        className="px-3 py-1.5 rounded-lg text-sm text-white border border-white/10 hover:bg-white/5 disabled:opacity-40 flex items-center gap-1.5">
                        {b === 'fetch' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}{t('team.catalog.fetch')}
                      </button>
                    )}
                  </>
                )}
                {state === 'author_only' && <span className="text-xs text-amber-300/90">{t('team.catalog.filesAt', { name: e.authorName })}</span>}
                {state === 'mine_local' && !isViewer && (
                  <button onClick={() => { void store(e); }} disabled={!!b || !storageReady}
                    title={storageReady ? undefined : t('team.catalog.needStorage')}
                    className="px-3 py-1.5 rounded-lg text-sm text-white border border-white/10 hover:bg-white/5 disabled:opacity-40 flex items-center gap-1.5">
                    {b === 'store' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                    {b === 'store' && pct !== undefined ? `${pct}%` : t('team.catalog.store')}
                  </button>
                )}
                {e.canEdit && (
                  <button onClick={() => remove(e)} disabled={!!b} className="ft-icon-btn ft-icon-btn-danger" aria-label={t('team.catalog.remove')} title={t('team.catalog.remove')}>
                    {b === 'remove' ? <Loader2 className="w-[16px] h-[16px] animate-spin" /> : <Trash2 className="w-[16px] h-[16px]" strokeWidth={1.8} />}
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </section>

      {shareOpen && (
        <ShareDialog
          apiKey={apiKey}
          team={team}
          authorKey={you?.key ?? (team.seatId ?? 'owner')}
          storageReady={storageReady}
          onClose={() => setShareOpen(false)}
          onShared={entry => { if (entry) setEntries(prev => mergeEntries(prev, [entry], false)); void sync(); }}
        />
      )}
    </div>
  );
}
