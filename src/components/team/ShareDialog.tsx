// Teilen-Dialog: lokale Modellversionen und Datasets im Team-Katalog zeigen,
// wahlweise mit Dateien im Team-Speicher.

import { useEffect, useMemo, useState } from 'react';
import { Database, Layers, Loader2, X } from 'lucide-react';
import ModalPortal from '../ui/ModalPortal';
import { useLanguage } from '../../contexts/LanguageContext';
import { useTheme } from '../../contexts/ThemeContext';
import {
  datasetPayload, formatBytes, versionPayload, versionTitle,
  type CatalogEntry, type LocalDataset, type LocalModel, type LocalVersion,
} from '../../team/catalog';
import type { TeamInfo } from '../../team/team';
import { loadLocalDatasets, loadLocalVersions, shareEntry, storeEntryFiles } from '../../team/teamApi';

interface Item {
  key: string;
  kind: 'version' | 'dataset';
  localRef: string;
  title: string;
  detail: string;
  payload: Record<string, unknown>;
  sourcePath: string;
  sizeBytes: number;
}

interface Props {
  apiKey: string;
  team: TeamInfo;
  authorKey: string;
  storageReady: boolean;
  onClose: () => void;
  onShared: (entry: CatalogEntry | null) => void;
}

export default function ShareDialog({ apiKey, team, authorKey, storageReady, onClose, onShared }: Props) {
  const { t, language } = useLanguage();
  const { currentTheme } = useTheme();
  const [items, setItems] = useState<Item[] | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [withFiles, setWithFiles] = useState(storageReady);
  const [running, setRunning] = useState<string | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    void (async () => {
      const [versions, datasets] = await Promise.all([loadLocalVersions(), loadLocalDatasets()]);
      const modelNames = new Map<string, string>(versions.map(({ model }) => [model.id, model.name]));
      const vItems: Item[] = versions.map(({ model, version }: { model: LocalModel; version: LocalVersion }) => ({
        key: `v:${version.id}`,
        kind: 'version',
        localRef: version.id,
        title: versionTitle(model, version),
        detail: [
          version.is_root ? t('team.share.base') : null,
          version.training_metrics ? `Loss ${(version.training_metrics.final_val_loss ?? version.training_metrics.final_train_loss).toFixed(3)}` : null,
          formatBytes(version.size_bytes),
        ].filter(Boolean).join(' · '),
        payload: versionPayload(model, version),
        sourcePath: version.path,
        sizeBytes: version.size_bytes,
      }));
      const dItems: Item[] = datasets.map((d: LocalDataset) => ({
        key: `d:${d.id}`,
        kind: 'dataset',
        localRef: d.id,
        title: d.name.slice(0, 120),
        detail: [modelNames.get(d.model_id), t('team.catalog.files', { n: d.file_count.toLocaleString(language === 'de' ? 'de-DE' : 'en-US') }), formatBytes(d.size_bytes)].filter(Boolean).join(' · '),
        payload: datasetPayload(d),
        sourcePath: d.storage_path,
        sizeBytes: d.size_bytes,
      }));
      setItems([...vItems, ...dItems]);
    })();
  }, [t, language]);

  const totalBytes = useMemo(
    () => (items ?? []).filter(i => selected.has(i.key)).reduce((s, i) => s + i.sizeBytes, 0),
    [items, selected],
  );

  const toggle = (key: string) => setSelected(prev => {
    const next = new Set(prev);
    if (next.has(key)) next.delete(key); else next.add(key);
    return next;
  });

  const submit = async () => {
    if (!items) return;
    setError('');
    for (const item of items.filter(i => selected.has(i.key))) {
      setRunning(item.key);
      try {
        const entry = withFiles && storageReady
          ? await storeEntryFiles(apiKey, team, { ...item, authorKey })
          : await shareEntry(apiKey, { kind: item.kind, localRef: item.localRef, title: item.title, payload: item.payload, sizeBytes: item.sizeBytes });
        onShared(entry);
        setSelected(prev => { const n = new Set(prev); n.delete(item.key); return n; });
      } catch (e) {
        setError(`${item.title}: ${typeof e === 'string' ? e : e instanceof Error ? e.message : t('team.catalog.genericError')}`);
        setRunning(null);
        return;
      }
    }
    setRunning(null);
    onClose();
  };

  const section = (kind: 'version' | 'dataset') => (items ?? []).filter(i => i.kind === kind);

  return (
    <ModalPortal>
      <div className="fixed inset-0 z-[9000] flex items-center justify-center p-6" style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(6px)' }}
        onMouseDown={e => { if (e.target === e.currentTarget && !running) onClose(); }}>
        <div role="dialog" aria-modal="true" aria-labelledby="team-share-title"
          className="w-full max-w-xl max-h-[85vh] flex flex-col rounded-2xl border border-white/10 bg-slate-900 shadow-2xl overflow-hidden">
          <div className="flex items-center justify-between px-5 py-4 border-b border-white/10">
            <h2 id="team-share-title" className="text-[17px] font-semibold text-white">{t('team.share.title')}</h2>
            <button onClick={onClose} disabled={!!running} className="ft-icon-btn" aria-label={t('team.share.close')}><X className="w-[17px] h-[17px]" /></button>
          </div>

          <div className="flex-1 min-h-0 overflow-y-auto px-5 py-4 space-y-4">
            {!items && <div className="text-sm text-gray-400 flex items-center gap-2"><Loader2 className="w-4 h-4 animate-spin" />{t('team.loading')}</div>}
            {items && items.length === 0 && <p className="text-sm text-gray-400">{t('team.share.empty')}</p>}
            {(['version', 'dataset'] as const).map(kind => section(kind).length > 0 && (
              <div key={kind}>
                <div className="ft-nav-group !px-0">{kind === 'version' ? t('team.share.versions') : t('team.share.datasets')}</div>
                <div className="rounded-xl border border-white/10">
                  {section(kind).map((item, i) => (
                    <label key={item.key} className={`flex items-center gap-3 px-3 py-2.5 cursor-pointer hover:bg-white/[0.03] ${i ? 'border-t border-white/10' : ''}`}>
                      <input type="checkbox" checked={selected.has(item.key)} onChange={() => toggle(item.key)} disabled={!!running} />
                      {kind === 'version' ? <Layers className="w-4 h-4 text-gray-400 flex-shrink-0" /> : <Database className="w-4 h-4 text-gray-400 flex-shrink-0" />}
                      <span className="min-w-0 flex-1">
                        <span className="block text-sm text-white truncate">{item.title}</span>
                        <span className="block text-xs text-gray-500 truncate">{item.detail}</span>
                      </span>
                      {running === item.key && <Loader2 className="w-4 h-4 animate-spin text-gray-400" />}
                    </label>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div className="px-5 py-4 border-t border-white/10 space-y-3">
            <label className={`flex items-start gap-2 text-sm ${storageReady ? 'text-gray-300 cursor-pointer' : 'text-gray-500'}`}>
              <input type="checkbox" className="mt-0.5" checked={withFiles && storageReady} disabled={!storageReady || !!running} onChange={e => setWithFiles(e.target.checked)} />
              <span>
                {t('team.share.withFiles')}
                <span className="block text-xs text-gray-500">
                  {storageReady ? t('team.share.withFilesHint', { size: formatBytes(totalBytes) || '0 B' }) : t('team.share.noStorage')}
                </span>
              </span>
            </label>
            {error && <p role="alert" className="text-sm text-red-300">{error}</p>}
            <div className="flex justify-end gap-2">
              <button onClick={onClose} disabled={!!running} className="px-4 py-2 rounded-xl text-sm text-gray-300 hover:text-white">{t('team.share.cancel')}</button>
              <button onClick={() => { void submit(); }} disabled={!selected.size || !!running}
                className={`px-4 py-2 rounded-xl text-sm font-medium text-white flex items-center gap-2 bg-gradient-to-r ${currentTheme.colors.gradient} disabled:opacity-50`}>
                {running && <Loader2 className="w-4 h-4 animate-spin" />}
                {t('team.share.submit', { n: selected.size })}
              </button>
            </div>
          </div>
        </div>
      </div>
    </ModalPortal>
  );
}
