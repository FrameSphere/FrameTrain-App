// Dialog "Modell hosten": Modell und Version aus Bibliothek bzw. Training
// waehlen. Die Groesse auf der Platte dient als RAM-Richtwert; wird es mit den
// schon laufenden Modellen knapp, warnt der Dialog vorher.

import { useEffect, useMemo, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Loader2, Search, X, Server, AlertTriangle, Check } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useTheme } from '../../contexts/ThemeContext';
import ModalPortal from '../ui/ModalPortal';
import { useEscapeKey } from '../../hooks/useEscapeKey';
import type { HostInfo } from './hostingModel';

interface ModelInfo { id: string; name: string; size_bytes?: number; model_type: string | null; source: string }
interface VersionItem { id: string; name: string; is_root: boolean; version_number: number }
interface ModelTree { id: string; name: string; versions: VersionItem[] }

export function hostName(model: { name: string }, v: VersionItem): string {
  if (v.is_root) return model.name;
  return v.name.toLowerCase().includes(model.name.toLowerCase()) ? v.name : `${model.name} · ${v.name}`;
}

export default function HostModelDialog({ hosts, onClose, onHosted }: {
  hosts: HostInfo[];
  onClose: () => void;
  onHosted: (versionId: string) => void;
}) {
  const { t } = useLanguage();
  const { currentTheme } = useTheme();
  useEscapeKey(onClose);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [trees, setTrees] = useState<ModelTree[]>([]);
  const [ramGb, setRamGb] = useState(0);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState('');
  const [selModel, setSelModel] = useState<string | null>(null);
  const [selVersion, setSelVersion] = useState<string | null>(null);
  const [makeDefault, setMakeDefault] = useState(hosts.length === 0);
  const [autoload, setAutoload] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      invoke<ModelInfo[]>('list_models').catch(() => []),
      invoke<ModelTree[]>('list_models_with_version_tree').catch(() => []),
      invoke<number>('get_system_ram_gb').catch(() => 0),
    ]).then(([m, tr, ram]) => { setModels(m); setTrees(tr); setRamGb(ram); }).finally(() => setLoading(false));
  }, []);

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return trees
      .filter(tr => tr.versions.length > 0)
      .filter(tr => !q || tr.name.toLowerCase().includes(q))
      .map(tr => {
        const info = models.find(m => m.id === tr.id);
        const versions = [...tr.versions].sort((a, b) => b.version_number - a.version_number);
        return { tree: tr, info, versions, trained: versions.some(v => !v.is_root) };
      })
      .sort((a, b) => Number(b.trained) - Number(a.trained) || a.tree.name.localeCompare(b.tree.name));
  }, [trees, models, query]);

  const current = rows.find(r => r.tree.id === selModel) ?? null;
  const version = current?.versions.find(v => v.id === selVersion) ?? current?.versions[0] ?? null;
  const sizeGb = current?.info?.size_bytes ? current.info.size_bytes / 1e9 : null;
  const runningGb = hosts.filter(h => h.status === 'ready' || h.status === 'loading').reduce((s, h) => s + (h.size_gb ?? 0), 0);
  // Grob: Gewichte ~1,3x im RAM (fp32 + Aktivierungen); Warnung ab 75 % des Arbeitsspeichers.
  const tight = !!(sizeGb && ramGb && (runningGb + sizeGb * 1.3) > ramGb * 0.75);
  const alreadyHosted = !!version && hosts.some(h => h.id === version.id);

  const submit = async () => {
    if (!current || !version) return;
    setBusy(true);
    setError(null);
    try {
      await invoke('hosting_start', {
        versionId: version.id,
        modelId: current.tree.id,
        name: hostName(current.tree, version),
        autoload,
        makeDefault,
      });
      onHosted(version.id);
      onClose();
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  };

  return (
    <ModalPortal>
      <div className="fixed inset-0 z-[9000] flex items-center justify-center p-6" style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(6px)' }} onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}>
        <div className="w-full max-w-2xl max-h-[85vh] flex flex-col rounded-2xl border border-white/10 bg-slate-900 shadow-2xl overflow-hidden">
          <div className="flex items-center justify-between px-5 py-4 border-b border-white/10">
            <div className="flex items-center gap-2.5">
              <Server className="w-5 h-5 text-gray-300" />
              <h2 className="text-white font-semibold">{t('hosting.dialog.title')}</h2>
            </div>
            <button onClick={onClose} className="text-gray-500 hover:text-white" aria-label={t('common.close', 'Schließen')}><X className="w-5 h-5" /></button>
          </div>

          <div className="px-5 pt-4">
            <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-black/20 px-3 py-2">
              <Search className="w-4 h-4 text-gray-500" />
              <input autoFocus value={query} onChange={e => setQuery(e.target.value)} placeholder={t('hosting.dialog.search')} className="flex-1 bg-transparent outline-none text-sm text-white placeholder-gray-500" />
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-5 py-3 space-y-1.5 min-h-[200px]">
            {loading && <div className="flex items-center gap-2 text-sm text-gray-400 py-6 justify-center"><Loader2 className="w-4 h-4 animate-spin" />{t('hosting.dialog.loading')}</div>}
            {!loading && rows.length === 0 && <p className="text-sm text-gray-500 py-6 text-center">{t('hosting.dialog.empty')}</p>}
            {rows.map(r => {
              const active = r.tree.id === selModel;
              const hostedCount = r.versions.filter(v => hosts.some(h => h.id === v.id)).length;
              return (
                <button
                  key={r.tree.id}
                  type="button"
                  onClick={() => { setSelModel(r.tree.id); setSelVersion(r.versions[0]?.id ?? null); }}
                  className={`w-full text-left px-3.5 py-2.5 rounded-xl border transition-colors ${active ? 'border-white/30 bg-white/10' : 'border-white/5 bg-white/[0.03] hover:bg-white/[0.06]'}`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <p className="text-sm text-white truncate">{r.tree.name}</p>
                      <p className="text-xs text-gray-500 truncate">
                        {r.trained ? t('hosting.dialog.versions', { count: r.versions.length }) : t('hosting.dialog.library')}
                        {r.info?.model_type ? ` · ${r.info.model_type}` : ''}
                        {hostedCount > 0 ? ` · ${t('hosting.dialog.hostedCount', { count: hostedCount })}` : ''}
                      </p>
                    </div>
                    {r.info?.size_bytes ? <span className="text-xs text-gray-500 tabular-nums flex-shrink-0">{(r.info.size_bytes / 1e9).toFixed(2)} GB</span> : null}
                  </div>
                </button>
              );
            })}
          </div>

          {current && (
            <div className="px-5 py-4 border-t border-white/10 space-y-3">
              <div className="flex items-center gap-3">
                <label className="text-xs text-gray-400 w-20">{t('hosting.dialog.version')}</label>
                <select value={version?.id ?? ''} onChange={e => setSelVersion(e.target.value)} className="flex-1 rounded-lg bg-black/30 border border-white/10 text-sm text-white px-2.5 py-1.5">
                  {current.versions.map(v => (
                    <option key={v.id} value={v.id}>{v.is_root ? t('hosting.dialog.original') : v.name}{hosts.some(h => h.id === v.id) ? ` (${t('hosting.dialog.alreadyHosted')})` : ''}</option>
                  ))}
                </select>
              </div>
              <div className="flex flex-wrap gap-x-5 gap-y-2">
                <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer"><input type="checkbox" checked={makeDefault} onChange={e => setMakeDefault(e.target.checked)} />{t('hosting.dialog.makeDefault')}</label>
                <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer"><input type="checkbox" checked={autoload} onChange={e => setAutoload(e.target.checked)} />{t('hosting.dialog.autoload')}</label>
              </div>
              {tight && (
                <div className="flex items-start gap-2 px-3 py-2 rounded-xl bg-amber-500/10 border border-amber-500/20 text-xs text-amber-300">
                  <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                  {t('hosting.dialog.ramTight', { need: ((sizeGb ?? 0) * 1.3).toFixed(1), running: runningGb.toFixed(1), total: ramGb.toFixed(0) })}
                </div>
              )}
              {error && <p className="text-xs text-red-300">{error}</p>}
            </div>
          )}

          <div className="flex justify-end gap-2 px-5 py-3.5 border-t border-white/10 bg-black/10">
            <button onClick={onClose} className="px-4 py-2 rounded-xl text-sm text-gray-300 bg-white/5 hover:bg-white/10 border border-white/10">{t('hosting.dialog.cancel')}</button>
            <button
              onClick={submit}
              disabled={!version || busy}
              className={`px-4 py-2 rounded-xl text-sm font-medium text-white flex items-center gap-2 disabled:opacity-40 bg-gradient-to-r ${currentTheme.colors.gradient}`}
            >
              {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : alreadyHosted ? <Check className="w-4 h-4" /> : <Server className="w-4 h-4" />}
              {alreadyHosted ? t('hosting.dialog.reload') : t('hosting.dialog.submit')}
            </button>
          </div>
        </div>
      </div>
    </ModalPortal>
  );
}
