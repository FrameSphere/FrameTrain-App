// Hosting: trainierte oder importierte Modelle lokal bereithalten und direkt
// befragen — hier im Hauptfenster, per Schnell-Chat (Kuerzel/Tray) oder ueber
// die lokale API.

import { useEffect, useMemo, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Server, Plus, Play, Square, Trash2, Star, Loader2, AlertTriangle, Moon, MessageSquarePlus, Settings2, ChevronDown } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useTheme } from '../../contexts/ThemeContext';
import HostModelDialog from './HostModelDialog';
import HostingSettingsPanel, { IS_MAC } from './HostingSettingsPanel';
import { Composer, MessageList, useHostChat } from './HostChat';
import { useHosting, useHostingSettings } from './useHosting';
import { shortcutLabel, taskKey, type HostInfo, type HostStatus } from './hostingModel';

export function StatusPill({ status, busy }: { status: HostStatus; busy?: boolean }) {
  const { t } = useLanguage();
  const cls: Record<HostStatus, string> = {
    ready: 'bg-emerald-500/15 text-emerald-300',
    loading: 'bg-amber-500/15 text-amber-300',
    sleeping: 'bg-sky-500/15 text-sky-300',
    error: 'bg-red-500/15 text-red-300',
    idle: 'bg-white/10 text-gray-400',
  };
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] ${cls[status]}`}>
      {(status === 'loading' || busy) && <Loader2 className="w-3 h-3 animate-spin" />}
      {status === 'sleeping' && <Moon className="w-3 h-3" />}
      {busy && status === 'ready' ? t('hosting.status.busy') : t(`hosting.status.${status}`)}
    </span>
  );
}

function HostCard({ h, active, onClick }: { h: HostInfo; active: boolean; onClick: () => void }) {
  const { t } = useLanguage();
  return (
    <button type="button" onClick={onClick}
      className={`w-full text-left rounded-xl border px-3 py-2.5 transition-colors ${active ? 'border-white/25 bg-white/10' : 'border-white/5 bg-white/[0.03] hover:bg-white/[0.06]'}`}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm text-white truncate">{h.name}</span>
        {h.is_default && <Star className="w-3.5 h-3.5 text-amber-300 flex-shrink-0" aria-label={t('hosting.page.default')} />}
      </div>
      <div className="flex items-center gap-2 mt-1.5">
        <StatusPill status={h.status} busy={h.busy} />
        {h.modality && <span className="text-[11px] text-gray-500 truncate">{t(`hosting.task.${taskKey(h)}`, h.modality)}</span>}
      </div>
    </button>
  );
}

function HostDetail({ host, shortcut }: { host: HostInfo; shortcut: string }) {
  const { t } = useLanguage();
  const chat = useHostChat(host, true);
  const act = (cmd: string, args: Record<string, unknown>) => invoke(cmd, args).catch(e => console.error(`[Hosting] ${cmd}:`, e));
  const loaded = host.status === 'ready' || host.status === 'loading';

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-center justify-between gap-3 px-5 py-3.5 border-b border-white/10">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h2 className="text-white font-semibold truncate">{host.name}</h2>
            <StatusPill status={host.status} busy={host.busy} />
          </div>
          <p className="text-xs text-gray-500 mt-0.5 truncate">
            {host.modality ? t(`hosting.task.${taskKey(host)}`, host.modality) : t('hosting.page.notLoadedYet')}
            {host.size_gb ? ` · ${host.size_gb.toFixed(2)} GB` : ''}
            {host.requests ? ` · ${t('hosting.page.requests', { count: host.requests })}` : ''}
            {` · API: ${host.api_name}`}
          </p>
        </div>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <button onClick={() => act('hosting_update_model', { id: host.id, makeDefault: true })} disabled={host.is_default}
            title={t('hosting.page.makeDefaultHint', { shortcut })}
            className={`px-2.5 py-1.5 rounded-lg text-xs border flex items-center gap-1.5 ${host.is_default ? 'border-amber-400/30 text-amber-300 bg-amber-500/10' : 'border-white/10 text-gray-300 hover:bg-white/5'}`}>
            <Star className="w-3.5 h-3.5" />{host.is_default ? t('hosting.page.isDefault') : t('hosting.page.makeDefault')}
          </button>
          <label className="px-2.5 py-1.5 rounded-lg text-xs border border-white/10 text-gray-300 flex items-center gap-1.5 cursor-pointer hover:bg-white/5">
            <input type="checkbox" checked={host.autoload} onChange={e => act('hosting_update_model', { id: host.id, autoload: e.target.checked })} />
            {t('hosting.page.autoload')}
          </label>
          {loaded ? (
            <button onClick={() => act('hosting_stop', { id: host.id })} className="px-2.5 py-1.5 rounded-lg text-xs border border-white/10 text-gray-300 hover:bg-white/5 flex items-center gap-1.5">
              <Square className="w-3.5 h-3.5" />{t('hosting.page.stop')}
            </button>
          ) : (
            <button onClick={() => act('hosting_start', { versionId: host.id, modelId: host.model_id, name: host.name })} className="px-2.5 py-1.5 rounded-lg text-xs border border-emerald-400/30 text-emerald-300 bg-emerald-500/10 hover:bg-emerald-500/20 flex items-center gap-1.5">
              <Play className="w-3.5 h-3.5" />{t('hosting.page.load')}
            </button>
          )}
          <button onClick={() => { if (window.confirm(t('hosting.page.removeConfirm', { name: host.name }))) act('hosting_remove', { id: host.id }); }}
            className="p-1.5 rounded-lg text-gray-500 hover:text-red-300 hover:bg-red-500/10" title={t('hosting.page.remove')} aria-label={t('hosting.page.remove')}>
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {host.status === 'error' && host.error && (
        <div className="mx-5 mt-3 flex items-start gap-2 px-3 py-2 rounded-xl bg-red-500/10 border border-red-500/20 text-xs text-red-300">
          <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
          <span className="whitespace-pre-wrap break-words">{host.error}</span>
        </div>
      )}

      <div className="flex-1 overflow-y-auto px-5 py-4 min-h-0">
        {chat.messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center gap-2 text-gray-500">
            <MessageSquarePlus className="w-8 h-8 text-gray-600" />
            <p className="text-sm">{t('hosting.page.emptyChat')}</p>
            {shortcut && <p className="text-xs">{t('hosting.page.quickHint', { shortcut })}</p>}
          </div>
        ) : (
          <MessageList messages={chat.messages} host={host} />
        )}
      </div>

      <div className="px-5 pb-4 pt-2 border-t border-white/5">
        <Composer host={host} sending={chat.sending} onSend={chat.send} autoFocus />
        <div className="flex items-center justify-between mt-2 text-[11px] text-gray-500">
          <span>{t('hosting.page.composerHint')}</span>
          {chat.messages.length > 0 && <button onClick={chat.clear} className="hover:text-gray-300">{t('hosting.page.clear')}</button>}
        </div>
      </div>
    </div>
  );
}

export default function HostingPanel() {
  const { t } = useLanguage();
  const { currentTheme } = useTheme();
  const { hosts, loaded } = useHosting();
  const { settings, status, save, rotateToken, reloadStatus } = useHostingSettings();
  const [selected, setSelected] = useState<string | null>(null);
  const [dialog, setDialog] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => {
    if (!hosts.length) { setSelected(null); return; }
    if (!selected || !hosts.some(h => h.id === selected)) {
      setSelected((hosts.find(h => h.is_default) ?? hosts[0]).id);
    }
  }, [hosts, selected]);

  useEffect(() => { void reloadStatus(); }, [hosts.length, reloadStatus]);

  const host = hosts.find(h => h.id === selected) ?? null;
  const running = hosts.filter(h => h.status === 'ready');
  const ramGb = useMemo(() => running.reduce((s, h) => s + (h.size_gb ?? 0), 0), [running]);
  const shortcut = settings?.shortcut ? shortcutLabel(settings.shortcut, IS_MAC) : '';

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3"><Server className="w-8 h-8" />{t('hosting.page.title')}</h1>
          <p className="text-gray-400 mt-1">{t('hosting.page.subtitle')}</p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => setShowSettings(s => !s)} className="px-3.5 py-2 rounded-xl text-sm text-gray-300 bg-white/5 hover:bg-white/10 border border-white/10 flex items-center gap-2">
            <Settings2 className="w-4 h-4" />{t('hosting.page.settings')}
            <ChevronDown className={`w-4 h-4 transition-transform ${showSettings ? 'rotate-180' : ''}`} />
          </button>
          <button onClick={() => setDialog(true)} className={`px-4 py-2 rounded-xl text-sm font-medium text-white flex items-center gap-2 bg-gradient-to-r ${currentTheme.colors.gradient}`}>
            <Plus className="w-4 h-4" />{t('hosting.page.add')}
          </button>
        </div>
      </div>

      {showSettings && settings && (
        <HostingSettingsPanel settings={settings} status={status} save={save} rotateToken={rotateToken} host={host} />
      )}

      {loaded && hosts.length === 0 ? (
        <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-8 py-14 text-center">
          <Server className="w-10 h-10 text-gray-600 mx-auto" />
          <h2 className="text-white font-semibold mt-3">{t('hosting.page.emptyTitle')}</h2>
          <p className="text-sm text-gray-400 mt-1.5 max-w-lg mx-auto leading-relaxed">{t('hosting.page.emptyText', { shortcut: shortcut || '—' })}</p>
          <button onClick={() => setDialog(true)} className={`mt-5 px-4 py-2 rounded-xl text-sm font-medium text-white inline-flex items-center gap-2 bg-gradient-to-r ${currentTheme.colors.gradient}`}>
            <Plus className="w-4 h-4" />{t('hosting.page.add')}
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-[240px_minmax(0,1fr)] gap-4 h-[calc(100vh-13rem)] min-h-[480px]">
          <aside className="rounded-2xl border border-white/10 bg-white/[0.03] p-3 flex flex-col gap-2 min-h-0">
            <div className="flex items-center justify-between px-1 pb-1">
              <span className="text-xs font-medium text-gray-400">{t('hosting.page.hosted')}</span>
              <span className="text-[11px] text-gray-500 tabular-nums">{t('hosting.page.running', { count: running.length, gb: ramGb.toFixed(1) })}</span>
            </div>
            <div className="flex-1 overflow-y-auto space-y-1.5 min-h-0">
              {hosts.map(h => <HostCard key={h.id} h={h} active={h.id === selected} onClick={() => setSelected(h.id)} />)}
            </div>
            {shortcut && <p className="text-[11px] text-gray-500 px-1 pt-1 border-t border-white/5">{t('hosting.page.quickHint', { shortcut })}</p>}
          </aside>
          <section className="rounded-2xl border border-white/10 bg-white/[0.03] min-h-0 overflow-hidden">
            {host ? <HostDetail key={host.id} host={host} shortcut={shortcut} /> : null}
          </section>
        </div>
      )}

      {dialog && <HostModelDialog hosts={hosts} onClose={() => setDialog(false)} onHosted={id => setSelected(id)} />}
    </div>
  );
}
