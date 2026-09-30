// Schnell-Chat: eigenes, rahmenloses Fenster ("quickchat"). Oeffnet per
// Tastenkuerzel, Doppeltipp oder Tray-Symbol ueber jeder App, spricht das
// Standard-Modell des Hostings an und verschwindet bei Esc oder Fokusverlust.

import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { ChevronDown, Server, ExternalLink, RotateCcw, Check } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useTheme } from '../../contexts/ThemeContext';
import { Composer, MessageList, useHostChat } from './HostChat';
import { StatusPill } from './HostingPanel';
import { useHosting } from './useHosting';
import { blurGuarded } from './quickchatGuard';
import { taskKey, type HostInfo } from './hostingModel';

function ModelButton({ host, open, onToggle }: { host: HostInfo | null; open: boolean; onToggle: () => void }) {
  if (!host) return null;
  return (
    <button type="button" onClick={onToggle}
      className={`flex-shrink-0 self-center flex items-center gap-1.5 max-w-[190px] pl-2.5 pr-2 py-1 rounded-lg text-xs text-white ${open ? 'bg-white/20' : 'bg-white/10 hover:bg-white/15'}`}>
      <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${host.status === 'ready' ? 'bg-emerald-400' : host.status === 'error' ? 'bg-red-400' : 'bg-amber-400'}`} />
      <span className="truncate">{host.name}</span>
      <ChevronDown className={`w-3.5 h-3.5 text-gray-400 flex-shrink-0 transition-transform ${open ? 'rotate-180' : ''}`} />
    </button>
  );
}

/** Modellliste im Fluss statt als Popup — ein Popup wuerde am Fensterrand abgeschnitten. */
function ModelList({ hosts, host, onPick }: { hosts: HostInfo[]; host: HostInfo | null; onPick: (id: string) => void }) {
  const { t } = useLanguage();
  return (
    <div className="border-t border-white/10 py-1 max-h-64 overflow-y-auto">
      {hosts.map(h => (
        <button key={h.id} type="button" onClick={() => onPick(h.id)}
          className="w-full flex items-center justify-between gap-2 px-3.5 py-2 text-left hover:bg-white/5">
          <div className="min-w-0">
            <p className="text-sm text-white truncate">{h.name}</p>
            <p className="text-[11px] text-gray-500 truncate">{h.modality ? t(`hosting.task.${taskKey(h)}`, h.modality) : t('hosting.page.notLoadedYet')}</p>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <StatusPill status={h.status} busy={h.busy} />
            {h.id === host?.id && <Check className="w-3.5 h-3.5 text-gray-300" />}
          </div>
        </button>
      ))}
    </div>
  );
}

export default function QuickChatApp() {
  const { t } = useLanguage();
  const { currentTheme } = useTheme();
  const { hosts, loaded } = useHosting();
  const [pickedId, setPickedId] = useState<string | null>(null);
  const [focusSignal, setFocusSignal] = useState(1);
  const [pickerOpen, setPickerOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  const host = hosts.find(h => h.id === pickedId) ?? hosts.find(h => h.is_default) ?? hosts.find(h => h.status === 'ready') ?? hosts[0] ?? null;
  const chat = useHostChat(host, false);

  // Durchsichtiges Fenster: kein Seitenhintergrund
  useLayoutEffect(() => {
    for (const el of [document.documentElement, document.body]) {
      el.style.background = 'transparent';
      el.style.overflow = 'hidden';
    }
  }, []);

  // Fensterhoehe folgt dem Inhalt
  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    const report = () => { void invoke('hosting_quickchat_resize', { height: Math.ceil(el.getBoundingClientRect().height) + 12 }).catch(() => {}); };
    report();
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(report);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Oeffnen → Fokus ins Eingabefeld; Fokusverlust → ausblenden
  useEffect(() => {
    const offs: (() => void)[] = [];
    let disposed = false;
    const keep = (p: Promise<() => void>) => p.then(fn => { if (disposed) fn(); else offs.push(fn); }).catch(() => {});
    keep(listen('quickchat-shown', () => { setPickerOpen(false); setFocusSignal(n => n + 1); }));
    keep(getCurrentWindow().onFocusChanged(({ payload: focused }) => {
      if (!focused && !blurGuarded()) void invoke('hosting_quickchat_blur').catch(() => {});
    }));
    return () => { disposed = true; offs.forEach(f => f()); };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); void invoke('hosting_hide_quickchat').catch(() => {}); }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'n') { e.preventDefault(); chat.clear(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [chat]);

  const pick = (id: string) => {
    setPickedId(id);
    setPickerOpen(false);
    void invoke('hosting_update_model', { id, makeDefault: true }).catch(() => {});
    setFocusSignal(n => n + 1);
  };

  const openMain = () => { void invoke('hosting_open_main', { view: 'hosting' }).catch(() => {}); };

  return (
    <div className="p-1.5">
      <div ref={rootRef} className="rounded-2xl border border-white/15 bg-slate-900/95 backdrop-blur-xl shadow-[0_18px_50px_rgba(0,0,0,0.55)] overflow-hidden text-white">
        {loaded && hosts.length === 0 ? (
          <div className="flex items-center gap-3 px-4 py-3.5">
            <Server className="w-5 h-5 text-gray-400 flex-shrink-0" />
            <p className="flex-1 text-sm text-gray-300">{t('hosting.quick.noModels')}</p>
            <button onClick={openMain} className={`px-3 py-1.5 rounded-lg text-xs font-medium text-white bg-gradient-to-r ${currentTheme.colors.gradient}`}>{t('hosting.page.add')}</button>
          </div>
        ) : (
          <>
            <Composer host={host} sending={chat.sending} onSend={chat.send} compact autoFocus focusSignal={focusSignal}
              leading={<ModelButton host={host} open={pickerOpen} onToggle={() => setPickerOpen(o => !o)} />} />
            {pickerOpen && <ModelList hosts={hosts} host={host} onPick={pick} />}
            {chat.messages.length > 0 && (
              <div className="border-t border-white/10 px-4 py-3 max-h-[440px] overflow-y-auto">
                <MessageList messages={chat.messages} host={host} compact />
              </div>
            )}
            <div data-tauri-drag-region className="flex items-center justify-between gap-3 px-3.5 py-1.5 border-t border-white/5 text-[11px] text-gray-500 select-none">
              <span data-tauri-drag-region>{t('hosting.quick.keys')}</span>
              <div className="flex items-center gap-3">
                {chat.messages.length > 0 && (
                  <button onClick={chat.clear} className="flex items-center gap-1 hover:text-gray-300"><RotateCcw className="w-3 h-3" />{t('hosting.quick.new')}</button>
                )}
                <button onClick={openMain} className="flex items-center gap-1 hover:text-gray-300">{t('hosting.quick.openApp')}<ExternalLink className="w-3 h-3" /></button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
