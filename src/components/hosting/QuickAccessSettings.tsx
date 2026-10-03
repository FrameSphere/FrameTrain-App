// Schnell-Zugriff & API: einmal in den Einstellungen (oder beim ersten Start)
// einrichten, die Hosting-Seite zeigt danach nur noch den Stand.

import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Zap, Plug, RefreshCw, Eye, EyeOff, AlertTriangle, CheckCircle2, MousePointerClick, Keyboard, PanelTop, Timer, MessageSquarePlus } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { CopyButton } from './HostResultView';
import { useHosting, useHostingSettings } from './useHosting';
import {
  acceleratorFromEvent, apiSnippets, heldModifiers, knownConflict, shortcutLabel,
  type DesktopStatus, type HostingSettings, type QuickPolicy,
} from './hostingModel';

export const IS_MAC = typeof navigator !== 'undefined' && /Mac/i.test(navigator.platform || navigator.userAgent);
export const DEFAULT_SHORTCUT = IS_MAC ? 'Super+Shift+Space' : 'Control+Shift+Space';

/** Kurzform fuer Anzeigen: "2× ⌃ · ⌘⇧␣" */
export function quickSummary(s: HostingSettings | null, t: (k: string, p?: Record<string, string | number>) => string): string {
  if (!s || !s.quick_enabled) return t('hosting.quick.off');
  const parts: string[] = [];
  if (s.double_tap !== 'off') parts.push(t('hosting.settings.tapShort', { key: t(`hosting.settings.tapKey.${s.double_tap}${IS_MAC ? 'Mac' : ''}`) }));
  if (s.shortcut) parts.push(shortcutLabel(s.shortcut, IS_MAC));
  return parts.join(` ${t('hosting.settings.or')} `) || t('hosting.settings.trayOnly');
}

export function Toggle({ on, onChange, label }: { on: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button type="button" role="switch" aria-checked={on} aria-label={label} onClick={() => onChange(!on)}
      className={`w-11 h-6 rounded-full relative transition-colors flex-shrink-0 ${on ? 'bg-emerald-500/80' : 'bg-white/15'}`}>
      <span className={`absolute top-0.5 w-5 h-5 rounded-full bg-white transition-all ${on ? 'left-[22px]' : 'left-0.5'}`} />
    </button>
  );
}

function Row({ icon: Icon, label, hint, children }: { icon: typeof Zap; label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-6 py-3 border-b border-white/5 last:border-0">
      <div className="flex items-start gap-3 min-w-0">
        <Icon className="w-4 h-4 text-gray-400 mt-0.5 flex-shrink-0" />
        <div className="min-w-0">
          <p className="text-sm text-white">{label}</p>
          {hint && <p className="text-xs text-gray-500 mt-0.5 leading-relaxed">{hint}</p>}
        </div>
      </div>
      <div className="flex-shrink-0 flex items-center gap-2">{children}</div>
    </div>
  );
}

function ShortcutRecorder({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const { t } = useLanguage();
  const [recording, setRecording] = useState(false);
  const [held, setHeld] = useState('');
  useEffect(() => {
    if (!recording) return;
    // Das bisherige Kuerzel aussetzen — sonst faengt das System genau diesen Tastendruck ab.
    void invoke('hosting_pause_shortcut', { paused: true }).catch(() => {});
    const onKey = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.key === 'Escape') { setRecording(false); return; }
      const acc = acceleratorFromEvent(e);
      if (acc) { onChange(acc); setRecording(false); return; }
      setHeld(heldModifiers(e, IS_MAC));
    };
    const onUp = (e: KeyboardEvent) => setHeld(heldModifiers(e, IS_MAC));
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('keyup', onUp, true);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      window.removeEventListener('keyup', onUp, true);
      setHeld('');
      void invoke('hosting_pause_shortcut', { paused: false }).catch(() => {});
    };
  }, [recording, onChange]);
  return (
    <button type="button" onClick={() => setRecording(r => !r)} onBlur={() => setRecording(false)}
      className={`min-w-[150px] px-3 py-1.5 rounded-lg border text-sm tabular-nums ${recording ? 'border-amber-400/60 text-amber-300 bg-amber-500/10' : 'border-white/15 text-white bg-black/20 hover:bg-white/5'}`}>
      {recording ? (held || t('hosting.settings.pressKeys')) : value ? shortcutLabel(value, IS_MAC) : t('hosting.settings.off')}
    </button>
  );
}

/** Schnell-Zugriff: Hauptschalter, Doppeltipp, Kuerzel, Menueleiste. Auch im Ersteinrichtungs-Assistenten. */
export function QuickAccessCard({ settings, status, save, compact = false }: {
  settings: HostingSettings;
  status: DesktopStatus | null;
  save: (patch: Partial<HostingSettings>) => void;
  compact?: boolean;
}) {
  const { t } = useLanguage();
  const conflict = settings.shortcut ? knownConflict(settings.shortcut, IS_MAC) : null;
  const tapKeys: HostingSettings['double_tap'][] = ['control', 'shift', 'meta', 'alt', 'off'];
  const tapUnsupported = !!status && !status.double_tap_supported;
  const platform = status?.platform === 'windows' ? 'windows' : status?.platform === 'linux' ? 'linux' : 'macos';

  return (
    <section className="rounded-2xl border border-white/10 bg-white/[0.03] px-5 py-2">
      <div className="flex items-center justify-between gap-4 py-3">
        <div className="flex items-start gap-3">
          <Zap className="w-5 h-5 text-amber-300 mt-0.5" />
          <div>
            <h3 className="text-sm font-semibold text-white">{t('hosting.settings.quickTitle')}</h3>
            {!compact && <p className="text-xs text-gray-500 mt-0.5 leading-relaxed">{t('hosting.settings.quickIntro')}</p>}
          </div>
        </div>
        <Toggle on={settings.quick_enabled} onChange={v => save({ quick_enabled: v })} label={t('hosting.settings.quickTitle')} />
      </div>

      {settings.quick_enabled && (
        <div className="border-t border-white/5">
          <Row icon={MousePointerClick} label={t('hosting.settings.doubleTap')}
            hint={tapUnsupported ? t('hosting.settings.doubleTapWayland') : t('hosting.settings.doubleTapHint')}>
            <select value={tapUnsupported ? 'off' : settings.double_tap} disabled={tapUnsupported}
              onChange={e => save({ double_tap: e.target.value as HostingSettings['double_tap'] })}
              className="rounded-lg bg-black/30 border border-white/15 text-sm text-white px-2.5 py-1.5 disabled:opacity-40">
              {tapKeys.map(k => <option key={k} value={k}>{t(`hosting.settings.tap.${k}${IS_MAC && (k === 'alt' || k === 'meta' || k === 'control') ? 'Mac' : ''}`)}</option>)}
            </select>
          </Row>
          <Row icon={Keyboard} label={t('hosting.settings.shortcut')} hint={t('hosting.settings.shortcutHint')}>
            <ShortcutRecorder value={settings.shortcut} onChange={v => save({ shortcut: v })} />
            {settings.shortcut !== DEFAULT_SHORTCUT && (
              <button className="text-xs text-gray-500 hover:text-white" onClick={() => save({ shortcut: DEFAULT_SHORTCUT })}>{t('hosting.settings.reset')}</button>
            )}
          </Row>
          {(conflict || status?.shortcut_error) && (
            <p className="flex items-start gap-1.5 text-xs text-amber-300 pb-2">
              <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
              {status?.shortcut_error ? t('hosting.settings.shortcutTaken') : t('hosting.settings.conflict', { app: t(`hosting.conflict.${conflict}`) })}
            </p>
          )}
          <Row icon={PanelTop} label={t('hosting.settings.tray')} hint={t(`hosting.settings.trayHint.${platform}`)}>
            <Toggle on={settings.tray_enabled} onChange={v => save({ tray_enabled: v })} label={t('hosting.settings.tray')} />
          </Row>
          {!compact && (
            <Row icon={MessageSquarePlus} label={t('hosting.settings.session')} hint={t('hosting.settings.sessionHint')}>
              <select value={settings.quick_session ?? 'smart'} onChange={e => save({ quick_session: e.target.value as QuickPolicy })}
                className="rounded-lg bg-black/30 border border-white/15 text-sm text-white px-2.5 py-1.5 max-w-[240px]">
                {(['smart', 'always', 'never'] as QuickPolicy[]).map(k => <option key={k} value={k}>{t(`hosting.settings.sessionOpt.${k}`)}</option>)}
              </select>
            </Row>
          )}
          <div className="flex items-center justify-between gap-3 py-3">
            <p className="text-xs text-gray-400">{t('hosting.settings.tryHint', { how: quickSummary(settings, t) })}</p>
            <button onClick={() => { void invoke('hosting_show_quickchat').catch(() => {}); }}
              className="px-3 py-1.5 rounded-lg text-xs border border-white/15 text-white hover:bg-white/10 flex-shrink-0">
              {t('hosting.settings.tryNow')}
            </button>
          </div>
        </div>
      )}
    </section>
  );
}

function ApiCard({ settings, status, save, rotateToken }: {
  settings: HostingSettings; status: DesktopStatus | null;
  save: (patch: Partial<HostingSettings>) => void; rotateToken: () => void;
}) {
  const { t } = useLanguage();
  const { hosts } = useHosting();
  const [showToken, setShowToken] = useState(false);
  const [lang, setLang] = useState<'curl' | 'python'>('curl');
  const [port, setPort] = useState(String(settings.api_port));
  useEffect(() => setPort(String(settings.api_port)), [settings.api_port]);
  const host = hosts.find(h => h.is_default) ?? hosts[0] ?? null;
  const apiUrl = status?.api.url ?? `http://127.0.0.1:${settings.api_port}/v1`;
  const snippets = host ? apiSnippets(apiUrl, showToken ? settings.api_token : '$FRAMETRAIN_TOKEN', host) : null;

  return (
    <section className="rounded-2xl border border-white/10 bg-white/[0.03] px-5 py-2">
      <div className="flex items-center justify-between gap-4 py-3">
        <div className="flex items-start gap-3">
          <Plug className="w-5 h-5 text-gray-300 mt-0.5" />
          <div>
            <h3 className="text-sm font-semibold text-white">{t('hosting.settings.apiTitle')}</h3>
            <p className="text-xs text-gray-500 mt-0.5 leading-relaxed">{t('hosting.settings.apiHint')}</p>
          </div>
        </div>
        <Toggle on={settings.api_enabled} onChange={v => save({ api_enabled: v })} label={t('hosting.settings.apiTitle')} />
      </div>
      {settings.api_enabled && (
        <div className="border-t border-white/5 pt-2">
          <div className={`flex items-center gap-2 text-xs py-2 ${status?.api.error ? 'text-red-300' : 'text-emerald-300'}`}>
            {status?.api.error ? <AlertTriangle className="w-3.5 h-3.5" /> : <CheckCircle2 className="w-3.5 h-3.5" />}
            <span className="break-all">{status?.api.error ?? t('hosting.settings.apiRunning', { url: apiUrl })}</span>
          </div>
          <Row icon={Plug} label={t('hosting.settings.port')}>
            <input value={port} onChange={e => setPort(e.target.value.replace(/\D/g, '').slice(0, 5))}
              onBlur={() => { const p = Number(port); if (p >= 1024 && p <= 65535 && p !== settings.api_port) save({ api_port: p }); else setPort(String(settings.api_port)); }}
              className="w-24 rounded-lg bg-black/30 border border-white/15 text-sm text-white px-2.5 py-1.5 font-mono" />
          </Row>
          <Row icon={Plug} label={t('hosting.settings.token')} hint={t('hosting.settings.tokenHint')}>
            <code className="text-xs text-gray-300 font-mono max-w-[190px] truncate">{showToken ? settings.api_token : '••••••••••••••••'}</code>
            <button onClick={() => setShowToken(s => !s)} className="text-gray-500 hover:text-white" aria-label={t('hosting.settings.showToken')}>{showToken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}</button>
            <CopyButton text={settings.api_token} label="" />
            <button onClick={rotateToken} className="text-gray-500 hover:text-white" title={t('hosting.settings.rotate')} aria-label={t('hosting.settings.rotate')}><RefreshCw className="w-4 h-4" /></button>
          </Row>
          {snippets && host && (
            <div className="py-3 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex gap-1">
                  {(['curl', 'python'] as const).map(l => (
                    <button key={l} onClick={() => setLang(l)} className={`px-2.5 py-1 rounded-md text-xs ${lang === l ? 'bg-white/10 text-white' : 'text-gray-500 hover:text-gray-300'}`}>{l === 'curl' ? 'curl' : 'Python'}</button>
                  ))}
                </div>
                <CopyButton text={snippets[lang]} label={t('hosting.result.copy')} />
              </div>
              <pre className="text-[11px] leading-relaxed text-gray-300 bg-black/40 border border-white/5 rounded-xl p-3 overflow-x-auto whitespace-pre">{snippets[lang]}</pre>
              <p className="text-[11px] text-gray-500">{t('hosting.settings.snippetFor', { name: host.api_name })}</p>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

/** Ganzer Einstellungs-Tab "Schnell-Zugriff & API". */
export default function QuickAccessSettings() {
  const { t } = useLanguage();
  const { settings, status, save, rotateToken } = useHostingSettings();
  if (!settings) return null;
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-2xl font-bold text-white">{t('hosting.settings.tabTitle')}</h2>
        <p className="text-gray-400 text-sm mt-1">{t('hosting.settings.tabIntro')}</p>
      </div>
      <QuickAccessCard settings={settings} status={status} save={save} />
      <section className="rounded-2xl border border-white/10 bg-white/[0.03] px-5 py-2">
        <Row icon={Timer} label={t('hosting.settings.idle')} hint={t('hosting.settings.idleHint')}>
          <select value={settings.idle_minutes} onChange={e => save({ idle_minutes: Number(e.target.value) })}
            className="rounded-lg bg-black/30 border border-white/15 text-sm text-white px-2.5 py-1.5">
            {[0, 10, 30, 60, 120, 480].map(m => <option key={m} value={m}>{m === 0 ? t('hosting.settings.never') : t('hosting.settings.minutes', { count: m })}</option>)}
          </select>
        </Row>
      </section>
      <ApiCard settings={settings} status={status} save={save} rotateToken={rotateToken} />
    </div>
  );
}
