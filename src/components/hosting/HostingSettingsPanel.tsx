// Einstellungen fuer Schnell-Zugriff (Kuerzel, Doppeltipp, Tray), Leerlauf
// und die lokale API — mit Code-Beispielen fuer das gewaehlte Modell.

import { useEffect, useState } from 'react';
import { Keyboard, Plug, RefreshCw, Eye, EyeOff, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { CopyButton } from './HostResultView';
import { acceleratorFromEvent, apiSnippets, knownConflict, shortcutLabel, type HostInfo, type HostingSettings, type DesktopStatus } from './hostingModel';

export const IS_MAC = typeof navigator !== 'undefined' && /Mac/i.test(navigator.platform || navigator.userAgent);
export const DEFAULT_SHORTCUT = IS_MAC ? 'Control+Alt+Super+K' : 'Control+Alt+Shift+K';

function Row({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-6 py-3 border-b border-white/5 last:border-0">
      <div className="min-w-0">
        <p className="text-sm text-white">{label}</p>
        {hint && <p className="text-xs text-gray-500 mt-0.5 leading-relaxed">{hint}</p>}
      </div>
      <div className="flex-shrink-0 flex items-center gap-2">{children}</div>
    </div>
  );
}

function Toggle({ on, onChange, label }: { on: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button type="button" role="switch" aria-checked={on} aria-label={label} onClick={() => onChange(!on)}
      className={`w-10 h-6 rounded-full relative transition-colors ${on ? 'bg-emerald-500/80' : 'bg-white/15'}`}>
      <span className={`absolute top-0.5 w-5 h-5 rounded-full bg-white transition-all ${on ? 'left-[18px]' : 'left-0.5'}`} />
    </button>
  );
}

function ShortcutRecorder({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const { t } = useLanguage();
  const [recording, setRecording] = useState(false);
  useEffect(() => {
    if (!recording) return;
    const onKey = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.key === 'Escape') { setRecording(false); return; }
      const acc = acceleratorFromEvent(e);
      if (acc) { onChange(acc); setRecording(false); }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [recording, onChange]);
  return (
    <button type="button" onClick={() => setRecording(r => !r)}
      className={`min-w-[120px] px-3 py-1.5 rounded-lg border text-sm font-mono tabular-nums ${recording ? 'border-amber-400/60 text-amber-300 bg-amber-500/10' : 'border-white/15 text-white bg-black/20 hover:bg-white/5'}`}>
      {recording ? t('hosting.settings.pressKeys') : value ? shortcutLabel(value, IS_MAC) : t('hosting.settings.off')}
    </button>
  );
}

export default function HostingSettingsPanel({ settings, status, save, rotateToken, host }: {
  settings: HostingSettings;
  status: DesktopStatus | null;
  save: (patch: Partial<HostingSettings>) => void;
  rotateToken: () => void;
  host: HostInfo | null;
}) {
  const { t } = useLanguage();
  const [showToken, setShowToken] = useState(false);
  const [lang, setLang] = useState<'curl' | 'python'>('curl');
  const [port, setPort] = useState(String(settings.api_port));
  useEffect(() => setPort(String(settings.api_port)), [settings.api_port]);

  const conflict = settings.shortcut ? knownConflict(settings.shortcut, IS_MAC) : null;
  const apiUrl = status?.api.url ?? `http://127.0.0.1:${settings.api_port}/v1`;
  const snippets = host ? apiSnippets(apiUrl, showToken ? settings.api_token : '$FRAMETRAIN_TOKEN', host) : null;
  const tapKeys: HostingSettings['double_tap'][] = ['off', 'control', 'alt', 'shift', 'meta'];

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <section className="rounded-2xl border border-white/10 bg-white/[0.03] px-5 py-2">
        <div className="flex items-center gap-2 pt-3 pb-1">
          <Keyboard className="w-4 h-4 text-gray-400" />
          <h3 className="text-sm font-semibold text-white">{t('hosting.settings.quickTitle')}</h3>
        </div>
        <Row label={t('hosting.settings.shortcut')} hint={t('hosting.settings.shortcutHint', { def: shortcutLabel(DEFAULT_SHORTCUT, IS_MAC) })}>
          <ShortcutRecorder value={settings.shortcut} onChange={v => save({ shortcut: v })} />
          {settings.shortcut !== DEFAULT_SHORTCUT && (
            <button className="text-xs text-gray-500 hover:text-white" onClick={() => save({ shortcut: DEFAULT_SHORTCUT })}>{t('hosting.settings.reset')}</button>
          )}
          {settings.shortcut && (
            <button className="text-xs text-gray-500 hover:text-white" onClick={() => save({ shortcut: '' })}>{t('hosting.settings.disable')}</button>
          )}
        </Row>
        {(conflict || status?.shortcut_error) && (
          <p className="flex items-start gap-1.5 text-xs text-amber-300 -mt-1 mb-2">
            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
            {status?.shortcut_error ? t('hosting.settings.shortcutTaken') : t('hosting.settings.conflict', { app: t(`hosting.conflict.${conflict}`) })}
          </p>
        )}
        <Row label={t('hosting.settings.doubleTap')} hint={status && !status.double_tap_supported ? t('hosting.settings.doubleTapWayland') : t('hosting.settings.doubleTapHint')}>
          <select value={settings.double_tap} disabled={!!status && !status.double_tap_supported}
            onChange={e => save({ double_tap: e.target.value as HostingSettings['double_tap'] })}
            className="rounded-lg bg-black/30 border border-white/15 text-sm text-white px-2.5 py-1.5 disabled:opacity-40">
            {tapKeys.map(k => <option key={k} value={k}>{t(`hosting.settings.tap.${k}${IS_MAC && (k === 'alt' || k === 'meta') ? 'Mac' : ''}`)}</option>)}
          </select>
        </Row>
        <Row label={t('hosting.settings.tray')} hint={t(`hosting.settings.trayHint.${status?.platform === 'windows' ? 'windows' : status?.platform === 'linux' ? 'linux' : 'macos'}`)}>
          <Toggle on={settings.tray_enabled} onChange={v => save({ tray_enabled: v })} label={t('hosting.settings.tray')} />
        </Row>
        <Row label={t('hosting.settings.idle')} hint={t('hosting.settings.idleHint')}>
          <select value={settings.idle_minutes} onChange={e => save({ idle_minutes: Number(e.target.value) })}
            className="rounded-lg bg-black/30 border border-white/15 text-sm text-white px-2.5 py-1.5">
            {[0, 10, 30, 60, 120, 480].map(m => <option key={m} value={m}>{m === 0 ? t('hosting.settings.never') : t('hosting.settings.minutes', { count: m })}</option>)}
          </select>
        </Row>
      </section>

      <section className="rounded-2xl border border-white/10 bg-white/[0.03] px-5 py-2">
        <div className="flex items-center justify-between pt-3 pb-1">
          <div className="flex items-center gap-2">
            <Plug className="w-4 h-4 text-gray-400" />
            <h3 className="text-sm font-semibold text-white">{t('hosting.settings.apiTitle')}</h3>
          </div>
          <Toggle on={settings.api_enabled} onChange={v => save({ api_enabled: v })} label={t('hosting.settings.apiTitle')} />
        </div>
        <p className="text-xs text-gray-500 leading-relaxed pb-2">{t('hosting.settings.apiHint')}</p>
        {settings.api_enabled && (
          <>
            <div className={`flex items-center gap-2 text-xs mb-2 ${status?.api.error ? 'text-red-300' : 'text-emerald-300'}`}>
              {status?.api.error ? <AlertTriangle className="w-3.5 h-3.5" /> : <CheckCircle2 className="w-3.5 h-3.5" />}
              <span className="break-all">{status?.api.error ?? t('hosting.settings.apiRunning', { url: apiUrl })}</span>
            </div>
            <Row label={t('hosting.settings.port')}>
              <input value={port} onChange={e => setPort(e.target.value.replace(/\D/g, '').slice(0, 5))}
                onBlur={() => { const p = Number(port); if (p >= 1024 && p <= 65535 && p !== settings.api_port) save({ api_port: p }); else setPort(String(settings.api_port)); }}
                className="w-24 rounded-lg bg-black/30 border border-white/15 text-sm text-white px-2.5 py-1.5 font-mono" />
            </Row>
            <Row label={t('hosting.settings.token')} hint={t('hosting.settings.tokenHint')}>
              <code className="text-xs text-gray-300 font-mono max-w-[170px] truncate">{showToken ? settings.api_token : '••••••••••••••••'}</code>
              <button onClick={() => setShowToken(s => !s)} className="text-gray-500 hover:text-white" aria-label={t('hosting.settings.showToken')}>{showToken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}</button>
              <CopyButton text={settings.api_token} label="" />
              <button onClick={rotateToken} className="text-gray-500 hover:text-white" title={t('hosting.settings.rotate')} aria-label={t('hosting.settings.rotate')}><RefreshCw className="w-4 h-4" /></button>
            </Row>
            {snippets && (
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
                <p className="text-[11px] text-gray-500">{t('hosting.settings.snippetFor', { name: host?.api_name ?? '' })}</p>
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}
