// Einstellungen > KI-Verbindungen: FrameTrain per MCP aus Claude, Cursor, ChatGPT steuern.
// Lokal (127.0.0.1, Token) und optional ueber frame-train.com (Relay).

import { useCallback, useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  Cable, Plug, KeyRound, Eye, EyeOff, RefreshCw, AlertTriangle, CheckCircle2, Loader2, Terminal,
  Laptop, Cloud, ShieldCheck, History, Trash2, Unplug,
} from 'lucide-react';
import { friendlyClientName } from '../../mcp/clientNames';
import { useLanguage } from '../../contexts/LanguageContext';
import { dateLocale } from '../../utils/dateLocale';
import { Toggle } from '../hosting/QuickAccessSettings';
import { CopyButton } from '../hosting/HostResultView';
import ModalPortal from '../ui/ModalPortal';
import {
  REMOTE_MCP_URL, TOKEN_MASK, claudeCodeCommand, cursorJson, connectErrorKey, endpointUrl, groupApprovals,
  loadApprovalsApi, parseRemoteStatus, type ApprovalsApi, type McpApproval, type McpStatus, type RemoteStatus,
} from './mcpConnections';

const POLL_MS = 5000;

function Card({ icon: Icon, title, hint, right, children }: {
  icon: typeof Plug; title: string; hint?: string; right?: React.ReactNode; children?: React.ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-white/10 bg-white/[0.03] px-5 py-2">
      <div className="flex items-center justify-between gap-4 py-3">
        <div className="flex items-start gap-3 min-w-0">
          <Icon className="w-5 h-5 text-gray-300 mt-0.5 flex-shrink-0" />
          <div className="min-w-0">
            <h3 className="text-sm font-semibold text-white">{title}</h3>
            {hint && <p className="text-xs text-gray-500 mt-0.5 leading-relaxed">{hint}</p>}
          </div>
        </div>
        {right && <div className="flex-shrink-0 flex items-center gap-2">{right}</div>}
      </div>
      {children && <div className="border-t border-white/5 py-2">{children}</div>}
    </section>
  );
}

function Row({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-6 py-2.5">
      <div className="min-w-0">
        <p className="text-sm text-white">{label}</p>
        {hint && <p className="text-xs text-gray-500 mt-0.5 leading-relaxed">{hint}</p>}
      </div>
      <div className="flex-shrink-0 flex items-center gap-2">{children}</div>
    </div>
  );
}

type ClaudePhase = 'idle' | 'waitQuit' | 'reopen' | 'done' | 'lost';
const CLAUDE_POLL_MS = 1500;
/** Claude Desktop schreibt die Konfiguration kurz nach dem Start; erst danach pruefen. */
const CLAUDE_SETTLE_MS = 5000;

function StatusLine({ ok, text }: { ok: boolean; text: string }) {
  return (
    <div className={`flex items-center gap-2 text-xs py-2 ${ok ? 'text-emerald-300' : 'text-amber-300'}`}>
      {ok ? <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0" /> : <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />}
      <span className="break-all">{text}</span>
    </div>
  );
}

function Snippet({ title, text, shown, copyLabel }: { title: string; text: string; shown: string; copyLabel: string }) {
  return (
    <div className="py-2 space-y-1.5">
      <div className="flex items-center justify-between">
        <p className="text-xs text-gray-400">{title}</p>
        <span className="text-gray-400"><CopyButton text={text} label={copyLabel} /></span>
      </div>
      <pre className="text-[11px] leading-relaxed text-gray-300 bg-black/40 border border-white/5 rounded-xl p-3 overflow-x-auto whitespace-pre">{shown}</pre>
    </div>
  );
}

export default function AIConnectionsSettings({ approvalsApi }: { approvalsApi?: ApprovalsApi | null } = {}) {
  const { t, language } = useLanguage();
  const [status, setStatus] = useState<McpStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showToken, setShowToken] = useState(false);
  const [port, setPort] = useState('');
  const [confirmRotate, setConfirmRotate] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [connectInfo, setConnectInfo] = useState<{ ok: boolean; text: string } | null>(null);
  // Claude Desktop ueberschreibt seine Konfiguration beim Beenden: erst eintragen, wenn es zu ist.
  const [claudePhase, setClaudePhase] = useState<ClaudePhase>('idle');
  const [remote, setRemote] = useState<RemoteStatus | null>(null);
  const [remoteError, setRemoteError] = useState<string | null>(null);
  const [remoteLoading, setRemoteLoading] = useState(false);
  const [approvals, setApprovals] = useState<McpApproval[]>([]);
  const [api, setApi] = useState<ApprovalsApi | null>(approvalsApi ?? null);

  const fmt = (iso: string | null | undefined) => {
    if (!iso) return '–';
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(dateLocale(language), { dateStyle: 'short', timeStyle: 'short' });
  };

  const refresh = useCallback(async () => {
    try {
      const s = await invoke<McpStatus>('mcp_get_status');
      setStatus(s);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
    const id = setInterval(() => void refresh(), POLL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  useEffect(() => { if (status) setPort(String(status.config.port)); }, [status?.config.port]); // eslint-disable-line react-hooks/exhaustive-deps

  // Freigaben (Strang A). Prop hat Vorrang (Tests), sonst optionaler Import.
  useEffect(() => {
    if (approvalsApi !== undefined) { setApi(approvalsApi); return; }
    let alive = true;
    void loadApprovalsApi().then(a => { if (alive) setApi(a); });
    return () => { alive = false; };
  }, [approvalsApi]);

  const reloadApprovals = useCallback(async () => {
    if (!api) { setApprovals([]); return; }
    try { setApprovals(await api.listApprovals()); } catch { setApprovals([]); }
  }, [api]);

  useEffect(() => {
    void reloadApprovals();
    const off = api?.onApprovalsChanged?.(() => void reloadApprovals());
    return () => { off?.(); };
  }, [api, reloadApprovals]);

  const loadRemote = useCallback(async () => {
    setRemoteLoading(true);
    try {
      setRemote(parseRemoteStatus(await invoke('mcp_remote_status')));
      setRemoteError(null);
    } catch (e) {
      setRemoteError(String(e));
    } finally {
      setRemoteLoading(false);
    }
  }, []);

  const remoteOn = status?.config.remoteEnabled ?? false;
  useEffect(() => { if (remoteOn) void loadRemote(); }, [remoteOn, loadRemote]);

  const save = async (patch: Partial<{ enabled: boolean; port: number; remoteEnabled: boolean }>) => {
    if (!status) return;
    const c = { ...status.config, ...patch };
    try {
      setStatus(await invoke<McpStatus>('mcp_save_settings', { enabled: c.enabled, port: c.port, remoteEnabled: c.remoteEnabled }));
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  };

  const rotate = async () => {
    setConfirmRotate(false);
    try { setStatus(await invoke<McpStatus>('mcp_rotate_token')); } catch (e) { setError(String(e)); }
  };

  const connectClaude = useCallback(async () => {
    setConnecting(true);
    setConnectInfo(null);
    try {
      const r = await invoke<{ configPath: string; backupPath: string | null; status: McpStatus }>('mcp_connect_claude_desktop');
      setStatus(r.status);
      setClaudePhase('reopen');
      setConnectInfo({
        ok: true,
        text: r.backupPath
          ? t('mcp.settings.claudeDesktop.doneBackup', { path: r.backupPath })
          : t('mcp.settings.claudeDesktop.done'),
      });
    } catch (e) {
      const err = String(e);
      if (err.startsWith('CLAUDE_RUNNING')) {
        setClaudePhase('waitQuit');
        return;
      }
      setClaudePhase('idle');
      const key = connectErrorKey(err);
      setConnectInfo({ ok: false, text: key ? t(key) : t('mcp.settings.claudeDesktop.failed', { error: err }) });
    } finally {
      setConnecting(false);
    }
  }, [t]);

  // Warten, bis Claude Desktop beendet ist, dann eintragen; danach pruefen, ob der Eintrag den Start uebersteht.
  useEffect(() => {
    if (claudePhase !== 'waitQuit' && claudePhase !== 'reopen') return;
    let cancelled = false;
    let seenRunningAt: number | null = null;
    const tick = async () => {
      let running: boolean;
      try { running = await invoke<boolean>('mcp_claude_desktop_running'); } catch { return; }
      if (cancelled) return;
      if (claudePhase === 'waitQuit') {
        if (!running) void connectClaude();
        return;
      }
      if (!running) { seenRunningAt = null; return; }
      seenRunningAt ??= Date.now();
      if (Date.now() - seenRunningAt < CLAUDE_SETTLE_MS) return;
      try {
        const s = await invoke<McpStatus>('mcp_get_status');
        if (cancelled) return;
        setStatus(s);
        setClaudePhase(s.claudeDesktop.configured ? 'done' : 'lost');
        setConnectInfo(s.claudeDesktop.configured
          ? { ok: true, text: t('mcp.settings.claudeDesktop.verified') }
          : { ok: false, text: t('mcp.settings.claudeDesktop.lost') });
      } catch { /* naechster Versuch */ }
    };
    const id = setInterval(() => void tick(), CLAUDE_POLL_MS);
    void tick();
    return () => { cancelled = true; clearInterval(id); };
  }, [claudePhase, connectClaude, t]);

  const disconnectRemote = async (id: string) => {
    try { await invoke('mcp_remote_disconnect', { connectionId: id }); } catch (e) { setRemoteError(String(e)); }
    void loadRemote();
  };

  const revoke = async (a: McpApproval) => {
    if (!api) return;
    try { await api.revokeApproval(a.client, a.tool); } finally { void reloadApprovals(); }
  };

  if (!status) {
    return (
      <div className="space-y-4">
        <h2 className="text-2xl font-bold text-white">{t('mcp.settings.title')}</h2>
        {error
          ? <StatusLine ok={false} text={error} />
          : <div className="flex items-center gap-2 text-sm text-gray-400"><Loader2 className="w-4 h-4 animate-spin" />{t('mcp.settings.loading')}</div>}
      </div>
    );
  }

  const cfg = status.config;
  const url = status.server.url ?? endpointUrl(cfg.port);
  const shownToken = showToken ? cfg.token : TOKEN_MASK;
  const grouped = groupApprovals(approvals);

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-2xl font-bold text-white">{t('mcp.settings.title')}</h2>
        <p className="text-gray-400 text-sm mt-1">{t('mcp.settings.intro')}</p>
      </div>
      {error && <StatusLine ok={false} text={error} />}

      {/* Lokaler Zugang */}
      <Card icon={Plug} title={t('mcp.settings.local.title')} hint={t('mcp.settings.local.hint')}
        right={<Toggle on={cfg.enabled} onChange={v => void save({ enabled: v })} label={t('mcp.settings.local.title')} />}>
        {cfg.enabled ? (
          <>
            <StatusLine ok={!status.server.error} text={status.server.error ?? t('mcp.settings.local.running', { url })} />
            {!status.manifestLoaded && <p className="text-xs text-gray-500 pb-1">{t('mcp.settings.local.noManifest')}</p>}
            <Row label={t('mcp.settings.local.port')}>
              <input aria-label={t('mcp.settings.local.port')} value={port} onChange={e => setPort(e.target.value.replace(/\D/g, '').slice(0, 5))}
                onBlur={() => { const p = Number(port); if (p >= 1024 && p <= 65535 && p !== cfg.port) void save({ port: p }); else setPort(String(cfg.port)); }}
                className="w-24 rounded-lg bg-black/30 border border-white/15 text-sm text-white px-2.5 py-1.5 font-mono" />
            </Row>
            <Row label={t('mcp.settings.local.token')} hint={t('mcp.settings.local.tokenHint')}>
              <code className="text-xs text-gray-300 font-mono max-w-[220px] truncate">{shownToken}</code>
              <button onClick={() => setShowToken(s => !s)} className="text-gray-500 hover:text-white" aria-label={t('mcp.settings.local.showToken')}>
                {showToken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
              <span className="text-gray-500"><CopyButton text={cfg.token} label="" /></span>
              <button onClick={() => setConfirmRotate(true)} className="text-gray-500 hover:text-white" title={t('mcp.settings.local.rotate')} aria-label={t('mcp.settings.local.rotate')}>
                <RefreshCw className="w-4 h-4" />
              </button>
            </Row>
          </>
        ) : (
          <p className="text-xs text-gray-500 py-1">{t('mcp.settings.local.offHint')}</p>
        )}
      </Card>

      {/* Programme verbinden */}
      <Card icon={Cable} title={t('mcp.settings.apps.title')} hint={t('mcp.settings.apps.hint')}>
        <Row label={t('mcp.settings.claudeDesktop.title')}
          hint={status.claudeDesktop.configured ? t('mcp.settings.claudeDesktop.configured') : t('mcp.settings.claudeDesktop.hint')}>
          {status.claudeDesktop.configured && <CheckCircle2 className="w-4 h-4 text-emerald-400" aria-label={t('mcp.settings.claudeDesktop.configured')} />}
          <button onClick={() => void connectClaude()} disabled={connecting || claudePhase === 'waitQuit' || !status.claudeDesktop.installed}
            className="inline-flex items-center gap-2 rounded-lg bg-white/10 hover:bg-white/15 disabled:opacity-40 disabled:cursor-not-allowed text-sm text-white px-3 py-1.5">
            {connecting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Laptop className="w-4 h-4" />}
            {status.claudeDesktop.configured ? t('mcp.settings.claudeDesktop.reconnect') : t('mcp.settings.claudeDesktop.connect')}
          </button>
        </Row>
        {!status.claudeDesktop.installed && <p className="text-xs text-gray-500 pb-2">{t('mcp.settings.claudeDesktop.notFound')}</p>}
        {status.exeTranslocated && <StatusLine ok={false} text={t('mcp.settings.claudeDesktop.translocated')} />}
        {claudePhase === 'waitQuit' && (
          <div className="flex items-center gap-2 text-xs py-2 text-amber-300" role="status">
            <Loader2 className="w-3.5 h-3.5 flex-shrink-0 animate-spin" />
            <span className="flex-1">{t('mcp.settings.claudeDesktop.quitFirst')}</span>
            <button onClick={() => setClaudePhase('idle')} className="text-gray-400 hover:text-white underline">
              {t('mcp.settings.claudeDesktop.cancelWait')}
            </button>
          </div>
        )}
        {claudePhase !== 'waitQuit' && connectInfo && <StatusLine ok={connectInfo.ok} text={connectInfo.text} />}
        <Snippet title={t('mcp.settings.claudeCode.title')} copyLabel={t('mcp.settings.copy')}
          text={claudeCodeCommand(cfg.port, cfg.token)} shown={claudeCodeCommand(cfg.port, shownToken)} />
        <Snippet title={t('mcp.settings.cursor.title')} copyLabel={t('mcp.settings.copy')}
          text={cursorJson(cfg.port, cfg.token)} shown={cursorJson(cfg.port, shownToken)} />
        <p className="text-[11px] text-gray-500 pb-2 flex items-center gap-1.5"><Terminal className="w-3.5 h-3.5" />{t('mcp.settings.apps.tokenNote')}</p>
      </Card>

      {/* Fernzugriff */}
      <Card icon={Cloud} title={t('mcp.settings.remote.title')} hint={t('mcp.settings.remote.hint')}
        right={<Toggle on={cfg.remoteEnabled} onChange={v => void save({ remoteEnabled: v })} label={t('mcp.settings.remote.title')} />}>
        {cfg.remoteEnabled ? (
          <div className="space-y-1">
            <StatusLine ok={status.relay.connected}
              text={status.relay.connected
                ? t('mcp.settings.remote.online', { time: fmt(status.relay.lastHeartbeat) })
                : status.relay.lastError
                  ? t('mcp.settings.remote.offlineError', { error: status.relay.lastError })
                  : t('mcp.settings.remote.connecting')} />
            <Snippet title={t('mcp.settings.remote.connectorUrl')} copyLabel={t('mcp.settings.copy')} text={REMOTE_MCP_URL} shown={REMOTE_MCP_URL} />
            <div className="flex items-center justify-between pt-1">
              <p className="text-xs font-medium text-gray-300">{t('mcp.settings.remote.clients')}</p>
              <button onClick={() => void loadRemote()} className="text-gray-500 hover:text-white" aria-label={t('mcp.settings.remote.refresh')} title={t('mcp.settings.remote.refresh')}>
                <RefreshCw className={`w-3.5 h-3.5 ${remoteLoading ? 'animate-spin' : ''}`} />
              </button>
            </div>
            {remoteError && <StatusLine ok={false} text={remoteError} />}
            {remote && remote.connections.length === 0 && <p className="text-xs text-gray-500">{t('mcp.settings.remote.noClients')}</p>}
            {remote?.connections.map((c, i) => (
              <div key={c.id ?? `${c.clientName}-${i}`} className="flex items-center justify-between text-xs py-1">
                <span className="text-white">{friendlyClientName(c.clientName)}</span>
                <span className="flex items-center gap-3 text-gray-500">
                  {t('mcp.settings.remote.lastUsed', { time: fmt(c.lastUsedAt ?? c.createdAt) })}
                  {c.id && (
                    <button onClick={() => void disconnectRemote(c.id!)} className="inline-flex items-center gap-1 text-gray-400 hover:text-red-300">
                      <Unplug className="w-3.5 h-3.5" />{t('mcp.settings.remote.disconnect')}
                    </button>
                  )}
                </span>
              </div>
            ))}
            {remote && remote.recentJobs.length > 0 && (
              <>
                <p className="text-xs font-medium text-gray-300 pt-2">{t('mcp.settings.remote.recent')}</p>
                {remote.recentJobs.slice(0, 10).map((j, i) => (
                  <div key={`${j.createdAt}-${i}`} className="grid grid-cols-[1fr_auto_auto] gap-3 text-xs py-0.5">
                    <span className="text-gray-300 font-mono truncate">{j.tool}</span>
                    <span className="text-gray-500">{friendlyClientName(j.clientName)} · {j.status}</span>
                    <span className="text-gray-600">{fmt(j.createdAt)}</span>
                  </div>
                ))}
              </>
            )}
          </div>
        ) : (
          <p className="text-xs text-gray-500 py-1">{t('mcp.settings.remote.offHint')}</p>
        )}
      </Card>

      {/* Letzte Aufrufe auf diesem Rechner */}
      <Card icon={History} title={t('mcp.settings.activity.title')} hint={t('mcp.settings.activity.hint')}>
        {status.recentCalls.length === 0
          ? <p className="text-xs text-gray-500 py-1">{t('mcp.settings.activity.empty')}</p>
          : status.recentCalls.slice(0, 12).map((c, i) => (
            <div key={`${c.at}-${i}`} className="grid grid-cols-[1fr_auto_auto] gap-3 text-xs py-0.5">
              <span className={`font-mono truncate ${c.isError ? 'text-amber-300' : 'text-gray-300'}`}>{c.tool}</span>
              <span className="text-gray-500">{friendlyClientName(c.clientName)} · {c.source === 'remote' ? t('mcp.settings.activity.remote') : t('mcp.settings.activity.local')}</span>
              <span className="text-gray-600">{fmt(c.at)}</span>
            </div>
          ))}
      </Card>

      {/* Freigaben */}
      <Card icon={ShieldCheck} title={t('mcp.settings.approvals.title')} hint={t('mcp.settings.approvals.hint')}>
        {grouped.length === 0
          ? <p className="text-xs text-gray-500 py-1">{t('mcp.settings.approvals.empty')}</p>
          : grouped.map(g => (
            <div key={g.client} className="py-1.5">
              <p className="text-xs font-medium text-white flex items-center gap-1.5"><KeyRound className="w-3.5 h-3.5 text-gray-400" />{g.client}</p>
              {g.items.map(a => (
                <div key={a.tool} className="flex items-center justify-between text-xs pl-5 py-0.5">
                  <span className="font-mono text-gray-300">{a.tool}</span>
                  <button onClick={() => void revoke(a)} className="inline-flex items-center gap-1 text-gray-400 hover:text-red-300"
                    aria-label={t('mcp.settings.approvals.revokeFor', { tool: a.tool, client: g.client })}>
                    <Trash2 className="w-3.5 h-3.5" />{t('mcp.settings.approvals.revoke')}
                  </button>
                </div>
              ))}
            </div>
          ))}
      </Card>

      {confirmRotate && (
        <ModalPortal>
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={() => setConfirmRotate(false)}>
            <div role="dialog" aria-modal="true" className="bg-white/10 backdrop-blur-sm rounded-2xl border border-white/10 max-w-md w-full p-6 space-y-4" onClick={e => e.stopPropagation()}>
              <h3 className="text-lg font-semibold text-white">{t('mcp.settings.local.rotateTitle')}</h3>
              <p className="text-sm text-gray-400 leading-relaxed">{t('mcp.settings.local.rotateText')}</p>
              <div className="flex justify-end gap-2">
                <button onClick={() => setConfirmRotate(false)} className="px-4 py-2 rounded-lg text-sm text-gray-300 hover:bg-white/5">{t('mcp.settings.cancel')}</button>
                <button onClick={() => void rotate()} className="px-4 py-2 rounded-lg text-sm bg-white/15 hover:bg-white/20 text-white">{t('mcp.settings.local.rotate')}</button>
              </div>
            </div>
          </div>
        </ModalPortal>
      )}
    </div>
  );
}
