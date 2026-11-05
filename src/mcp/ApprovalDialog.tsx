// Freigabe-Dialog fuer Werkzeug-Aufrufe von KI-Clients.
//
// Haengt einmal global in App.tsx. Zeigt immer die aelteste offene Anfrage:
// welcher Client, welches Werkzeug, mit welchen Argumenten. Antworten:
// Erlauben, Ablehnen, oder (nicht bei Loeschen) "Fuer diesen Client immer
// erlauben" — widerrufbar im Einstellungs-Tab "KI-Verbindungen".

import { useEffect, useState } from 'react';
import { Bot, Check, Globe, Laptop, ShieldAlert, ShieldCheck, Trash2, X } from 'lucide-react';
import ModalPortal from '../components/ui/ModalPortal';
import { useLanguage } from '../contexts/LanguageContext';
import { answerApproval, isDestructive, subscribeApprovals } from './approval';
import type { ApprovalRequest } from './types';

/** Argumente als lesbare Zeilen; lange Werte (Skripte, Bilder) gekuerzt. */
export function formatArgs(args: Record<string, unknown>, max = 400): { key: string; value: string }[] {
  return Object.entries(args).map(([key, v]) => {
    let value: string;
    if (typeof v === 'string') value = v;
    else {
      try { value = JSON.stringify(v, null, 1); } catch { value = String(v); }
    }
    if (key.endsWith('base64') && typeof v === 'string') value = `(${Math.round(v.length * 0.75 / 1024)} KB)`;
    if (value.length > max) value = `${value.slice(0, max)} … (+${value.length - max})`;
    return { key, value };
  });
}

/** Werkzeugtitel in der UI-Sprache; das Manifest selbst bleibt englisch (fuer die KI). */
export function toolTitle(t: (key: string) => string, name: string, fallback: string): string {
  const key = `mcp.tools.titles.${name}`;
  const translated = t(key);
  return translated && translated !== key ? translated : fallback;
}

export default function ApprovalDialog() {
  const { t } = useLanguage();
  const [request, setRequest] = useState<ApprovalRequest | null>(null);
  const [waiting, setWaiting] = useState(0);

  useEffect(() => subscribeApprovals((head, count) => {
    setRequest(head);
    setWaiting(count);
  }), []);

  useEffect(() => {
    if (!request) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') answerApproval(request.id, 'deny');
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [request]);

  if (!request) return null;
  const destructive = isDestructive(request.tool);
  const rows = formatArgs(request.args);
  const SourceIcon = request.source === 'remote' ? Globe : Laptop;

  return (
    <ModalPortal>
      <div
        className="fixed inset-0 z-[10000] flex items-center justify-center p-4"
        style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(6px)' }}
        role="dialog"
        aria-modal="true"
        aria-labelledby="mcp-approval-title"
      >
        <div className="ft-sheet w-full max-w-lg rounded-2xl border border-white/10 bg-slate-900 shadow-2xl overflow-hidden">
          <div className="p-6 space-y-4">
            <div className="flex items-start gap-3">
              <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 border ${destructive ? 'bg-red-500/15 border-red-500/30' : 'bg-violet-500/15 border-violet-500/30'}`}>
                {destructive ? <ShieldAlert className="w-5 h-5 text-red-400" /> : <Bot className="w-5 h-5 text-violet-300" />}
              </div>
              <div className="min-w-0">
                <h2 id="mcp-approval-title" className="text-white font-semibold text-lg leading-tight">
                  {t('mcp.tools.approval.title', { client: request.client.name })}
                </h2>
                <p className="text-sm text-gray-400 mt-1 flex items-center gap-1.5">
                  <SourceIcon className="w-3.5 h-3.5" />
                  <span>{t(request.source === 'remote' ? 'mcp.tools.approval.sourceRemote' : 'mcp.tools.approval.sourceLocal')}</span>
                  {request.client.version && <span className="text-gray-500">· {request.client.version}</span>}
                </p>
              </div>
            </div>

            <div className="rounded-xl border border-white/10 bg-white/5 p-3 space-y-1">
              <div className="text-xs uppercase tracking-wide text-gray-500">{t('mcp.tools.approval.action')}</div>
              <div className="text-white font-medium">{toolTitle(t, request.tool.name, request.tool.title)}</div>
              <code className="text-xs text-gray-400">{request.tool.name}</code>
            </div>

            {rows.length > 0 && (
              <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500 mb-2">{t('mcp.tools.approval.arguments')}</div>
                <dl className="space-y-1.5 max-h-56 overflow-auto text-sm">
                  {rows.map((r) => (
                    <div key={r.key} className="grid grid-cols-[minmax(0,9rem)_1fr] gap-2">
                      <dt className="text-gray-400 truncate" title={r.key}>{r.key}</dt>
                      <dd className="text-gray-100 font-mono text-xs whitespace-pre-wrap break-all">{r.value}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            )}

            {destructive && (
              <p className="text-sm text-red-300 flex items-center gap-2">
                <Trash2 className="w-4 h-4 flex-shrink-0" />
                {t('mcp.tools.approval.destructiveHint')}
              </p>
            )}
            {waiting > 1 && (
              <p className="text-xs text-gray-500">{t('mcp.tools.approval.queue', { count: waiting - 1 })}</p>
            )}

            <div className="flex flex-col sm:flex-row gap-2 pt-1">
              <button
                type="button"
                className="ft-btn ft-btn-ghost flex-1 inline-flex items-center justify-center gap-2"
                onClick={() => answerApproval(request.id, 'deny')}
              >
                <X className="w-4 h-4" />{t('mcp.tools.approval.deny')}
              </button>
              <button
                type="button"
                autoFocus
                className={`ft-btn flex-1 inline-flex items-center justify-center gap-2 ${destructive ? 'bg-red-600 hover:bg-red-500 text-white' : 'ft-btn-primary'}`}
                onClick={() => answerApproval(request.id, 'allow')}
              >
                <Check className="w-4 h-4" />{t('mcp.tools.approval.allow')}
              </button>
            </div>
            {request.canRemember && (
              <button
                type="button"
                className="w-full text-sm text-gray-400 hover:text-white inline-flex items-center justify-center gap-2 py-1"
                onClick={() => answerApproval(request.id, 'always')}
              >
                <ShieldCheck className="w-4 h-4" />{t('mcp.tools.approval.always', { client: request.client.name })}
              </button>
            )}
            <p className="text-xs text-gray-500 text-center">{t('mcp.tools.approval.revokeHint')}</p>
          </div>
        </div>
      </div>
    </ModalPortal>
  );
}
