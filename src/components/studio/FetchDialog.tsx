// Adressen aus dem Netz holen.
//
// Kein Knopf, der das Netz absaugt: eine Liste von Adressen, robots.txt wird
// befolgt, je Server wird gewartet, und jede Datei traegt ihre Herkunft samt
// angegebener Lizenz mit. Ohne diese Herkunft darf ein gesammelter Datensatz
// dieses Geraet nie verlassen — und das merkt man sonst erst zu spaet.

import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { Loader2, Globe, ShieldCheck } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useNotification } from '../../contexts/NotificationContext';
import type { StudioProject } from './studioTypes';

export interface FetchReport {
  fetched:      number;
  duplicates:   number;
  blocked:      string[];
  failed:       string[];
  skipped_type: string[];
}

export default function FetchDialog({ project, onClose, onDone }: {
  project: StudioProject;
  onClose: () => void;
  onDone: () => void;
}) {
  const { t } = useLanguage();
  const { error } = useNotification();
  const [text, setText] = useState('');
  const [license, setLicense] = useState('');
  const [busy, setBusy] = useState(false);
  const [fortschritt, setFortschritt] = useState<{ cur: number; total: number } | null>(null);
  const [report, setReport] = useState<FetchReport | null>(null);

  const adressen = text.split(/[\s,]+/).map(u => u.trim())
    .filter(u => u.startsWith('http://') || u.startsWith('https://'));

  useEffect(() => {
    const un = listen<{ project_id: string; current: number; total: number; done?: boolean }>(
      'studio-fetch-progress', ev => {
        if (ev.payload.project_id !== project.id) return;
        setFortschritt(ev.payload.done ? null : { cur: ev.payload.current, total: ev.payload.total });
      });
    return () => { void un.then(f => f()); };
  }, [project.id]);

  const run = async () => {
    if (adressen.length === 0) return;
    setBusy(true);
    try {
      setReport(await invoke<FetchReport>('studio_fetch_urls', {
        projectId: project.id, urls: adressen, license: license.trim() || null,
      }));
    } catch (err: unknown) {
      error(t('studio.fetch.errorTitle'), String(err));
    } finally {
      setBusy(false);
      setFortschritt(null);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
      onClick={busy ? undefined : onClose}>
      <div className="w-full max-w-lg rounded-2xl border border-white/10 bg-[#101218] p-6 space-y-4 max-h-[85vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}>
        <div>
          <h3 className="text-white font-semibold">{t('studio.fetch.title')}</h3>
          <p className="text-gray-500 text-xs mt-1">
            {t(project.modality === 'text' ? 'studio.fetch.subtitleText' : 'studio.fetch.subtitleImage')}
          </p>
        </div>

        {report ? (
          <div className="space-y-3">
            <div className="rounded-lg bg-white/[0.04] border border-white/10 p-3 space-y-1.5">
              {([['fetched', report.fetched], ['duplicates', report.duplicates]] as const).map(([key, val]) => (
                <div key={key} className="flex items-center justify-between text-xs">
                  <span className="text-gray-400">{t(`studio.fetch.report.${key}`)}</span>
                  <span className="text-gray-200 tabular-nums">{val}</span>
                </div>
              ))}
              {([['blocked', report.blocked], ['failed', report.failed],
                 ['skippedType', report.skipped_type]] as const).map(([key, liste]) => liste.length > 0 && (
                <div key={key} className="flex items-center justify-between text-xs">
                  <span className={key === 'blocked' ? 'text-amber-300' : 'text-gray-400'}>
                    {t(`studio.fetch.report.${key}`)}
                  </span>
                  <span className={`tabular-nums ${key === 'blocked' ? 'text-amber-300' : 'text-gray-200'}`}>
                    {liste.length}
                  </span>
                </div>
              ))}
            </div>

            {report.blocked.length > 0 && (
              <div className="rounded-lg bg-amber-500/10 border border-amber-500/25 p-3">
                <p className="text-amber-200 text-xs font-medium mb-1">{t('studio.fetch.blockedTitle')}</p>
                <p className="text-amber-200/80 text-[11px] break-all">
                  {report.blocked.slice(0, 5).join(', ')}
                </p>
              </div>
            )}

            <button onClick={onDone}
              className="w-full py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm">
              {t('studio.suggest.report.close')}
            </button>
          </div>
        ) : (
          <>
            <label className="block">
              <span className="text-gray-400 text-xs">{t('studio.fetch.urlsLabel')}</span>
              <textarea value={text} onChange={e => setText(e.target.value)} rows={7} autoFocus
                placeholder={"https://…\nhttps://…"}
                className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-xs placeholder-gray-600 focus:outline-none focus:border-white/25 resize-none font-mono" />
            </label>

            <label className="block">
              <span className="text-gray-400 text-xs">{t('studio.fetch.licenseLabel')}</span>
              <input value={license} onChange={e => setLicense(e.target.value)}
                placeholder="CC-BY-4.0"
                className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm placeholder-gray-600 focus:outline-none focus:border-white/25" />
              <span className="text-gray-600 text-[11px]">{t('studio.fetch.licenseHint')}</span>
            </label>

            <div className="rounded-lg bg-white/[0.04] border border-white/10 p-3 flex items-start gap-2">
              <ShieldCheck className="w-4 h-4 text-gray-400 flex-shrink-0 mt-0.5" />
              <p className="text-gray-400 text-xs">{t('studio.fetch.manners')}</p>
            </div>

            <p className="text-gray-500 text-xs">
              {fortschritt
                ? t('studio.fetch.progress', { current: fortschritt.cur, total: fortschritt.total })
                : t('studio.fetch.count', { count: adressen.length })}
            </p>

            <div className="flex gap-2">
              <button onClick={onClose} disabled={busy}
                className="flex-1 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white text-sm disabled:opacity-40">
                {t('common.cancel', 'Abbrechen')}
              </button>
              <button onClick={() => void run()} disabled={busy || adressen.length === 0}
                className="flex-1 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm inline-flex items-center justify-center gap-2 disabled:opacity-40">
                {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Globe className="w-4 h-4" />}
                {t('studio.fetch.runButton')}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
