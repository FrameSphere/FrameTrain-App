// Daten aus dem Netz holen.
//
// Drei Wege: einzelne Adressen, eine Website ab Startseiten durchsuchen, oder
// die Seiten ihrer Sitemap. Eine Seitenadresse wird dabei nach dem
// durchsucht, was das Projekt braucht — Bilder, Audio, Video oder Absaetze.
// robots.txt wird befolgt, je Server gewartet, und jede Datei traegt Adresse,
// Fundseite und Lizenz mit. Ohne diese Herkunft darf ein gesammelter
// Datensatz dieses Geraet nie verlassen — und das merkt man sonst erst zu spaet.

import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { Loader2, Globe, ShieldCheck, ChevronDown, Square } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useNotification } from '../../contexts/NotificationContext';
import type { StudioProject } from './studioTypes';
import ModalPortal from '../ui/ModalPortal';
import { useEscape } from './useEscape';

export interface FetchReport {
  fetched:           number;
  duplicates:        number;
  blocked:           string[];
  failed:            { url: string; reason: string }[];
  skipped_type:      string[];
  pages_visited:     number;
  too_large:         string[];
  too_small:         number;
  outside_allowlist: number;
  limit_reached:     boolean;
  cancelled:         boolean;
}

export type FetchMode = 'urls' | 'crawl' | 'sitemap';

export interface WebOptions {
  mode: FetchMode;
  max_depth: number;
  max_pages: number;
  max_files: number;
  allowlist: string[];
  max_mb: number;
  min_side: number;
  paragraphs: boolean;
  min_chars: number;
  license: string | null;
  label: string | null;
}

/** Adressen aus einem Eingabefeld: je Zeile, mit Leerzeichen oder Komma getrennt. */
export function adressenAus(text: string): string[] {
  return text.split(/[\s,]+/).map(u => u.trim())
    .filter(u => u.startsWith('http://') || u.startsWith('https://'));
}

/** Startwerte je Projektart. Videos sind gross und wenige, Bilder klein und
 *  viele; Absaetze einer Website schnell ein paar tausend. Geladen wird als
 *  Stream auf die Platte — das Groessenlimit schuetzt nur noch den Speicherplatz. */
export function startGrenzen(modality: string): { files: number; mb: number; pages: number } {
  switch (modality) {
    case 'video': return { files: 100, mb: 4000, pages: 50 };
    case 'audio': return { files: 300, mb: 200, pages: 50 };
    case 'text':  return { files: 2000, mb: 25, pages: 50 };
    default:      return { files: 500, mb: 25, pages: 50 };
  }
}

type Fortschritt = { files: number; maxFiles: number; pages: number; maxPages: number; url?: string };

export default function FetchDialog({ project, onClose, onDone }: {
  project: StudioProject;
  onClose: () => void;
  onDone: () => void;
}) {
  const { t } = useLanguage();
  const { error } = useNotification();
  // Grund aus dem Backend ("timeout", "connect", "http 403" …) in Klartext.
  const warum = (reason: string) => {
    const code = reason.match(/^http (\d+)$/)?.[1];
    if (!code) return t(`studio.fetch.why.${reason}`);
    return ['403', '404', '429'].includes(code)
      ? t(`studio.fetch.why.http${code}`) : t('studio.fetch.why.http', { code });
  };
  const ist = project.modality;
  const [mode, setMode] = useState<FetchMode>('urls');
  const [text, setText] = useState('');
  const [license, setLicense] = useState('');
  const [allowlist, setAllowlist] = useState('');
  const [depth, setDepth] = useState(1);
  const start = startGrenzen(ist);
  const [maxPages, setMaxPages] = useState(start.pages);
  const [maxFiles, setMaxFiles] = useState(start.files);
  const [maxMb, setMaxMb] = useState(start.mb);
  const [minSide, setMinSide] = useState(64);
  const [paragraphs, setParagraphs] = useState(true);
  const [minChars, setMinChars] = useState(40);
  const [showLimits, setShowLimits] = useState(false);
  // Klasse fuer alles Geholte — nur wo es Klassen gibt (nicht bei Boxen,
  // Paaren und Transkripten).
  const klassenProjekt = project.classes.length > 0
    && !['bbox', 'pairs', 'transcript'].includes(project.task);
  const [label, setLabel] = useState('');
  const [busy, setBusy] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [fortschritt, setFortschritt] = useState<Fortschritt | null>(null);
  const [report, setReport] = useState<FetchReport | null>(null);

  const adressen = adressenAus(text);

  useEffect(() => {
    const un = listen<{ project_id: string; current: number; total: number; pages?: number; max_pages?: number; url?: string; done?: boolean }>(
      'studio-fetch-progress', ev => {
        if (ev.payload.project_id !== project.id) return;
        setFortschritt(ev.payload.done ? null : {
          files: ev.payload.current, maxFiles: ev.payload.total,
          pages: ev.payload.pages ?? 0, maxPages: ev.payload.max_pages ?? 0, url: ev.payload.url,
        });
      });
    return () => { void un.then(f => f()); };
  }, [project.id]);

  const run = async () => {
    if (adressen.length === 0) return;
    setBusy(true);
    setStopping(false);
    const options: WebOptions = {
      mode, max_depth: depth, max_pages: maxPages, max_files: maxFiles,
      allowlist: allowlist.split(/[\s,]+/).map(d => d.trim()).filter(Boolean),
      max_mb: maxMb, min_side: minSide, paragraphs, min_chars: minChars,
      license: license.trim() || null,
      label: klassenProjekt && label ? label : null,
    };
    try {
      setReport(await invoke<FetchReport>('studio_fetch_web', {
        projectId: project.id, urls: adressen, options,
      }));
    } catch (err: unknown) {
      error(t('studio.fetch.errorTitle'), String(err));
    } finally {
      setBusy(false);
      setFortschritt(null);
    }
  };

  const stop = async () => {
    setStopping(true);
    try { await invoke('studio_fetch_cancel'); } catch { /* der Lauf endet ohnehin */ }
  };

  // Nach "Anhalten" darf man den Dialog schliessen, auch wenn der Lauf noch
  // die aktuelle Datei fertig laedt — er endet ohnehin von selbst.
  useEscape(onClose, !busy || stopping);

  const untertitel = ist === 'text' ? 'studio.fetch.subtitleText'
    : ist === 'audio' ? 'studio.fetch.subtitleAudio'
      : ist === 'video' ? 'studio.fetch.subtitleVideo' : 'studio.fetch.subtitleImage';
  const zahl = (label: string, wert: number, setzen: (n: number) => void, min: number, max: number, step = 1) => (
    <label className="block">
      <span className="text-gray-400 text-[11px]">{label}</span>
      <input type="number" value={wert} min={min} max={max} step={step} disabled={busy}
        onChange={e => setzen(Math.min(max, Math.max(min, Number(e.target.value) || min)))}
        className="mt-1 w-full px-2 py-1.5 rounded-lg bg-white/5 border border-white/10 text-white text-xs focus:outline-none focus:border-white/25 tabular-nums" />
    </label>
  );

  return (
    <ModalPortal><div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
      onClick={busy && !stopping ? undefined : onClose}>
      <div className="w-full max-w-lg rounded-2xl border border-white/10 bg-[#101218] p-6 space-y-4 max-h-[85vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}>
        <div>
          <h3 className="text-white font-semibold">{t('studio.fetch.title')}</h3>
          <p className="text-gray-500 text-xs mt-1">{t(untertitel)}</p>
        </div>

        {report ? (
          <div className="space-y-3">
            <div className="rounded-lg bg-white/[0.04] border border-white/10 p-3 space-y-1.5">
              {([['fetched', report.fetched], ['duplicates', report.duplicates],
                 ['pagesVisited', report.pages_visited]] as const).map(([key, val]) => (
                <div key={key} className="flex items-center justify-between text-xs">
                  <span className="text-gray-400">{t(`studio.fetch.report.${key}`)}</span>
                  <span className="text-gray-200 tabular-nums">{val}</span>
                </div>
              ))}
              {([['blocked', report.blocked.length], ['failed', report.failed.length],
                 ['skippedType', report.skipped_type.length], ['tooLarge', report.too_large.length],
                 ['tooSmall', report.too_small], ['outsideAllowlist', report.outside_allowlist]] as const)
                .filter(([, n]) => n > 0).map(([key, n]) => (
                <div key={key} className="flex items-center justify-between text-xs">
                  <span className={key === 'blocked' ? 'text-amber-300' : 'text-gray-400'}>
                    {t(`studio.fetch.report.${key}`)}
                  </span>
                  <span className={`tabular-nums ${key === 'blocked' ? 'text-amber-300' : 'text-gray-200'}`}>{n}</span>
                </div>
              ))}
            </div>

            {(report.limit_reached || report.cancelled) && (
              <p className="text-gray-400 text-xs">
                {t(report.cancelled ? 'studio.fetch.report.cancelled' : 'studio.fetch.report.limitReached')}
              </p>
            )}

            {report.blocked.length > 0 && (
              <div className="rounded-lg bg-amber-500/10 border border-amber-500/25 p-3">
                <p className="text-amber-200 text-xs font-medium mb-1">{t('studio.fetch.blockedTitle')}</p>
                <p className="text-amber-200/80 text-[11px] break-all">{report.blocked.slice(0, 5).join(', ')}</p>
              </div>
            )}
            {report.failed.length > 0 && (
              <div className="rounded-lg bg-white/[0.04] border border-white/10 p-3 space-y-1" data-testid="fetch-failed">
                <p className="text-gray-300 text-xs font-medium">{t('studio.fetch.failedTitle')}</p>
                {report.failed.slice(0, 5).map(f => (
                  <p key={f.url} className="text-[11px] break-all">
                    <span className="text-gray-500">{f.url}</span>
                    <span className="text-gray-300"> – {warum(f.reason)}</span>
                  </p>
                ))}
              </div>
            )}
            {report.too_large.length > 0 && (
              <div className="rounded-lg bg-white/[0.04] border border-white/10 p-3">
                <p className="text-gray-300 text-xs font-medium mb-1">{t('studio.fetch.tooLargeTitle', { mb: maxMb })}</p>
                <p className="text-gray-500 text-[11px] break-all">{report.too_large.slice(0, 5).join(', ')}</p>
              </div>
            )}

            <button onClick={onDone}
              className="w-full py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm">
              {t('studio.suggest.report.close')}
            </button>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-3 gap-1.5" role="tablist">
              {(['urls', 'crawl', 'sitemap'] as const).map(m => (
                <button key={m} role="tab" aria-selected={mode === m} onClick={() => setMode(m)} disabled={busy}
                  className={`px-2 py-2 rounded-lg border text-xs transition-all ${mode === m ? 'bg-white/10 border-white/25 text-white' : 'bg-white/[0.03] border-white/10 text-gray-400 hover:bg-white/[0.06]'}`}>
                  {t(`studio.fetch.mode.${m}`)}
                </button>
              ))}
            </div>
            <p className="text-gray-500 text-[11px] -mt-2">{t(`studio.fetch.modeHint.${mode}`)}</p>

            <label className="block">
              <span className="text-gray-400 text-xs">
                {t(mode === 'urls' ? 'studio.fetch.urlsLabel' : mode === 'crawl' ? 'studio.fetch.startLabel' : 'studio.fetch.sitemapLabel')}
              </span>
              <textarea value={text} onChange={e => setText(e.target.value)} rows={mode === 'urls' ? 6 : 3} autoFocus disabled={busy}
                placeholder={mode === 'sitemap' ? 'https://example.org\nhttps://example.org/sitemap.xml' : 'https://…\nhttps://…'}
                className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-xs placeholder-gray-600 focus:outline-none focus:border-white/25 resize-none font-mono" />
            </label>

            {klassenProjekt && (
              <label className="block">
                <span className="text-gray-400 text-xs">{t('studio.fetch.labelLabel')}</span>
                <select value={label} onChange={e => setLabel(e.target.value)} disabled={busy}
                  className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-white/25">
                  <option value="" className="bg-[#101218]">{t('studio.fetch.labelNone')}</option>
                  {project.classes.map(c => <option key={c} value={c} className="bg-[#101218]">{c}</option>)}
                </select>
                <span className="text-gray-600 text-[11px]">{t('studio.fetch.labelHint')}</span>
              </label>
            )}

            {ist === 'text' && (
              <label className="flex items-start gap-2 cursor-pointer">
                <input type="checkbox" checked={paragraphs} onChange={e => setParagraphs(e.target.checked)} className="mt-0.5" disabled={busy} />
                <span className="text-gray-300 text-xs">{t('studio.fetch.paragraphsLabel')}</span>
              </label>
            )}

            <div className="rounded-lg border border-white/10">
              <button onClick={() => setShowLimits(v => !v)} className="w-full flex items-center gap-2 px-3 py-2 text-left">
                <span className="text-gray-400 text-xs flex-1">{t('studio.fetch.limitsTitle')}</span>
                <span className="text-gray-600 text-[11px]">
                  {[
                    mode !== 'urls' ? t('studio.fetch.limitPages', { n: maxPages }) : null,
                    t(ist === 'text' ? 'studio.fetch.limitParagraphs' : 'studio.fetch.limitFiles', { n: maxFiles }),
                    ist !== 'text' ? t('studio.fetch.limitMb', { n: maxMb >= 1000 ? `${(maxMb / 1000).toLocaleString()} GB` : `${maxMb} MB` }) : null,
                  ].filter(Boolean).join(' · ')}
                </span>
                <ChevronDown className={`w-3.5 h-3.5 text-gray-500 transition-transform ${showLimits ? 'rotate-180' : ''}`} />
              </button>
              {showLimits && (
                <div className="px-3 pb-3 space-y-3">
                  <p className="text-gray-500 text-[11px]">{t('studio.fetch.limitsWhy')}</p>
                  <div className="grid grid-cols-3 gap-2">
                    {mode === 'crawl' && zahl(t('studio.fetch.depthLabel'), depth, setDepth, 0, 5)}
                    {mode !== 'urls' && zahl(t('studio.fetch.maxPagesLabel'), maxPages, setMaxPages, 1, 5000)}
                    {zahl(t(ist === 'text' ? 'studio.fetch.maxParagraphsLabel' : 'studio.fetch.maxFilesLabel'), maxFiles, setMaxFiles, 1, 20000)}
                    {ist !== 'text' && zahl(t('studio.fetch.maxMbLabel'), maxMb, setMaxMb, 1, 50000)}
                    {ist === 'image' && zahl(t('studio.fetch.minSideLabel'), minSide, setMinSide, 0, 4000)}
                    {ist === 'text' && paragraphs && zahl(t('studio.fetch.minCharsLabel'), minChars, setMinChars, 1, 2000)}
                  </div>
                  <label className="block">
                    <span className="text-gray-400 text-[11px]">{t('studio.fetch.allowlistLabel')}</span>
                    <input value={allowlist} onChange={e => setAllowlist(e.target.value)} disabled={busy}
                      placeholder="example.org, wikimedia.org"
                      className="mt-1 w-full px-2 py-1.5 rounded-lg bg-white/5 border border-white/10 text-white text-xs placeholder-gray-600 focus:outline-none focus:border-white/25" />
                    <span className="text-gray-600 text-[11px]">
                      {t(mode === 'urls' ? 'studio.fetch.allowlistHintUrls' : 'studio.fetch.allowlistHint')}
                    </span>
                  </label>
                </div>
              )}
            </div>

            <label className="block">
              <span className="text-gray-400 text-xs">{t('studio.fetch.licenseLabel')}</span>
              <input value={license} onChange={e => setLicense(e.target.value)} disabled={busy}
                placeholder="CC-BY-4.0"
                className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm placeholder-gray-600 focus:outline-none focus:border-white/25" />
              <span className="text-gray-600 text-[11px]">{t('studio.fetch.licenseHint')}</span>
            </label>

            <div className="rounded-lg bg-white/[0.04] border border-white/10 p-3 flex items-start gap-2">
              <ShieldCheck className="w-4 h-4 text-gray-400 flex-shrink-0 mt-0.5" />
              <p className="text-gray-400 text-xs">{t('studio.fetch.manners')}</p>
            </div>

            <p className="text-gray-500 text-xs truncate" role="status">
              {busy
                ? fortschritt
                  ? t('studio.fetch.progressDetail', { pages: fortschritt.pages, files: fortschritt.files })
                    + (fortschritt.url ? ` · ${fortschritt.url}` : '')
                  : t('studio.suggest.starting')
                : t('studio.fetch.count', { count: adressen.length })}
            </p>

            <div className="flex gap-2">
              {busy ? (
                <button onClick={() => void stop()} disabled={stopping}
                  className="flex-1 py-2.5 rounded-xl bg-red-500/15 hover:bg-red-500/25 border border-red-500/30 text-red-200 text-sm inline-flex items-center justify-center gap-2 disabled:opacity-50">
                  <Square className="w-4 h-4" /> {t(stopping ? 'studio.fetch.stopping' : 'studio.fetch.stopButton')}
                </button>
              ) : (
                <button onClick={onClose}
                  className="flex-1 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white text-sm">
                  {t('common.cancel', 'Abbrechen')}
                </button>
              )}
              <button onClick={() => void run()} disabled={busy || adressen.length === 0}
                className="flex-1 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm inline-flex items-center justify-center gap-2 disabled:opacity-40">
                {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Globe className="w-4 h-4" />}
                {t('studio.fetch.runButton')}
              </button>
            </div>
          </>
        )}
      </div>
    </div></ModalPortal>
  );
}
