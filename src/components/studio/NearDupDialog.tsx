// Beinahe-Dubletten finden und aussortieren.
//
// Exakte Kopien faengt der Inhalts-Hash schon beim Import. Beinahe gleiche —
// dasselbe Foto verkleinert, derselbe Satz mit anderem Satzzeichen — fallen
// erst hier auf. Liegen sie spaeter in Train und Val, misst die Validierung
// Wiedererkennen statt Koennen. Je Gruppe bleibt das erste Sample, die
// anderen sind zum Entfernen vorgemerkt; entfernt wird erst auf Knopfdruck.

import { useState, useEffect, useCallback } from 'react';
import { invoke, convertFileSrc } from '@tauri-apps/api/core';
import { Loader2, Trash2 } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useNotification } from '../../contexts/NotificationContext';
import type { StudioProject, StudioSample, SamplePage } from './studioTypes';
import ModalPortal from '../ui/ModalPortal';
import { useEscape } from './useEscape';
import { computeDHash } from './dhash';

export interface NearDupReport {
  kind:    'text' | 'image' | 'exact_only';
  groups:  string[][];
  checked: number;
  missing: StudioSample[];
}

/** Fehlende Bild-Hashes rechnen und ablegen; gibt die Zahl der neuen zurueck. */
export async function hashesNachrechnen(
  projectId: string, missing: StudioSample[],
  hash: (url: string) => Promise<string> = computeDHash,
  onProgress?: (done: number, total: number) => void,
): Promise<number> {
  const neu: Record<string, string> = {};
  let fertig = 0;
  for (const s of missing) {
    try { neu[s.media] = await hash(convertFileSrc(s.abs_path)); }
    catch { /* unlesbares Bild: bleibt ohne Hash und zaehlt nicht mit */ }
    fertig += 1;
    if (fertig % 10 === 0) onProgress?.(fertig, missing.length);
  }
  if (Object.keys(neu).length > 0) await invoke('studio_save_hashes', { projectId, hashes: neu });
  return Object.keys(neu).length;
}

export default function NearDupDialog({ project, onClose, onDone, hash }: {
  project: StudioProject; onClose: () => void; onDone: (removed: number) => void;
  /** Fuer Tests austauschbar. */
  hash?: (url: string) => Promise<string>;
}) {
  const { t } = useLanguage();
  const { error, success } = useNotification();
  const [report, setReport] = useState<NearDupReport | null>(null);
  const [byId, setById] = useState<Map<string, StudioSample>>(new Map());
  const [weg, setWeg] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(true);
  const [hashing, setHashing] = useState<{ done: number; total: number } | null>(null);

  const pruefen = useCallback(async () => {
    setBusy(true);
    try {
      let r = await invoke<NearDupReport>('studio_near_duplicates', { projectId: project.id });
      if (r.kind === 'image' && r.missing.length > 0) {
        setHashing({ done: 0, total: r.missing.length });
        await hashesNachrechnen(project.id, r.missing, hash, (done, total) => setHashing({ done, total }));
        setHashing(null);
        r = await invoke<NearDupReport>('studio_near_duplicates', { projectId: project.id });
      }
      // Die Samples der Gruppen fuer die Vorschau — seitenweise, bis alle da sind.
      const gesucht = new Set(r.groups.flat());
      const m = new Map<string, StudioSample>();
      for (let offset = 0; gesucht.size > 0; offset += 500) {
        const page = await invoke<SamplePage>('studio_list_samples', {
          projectId: project.id, status: 'all', offset, limit: 500,
        });
        for (const s of page.items) if (gesucht.delete(s.id)) m.set(s.id, s);
        if (offset + 500 >= page.total) break;
      }
      setById(m);
      setReport(r);
      setWeg(new Set(r.groups.flatMap(g => g.slice(1))));
    } catch (err: unknown) {
      error(t('studio.nearDup.errorTitle'), String(err));
      onClose();
    } finally {
      setBusy(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.id]);

  useEffect(() => { void pruefen(); }, [pruefen]);
  useEscape(onClose, !busy);

  const entfernen = async () => {
    if (weg.size === 0) return;
    setBusy(true);
    try {
      await invoke('studio_delete_samples', { projectId: project.id, sampleIds: [...weg] });
      success(t('studio.nearDup.removedTitle'), t('studio.nearDup.removedDetail', { count: weg.size }));
      onDone(weg.size);
    } catch (err: unknown) {
      error(t('studio.remove.errorTitle'), String(err));
      setBusy(false);
    }
  };

  const vorschau = (id: string) => {
    const s = byId.get(id);
    if (!s) return <span className="text-gray-500 text-xs">{id}</span>;
    if (s.content != null) return <span className="text-gray-300 text-xs line-clamp-3">{s.content}</span>;
    if (s.mime.startsWith('image/')) {
      return <img src={convertFileSrc(s.abs_path)} alt="" className="h-20 w-auto rounded border border-white/10 object-contain" />;
    }
    return <span className="text-gray-300 text-xs">{s.src.origin?.split(/[\\/]/).pop() ?? id}</span>;
  };

  return (
    <ModalPortal><div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
      onClick={busy ? undefined : onClose}>
      <div className="w-full max-w-2xl rounded-2xl border border-white/10 bg-[#101218] p-6 space-y-4 max-h-[85vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}>
        <div>
          <h3 className="text-white font-semibold">{t('studio.nearDup.title')}</h3>
          <p className="text-gray-500 text-xs mt-1">{t('studio.nearDup.subtitle')}</p>
        </div>

        {busy && !report ? (
          <p className="text-gray-400 text-xs inline-flex items-center gap-2" role="status">
            <Loader2 className="w-4 h-4 animate-spin" />
            {hashing ? t('studio.nearDup.hashing', { done: hashing.done, total: hashing.total }) : t('studio.nearDup.checking')}
          </p>
        ) : report && report.kind === 'exact_only' ? (
          <p className="text-gray-400 text-xs">{t('studio.nearDup.exactOnly')}</p>
        ) : report && report.groups.length === 0 ? (
          <p className="text-emerald-300/80 text-xs">{t('studio.nearDup.none', { count: report.checked })}</p>
        ) : report && (
          <div className="space-y-3">
            <p className="text-gray-400 text-xs">
              {t('studio.nearDup.found', { groups: report.groups.length, checked: report.checked })}
            </p>
            {report.groups.map((g, gi) => (
              <div key={gi} className="rounded-lg border border-white/10 bg-white/[0.03] p-3 space-y-2">
                {g.map((id, i) => (
                  <label key={id} className="flex items-start gap-3 cursor-pointer">
                    <input type="checkbox" checked={weg.has(id)} className="mt-1"
                      aria-label={t('studio.nearDup.markRemove')}
                      onChange={e => setWeg(prev => {
                        const n = new Set(prev);
                        if (e.target.checked) n.add(id); else n.delete(id);
                        return n;
                      })} />
                    <div className="min-w-0 flex-1">{vorschau(id)}</div>
                    {i === 0 && <span className="text-emerald-300/80 text-[11px] whitespace-nowrap">{t('studio.nearDup.keep')}</span>}
                  </label>
                ))}
              </div>
            ))}
          </div>
        )}

        <div className="flex gap-2">
          <button onClick={onClose} disabled={busy && !report}
            className="flex-1 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white text-sm disabled:opacity-40">
            {t('studio.suggest.report.close')}
          </button>
          {report && report.groups.length > 0 && (
            <button onClick={() => void entfernen()} disabled={busy || weg.size === 0}
              className="flex-1 py-2.5 rounded-xl bg-red-500/15 hover:bg-red-500/25 border border-red-500/30 text-red-200 text-sm inline-flex items-center justify-center gap-2 disabled:opacity-40">
              {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
              {t('studio.nearDup.removeButton', { count: weg.size })}
            </button>
          )}
        </div>
      </div>
    </div></ModalPortal>
  );
}
