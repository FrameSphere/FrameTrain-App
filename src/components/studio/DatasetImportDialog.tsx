// Ein vorhandenes Dataset ins Projekt holen — mit Klassenzuordnung und
// Stichprobe. Bisher konnte das nur ein KI-Client; in der Oberflaeche kam ein
// Dataset nur ganz oder gar nicht, und nur in Box-Projekte.

import { useEffect, useMemo, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { ArrowLeft, Database, Loader2 } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useNotification } from '../../contexts/NotificationContext';
import ModalPortal from '../ui/ModalPortal';
import { useEscape } from './useEscape';
import {
  importDatasetIntoProject, studioKindOf, type DatasetImportReport, type StagedResult,
} from './datasetImport';
import type { StudioProject } from './studioTypes';

export interface DatasetChoice {
  id: string;
  name: string;
  model_id?: string;
  storage_path: string;
  file_count: number;
  dataset_type?: string;
  extensions?: string[];
}

interface Inspection {
  classes?: string[];
  table?: { labels?: Record<string, number>; task?: string; columns?: string[] } | null;
}

const ENDUNGEN: Record<string, string[]> = {
  image: ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif', '.parquet'],
  audio: ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.webm', '.aiff', '.aif', '.parquet'],
  video: ['.mp4', '.mov', '.m4v', '.webm', '.mkv', '.avi'],
  text: ['.csv', '.tsv', '.jsonl', '.json', '.parquet', '.txt'],
};

/** Kann dieses Dataset Samples fuer das Projekt liefern? Ohne bekannte Endungen
 *  bleibt es waehlbar: dann laesst sich nichts sagen. */
export function passtZurQuelle(d: Pick<DatasetChoice, 'extensions' | 'dataset_type'>, p: Pick<StudioProject, 'modality' | 'task'>): boolean {
  // Klassenordner haben keine Boxen und umgekehrt.
  if (p.modality === 'image' && p.task === 'bbox' && d.dataset_type === 'folder_class') return false;
  if (p.modality === 'image' && p.task === 'classify' && (d.dataset_type === 'yolo_bbox' || d.dataset_type === 'pascal_voc')) return false;
  if (!d.extensions || d.extensions.length === 0) return true;
  const ok = new Set(ENDUNGEN[p.modality] ?? []);
  const endungen = d.extensions.map(e => (e.startsWith('.') ? e : `.${e}`).toLowerCase());
  // Ein Bild-Dataset bringt oft eine CSV mit; als Textquelle taugt es deshalb nicht.
  if (p.modality === 'text' && endungen.some(e => ENDUNGEN.image.includes(e) && e !== '.parquet')) return false;
  return endungen.some(e => ok.has(e));
}

/** '' = unter eigenem Namen, WEG = weglassen, sonst die Projektklasse. */
export const WEG = '\u0000weg';

/** Auswahl der Zeilen in class_map/keep_unmapped des Backends uebersetzen. */
export function zuordnung(
  quelle: string[], wahl: Record<string, string>, restWeg: boolean, projektklassen: string[],
): { classMap: Record<string, string | null>; keepUnmapped: boolean } {
  const classMap: Record<string, string | null> = {};
  for (const k of quelle) {
    const w = wahl[k] ?? '';
    if (w === WEG) classMap[k] = null;
    else if (w) classMap[k] = w;
    else if (restWeg) {
      // Gleicher Name im Projekt gilt als zugeordnet.
      const gleich = projektklassen.find(p => p.trim().toLowerCase() === k.trim().toLowerCase());
      if (gleich) classMap[k] = gleich;
    }
  }
  return { classMap, keepUnmapped: !restWeg };
}

export default function DatasetImportDialog({ project, onClose, onDone }: {
  project: StudioProject;
  onClose: () => void;
  onDone: (report: DatasetImportReport & { staged: StagedResult }) => void;
}) {
  const { t } = useLanguage();
  const { error } = useNotification();
  const [datasets, setDatasets] = useState<DatasetChoice[] | null>(null);
  const [modelle, setModelle] = useState<Record<string, string>>({});
  const [gewaehlt, setGewaehlt] = useState<DatasetChoice | null>(null);
  const [klassen, setKlassen] = useState<string[] | null>(null);
  const [wahl, setWahl] = useState<Record<string, string>>({});
  const [restWeg, setRestWeg] = useState(false);
  const [max, setMax] = useState('');
  const [ausgewogen, setAusgewogen] = useState(true);
  const [busy, setBusy] = useState(false);
  const mitKlassen = project.task !== 'pairs' && project.task !== 'transcript';

  useEffect(() => {
    invoke<DatasetChoice[]>('list_all_datasets')
      .then(l => setDatasets((Array.isArray(l) ? l : []).filter(d => d.storage_path && passtZurQuelle(d, project))))
      .catch(() => setDatasets([]));
    invoke<{ id: string; name: string }[]>('list_models')
      .then(l => setModelle(Object.fromEntries((Array.isArray(l) ? l : []).map(m => [m.id, m.name]))))
      .catch(() => { /* dann ohne Modellnamen */ });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.id]);

  useEffect(() => {
    if (!gewaehlt) return;
    let active = true;
    setKlassen(null); setWahl({}); setRestWeg(false);
    if (!mitKlassen) { setKlassen([]); return; }
    invoke<Inspection>('inspect_dataset', { datasetId: gewaehlt.id })
      .then(r => { if (active) setKlassen(r?.classes?.length ? r.classes : Object.keys(r?.table?.labels ?? {})); })
      // Ohne Klassenliste geht der Import trotzdem, nur ohne Zuordnung.
      .catch(() => { if (active) setKlassen([]); });
    return () => { active = false; };
  }, [gewaehlt, mitKlassen]);

  const run = async () => {
    if (!gewaehlt || !studioKindOf(project)) return;
    setBusy(true);
    try {
      const n = Number(max);
      const { classMap, keepUnmapped } = zuordnung(klassen ?? [], wahl, restWeg, project.classes);
      onDone(await importDatasetIntoProject(invoke, project, {
        datasetId: gewaehlt.id, datasetName: gewaehlt.name, classMap, keepUnmapped,
        maxSamples: Number.isFinite(n) && n > 0 ? Math.floor(n) : undefined,
        perClass: mitKlassen && ausgewogen && n > 0,
      }));
    } catch (err: unknown) {
      error(t('studio.import.errorTitle'), String(err));
      setBusy(false);
    }
  };

  const uebernommen = useMemo(() => {
    if (!klassen) return 0;
    const { classMap, keepUnmapped } = zuordnung(klassen, wahl, restWeg, project.classes);
    return klassen.filter(k => (k in classMap ? classMap[k] !== null : keepUnmapped)).length;
  }, [klassen, wahl, restWeg, project.classes]);

  useEscape(onClose, !busy);

  return (
    <ModalPortal><div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
      onClick={busy ? undefined : onClose}>
      <div className="w-full max-w-lg rounded-2xl border border-white/10 bg-[#101218] p-6 space-y-4 max-h-[85vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}>
        <div>
          <h3 className="text-white font-semibold">{t('studio.datasetImport.title')}</h3>
          <p className="text-gray-500 text-xs mt-1">
            {gewaehlt ? gewaehlt.name : t('studio.datasetImport.subtitle')}
          </p>
        </div>

        {!gewaehlt ? (
          <div className="space-y-1.5">
            {datasets === null && (
              <p className="text-gray-500 text-xs flex items-center gap-2">
                <Loader2 className="w-3.5 h-3.5 animate-spin" /> {t('common.loading', 'Lädt…')}
              </p>
            )}
            {datasets?.length === 0 && <p className="text-gray-500 text-xs">{t('studio.datasetImport.none')}</p>}
            {datasets?.map(d => (
              <button key={d.id} onClick={() => setGewaehlt(d)}
                className="w-full flex items-center gap-3 p-2.5 rounded-lg bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 text-left transition-all">
                <Database className="w-4 h-4 text-gray-400 flex-shrink-0" />
                <span className="min-w-0 flex-1">
                  <span className="text-gray-200 text-xs block truncate">{d.name}</span>
                  <span className="text-gray-500 text-[11px] block truncate">
                    {[d.model_id ? modelle[d.model_id] : null, t('studio.source.datasetMeta', { files: d.file_count, type: d.dataset_type ?? '' })]
                      .filter(Boolean).join(' · ')}
                  </span>
                </span>
              </button>
            ))}
          </div>
        ) : (
          <>
            <button onClick={() => setGewaehlt(null)} disabled={busy}
              className="text-gray-400 hover:text-white text-xs inline-flex items-center gap-1.5 disabled:opacity-40">
              <ArrowLeft className="w-3.5 h-3.5" /> {t('studio.source.back')}
            </button>

            {klassen === null ? (
              <p className="text-gray-500 text-xs flex items-center gap-2">
                <Loader2 className="w-3.5 h-3.5 animate-spin" /> {t('studio.datasetImport.reading')}
              </p>
            ) : klassen.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-gray-400 text-xs">{t('studio.datasetImport.mapTitle')}</span>
                  <span className="text-gray-500 text-[11px] tabular-nums">
                    {t('studio.datasetImport.mapCount', { cur: uebernommen, total: klassen.length })}
                  </span>
                </div>
                <div className="max-h-56 overflow-y-auto space-y-1 pr-1">
                  {klassen.map(k => (
                    <label key={k} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] items-center gap-2">
                      <span className="text-gray-200 text-xs truncate" title={k}>{k}</span>
                      <select value={wahl[k] ?? ''} onChange={e => setWahl(w => ({ ...w, [k]: e.target.value }))}
                        aria-label={t('studio.datasetImport.mapFor', { klasse: k })}
                        className="px-2 py-1 rounded-lg bg-white/5 border border-white/10 text-white text-xs focus:outline-none focus:border-white/25">
                        <option value="" className="bg-[#101218]">
                          {t(restWeg ? 'studio.datasetImport.mapAuto' : 'studio.datasetImport.mapOwn')}
                        </option>
                        {project.classes.map(c => <option key={c} value={c} className="bg-[#101218]">{c}</option>)}
                        <option value={WEG} className="bg-[#101218]">{t('studio.datasetImport.mapDrop')}</option>
                      </select>
                    </label>
                  ))}
                </div>
                <label className="flex items-start gap-2 cursor-pointer">
                  <input type="checkbox" checked={restWeg} onChange={e => setRestWeg(e.target.checked)} className="mt-0.5" />
                  <span className="text-gray-300 text-xs">{t('studio.datasetImport.dropRest')}</span>
                </label>
              </div>
            )}

            <div className="flex flex-wrap items-end gap-4">
              <label className="block">
                <span className="text-gray-400 text-xs block">{t('studio.datasetImport.maxLabel')}</span>
                <input type="number" min={1} value={max} onChange={e => setMax(e.target.value)}
                  placeholder={t('studio.datasetImport.maxAll')}
                  className="mt-1 w-32 px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm placeholder-gray-600 focus:outline-none focus:border-white/25" />
              </label>
              {mitKlassen && max && (
                <label className="flex items-center gap-2 cursor-pointer pb-2.5">
                  <input type="checkbox" checked={ausgewogen} onChange={e => setAusgewogen(e.target.checked)} />
                  <span className="text-gray-300 text-xs">{t('studio.datasetImport.balanced')}</span>
                </label>
              )}
            </div>
          </>
        )}

        <div className="flex gap-2">
          <button onClick={onClose} disabled={busy}
            className="flex-1 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white text-sm disabled:opacity-40">
            {t('common.cancel', 'Abbrechen')}
          </button>
          {gewaehlt && (
            <button onClick={() => void run()}
              disabled={busy || klassen === null || (klassen.length > 0 && uebernommen === 0)}
              className="flex-1 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm inline-flex items-center justify-center gap-2 disabled:opacity-40">
              {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Database className="w-4 h-4" />}
              {t('studio.import.startButton')}
            </button>
          )}
        </div>
      </div>
    </div></ModalPortal>
  );
}
