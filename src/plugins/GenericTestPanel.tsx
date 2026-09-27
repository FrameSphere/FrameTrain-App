// Gemeinsame Test-Oberfläche für die Plugins, die über die Test-Engine laufen.
//
// Bild, Audio und Seq2Seq unterscheiden sich nur in Kleinigkeiten: was als
// Einzel-Eingabe zählt (Dateipfad oder Text) und wie das Ergebnis heißt.
// Alles andere — Datensatz-Lauf, Fortschritt, Abbruch, Fehleranzeige — ist
// identisch und liegt deshalb hier statt dreimal kopiert.

import { useEffect, useRef, useState } from 'react';
import { invoke, convertFileSrc } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { AlertTriangle, Loader2, Play, Square } from 'lucide-react';
import type { TestPluginProps } from './types';
import { useLanguage } from '../contexts/LanguageContext';

interface TopPred { label?: string; score?: number }

interface GenericTestPanelProps extends TestPluginProps {
  taskType: string;
  /** 'text' = Freitext-Feld, 'file' = Pfad zu einer Datei. */
  inputKind: 'text' | 'file';
  singleLabel: string;
  singlePlaceholder: string;
  resultLabel: string;
  /** Seq2Seq liefert freien Text statt Klassen — dann keine Konfidenz zeigen. */
  showConfidence?: boolean;
  /**
   * Liste unter dem Ergebnis (Entitaeten, aehnlichste Texte) auch ohne
   * Konfidenz zeigen. Standard: wie showConfidence.
   */
  showTopList?: boolean;
  /** 'raw' zeigt Scores als 0.834 (Kosinus), 'percent' als 83.4 %. */
  scoreFormat?: 'percent' | 'raw';
  /**
   * Einzel-Eingabe bekommt das gewaehlte Dataset als corpus_path mit — das
   * Embedding-Plugin sucht dort die aehnlichsten Texte.
   */
  singleUsesDataset?: boolean;
  pluginConfig?: Record<string, unknown>;
  /**
   * 'image': das Ergebnis ist der Pfad eines erzeugten Bildes (Text-to-Image)
   * und wird als Bild angezeigt statt als Text.
   */
  resultKind?: 'text' | 'image';
  /**
   * Optionales zweites Feld (z. B. die Frage zum Bild bei VLMs). Der Wert geht
   * als plugin_config[configKey] an die Engine — single_input bleibt der Pfad,
   * Rust und die anderen Plugins merken davon nichts.
   */
  secondaryInput?: { label: string; placeholder: string; configKey: string };
}

/** recall_at_1 -> Recall@1, entity_f1 -> Entity F1 — lesbar ohne Uebersetzungstabelle. */
export function metricLabel(key: string): string {
  const s = key.replace(/_at_(\d+)/g, '@$1').replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export default function GenericTestPanel({
  versionId, modelId, modelName, versionName, datasets,
  taskType, inputKind, singleLabel, singlePlaceholder, resultLabel,
  showConfidence = true, showTopList, scoreFormat = 'percent', singleUsesDataset = false,
  resultKind = 'text', secondaryInput,
  pluginConfig = {},
}: GenericTestPanelProps) {
  const topList = showTopList ?? showConfidence;
  const fmtScore = (v?: number) => scoreFormat === 'raw'
    ? (v ?? 0).toFixed(3)
    : `${((v ?? 0) * 100).toFixed(1)} %`;
  const { t } = useLanguage();
  const [input, setInput] = useState('');
  const [secondary, setSecondary] = useState('');
  const [singleBusy, setSingleBusy] = useState(false);
  const [single, setSingle] = useState<{ predicted: string; confidence?: number; top: TopPred[]; ms: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Die Engine meldet waehrend des Ladens Status-Zeilen ("Lade Bildmodell...").
  // Ohne sie sieht der Nutzer bei einem langsamen Kaltstart nur einen Spinner.
  const [status, setStatus] = useState<string | null>(null);

  const [datasetId, setDatasetId] = useState(datasets[0]?.id ?? '');
  const [maxSamples, setMaxSamples] = useState<number | ''>(50);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<{ current: number; total: number } | null>(null);
  const [summary, setSummary] = useState<{
    total: number; accuracy: number | null; correct: number | null;
    metrics?: Record<string, number>; images?: string[];
  } | null>(null);

  const unlistenRef = useRef<Array<() => void>>([]);
  useEffect(() => () => { unlistenRef.current.forEach(fn => fn()); }, []);

  const runSingle = async () => {
    if (!input.trim()) { setError(t('testPlugins.generic.inputMissing', { label: singleLabel })); return; }
    setError(null); setSingle(null); setStatus(null); setSingleBusy(true);
    try {
      const corpus = singleUsesDataset ? datasets.find(d => d.id === datasetId)?.storage_path : undefined;
      const testId = await invoke<string>('test_single_input', {
        versionId,
        singleInput: input.trim(),
        singleInputType: inputKind,
        taskType,
        pluginConfig: {
          ...pluginConfig,
          ...(corpus ? { corpus_path: corpus } : {}),
          ...(secondaryInput && secondary.trim() ? { [secondaryInput.configKey]: secondary.trim() } : {}),
        },
      });
      const off = await listen<{ test_id: string; data?: { predicted_output?: string; confidence?: number; top_predictions?: TopPred[]; inference_time?: number } }>(
        'test-single-complete', e => {
          if (e.payload.test_id !== testId) return;
          const d = e.payload.data;
          setSingle({
            predicted: d?.predicted_output ?? '—',
            confidence: d?.confidence ?? undefined,
            top: d?.top_predictions ?? [],
            ms: Math.round((d?.inference_time ?? 0) * 1000),
          });
          setStatus(null);
          setSingleBusy(false);
        });
      const offErr = await listen<{ test_id?: string; data?: { error?: string } }>(
        'test-error', e => {
          setError(e.payload.data?.error ?? t('testPlugins.common.unknownError'));
          setStatus(null);
          setSingleBusy(false);
        });
      const offSt = await listen<{ data?: { message?: string } }>(
        'test-status', e => { setStatus(e.payload.data?.message ?? null); });
      unlistenRef.current.push(off, offErr, offSt);
    } catch (e) {
      setError(String(e)); setSingleBusy(false);
    }
  };

  const runDataset = async () => {
    const ds = datasets.find(d => d.id === datasetId);
    if (!ds) { setError(t('testPlugins.generic.noDatasetSelected')); return; }
    setError(null); setSummary(null); setProgress(null); setStatus(null); setRunning(true);
    try {
      const job = await invoke<{ id: string }>('start_test', {
        modelId, modelName, versionId, versionName,
        datasetId: ds.id, datasetName: ds.name,
        batchSize: 8,
        maxSamples: maxSamples === '' ? null : maxSamples,
        taskType,
        pluginConfig,
      });
      const offP = await listen<{ test_id?: string; data?: { current_sample?: number; total_samples?: number } }>(
        'test-progress', e => {
          const d = e.payload.data;
          if (d?.current_sample != null && d?.total_samples != null) {
            setProgress({ current: d.current_sample, total: d.total_samples });
          }
        });
      const offC = await listen<{ test_id?: string; data?: {
        total_samples?: number; accuracy?: number | null; correct_predictions?: number | null;
        metrics?: Record<string, unknown>; images?: string[];
      } }>(
        'test-complete', e => {
          const d = e.payload.data;
          // Zusatzkennzahlen (Entitaeten-F1, Recall@k, ROUGE-L …) — nur Zahlen.
          const metrics = d?.metrics ? Object.fromEntries(
            Object.entries(d.metrics).filter(([, v]) => typeof v === 'number'),
          ) as Record<string, number> : undefined;
          setSummary({
            total: d?.total_samples ?? 0,
            accuracy: d?.accuracy ?? null,
            correct: d?.correct_predictions ?? null,
            metrics,
            images: Array.isArray(d?.images) ? d.images : undefined,
          });
          setStatus(null);
          setRunning(false);
        });
      const offE = await listen<{ data?: { error?: string } }>('test-error', e => {
        setError(e.payload.data?.error ?? t('testPlugins.common.unknownError'));
        setStatus(null);
        setRunning(false);
      });
      const offSt = await listen<{ data?: { message?: string } }>(
        'test-status', e => { setStatus(e.payload.data?.message ?? null); });
      unlistenRef.current.push(offP, offC, offE, offSt);
      void job;
    } catch (e) {
      setError(String(e)); setRunning(false);
    }
  };

  return (
    <div className="space-y-4">
      {error && (
        <div className="flex items-start gap-2 px-3 py-2 rounded-xl bg-red-500/10 border border-red-500/30">
          <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
          <p className="text-red-200/90 text-xs break-words">{error}</p>
        </div>
      )}

      {/* Einzel-Eingabe */}
      <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-3">
        <p className="text-white font-medium text-sm">{singleLabel}</p>
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder={singlePlaceholder}
          rows={inputKind === 'text' ? 3 : 1}
          className="w-full px-3 py-2 bg-slate-900/60 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-emerald-500/50"
        />
        {secondaryInput && (
          <div className="space-y-1">
            <p className="text-gray-400 text-[11px]">{secondaryInput.label}</p>
            <input
              value={secondary}
              onChange={e => setSecondary(e.target.value)}
              placeholder={secondaryInput.placeholder}
              aria-label={secondaryInput.label}
              className="w-full px-3 py-2 bg-slate-900/60 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-emerald-500/50"
            />
          </div>
        )}
        <button
          onClick={runSingle}
          disabled={singleBusy}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 text-emerald-200 text-xs font-medium disabled:opacity-50"
        >
          {singleBusy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
          {t('testPlugins.generic.evaluate')}
        </button>

        {singleBusy && (
          <p className="text-gray-400 text-[11px]">
            {status ?? t('testPlugins.common.engineStarting')}
          </p>
        )}

        {single && (
          <div className="rounded-xl bg-slate-900/60 border border-white/10 p-4 space-y-2">
            <p className="text-gray-400 text-[11px]">{resultLabel}</p>
            {resultKind === 'image' && single.predicted !== '—' ? (
              <div className="space-y-1">
                <img
                  src={convertFileSrc(single.predicted)}
                  alt={input}
                  className="max-h-80 w-auto rounded-lg border border-white/10"
                />
                <p className="text-gray-500 text-[11px] break-all">{single.predicted}</p>
              </div>
            ) : (
              <p className="text-white text-sm break-words">{single.predicted}</p>
            )}
            <p className="text-gray-500 text-[11px]">
              {showConfidence && single.confidence != null
                ? t('testPlugins.generic.confidence', { value: (single.confidence * 100).toFixed(1) })
                : ''}
              {single.ms} ms
            </p>
            {topList && single.top.length > (showConfidence ? 1 : 0) && (
              <div className="space-y-1 pt-1">
                {single.top.map((pred, i) => (
                  <div key={i} className="flex items-center justify-between gap-3 text-[11px] text-gray-400">
                    <span className="break-words">{pred.label}</span>
                    <span className="tabular-nums flex-shrink-0">{fmtScore(pred.score)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Datensatz-Lauf */}
      <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-3">
        <p className="text-white font-medium text-sm">{t('testPlugins.generic.datasetTitle')}</p>
        {datasets.length === 0 ? (
          <p className="text-gray-500 text-xs">{t('testPlugins.common.noDatasetForModel')}</p>
        ) : (
          <>
            <div className="flex gap-2">
              <select
                value={datasetId}
                onChange={e => setDatasetId(e.target.value)}
                className="flex-1 px-3 py-2 bg-slate-900/60 border border-white/10 rounded-xl text-white text-xs"
              >
                {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
              </select>
              <input
                type="number"
                value={maxSamples}
                onChange={e => setMaxSamples(e.target.value === '' ? '' : Number(e.target.value))}
                min={1}
                className="w-28 px-3 py-2 bg-slate-900/60 border border-white/10 rounded-xl text-white text-xs"
                title={t('testPlugins.generic.maxSamplesTitle')}
              />
            </div>
            <div className="flex gap-2">
              <button
                onClick={runDataset}
                disabled={running}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 text-emerald-200 text-xs font-medium disabled:opacity-50"
              >
                {running ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                {t('testPlugins.generic.startTest')}
              </button>
              {running && (
                <button
                  onClick={() => { invoke('stop_test').catch(() => {}); }}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl bg-red-500/15 hover:bg-red-500/25 border border-red-500/30 text-red-200 text-xs font-medium"
                >
                  <Square className="w-3.5 h-3.5" /> {t('testPlugins.generic.stop')}
                </button>
              )}
            </div>

            {running && !progress && (
              <p className="text-gray-400 text-[11px]">
                {status ?? t('testPlugins.common.engineStarting')}
              </p>
            )}

            {progress && (
              <div className="space-y-1">
                <p className="text-gray-400 text-[11px]">{progress.current} / {progress.total}</p>
                <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-emerald-500 transition-all"
                    style={{ width: `${progress.total ? (progress.current / progress.total) * 100 : 0}%` }}
                  />
                </div>
              </div>
            )}

            {summary && (
              <div className="rounded-xl bg-slate-900/60 border border-white/10 p-4 text-xs text-gray-300 space-y-1">
                <p>{t('testPlugins.generic.evaluated', { n: summary.total })}</p>
                {summary.accuracy != null
                  ? <p className="text-white font-medium">{t('testPlugins.generic.hits', { value: (summary.accuracy * 100).toFixed(1), correct: summary.correct ?? 0 })}</p>
                  : !summary.metrics && <p className="text-gray-500">{t('testPlugins.generic.noExpected')}</p>}
                {summary.metrics && Object.keys(summary.metrics).length > 0 && (
                  <div className="pt-1 space-y-0.5">
                    <p className="text-gray-400 text-[11px]">{t('testPlugins.generic.metrics')}</p>
                    {Object.entries(summary.metrics).map(([k, v]) => (
                      <div key={k} className="flex items-center justify-between text-[11px]">
                        <span className="text-gray-400">{metricLabel(k)}</span>
                        <span className="tabular-nums text-white">{typeof v === 'number' ? v.toFixed(3) : String(v)}</span>
                      </div>
                    ))}
                  </div>
                )}
                {summary.images && summary.images.length > 0 && (
                  <div className="grid grid-cols-3 sm:grid-cols-4 gap-2 pt-2">
                    {summary.images.slice(0, 12).map(p => (
                      <img key={p} src={convertFileSrc(p)} alt="" className="w-full rounded-md border border-white/10" />
                    ))}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
