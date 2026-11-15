// Test-Oberflaeche fuer semantische Segmentierung.
//
// Einzelbild: Originalbild mit eingefaerbter Klassenmaske darueber (Deckkraft
// regelbar) und Legende mit Flaechenanteil je Klasse. Datensatz: mIoU,
// Pixel-Accuracy, IoU je Klasse und einige Ueberlagerungen. Die Engine liefert
// Pfade (image_path, color_mask_path, overlay_path) und die Legende.

import { useEffect, useRef, useState } from 'react';
import { invoke, convertFileSrc } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { AlertTriangle, Loader2, Play, Square } from 'lucide-react';
import type { TestPluginProps } from '../types';
import { useLanguage } from '../../contexts/LanguageContext';
import { engineErrorText, type EngineErrorData } from '../engineError';

export const TASK_TYPE = 'semantic_segmentation';

export interface LegendRow { id: number; label: string; color: string; share: number }
export interface ClassRow { id: number; label: string; color: string; iou: number | null; accuracy?: number | null; pixels?: number }

export interface SingleResult {
  imagePath: string;
  colorMaskPath: string;
  overlayPath: string;
  legend: LegendRow[];
  ms: number;
}

export interface DatasetSummary {
  total: number;
  split?: string;
  meanIou: number | null;
  pixelAccuracy: number | null;
  meanAccuracy: number | null;
  perClass: ClassRow[];
  overlays: string[];
}

/** Antwort der Engine (Einzelbild) in die Anzeigeform bringen. */
export function parseSingle(d: Record<string, unknown> | undefined): SingleResult | null {
  if (!d || typeof d.color_mask_path !== 'string' || typeof d.image_path !== 'string') return null;
  const legend = Array.isArray(d.legend) ? (d.legend as LegendRow[]) : [];
  return {
    imagePath: d.image_path,
    colorMaskPath: d.color_mask_path,
    overlayPath: typeof d.overlay_path === 'string' ? d.overlay_path : '',
    legend: legend.filter(r => r && typeof r.label === 'string'),
    ms: Math.round(Number(d.inference_time ?? 0) * 1000),
  };
}

/** Antwort der Engine (Datensatz) in die Anzeigeform bringen. */
export function parseDataset(d: Record<string, unknown> | undefined): DatasetSummary {
  const m = (d?.metrics ?? {}) as Record<string, unknown>;
  const num = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
  return {
    total: Number(d?.total_samples ?? 0),
    split: typeof d?.split === 'string' ? d.split : undefined,
    meanIou: num(m.mean_iou),
    pixelAccuracy: num(m.pixel_accuracy ?? d?.accuracy),
    meanAccuracy: num(m.mean_accuracy),
    perClass: Array.isArray(d?.per_class) ? (d!.per_class as ClassRow[]) : [],
    overlays: Array.isArray(d?.images) ? (d!.images as string[]).filter(p => typeof p === 'string') : [],
  };
}

export const pct = (v: number | null | undefined) => (v == null ? '—' : `${(v * 100).toFixed(1)} %`);

function Swatch({ color }: { color: string }) {
  return <span className="inline-block w-3 h-3 rounded-sm border border-white/20 flex-shrink-0" style={{ backgroundColor: color }} />;
}

export default function SemanticSegmentationTestPlugin({
  versionId, modelId, modelName, versionName, datasets,
}: TestPluginProps) {
  const { t } = useLanguage();
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [single, setSingle] = useState<SingleResult | null>(null);
  const [opacity, setOpacity] = useState(55);

  const [datasetId, setDatasetId] = useState(datasets[0]?.id ?? '');
  const [maxSamples, setMaxSamples] = useState<number | ''>(50);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<{ current: number; total: number } | null>(null);
  const [summary, setSummary] = useState<DatasetSummary | null>(null);

  const unlistenRef = useRef<Array<() => void>>([]);
  useEffect(() => () => { unlistenRef.current.forEach(fn => fn()); }, []);

  const onError = (data?: EngineErrorData) => {
    setError(engineErrorText(data, t, t('testPlugins.common.unknownError')));
    setStatus(null); setBusy(false); setRunning(false);
  };

  const runSingle = async () => {
    if (!input.trim()) { setError(t('testPlugins.segmentation.imageMissing')); return; }
    setError(null); setSingle(null); setStatus(null); setBusy(true);
    try {
      const testId = await invoke<string>('test_single_input', {
        versionId, singleInput: input.trim(), singleInputType: 'file', taskType: TASK_TYPE, pluginConfig: {},
      });
      const off = await listen<{ test_id: string; data?: Record<string, unknown> }>('test-single-complete', e => {
        if (e.payload.test_id !== testId) return;
        const r = parseSingle(e.payload.data);
        if (r) setSingle(r); else setError(t('testPlugins.segmentation.noMask'));
        setStatus(null); setBusy(false);
      });
      const offErr = await listen<{ data?: EngineErrorData }>('test-error', e => onError(e.payload.data));
      const offSt = await listen<{ data?: { message?: string } }>('test-status', e => setStatus(e.payload.data?.message ?? null));
      unlistenRef.current.push(off, offErr, offSt);
    } catch (e) { setError(String(e)); setBusy(false); }
  };

  const runDataset = async () => {
    const ds = datasets.find(d => d.id === datasetId);
    if (!ds) { setError(t('testPlugins.generic.noDatasetSelected')); return; }
    setError(null); setSummary(null); setProgress(null); setStatus(null); setRunning(true);
    try {
      await invoke('start_test', {
        modelId, modelName, versionId, versionName, datasetId: ds.id, datasetName: ds.name,
        batchSize: 1, maxSamples: maxSamples === '' ? null : maxSamples, taskType: TASK_TYPE, pluginConfig: {},
      });
      const offP = await listen<{ data?: { current_sample?: number; total_samples?: number } }>('test-progress', e => {
        const d = e.payload.data;
        if (d?.current_sample != null && d?.total_samples != null) setProgress({ current: d.current_sample, total: d.total_samples });
      });
      const offC = await listen<{ data?: Record<string, unknown> }>('test-complete', e => {
        setSummary(parseDataset(e.payload.data)); setStatus(null); setRunning(false);
      });
      const offE = await listen<{ data?: EngineErrorData }>('test-error', e => onError(e.payload.data));
      const offSt = await listen<{ data?: { message?: string } }>('test-status', e => setStatus(e.payload.data?.message ?? null));
      unlistenRef.current.push(offP, offC, offE, offSt);
    } catch (e) { setError(String(e)); setRunning(false); }
  };

  return (
    <div className="space-y-4">
      {error && (
        <div className="flex items-start gap-2 px-3 py-2 rounded-xl bg-red-500/10 border border-red-500/30">
          <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
          <p className="text-red-200/90 text-xs break-words">{error}</p>
        </div>
      )}

      <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-3">
        <p className="text-white font-medium text-sm">{t('testPlugins.segmentation.singleTitle')}</p>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder={t('testPlugins.generic.imagePlaceholder')}
          aria-label={t('testPlugins.segmentation.singleTitle')}
          className="w-full px-3 py-2 bg-slate-900/60 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-emerald-500/50"
        />
        <button
          onClick={runSingle}
          disabled={busy}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 text-emerald-200 text-xs font-medium disabled:opacity-50"
        >
          {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
          {t('testPlugins.segmentation.segment')}
        </button>
        {busy && <p className="text-gray-400 text-[11px]">{status ?? t('testPlugins.common.engineStarting')}</p>}

        {single && (
          <div className="rounded-xl bg-slate-900/60 border border-white/10 p-4 space-y-3" data-testid="seg-single">
            <div className="relative inline-block max-w-full">
              <img src={convertFileSrc(single.imagePath)} alt="" className="block max-h-96 w-auto rounded-lg" />
              <img
                src={convertFileSrc(single.colorMaskPath)}
                alt={t('testPlugins.segmentation.maskAlt')}
                className="absolute inset-0 w-full h-full rounded-lg pointer-events-none"
                style={{ opacity: opacity / 100, imageRendering: 'pixelated' }}
              />
            </div>
            <label className="flex items-center gap-3 text-[11px] text-gray-400">
              {t('testPlugins.segmentation.opacity')}
              <input type="range" min={0} max={100} value={opacity} onChange={e => setOpacity(Number(e.target.value))} className="flex-1" />
              <span className="tabular-nums w-10 text-right">{opacity} %</span>
            </label>
            <div className="space-y-1">
              <p className="text-gray-400 text-[11px]">{t('testPlugins.segmentation.legend')}</p>
              {single.legend.map(r => (
                <div key={r.id} className="flex items-center gap-2 text-[11px] text-gray-300">
                  <Swatch color={r.color} />
                  <span className="flex-1 break-words">{r.label}</span>
                  <span className="tabular-nums text-gray-400">{pct(r.share)}</span>
                </div>
              ))}
            </div>
            <p className="text-gray-500 text-[11px]">{single.ms} ms</p>
          </div>
        )}
      </div>

      <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-3">
        <p className="text-white font-medium text-sm">{t('testPlugins.generic.datasetTitle')}</p>
        {datasets.length === 0 ? (
          <p className="text-gray-500 text-xs">{t('testPlugins.common.noDatasetForModel')}</p>
        ) : (
          <>
            <div className="flex gap-2">
              <select value={datasetId} onChange={e => setDatasetId(e.target.value)}
                className="flex-1 px-3 py-2 bg-slate-900/60 border border-white/10 rounded-xl text-white text-xs">
                {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
              </select>
              <input type="number" min={1} value={maxSamples}
                onChange={e => setMaxSamples(e.target.value === '' ? '' : Number(e.target.value))}
                title={t('testPlugins.generic.maxSamplesTitle')}
                className="w-28 px-3 py-2 bg-slate-900/60 border border-white/10 rounded-xl text-white text-xs" />
            </div>
            <div className="flex gap-2">
              <button onClick={runDataset} disabled={running}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 text-emerald-200 text-xs font-medium disabled:opacity-50">
                {running ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                {t('testPlugins.generic.startTest')}
              </button>
              {running && (
                <button onClick={() => { invoke('stop_test').catch(() => {}); }}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl bg-red-500/15 hover:bg-red-500/25 border border-red-500/30 text-red-200 text-xs font-medium">
                  <Square className="w-3.5 h-3.5" /> {t('testPlugins.generic.stop')}
                </button>
              )}
            </div>
            {running && !progress && <p className="text-gray-400 text-[11px]">{status ?? t('testPlugins.common.engineStarting')}</p>}
            {progress && (
              <div className="space-y-1">
                <p className="text-gray-400 text-[11px]">{progress.current} / {progress.total}</p>
                <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
                  <div className="h-full rounded-full bg-emerald-500 transition-all"
                    style={{ width: `${progress.total ? (progress.current / progress.total) * 100 : 0}%` }} />
                </div>
              </div>
            )}

            {summary && (
              <div className="rounded-xl bg-slate-900/60 border border-white/10 p-4 text-xs text-gray-300 space-y-3" data-testid="seg-summary">
                <p>{t('testPlugins.generic.evaluated', { n: summary.total })}{summary.split ? ` · ${t('testPlugins.segmentation.split', { split: summary.split })}` : ''}</p>
                <div className="grid grid-cols-3 gap-2">
                  {([['mIoU', summary.meanIou], [t('testPlugins.segmentation.pixelAccuracy'), summary.pixelAccuracy], [t('testPlugins.segmentation.meanAccuracy'), summary.meanAccuracy]] as const).map(([label, v]) => (
                    <div key={label} className="rounded-lg bg-white/5 px-3 py-2">
                      <p className="text-gray-400 text-[11px]">{label}</p>
                      <p className="text-white text-sm font-medium tabular-nums">{pct(v)}</p>
                    </div>
                  ))}
                </div>
                {summary.perClass.length > 0 && (
                  <div className="space-y-1">
                    <p className="text-gray-400 text-[11px]">{t('testPlugins.segmentation.perClass')}</p>
                    {summary.perClass.map(c => (
                      <div key={c.id} className="flex items-center gap-2 text-[11px]">
                        <Swatch color={c.color} />
                        <span className="flex-1 break-words text-gray-300">{c.label}</span>
                        <span className="tabular-nums text-white">{c.iou == null ? t('testPlugins.segmentation.notPresent') : pct(c.iou)}</span>
                      </div>
                    ))}
                  </div>
                )}
                {summary.overlays.length > 0 && (
                  <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
                    {summary.overlays.slice(0, 12).map(p => (
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
