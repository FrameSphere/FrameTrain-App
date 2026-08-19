// Gemeinsame Test-Oberfläche für die Plugins, die über die Test-Engine laufen.
//
// Bild, Audio und Seq2Seq unterscheiden sich nur in Kleinigkeiten: was als
// Einzel-Eingabe zählt (Dateipfad oder Text) und wie das Ergebnis heißt.
// Alles andere — Datensatz-Lauf, Fortschritt, Abbruch, Fehleranzeige — ist
// identisch und liegt deshalb hier statt dreimal kopiert.

import { useEffect, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { AlertTriangle, Loader2, Play, Square } from 'lucide-react';
import type { TestPluginProps } from './types';

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
  pluginConfig?: Record<string, unknown>;
}

export default function GenericTestPanel({
  versionId, modelId, modelName, versionName, datasets,
  taskType, inputKind, singleLabel, singlePlaceholder, resultLabel,
  showConfidence = true, pluginConfig = {},
}: GenericTestPanelProps) {
  const [input, setInput] = useState('');
  const [singleBusy, setSingleBusy] = useState(false);
  const [single, setSingle] = useState<{ predicted: string; confidence?: number; top: TopPred[]; ms: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [datasetId, setDatasetId] = useState(datasets[0]?.id ?? '');
  const [maxSamples, setMaxSamples] = useState<number | ''>(50);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<{ current: number; total: number } | null>(null);
  const [summary, setSummary] = useState<{ total: number; accuracy: number | null; correct: number | null } | null>(null);

  const unlistenRef = useRef<Array<() => void>>([]);
  useEffect(() => () => { unlistenRef.current.forEach(fn => fn()); }, []);

  const runSingle = async () => {
    if (!input.trim()) { setError(singleLabel + ' fehlt.'); return; }
    setError(null); setSingle(null); setSingleBusy(true);
    try {
      const testId = await invoke<string>('test_single_input', {
        versionId,
        singleInput: input.trim(),
        singleInputType: inputKind,
        taskType,
        pluginConfig,
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
          setSingleBusy(false);
        });
      const offErr = await listen<{ test_id?: string; data?: { error?: string } }>(
        'test-error', e => {
          setError(e.payload.data?.error ?? 'Unbekannter Fehler');
          setSingleBusy(false);
        });
      unlistenRef.current.push(off, offErr);
    } catch (e) {
      setError(String(e)); setSingleBusy(false);
    }
  };

  const runDataset = async () => {
    const ds = datasets.find(d => d.id === datasetId);
    if (!ds) { setError('Kein Dataset ausgewählt.'); return; }
    setError(null); setSummary(null); setProgress(null); setRunning(true);
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
      const offC = await listen<{ test_id?: string; data?: { total_samples?: number; accuracy?: number | null; correct_predictions?: number | null } }>(
        'test-complete', e => {
          const d = e.payload.data;
          setSummary({
            total: d?.total_samples ?? 0,
            accuracy: d?.accuracy ?? null,
            correct: d?.correct_predictions ?? null,
          });
          setRunning(false);
        });
      const offE = await listen<{ data?: { error?: string } }>('test-error', e => {
        setError(e.payload.data?.error ?? 'Unbekannter Fehler');
        setRunning(false);
      });
      unlistenRef.current.push(offP, offC, offE);
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
        <button
          onClick={runSingle}
          disabled={singleBusy}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 text-emerald-200 text-xs font-medium disabled:opacity-50"
        >
          {singleBusy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
          Auswerten
        </button>

        {single && (
          <div className="rounded-xl bg-slate-900/60 border border-white/10 p-4 space-y-2">
            <p className="text-gray-400 text-[11px]">{resultLabel}</p>
            <p className="text-white text-sm break-words">{single.predicted}</p>
            <p className="text-gray-500 text-[11px]">
              {showConfidence && single.confidence != null
                ? `Konfidenz ${(single.confidence * 100).toFixed(1)} % · `
                : ''}
              {single.ms} ms
            </p>
            {showConfidence && single.top.length > 1 && (
              <div className="space-y-1 pt-1">
                {single.top.map((t, i) => (
                  <div key={i} className="flex items-center justify-between text-[11px] text-gray-400">
                    <span>{t.label}</span>
                    <span className="tabular-nums">{((t.score ?? 0) * 100).toFixed(1)} %</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Datensatz-Lauf */}
      <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-3">
        <p className="text-white font-medium text-sm">Auf einem Dataset auswerten</p>
        {datasets.length === 0 ? (
          <p className="text-gray-500 text-xs">Kein Dataset für dieses Modell vorhanden.</p>
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
                title="Maximale Anzahl Beispiele"
              />
            </div>
            <div className="flex gap-2">
              <button
                onClick={runDataset}
                disabled={running}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 text-emerald-200 text-xs font-medium disabled:opacity-50"
              >
                {running ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                Test starten
              </button>
              {running && (
                <button
                  onClick={() => { invoke('stop_test').catch(() => {}); }}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl bg-red-500/15 hover:bg-red-500/25 border border-red-500/30 text-red-200 text-xs font-medium"
                >
                  <Square className="w-3.5 h-3.5" /> Stoppen
                </button>
              )}
            </div>

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
                <p>{summary.total} Beispiele ausgewertet</p>
                {summary.accuracy != null
                  ? <p className="text-white font-medium">Treffer: {(summary.accuracy * 100).toFixed(1)} % ({summary.correct} richtig)</p>
                  : <p className="text-gray-500">Ohne erwartete Werte im Dataset lässt sich keine Trefferquote berechnen.</p>}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
