// XLM-RoBERTa – Test Plugin UI
// Verwendet test_single_input (Text) und start_test (Dataset)

import { useState, useEffect, useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import {
  Loader2, Square, FileText, Type, CheckCircle,
  AlertCircle, BarChart3, ChevronDown, ChevronUp,
} from 'lucide-react';
import type { TestPluginProps } from '../types';

// ── Typen ─────────────────────────────────────────────────────────────────

interface TopPred { label: string; score: number; }

interface SingleResult {
  predicted_output: string;
  confidence?: number;
  top_predictions?: TopPred[];
  inference_time: number;
}

interface DatasetProgress {
  current_sample: number;
  total_samples: number;
  progress_percent: number;
  samples_per_second: number;
}

interface PredRow {
  input_text: string;
  expected_output?: string;
  predicted_output: string;
  is_correct: boolean;
  confidence?: number;
  inference_time: number;
}

interface DatasetResults {
  total_samples: number;
  correct_predictions?: number;
  accuracy?: number;
  average_inference_time: number;
  samples_per_second?: number;
  predictions?: PredRow[];
}

// ── Helpers ───────────────────────────────────────────────────────────────

function formatBytes(b: number) {
  if (!b) return '0 B';
  const k = 1024, s = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(b) / Math.log(k));
  return (b / Math.pow(k, i)).toFixed(1) + ' ' + s[i];
}

type UnlistenFn = () => void;

// ── Hauptkomponente ────────────────────────────────────────────────────────

export default function XLMRobertaTestPlugin({
  versionId,
  modelId,
  modelName,
  versionName,
  datasets,
}: TestPluginProps) {

  const [tab, setTab] = useState<'text' | 'dataset'>('text');

  // -- Text-Modus --
  const [inputText, setInputText]     = useState('');
  const [singleLoading, setSingleLoading] = useState(false);
  const [singleResult, setSingleResult]   = useState<SingleResult | null>(null);
  const [singleError, setSingleError]     = useState<string | null>(null);
  const [showAllPreds, setShowAllPreds]   = useState(false);

  // -- Dataset-Modus --
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [batchSize, setBatchSize]     = useState(16);
  const [maxSamples, setMaxSamples]   = useState<number | ''>('');
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [datasetProgress, setDatasetProgress] = useState<DatasetProgress | null>(null);
  const [datasetResults, setDatasetResults]   = useState<DatasetResults | null>(null);
  const [datasetError, setDatasetError]       = useState<string | null>(null);
  const [showPredTable, setShowPredTable]     = useState(false);

  const unlistenRef = useRef<UnlistenFn[]>([]);

  // Wenn datasets geladen werden, bestes Dataset auto-selektieren
  useEffect(() => {
    if (!datasets.length) return;
    const split = datasets.find(d => d.status === 'split');
    setSelectedDatasetId(prev => prev || split?.id || datasets[0].id);
  }, [datasets]);

  // Cleanup on unmount
  useEffect(() => () => { unlistenRef.current.forEach(fn => fn()); }, []);

  // ── Text-Test ─────────────────────────────────────────────────────────────

  const handleSingleTest = useCallback(async () => {
    if (!inputText.trim()) return;
    setSingleError(null);
    setSingleResult(null);
    setSingleLoading(true);
    setShowAllPreds(false);

    // Alte Listener entfernen
    unlistenRef.current.forEach(fn => fn());
    unlistenRef.current = [];

    try {
      const testId = await invoke<string>('test_single_input', {
        versionId,
        singleInput: inputText.trim(),
        singleInputType: 'text',
      });

      const u1 = await listen<{
        test_id: string;
        data?: { predicted_output?: string; confidence?: number; top_predictions?: TopPred[]; inference_time?: number };
      }>('test-single-complete', (e) => {
        if (e.payload.test_id !== testId) return;
        const d = e.payload.data;
        if (d?.predicted_output !== undefined) {
          setSingleResult({
            predicted_output: d.predicted_output,
            confidence: d.confidence,
            top_predictions: d.top_predictions,
            inference_time: d.inference_time ?? 0,
          });
        } else {
          setSingleError('Keine Ergebnisse vom Modell erhalten.');
        }
        setSingleLoading(false);
      });

      const u2 = await listen<{ test_id: string; data?: { error?: string } }>('test-error', (e) => {
        if (e.payload.test_id !== testId) return;
        setSingleError(e.payload.data?.error ?? 'Unbekannter Fehler beim Test.');
        setSingleLoading(false);
      });

      const u3 = await listen<{ test_id: string }>('test-finished', (e) => {
        if (e.payload.test_id !== testId) return;
        // Falls kein Ergebnis-Event ankam
        setSingleLoading(false);
      });

      unlistenRef.current = [u1, u2, u3];
    } catch (e: unknown) {
      setSingleError(String(e));
      setSingleLoading(false);
    }
  }, [inputText, versionId]);

  // ── Dataset-Test ──────────────────────────────────────────────────────────

  const handleStartDatasetTest = useCallback(async () => {
    const ds = datasets.find(d => d.id === selectedDatasetId);
    if (!ds) return;

    setDatasetError(null);
    setDatasetResults(null);
    setDatasetProgress(null);
    setDatasetLoading(true);
    setShowPredTable(false);

    unlistenRef.current.forEach(fn => fn());
    unlistenRef.current = [];

    try {
      const job = await invoke<{ id: string }>('start_test', {
        modelId,
        modelName,
        versionId,
        versionName,
        datasetId: selectedDatasetId,
        datasetName: ds.name,
        batchSize,
        maxSamples: maxSamples === '' ? null : maxSamples,
      });
      const jobId = job.id;

      const u1 = await listen<{ test_id: string; data?: DatasetProgress }>('test-progress', (e) => {
        if (e.payload.test_id !== jobId) return;
        if (e.payload.data) setDatasetProgress(e.payload.data);
      });

      const u2 = await listen<{
        test_id: string;
        data?: { accuracy?: number; correct_predictions?: number; total_samples?: number; average_inference_time?: number; samples_per_second?: number; predictions?: PredRow[] };
      }>('test-complete', (e) => {
        if (e.payload.test_id !== jobId) return;
        const d = e.payload.data;
        setDatasetResults({
          total_samples:           d?.total_samples ?? 0,
          correct_predictions:     d?.correct_predictions,
          accuracy:                d?.accuracy,
          average_inference_time:  d?.average_inference_time ?? 0,
          samples_per_second:      d?.samples_per_second,
          predictions:             d?.predictions,
        });
        setDatasetLoading(false);
      });

      const u3 = await listen<{ test_id: string; data?: { error?: string } }>('test-error', (e) => {
        if (e.payload.test_id !== jobId) return;
        setDatasetError(e.payload.data?.error ?? 'Unbekannter Fehler.');
        setDatasetLoading(false);
      });

      const u4 = await listen<{ test_id: string }>('test-finished', (e) => {
        if (e.payload.test_id !== jobId) return;
        setDatasetLoading(false);
      });

      unlistenRef.current = [u1, u2, u3, u4];
    } catch (e: unknown) {
      setDatasetError(String(e));
      setDatasetLoading(false);
    }
  }, [selectedDatasetId, datasets, modelId, modelName, versionId, versionName, batchSize, maxSamples]);

  const handleStopTest = async () => {
    try { await invoke('stop_test'); } catch { /* ignore */ }
    setDatasetLoading(false);
  };

  const selectedDataset = datasets.find(d => d.id === selectedDatasetId);

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-5">

      {/* Header */}
      <div className="flex items-center gap-3 p-4 rounded-2xl border border-amber-500/30 bg-amber-500/10">
        <div className="w-10 h-10 rounded-xl bg-amber-500/20 border border-amber-500/30 flex items-center justify-center text-xl">🧪</div>
        <div className="min-w-0">
          <p className="text-amber-300 text-sm font-medium">XLM-RoBERTa · Keyword Recognition</p>
          <p className="text-gray-400 text-xs truncate">{modelName} · {versionName}</p>
        </div>
      </div>

      {/* Tab-Auswahl */}
      <div className="flex items-center gap-1 p-1 rounded-xl bg-white/5 border border-white/10">
        {([
          { key: 'text',    label: 'Text-Eingabe', icon: <Type    className="w-3.5 h-3.5" /> },
          { key: 'dataset', label: 'Dataset',      icon: <FileText className="w-3.5 h-3.5" /> },
        ] as const).map(t => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-sm font-medium transition-all ${
              tab === t.key
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            {t.icon}{t.label}
          </button>
        ))}
      </div>

      {/* ── Tab: Text-Eingabe ─────────────────────────────────────────────── */}
      {tab === 'text' && (
        <div className="space-y-4">
          {singleError && (
            <div className="flex items-start gap-3 p-4 rounded-xl border border-red-500/40 bg-red-500/10 text-red-300 text-sm">
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              {singleError}
            </div>
          )}

          <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-3">
            <label className="block text-white text-sm font-medium">Eingabetext</label>
            <textarea
              rows={4}
              value={inputText}
              onChange={(e) => { setInputText(e.target.value); setSingleResult(null); setSingleError(null); }}
              onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSingleTest(); }}
              placeholder="Text zum Testen eingeben… (⌘/Strg + Enter zum Starten)"
              disabled={singleLoading}
              className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-amber-500/50 resize-none disabled:opacity-50 transition-all"
            />
            <button
              onClick={handleSingleTest}
              disabled={singleLoading || !inputText.trim()}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-300 font-medium text-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {singleLoading
                ? <><Loader2 className="w-4 h-4 animate-spin" />Analysiere…</>
                : '▶ Testen'}
            </button>
          </div>

          {/* Ergebnis-Karte */}
          {singleResult && (
            <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-amber-400" />
                  <span className="text-white font-medium text-sm">Ergebnis</span>
                </div>
                <span className="text-gray-500 text-xs">{(singleResult.inference_time * 1000).toFixed(0)} ms</span>
              </div>

              {/* Beste Klasse */}
              <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-amber-500/10 border border-amber-500/20">
                <span className="text-amber-300 text-base font-semibold">{singleResult.predicted_output}</span>
                {singleResult.confidence != null && (
                  <span className="ml-auto text-amber-400 text-sm font-mono tabular-nums">
                    {(singleResult.confidence * 100).toFixed(1)}%
                  </span>
                )}
              </div>

              {/* Alle Klassen aufklappen */}
              {singleResult.top_predictions && singleResult.top_predictions.length > 1 && (
                <div className="space-y-2">
                  <button
                    onClick={() => setShowAllPreds(v => !v)}
                    className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-white transition-colors"
                  >
                    <BarChart3 className="w-3.5 h-3.5" />
                    Alle {singleResult.top_predictions.length} Klassen
                    {showAllPreds ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                  </button>

                  {showAllPreds && (
                    <div className="space-y-1.5 pt-1">
                      {[...singleResult.top_predictions]
                        .sort((a, b) => b.score - a.score)
                        .map((p) => (
                          <div key={p.label} className="flex items-center gap-3">
                            <span className="text-gray-300 text-xs w-36 truncate">{p.label}</span>
                            <div className="flex-1 h-1.5 rounded-full bg-white/10 overflow-hidden">
                              <div
                                className="h-full rounded-full bg-amber-400 transition-all duration-500"
                                style={{ width: `${(p.score * 100).toFixed(1)}%` }}
                              />
                            </div>
                            <span className="text-gray-400 text-xs w-12 text-right font-mono tabular-nums">
                              {(p.score * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── Tab: Dataset ──────────────────────────────────────────────────── */}
      {tab === 'dataset' && (
        <div className="space-y-4">
          {datasetError && (
            <div className="flex items-start gap-3 p-4 rounded-xl border border-red-500/40 bg-red-500/10 text-red-300 text-sm">
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              {datasetError}
            </div>
          )}

          {/* Dataset-Konfiguration */}
          <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-4">

            <div className="space-y-1.5">
              <label className="block text-white text-sm font-medium">Dataset</label>
              {datasets.length === 0 ? (
                <p className="text-gray-500 text-sm py-2">Kein Dataset für dieses Modell vorhanden.</p>
              ) : (
                <select
                  value={selectedDatasetId}
                  onChange={e => setSelectedDatasetId(e.target.value)}
                  disabled={datasetLoading}
                  className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-amber-500/50 appearance-none disabled:opacity-50 transition-all"
                >
                  {datasets.map(d => (
                    <option key={d.id} value={d.id} className="bg-slate-900">
                      {d.name} {d.status === 'split' ? '✅' : '⚠️'} · {d.file_count} Dateien · {formatBytes(d.size_bytes)}
                    </option>
                  ))}
                </select>
              )}
              {selectedDataset?.status === 'unused' && (
                <p className="text-amber-400 text-xs">⚠️ Dataset hat noch keinen Split – erst im Dataset-Manager aufteilen.</p>
              )}
            </div>

            {/* Optionen */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label className="text-xs text-gray-400">Batch Size</label>
                <input
                  type="number"
                  value={batchSize}
                  min={1} max={128}
                  disabled={datasetLoading}
                  onChange={e => setBatchSize(Math.max(1, parseInt(e.target.value) || 1))}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-amber-500/50 disabled:opacity-50"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs text-gray-400">
                  Max. Samples <span className="text-gray-600">(leer = alle)</span>
                </label>
                <input
                  type="number"
                  value={maxSamples}
                  min={1}
                  placeholder="Alle"
                  disabled={datasetLoading}
                  onChange={e => setMaxSamples(e.target.value === '' ? '' : Math.max(1, parseInt(e.target.value) || 1))}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-amber-500/50 disabled:opacity-50 placeholder:text-gray-600"
                />
              </div>
            </div>

            {/* Start / Stop */}
            {datasetLoading ? (
              <button
                onClick={handleStopTest}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-red-300 font-medium text-sm transition-all"
              >
                <Square className="w-4 h-4" /> Test stoppen
              </button>
            ) : (
              <button
                onClick={handleStartDatasetTest}
                disabled={!selectedDatasetId || selectedDataset?.status !== 'split' || datasets.length === 0}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-300 font-medium text-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed"
              >
                ▶ Dataset testen
              </button>
            )}
          </div>

          {/* Fortschritt */}
          {datasetLoading && (
            <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-3">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin text-amber-400" />
                  <span className="text-white">
                    {datasetProgress
                      ? `Teste… ${datasetProgress.current_sample}/${datasetProgress.total_samples}`
                      : 'Starte Test-Engine…'}
                  </span>
                </div>
                {datasetProgress && (
                  <span className="text-gray-400 text-xs tabular-nums">
                    {datasetProgress.samples_per_second.toFixed(1)} S/s
                  </span>
                )}
              </div>
              {datasetProgress && (
                <>
                  <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-amber-400 transition-all duration-300"
                      style={{ width: `${datasetProgress.progress_percent}%` }}
                    />
                  </div>
                  <p className="text-gray-500 text-xs text-right tabular-nums">
                    {datasetProgress.progress_percent.toFixed(1)}%
                  </p>
                </>
              )}
            </div>
          )}

          {/* Ergebnis-Metriken */}
          {datasetResults && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                {[
                  {
                    label: 'Genauigkeit',
                    value: datasetResults.accuracy != null
                      ? `${(datasetResults.accuracy * 100).toFixed(1)}%`
                      : '–',
                    color: datasetResults.accuracy != null
                      ? (datasetResults.accuracy > 0.9 ? 'text-emerald-400'
                        : datasetResults.accuracy > 0.7 ? 'text-amber-400'
                        : 'text-red-400')
                      : 'text-gray-400',
                  },
                  {
                    label: 'Richtig / Gesamt',
                    value: `${datasetResults.correct_predictions ?? '–'} / ${datasetResults.total_samples}`,
                    color: 'text-white',
                  },
                  {
                    label: 'Ø Inferenzzeit',
                    value: `${(datasetResults.average_inference_time * 1000).toFixed(0)} ms`,
                    color: 'text-blue-400',
                  },
                  {
                    label: 'Samples / Sek.',
                    value: datasetResults.samples_per_second != null
                      ? datasetResults.samples_per_second.toFixed(1)
                      : '–',
                    color: 'text-purple-400',
                  },
                ].map((m) => (
                  <div key={m.label} className="rounded-xl border border-white/10 bg-white/5 p-4">
                    <p className="text-gray-400 text-xs mb-1">{m.label}</p>
                    <p className={`text-xl font-bold tabular-nums ${m.color}`}>{m.value}</p>
                  </div>
                ))}
              </div>

              {/* Predictions-Tabelle */}
              {datasetResults.predictions && datasetResults.predictions.length > 0 && (
                <div className="rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
                  <button
                    onClick={() => setShowPredTable(v => !v)}
                    className="w-full flex items-center justify-between px-5 py-3.5 hover:bg-white/5 transition-colors"
                  >
                    <span className="text-white text-sm font-medium">
                      Einzelne Vorhersagen ({datasetResults.predictions.length})
                    </span>
                    {showPredTable
                      ? <ChevronUp className="w-4 h-4 text-gray-400" />
                      : <ChevronDown className="w-4 h-4 text-gray-400" />}
                  </button>

                  {showPredTable && (
                    <div className="overflow-x-auto max-h-80 overflow-y-auto">
                      <table className="w-full text-xs">
                        <thead className="sticky top-0 bg-slate-900/90 backdrop-blur-sm">
                          <tr className="text-left text-gray-400 border-b border-white/10">
                            {['#', 'Eingabe', 'Erwartet', 'Vorhergesagt', 'OK', 'ms'].map(h => (
                              <th key={h} className="px-4 py-2.5 font-medium">{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {datasetResults.predictions.map((p, i) => (
                            <tr
                              key={i}
                              className={`border-b border-white/5 transition-colors ${
                                p.is_correct ? 'hover:bg-emerald-500/5' : 'bg-red-500/[0.03] hover:bg-red-500/5'
                              }`}
                            >
                              <td className="px-4 py-2 text-gray-500 tabular-nums">{i + 1}</td>
                              <td className="px-4 py-2 text-gray-300 max-w-48 truncate">{p.input_text || '–'}</td>
                              <td className="px-4 py-2 text-gray-400">{p.expected_output || '–'}</td>
                              <td className="px-4 py-2 text-white font-medium">{p.predicted_output}</td>
                              <td className="px-4 py-2">
                                {p.is_correct
                                  ? <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                                  : <AlertCircle className="w-3.5 h-3.5 text-red-400" />}
                              </td>
                              <td className="px-4 py-2 text-gray-500 font-mono tabular-nums">
                                {(p.inference_time * 1000).toFixed(0)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
