// HF Encoder – Test Plugin UI
//
// Generische Test-UI für alle Encoder-Modelle die über `seq_classification` laufen.

import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { Loader2, Play, Square, RefreshCw } from 'lucide-react';
import type { TestPluginProps } from '../types';

type TopPred = { label?: string; score?: number };

type DatasetProgress = {
  current_sample?: number;
  total_samples?: number;
  progress_percent?: number;
  samples_per_second?: number;
  estimated_time_remaining?: number;
};

type PredRow = {
  sample_id: number;
  input_text?: string;
  expected_output?: string | null;
  predicted_output: string;
  is_correct?: boolean;
  confidence?: number | null;
  inference_time?: number;
};

export default function HFEncoderTestPlugin({
  versionId,
  modelId,
  modelName,
  versionName,
  datasets,
}: TestPluginProps) {
  // ── Single Input ────────────────────────────────────────────────────────
  const [inputText, setInputText] = useState('');
  const [singleLoading, setSingleLoading] = useState(false);
  const [singleError, setSingleError] = useState<string | null>(null);
  const [singleResult, setSingleResult] = useState<{ predicted_output: string; confidence?: number; top_predictions?: TopPred[]; inference_time: number } | null>(null);
  const [showAllPreds, setShowAllPreds] = useState(false);

  // ── Dataset Test ────────────────────────────────────────────────────────
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>(datasets[0]?.id ?? '');
  const [batchSize, setBatchSize] = useState<number>(16);
  const [maxSamples, setMaxSamples] = useState<string>('');
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [datasetProgress, setDatasetProgress] = useState<DatasetProgress | null>(null);
  const [datasetResults, setDatasetResults] = useState<{ total_samples: number; correct_predictions?: number; accuracy?: number; average_inference_time: number; samples_per_second?: number; predictions?: PredRow[] } | null>(null);
  const [showPredTable, setShowPredTable] = useState(false);

  // ── Listener Cleanup ────────────────────────────────────────────────────
  const unlistenRef = useRef<(() => void)[]>([]);
  useEffect(() => () => { unlistenRef.current.forEach(fn => fn()); }, []);

  useEffect(() => {
    if (!selectedDatasetId && datasets.length > 0) setSelectedDatasetId(datasets[0].id);
  }, [datasets, selectedDatasetId]);

  const headerModel = useMemo(() => `${modelName} · ${versionName}`, [modelName, versionName]);

  // ── Text-Test ────────────────────────────────────────────────────────────
  const handleSingleTest = useCallback(async () => {
    if (!inputText.trim()) return;
    setSingleError(null);
    setSingleResult(null);
    setSingleLoading(true);
    setShowAllPreds(false);

    unlistenRef.current.forEach(fn => fn());
    unlistenRef.current = [];

    try {
      const testId = await invoke<string>('test_single_input', {
        versionId,
        singleInput: inputText.trim(),
        singleInputType: 'text',
        taskType: 'seq_classification',
        pluginConfig: {},
      });

      const u1 = await listen<{ test_id: string; data?: { predicted_output?: string; confidence?: number; top_predictions?: TopPred[]; inference_time?: number } }>(
        'test-single-complete',
        (e) => {
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
        },
      );

      const u2 = await listen<{ test_id: string; data?: { error?: string } }>('test-error', (e) => {
        if (e.payload.test_id !== testId) return;
        setSingleError(e.payload.data?.error ?? 'Unbekannter Fehler.');
        setSingleLoading(false);
      });

      unlistenRef.current = [u1, u2];
    } catch (e: unknown) {
      setSingleError(String(e));
      setSingleLoading(false);
    }
  }, [inputText, versionId]);

  // ── Dataset-Test ─────────────────────────────────────────────────────────
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
        taskType: 'seq_classification',
        pluginConfig: {},
      });
      const jobId = job.id;

      const u1 = await listen<{ test_id: string; data?: DatasetProgress }>('test-progress', (e) => {
        if (e.payload.test_id !== jobId) return;
        if (e.payload.data) setDatasetProgress(e.payload.data);
      });

      const u2 = await listen<{ test_id: string; data?: { accuracy?: number; correct_predictions?: number; total_samples?: number; average_inference_time?: number; samples_per_second?: number; predictions?: PredRow[] } }>(
        'test-complete',
        (e) => {
          if (e.payload.test_id !== jobId) return;
          const d = e.payload.data;
          setDatasetResults({
            total_samples: d?.total_samples ?? 0,
            correct_predictions: d?.correct_predictions,
            accuracy: d?.accuracy,
            average_inference_time: d?.average_inference_time ?? 0,
            samples_per_second: d?.samples_per_second,
            predictions: d?.predictions,
          });
          setDatasetLoading(false);
        },
      );

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
  }, [datasets, selectedDatasetId, modelId, modelName, versionId, versionName, batchSize, maxSamples]);

  const handleResetDataset = () => {
    setDatasetError(null);
    setDatasetResults(null);
    setDatasetProgress(null);
    setDatasetLoading(false);
    setShowPredTable(false);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3 p-4 rounded-2xl border border-amber-500/30 bg-amber-500/10">
        <div className="w-10 h-10 rounded-xl bg-amber-500/20 border border-amber-500/30 flex items-center justify-center text-xl">
          🧪
        </div>
        <div className="min-w-0">
          <p className="text-amber-300 text-sm font-medium truncate">{headerModel}</p>
          <p className="text-gray-400 text-xs">Sequence Classification (Encoder)</p>
        </div>
      </div>

      {/* Single input */}
      <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-3">
        <p className="text-white text-sm font-medium">Einzelner Text</p>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Text eingeben…"
          className="w-full min-h-[90px] bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-amber-500/40"
        />

        <div className="flex items-center gap-2">
          <button
            onClick={handleSingleTest}
            disabled={singleLoading || !inputText.trim()}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-300 text-sm font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {singleLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            Testen
          </button>
          <button
            onClick={() => { setInputText(''); setSingleResult(null); setSingleError(null); }}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-sm transition-all"
          >
            <RefreshCw className="w-4 h-4" />
            Reset
          </button>
        </div>

        {singleError && <div className="text-red-300 text-sm">⚠️ {singleError}</div>}
        {singleResult && (
          <div className="rounded-xl border border-white/10 bg-black/20 p-4 space-y-2">
            <div className="flex items-center justify-between">
              <p className="text-white text-sm font-medium">Prediction</p>
              <p className="text-gray-400 text-xs">Inference: {singleResult.inference_time.toFixed(0)}ms</p>
            </div>
            <p className="text-amber-200 font-mono text-sm">{singleResult.predicted_output}</p>
            {Array.isArray(singleResult.top_predictions) && singleResult.top_predictions.length > 0 && (
              <button
                onClick={() => setShowAllPreds(s => !s)}
                className="text-xs text-gray-400 hover:text-white"
              >
                {showAllPreds ? 'Top-Predictions ausblenden' : 'Top-Predictions anzeigen'}
              </button>
            )}
            {showAllPreds && (
              <div className="space-y-1">
                {(singleResult.top_predictions ?? []).slice(0, 10).map((p, idx) => (
                  <div key={idx} className="flex items-center justify-between text-xs text-gray-300">
                    <span className="truncate max-w-[70%]">{p.label ?? '—'}</span>
                    <span className="text-gray-500">{typeof p.score === 'number' ? p.score.toFixed(4) : '—'}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Dataset test */}
      <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-4">
        <div className="flex items-center justify-between">
          <p className="text-white text-sm font-medium">Dataset-Test</p>
          {datasetLoading && (
            <button
              onClick={() => invoke('stop_test').catch(() => {})}
              className="inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-red-500/15 hover:bg-red-500/20 border border-red-500/30 text-red-300 text-sm transition-all"
            >
              <Square className="w-4 h-4" />
              Stop
            </button>
          )}
        </div>

        <div className="grid grid-cols-3 gap-3">
          <div className="space-y-1">
            <label className="text-xs text-gray-400">Dataset</label>
            <select
              value={selectedDatasetId}
              onChange={(e) => setSelectedDatasetId(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-white text-sm focus:outline-none focus:border-amber-500/40"
            >
              {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-xs text-gray-400">Batch Size</label>
            <input
              type="number"
              min={1}
              value={batchSize}
              onChange={(e) => setBatchSize(Number(e.target.value))}
              className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-white text-sm focus:outline-none focus:border-amber-500/40"
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs text-gray-400">Max Samples</label>
            <input
              type="text"
              value={maxSamples}
              onChange={(e) => setMaxSamples(e.target.value)}
              placeholder="leer = alle"
              className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-amber-500/40"
            />
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleStartDatasetTest}
            disabled={datasetLoading || !selectedDatasetId}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-300 text-sm font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {datasetLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            Start
          </button>
          <button
            onClick={handleResetDataset}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-sm transition-all"
          >
            <RefreshCw className="w-4 h-4" />
            Reset
          </button>
        </div>

        {datasetProgress && (
          <div className="text-gray-300 text-sm">
            Fortschritt: {Math.round(datasetProgress.progress_percent ?? 0)}% ({datasetProgress.current_sample ?? 0}/{datasetProgress.total_samples ?? 0})
          </div>
        )}
        {datasetError && <div className="text-red-300 text-sm">⚠️ {datasetError}</div>}
        {datasetResults && (
          <div className="rounded-xl border border-white/10 bg-black/20 p-4 space-y-2">
            <p className="text-white text-sm font-medium">Ergebnisse</p>
            <div className="text-gray-300 text-sm">Samples: {datasetResults.total_samples}</div>
            {typeof datasetResults.accuracy === 'number' && (
              <div className="text-gray-300 text-sm">Accuracy: {(datasetResults.accuracy * 100).toFixed(2)}%</div>
            )}
            <div className="text-gray-300 text-sm">Avg inference: {datasetResults.average_inference_time.toFixed(0)}ms</div>
            {typeof datasetResults.samples_per_second === 'number' && (
              <div className="text-gray-300 text-sm">Speed: {datasetResults.samples_per_second.toFixed(2)} samples/s</div>
            )}
            {Array.isArray(datasetResults.predictions) && datasetResults.predictions.length > 0 && (
              <button className="text-xs text-gray-400 hover:text-white" onClick={() => setShowPredTable(s => !s)}>
                {showPredTable ? 'Predictions ausblenden' : 'Predictions anzeigen'}
              </button>
            )}
            {showPredTable && (
              <div className="max-h-72 overflow-auto rounded-lg border border-white/10">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-slate-900/80 backdrop-blur border-b border-white/10">
                    <tr>
                      <th className="text-left px-3 py-2 text-gray-400 font-medium">#</th>
                      <th className="text-left px-3 py-2 text-gray-400 font-medium">Expected</th>
                      <th className="text-left px-3 py-2 text-gray-400 font-medium">Predicted</th>
                      <th className="text-left px-3 py-2 text-gray-400 font-medium">Conf</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(datasetResults.predictions ?? []).slice(0, 200).map((p) => (
                      <tr key={p.sample_id} className="border-b border-white/5">
                        <td className="px-3 py-2 text-gray-500">{p.sample_id}</td>
                        <td className="px-3 py-2 text-gray-300">{p.expected_output ?? '—'}</td>
                        <td className="px-3 py-2 text-amber-200 font-mono">{p.predicted_output}</td>
                        <td className="px-3 py-2 text-gray-400">{typeof p.confidence === 'number' ? p.confidence.toFixed(3) : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

