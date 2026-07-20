import { useState, useCallback, useRef, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { ImageIcon, Loader2, Play, RefreshCw, AlertTriangle, Upload } from 'lucide-react';
import { open as openDialog } from '@tauri-apps/plugin-dialog';
import type { TestPluginProps } from '../types';

type Prediction = { label: string; confidence: number };

export default function ImageClassificationTestPlugin({ versionId, modelId, modelName, versionName, datasets }: TestPluginProps) {
  const [imagePath, setImagePath] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ top: Prediction[]; inference_time: number } | null>(null);
  const [dsLoading, setDsLoading] = useState(false);
  const [dsError, setDsError] = useState<string | null>(null);
  const [dsResult, setDsResult] = useState<{ accuracy: number; total: number } | null>(null);
  const [selectedDs, setSelectedDs] = useState(datasets[0]?.id ?? '');
  const unlistenRef = useRef<(() => void)[]>([]);
  useEffect(() => () => { unlistenRef.current.forEach(f => f()); }, []);

  const pickImage = useCallback(async () => {
    const path = await openDialog({ filters: [{ name: 'Bild', extensions: ['jpg','jpeg','png','bmp','webp'] }] });
    if (typeof path === 'string') setImagePath(path);
  }, []);

  const handleSingleTest = useCallback(async () => {
    if (!imagePath) return;
    setError(null); setResult(null); setLoading(true);
    unlistenRef.current.forEach(f => f()); unlistenRef.current = [];
    try {
      const testId = await invoke<string>('test_single_input', {
        versionId, singleInput: imagePath, singleInputType: 'image',
        taskType: 'image_classification', pluginConfig: {},
      });
      const u1 = await listen<{ test_id: string; data?: { top_predictions?: Prediction[]; inference_time?: number } }>(
        'test-single-complete', e => {
          if (e.payload.test_id !== testId) return;
          const d = e.payload.data;
          setResult({ top: d?.top_predictions ?? [], inference_time: d?.inference_time ?? 0 });
          setLoading(false);
        }
      );
      const u2 = await listen<{ test_id: string; data?: { error?: string } }>('test-error', e => {
        if (e.payload.test_id !== testId) return;
        setError(e.payload.data?.error ?? 'Fehler'); setLoading(false);
      });
      unlistenRef.current = [u1, u2];
    } catch (e) { setError(String(e)); setLoading(false); }
  }, [imagePath, versionId]);

  const handleDatasetTest = useCallback(async () => {
    const ds = datasets.find(d => d.id === selectedDs);
    if (!ds) return;
    setDsError(null); setDsResult(null); setDsLoading(true);
    unlistenRef.current.forEach(f => f()); unlistenRef.current = [];
    try {
      const job = await invoke<{ id: string }>('start_test', {
        modelId, modelName, versionId, versionName,
        datasetId: selectedDs, datasetName: ds.name,
        batchSize: 16, maxSamples: null,
        taskType: 'image_classification', pluginConfig: {},
      });
      const u1 = await listen<{ test_id: string; data?: { accuracy?: number; total_samples?: number } }>(
        'test-complete', e => {
          if (e.payload.test_id !== job.id) return;
          setDsResult({ accuracy: e.payload.data?.accuracy ?? 0, total: e.payload.data?.total_samples ?? 0 });
          setDsLoading(false);
        }
      );
      const u2 = await listen<{ test_id: string; data?: { error?: string } }>('test-error', e => {
        if (e.payload.test_id !== job.id) return;
        setDsError(e.payload.data?.error ?? 'Fehler'); setDsLoading(false);
      });
      unlistenRef.current = [u1, u2];
    } catch (e) { setDsError(String(e)); setDsLoading(false); }
  }, [datasets, selectedDs, modelId, modelName, versionId, versionName]);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 p-4 rounded-2xl border border-blue-500/30 bg-blue-500/10">
        <div className="w-10 h-10 rounded-xl bg-blue-500/20 border border-blue-500/30 flex items-center justify-center">
          <ImageIcon className="w-5 h-5 text-blue-300" />
        </div>
        <div>
          <p className="text-blue-300 text-sm font-medium">{modelName} · {versionName}</p>
          <p className="text-gray-400 text-xs">Image Classification</p>
        </div>
      </div>

      <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-4">
        <p className="text-white text-sm font-medium">Einzelbild testen</p>
        <div className="flex gap-2">
          <input value={imagePath} readOnly placeholder="Bild auswählen…"
            className="flex-1 bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-gray-300 text-sm placeholder:text-gray-600" />
          <button onClick={pickImage}
            className="px-3 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/10 text-gray-300 text-sm flex items-center gap-1.5 transition-all">
            <Upload className="w-4 h-4" /> Bild
          </button>
        </div>
        <div className="flex gap-2">
          <button onClick={handleSingleTest} disabled={loading || !imagePath}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/40 text-blue-300 text-sm font-medium transition-all disabled:opacity-50">
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            Klassifizieren
          </button>
          <button onClick={() => { setImagePath(''); setResult(null); setError(null); }}
            className="px-3 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-sm flex items-center gap-1.5 transition-all">
            <RefreshCw className="w-4 h-4" /> Reset
          </button>
        </div>
        {error && <p className="text-red-300 text-sm flex items-center gap-2"><AlertTriangle className="w-4 h-4"/>{error}</p>}
        {result && (
          <div className="rounded-xl border border-white/10 bg-black/20 p-4 space-y-2">
            <div className="flex justify-between text-xs text-gray-400 mb-1">
              <span>Top Predictions</span>
              <span>{result.inference_time.toFixed(0)}ms</span>
            </div>
            {result.top.slice(0, 5).map((p, i) => (
              <div key={i} className="flex items-center gap-3">
                <span className="text-xs text-gray-500 w-4">{i + 1}.</span>
                <span className="text-sm text-white flex-1 truncate">{p.label}</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-400 rounded-full" style={{ width: `${p.confidence * 100}%` }} />
                  </div>
                  <span className="text-xs text-gray-400 w-12 text-right">{(p.confidence * 100).toFixed(1)}%</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {datasets.length > 0 && (
        <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-4">
          <p className="text-white text-sm font-medium">Dataset-Accuracy</p>
          <select value={selectedDs} onChange={e => setSelectedDs(e.target.value)}
            className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-white text-sm focus:outline-none">
            {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
          </select>
          <button onClick={handleDatasetTest} disabled={dsLoading}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/40 text-blue-300 text-sm font-medium transition-all disabled:opacity-50">
            {dsLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            Accuracy messen
          </button>
          {dsError && <p className="text-red-300 text-sm flex items-center gap-2"><AlertTriangle className="w-4 h-4"/>{dsError}</p>}
          {dsResult && (
            <div className="rounded-xl border border-white/10 bg-black/20 p-4 space-y-1">
              <p className="text-2xl font-bold text-white">{(dsResult.accuracy * 100).toFixed(2)}%</p>
              <p className="text-gray-400 text-xs">Accuracy über {dsResult.total} Samples</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
