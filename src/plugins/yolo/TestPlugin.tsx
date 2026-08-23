// YOLO Plugin – TestPlugin.tsx
// Test-UI für YOLO Object Detection: Einzelbild-Inferenz

import { useState, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { Upload, Target, AlertTriangle, Loader2, FolderOpen } from 'lucide-react';
import type { TestPluginProps } from '../types';

interface Detection {
  label: string;
  confidence: number;
  bbox: [number, number, number, number]; // x1, y1, x2, y2
}

interface InferenceResult {
  detections: Detection[];
  inference_time_ms: number;
  image_path: string;
}

export default function YOLOTestPlugin({ modelPath, modelName, versionId }: TestPluginProps) {
  const [imagePath, setImagePath] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [iouThreshold, setIouThreshold] = useState(0.45);

  const handlePickImage = async () => {
    const sel = await open({
      filters: [{ name: 'Bilder', extensions: ['jpg', 'jpeg', 'png', 'bmp', 'webp'] }],
      multiple: false,
    });
    if (sel && typeof sel === 'string') setImagePath(sel);
  };

  const handleRunInference = async () => {
    if (!imagePath) return;
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const res = await invoke<InferenceResult>('run_yolo_inference', {
        modelPath,
        imagePath,
        confThreshold,
        iouThreshold,
        // Ohne die Version testet die Inferenz das Basismodell statt des
        // selbst trainierten Stands.
        versionId,
      });
      setResult(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-5 text-sm text-gray-300">

      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="p-2.5 rounded-xl bg-orange-500/15 border border-orange-500/20">
          <Target className="w-5 h-5 text-orange-400" />
        </div>
        <div>
          <h3 className="font-semibold text-white">{modelName} – Test</h3>
          <p className="text-xs text-gray-500">Einzelbild-Inferenz · Bounding Box Detection</p>
        </div>
      </div>

      {/* Schwellwerte */}
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <label className="text-xs text-gray-500">Konfidenz-Schwelle</label>
          <input
            type="range" min={0.01} max={0.99} step={0.01} value={confThreshold}
            onChange={e => setConfThreshold(Number(e.target.value))}
            className="w-full" style={{ accentColor: '#f97316' }}
          />
          <span className="text-xs text-orange-400">{(confThreshold * 100).toFixed(0)}%</span>
        </div>
        <div className="space-y-1">
          <label className="text-xs text-gray-500">IoU-Schwelle (NMS)</label>
          <input
            type="range" min={0.01} max={0.99} step={0.01} value={iouThreshold}
            onChange={e => setIouThreshold(Number(e.target.value))}
            className="w-full" style={{ accentColor: '#f97316' }}
          />
          <span className="text-xs text-orange-400">{(iouThreshold * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Bild auswählen */}
      <div
        className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all ${
          imagePath ? 'border-emerald-500/50 bg-emerald-500/5' : 'border-white/15 hover:border-white/30'
        }`}
      >
        {imagePath ? (
          <div className="space-y-2">
            <p className="text-white font-medium text-xs truncate">{imagePath.split('/').pop()}</p>
            <button
              onClick={handlePickImage}
              className="text-xs text-gray-400 hover:text-white underline transition-colors"
            >
              Anderes Bild wählen
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            <Upload className="w-8 h-8 text-gray-500 mx-auto" />
            <p className="text-gray-400 text-xs">Bild für Inferenz auswählen</p>
            <button
              onClick={handlePickImage}
              className="inline-flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/15 rounded-xl text-white text-xs transition-all"
            >
              <FolderOpen className="w-3.5 h-3.5" /> Bild öffnen
            </button>
          </div>
        )}
      </div>

      {/* Run */}
      <button
        onClick={handleRunInference}
        disabled={!imagePath || running}
        className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r from-orange-500 to-amber-500 text-white text-sm font-medium hover:opacity-90 transition-all disabled:opacity-40"
      >
        {running
          ? <><Loader2 className="w-4 h-4 animate-spin" /> Inferenz läuft…</>
          : <><Target className="w-4 h-4" /> Inferenz starten</>
        }
      </button>

      {/* Fehler */}
      {error && (
        <div className="flex items-start gap-2 p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-300 text-xs">
          <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {/* Ergebnis */}
      {result && (
        <div className="space-y-3 p-4 rounded-2xl border border-white/10 bg-white/5">
          <div className="flex items-center justify-between">
            <p className="font-medium text-white">
              {result.detections.length} Objekt{result.detections.length !== 1 ? 'e' : ''} erkannt
            </p>
            <span className="text-xs text-gray-500">{result.inference_time_ms.toFixed(1)} ms</span>
          </div>
          {result.detections.length === 0 ? (
            <p className="text-gray-500 text-xs">Keine Objekte über dem Konfidenz-Schwellwert.</p>
          ) : (
            <div className="space-y-1.5">
              {result.detections.map((det, i) => (
                <div key={i} className="flex items-center justify-between px-3 py-2 rounded-lg bg-white/5 border border-white/10">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-orange-400 flex-shrink-0" />
                    <span className="text-white text-xs font-medium">{det.label}</span>
                  </div>
                  <span className="text-orange-400 text-xs font-mono">
                    {(det.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
