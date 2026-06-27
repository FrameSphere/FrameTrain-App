// YOLO Plugin – TrainPlugin.tsx
// Trainings-UI für YOLOv5 / YOLOv8 / YOLOv9 / YOLO11 Object Detection

import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  Target, Layers, Cpu, Zap, AlertTriangle, Info,
  ChevronDown, CheckCircle,
} from 'lucide-react';
import type { TrainPluginProps } from '../types';

// ── Typen ──────────────────────────────────────────────────────────────────

interface DatasetInfo {
  id: string; name: string; status: string;
  file_count: number; size_bytes: number;
  dataset_type?: string; split_info?: {
    train_count: number; val_count: number; test_count: number;
  } | null;
  storage_path?: string;
  schema_hint?: Record<string, unknown> | null;
}

// ── Presets ────────────────────────────────────────────────────────────────

const YOLO_PRESETS = [
  {
    id: 'fast',
    label: 'Schnell (Prototyp)',
    icon: '⚡',
    desc: 'Wenige Epochen, kleine Bildgröße. Für erste Tests.',
    epochs: 30, imgsz: 416, batch: 16, lr0: 0.01, lrf: 0.01,
    patience: 10, optimizer: 'SGD', augment: true,
  },
  {
    id: 'balanced',
    label: 'Ausgewogen',
    icon: '⚖️',
    desc: 'Gute Balance zwischen Geschwindigkeit und Genauigkeit.',
    epochs: 100, imgsz: 640, batch: 16, lr0: 0.01, lrf: 0.01,
    patience: 50, optimizer: 'SGD', augment: true,
  },
  {
    id: 'accurate',
    label: 'Genau (lang)',
    icon: '🎯',
    desc: 'Mehr Epochen, höhere Auflösung. Besser für Produktion.',
    epochs: 300, imgsz: 640, batch: 8, lr0: 0.01, lrf: 0.001,
    patience: 100, optimizer: 'AdamW', augment: true,
  },
  {
    id: 'custom',
    label: 'Eigene Einstellungen',
    icon: '🔧',
    desc: 'Alle Parameter manuell konfigurieren.',
    epochs: 100, imgsz: 640, batch: 16, lr0: 0.01, lrf: 0.01,
    patience: 50, optimizer: 'SGD', augment: true,
  },
];

// ── Helpers ────────────────────────────────────────────────────────────────

function formatBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1073741824) return `${(b / 1048576).toFixed(1)} MB`;
  return `${(b / 1073741824).toFixed(2)} GB`;
}

// ── Main ───────────────────────────────────────────────────────────────────

export default function YOLOTrainPlugin({ modelPath }: TrainPluginProps) {
  const modelId = modelPath.split('/').pop() ?? modelPath;

  // Dataset
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [datasetsLoading, setDatasetsLoading] = useState(true);

  // Preset
  const [activePreset, setActivePreset] = useState('balanced');
  const preset = YOLO_PRESETS.find(p => p.id === activePreset) ?? YOLO_PRESETS[1];

  // Hyperparameter
  const [epochs, setEpochs]       = useState(preset.epochs);
  const [imgsz, setImgsz]         = useState(preset.imgsz);
  const [batch, setBatch]         = useState(preset.batch);
  const [lr0, setLr0]             = useState(preset.lr0);
  const [lrf, setLrf]             = useState(preset.lrf);
  const [patience, setPatience]   = useState(preset.patience);
  const [optimizer, setOptimizer] = useState(preset.optimizer);
  const [augment, setAugment]     = useState(preset.augment);

  // YOLO-Variante (wird aus Modellpfad geraten)
  const yoloVariant = (() => {
    const id = modelPath.toLowerCase();
    if (id.includes('yolo11') || id.includes('yolo_11')) return 'YOLO11';
    if (id.includes('yolov9'))  return 'YOLOv9';
    if (id.includes('yolov8'))  return 'YOLOv8';
    if (id.includes('yolov5'))  return 'YOLOv5';
    return 'YOLO';
  })();

  // Datasets laden
  useEffect(() => {
    invoke<DatasetInfo[]>('list_datasets_for_model', { modelId })
      .then(list => {
        setDatasets(list.filter(d => d.dataset_type === 'yolo_bbox' || d.dataset_type === 'pre_split' || d.dataset_type === 'unknown'));
        if (list.length > 0) setSelectedDataset(list[0].id);
      })
      .catch(() => {})
      .finally(() => setDatasetsLoading(false));
  }, [modelId]);

  // Preset anwenden
  const applyPreset = (pid: string) => {
    const p = YOLO_PRESETS.find(x => x.id === pid);
    if (!p) return;
    setActivePreset(pid);
    setEpochs(p.epochs); setImgsz(p.imgsz); setBatch(p.batch);
    setLr0(p.lr0); setLrf(p.lrf); setPatience(p.patience);
    setOptimizer(p.optimizer); setAugment(p.augment);
  };

  const ds = datasets.find(d => d.id === selectedDataset);
  const yamlPath = (ds?.schema_hint as Record<string, unknown> | null)?.dataset_yaml_path as string | undefined
    ?? (ds?.storage_path ? `${ds.storage_path}/dataset.yaml` : undefined);

  // YOLO braucht immer einen gesplitteten Datensatz
  const dsReady = ds && (ds.status === 'split' || ds.dataset_type === 'pre_split');

  const handleStartTraining = async () => {
    if (!selectedDataset || !dsReady) return;
    const pluginConfig = {
      task_type:     'detect',
      yolo_variant:  yoloVariant,
      epochs,
      imgsz,
      batch,
      lr0,
      lrf,
      patience,
      optimizer,
      augment,
      dataset_yaml_path: yamlPath ?? '',
    };
    try {
      await invoke('start_training', {
        modelId,
        modelName: `${yoloVariant} Object Detection`,
        datasetId: selectedDataset,
        datasetName: ds?.name ?? '',
        config: {
          task_type: 'yolo',
          dataset_path: ds?.storage_path ?? '',
          plugin_config: pluginConfig,
          epochs,
          batch_size: batch,
          learning_rate: lr0,
        },
      });
    } catch (e) {
      console.error('Training start error:', e);
    }
  };

  return (
    <div className="space-y-6 text-sm text-gray-300">

      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="p-2.5 rounded-xl bg-orange-500/15 border border-orange-500/20">
          <Target className="w-5 h-5 text-orange-400" />
        </div>
        <div>
          <h3 className="font-semibold text-white">{yoloVariant} Object Detection</h3>
          <p className="text-xs text-gray-500">Ultralytics · Bounding-Box-Erkennung</p>
        </div>
      </div>

      {/* Dataset auswählen */}
      <section className="space-y-2">
        <label className="block font-medium text-gray-300">Dataset</label>
        {datasetsLoading ? (
          <div className="text-gray-500 text-xs">Lade Datasets…</div>
        ) : datasets.length === 0 ? (
          <div className="flex items-start gap-2 p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs">
            <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span>Kein YOLO-Dataset vorhanden. Importiere zuerst ein Dataset (YOLO-Format) über den Dataset-Manager.</span>
          </div>
        ) : (
          <div className="relative">
            <select
              value={selectedDataset}
              onChange={e => setSelectedDataset(e.target.value)}
              className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm appearance-none focus:outline-none focus:border-white/30"
            >
              {datasets.map(d => (
                <option key={d.id} value={d.id} className="bg-slate-900">
                  {d.name} — {d.status === 'split' ? `✓ Split (${d.split_info?.train_count ?? '?'} Train)` : '⚠ Kein Split'} · {formatBytes(d.size_bytes)}
                </option>
              ))}
            </select>
            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
          </div>
        )}

        {/* Dataset-Status Banner */}
        {ds && !dsReady && (
          <div className="flex items-start gap-2 p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs">
            <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span>Dataset ist noch nicht gesplittet. Gehe zum Dataset-Manager und teile es in Train/Val/Test auf.</span>
          </div>
        )}
        {ds && dsReady && (
          <div className="flex items-start gap-2 p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-xs">
            <CheckCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              <span>Dataset bereit. </span>
              {ds.split_info && (
                <span>Train: {ds.split_info.train_count} · Val: {ds.split_info.val_count} · Test: {ds.split_info.test_count} Bilder</span>
              )}
              {yamlPath && (
                <p className="text-emerald-400/60 font-mono text-[10px] truncate">yaml: {yamlPath}</p>
              )}
            </div>
          </div>
        )}
      </section>

      {/* Presets */}
      <section className="space-y-2">
        <label className="block font-medium text-gray-300">Preset</label>
        <div className="grid grid-cols-2 gap-2">
          {YOLO_PRESETS.map(p => (
            <button
              key={p.id}
              onClick={() => applyPreset(p.id)}
              className={`flex items-start gap-2 p-3 rounded-xl border text-left transition-all ${
                activePreset === p.id
                  ? 'bg-orange-500/10 border-orange-500/30 text-orange-300'
                  : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
              }`}
            >
              <span className="text-lg leading-none flex-shrink-0">{p.icon}</span>
              <div>
                <p className="font-medium text-xs text-white">{p.label}</p>
                <p className="text-[10px] text-gray-500 mt-0.5 leading-tight">{p.desc}</p>
              </div>
            </button>
          ))}
        </div>
      </section>

      {/* Hyperparameter */}
      <section className="space-y-3">
        <label className="block font-medium text-gray-300 flex items-center gap-2">
          <Layers className="w-4 h-4 text-gray-500" /> Hyperparameter
        </label>

        <div className="grid grid-cols-2 gap-3">
          {/* Epochen */}
          <div className="space-y-1">
            <label className="block text-xs text-gray-500">Epochen</label>
            <input
              type="number" min={1} max={1000} value={epochs}
              onChange={e => setEpochs(Number(e.target.value))}
              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-white/30"
            />
          </div>

          {/* Bildgröße */}
          <div className="space-y-1">
            <label className="block text-xs text-gray-500">Bildgröße (imgsz)</label>
            <select
              value={imgsz}
              onChange={e => setImgsz(Number(e.target.value))}
              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm appearance-none focus:outline-none focus:border-white/30"
            >
              {[320, 416, 512, 640, 768, 1024, 1280].map(v => (
                <option key={v} value={v} className="bg-slate-900">{v}×{v}</option>
              ))}
            </select>
          </div>

          {/* Batch Size */}
          <div className="space-y-1">
            <label className="block text-xs text-gray-500">Batch-Size</label>
            <select
              value={batch}
              onChange={e => setBatch(Number(e.target.value))}
              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm appearance-none focus:outline-none focus:border-white/30"
            >
              {[1, 2, 4, 8, 16, 32, 64].map(v => (
                <option key={v} value={v} className="bg-slate-900">{v}</option>
              ))}
            </select>
          </div>

          {/* Optimizer */}
          <div className="space-y-1">
            <label className="block text-xs text-gray-500">Optimizer</label>
            <select
              value={optimizer}
              onChange={e => setOptimizer(e.target.value)}
              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm appearance-none focus:outline-none focus:border-white/30"
            >
              {['SGD', 'Adam', 'AdamW'].map(v => (
                <option key={v} value={v} className="bg-slate-900">{v}</option>
              ))}
            </select>
          </div>

          {/* LR Initial */}
          <div className="space-y-1">
            <label className="block text-xs text-gray-500">LR Start (lr0)</label>
            <input
              type="number" step={0.001} min={0.0001} max={0.1} value={lr0}
              onChange={e => setLr0(Number(e.target.value))}
              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-white/30"
            />
          </div>

          {/* LR Final */}
          <div className="space-y-1">
            <label className="block text-xs text-gray-500">LR Ende (lrf)</label>
            <input
              type="number" step={0.001} min={0.0001} max={0.1} value={lrf}
              onChange={e => setLrf(Number(e.target.value))}
              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-white/30"
            />
          </div>

          {/* Patience */}
          <div className="space-y-1">
            <label className="block text-xs text-gray-500">Early-Stop Patience</label>
            <input
              type="number" min={0} max={500} value={patience}
              onChange={e => setPatience(Number(e.target.value))}
              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-white/30"
            />
            <p className="text-[10px] text-gray-600">0 = kein Early Stopping</p>
          </div>

          {/* Augmentierung */}
          <div className="space-y-1 flex flex-col justify-end">
            <label className="flex items-center gap-2 cursor-pointer">
              <div
                onClick={() => setAugment(a => !a)}
                className={`relative w-10 h-5 rounded-full transition-colors flex-shrink-0 ${augment ? 'bg-orange-500' : 'bg-white/10'}`}
              >
                <span className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${augment ? 'translate-x-5' : 'translate-x-0.5'}`} />
              </div>
              <span className="text-xs text-gray-400">Augmentierung</span>
            </label>
            <p className="text-[10px] text-gray-600">Mosaic, Flip, HSV etc.</p>
          </div>
        </div>
      </section>

      {/* Info: DevTrain für Custom Scripts */}
      <div className="flex items-start gap-2 p-3 rounded-xl bg-blue-500/8 border border-blue-500/20 text-blue-300 text-xs">
        <Info className="w-4 h-4 flex-shrink-0 mt-0.5" />
        <div>
          <p className="font-medium">Erweitertes YOLO-Training</p>
          <p className="text-blue-400/60 mt-0.5">
            Für eigene Ultralytics-Skripte oder spezielle Konfigurationen nutze den <strong>DevTrain-Modus</strong>.
            Die dataset.yaml liegt unter: <code className="font-mono">{yamlPath ?? '<dataset_pfad>/dataset.yaml'}</code>
          </p>
        </div>
      </div>

      {/* Start-Button */}
      <button
        onClick={handleStartTraining}
        disabled={!dsReady || !selectedDataset}
        className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r from-orange-500 to-amber-500 text-white text-sm font-medium hover:opacity-90 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
      >
        <Cpu className="w-4 h-4" />
        {yoloVariant} Training starten
      </button>

      {!dsReady && selectedDataset && (
        <p className="text-xs text-center text-gray-500">
          Dataset muss zuerst gesplittet werden (Train/Val/Test).
        </p>
      )}
    </div>
  );
}
