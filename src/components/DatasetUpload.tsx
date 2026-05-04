// DatasetUpload.tsx – Dataset-Manager
// Portiert & erweitert aus desktop-app2

import { useState, useEffect, useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import {
  Upload, FolderOpen, Download, Trash2, Search,
  HardDrive, Cloud, CheckCircle, Loader2, Database,
  Calendar, ExternalLink, X, RefreshCw, ChevronDown,
  Scissors, Layers, FileText, Filter,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';
import { usePageContext } from '../contexts/PageContext';
import DatasetFileManager from './DatasetFileManager';

// ── Types ──────────────────────────────────────────────────────────────────

interface ModelInfo { id: string; name: string; source: string; }

interface SplitInfo {
  train_count: number; val_count: number; test_count: number;
  train_ratio: number; val_ratio: number; test_ratio: number;
}

interface DatasetInfo {
  id: string; name: string; model_id: string;
  source: 'local' | 'huggingface';
  source_path: string | null;
  size_bytes: number; file_count: number; created_at: string;
  status: 'unused' | 'split';
  split_info: SplitInfo | null;
  training_count: number; last_used_at: string | null;
}

interface HuggingFaceDataset {
  id: string; author?: string;
  downloads?: number; likes?: number; tags?: string[];
}

interface FilterOptions { tasks: string[]; languages: string[]; sizes: string[]; }

type ImportMode = 'local' | 'huggingface';

// ── Helpers ────────────────────────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(ds: string): string {
  return new Date(ds).toLocaleDateString('de-DE', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

function formatDownloads(n: number | undefined): string {
  if (!n) return '0';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return n.toString();
}

// ── Delete Dialog ──────────────────────────────────────────────────────────

function DeleteDialog({ name, onConfirm, onCancel }: { name: string; onConfirm: () => void; onCancel: () => void }) {
  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center" style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(8px)' }}>
      <div className="bg-slate-900 border border-white/10 rounded-2xl shadow-2xl w-full max-w-sm mx-4 overflow-hidden">
        <div className="h-1 bg-gradient-to-r from-red-500 to-orange-500" />
        <div className="p-6 space-y-5">
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 rounded-full bg-red-500/20 border border-red-500/40 flex items-center justify-center flex-shrink-0">
              <Trash2 className="w-5 h-5 text-red-400" />
            </div>
            <div>
              <h2 className="text-white font-semibold text-lg">Dataset löschen?</h2>
              <p className="text-gray-400 text-sm mt-1.5 leading-relaxed">
                <span className="text-white">„{name}"</span> und alle zugehörigen Dateien werden unwiderruflich entfernt.
              </p>
            </div>
          </div>
          <div className="flex gap-3">
            <button onClick={onCancel} className="flex-1 py-2.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-white text-sm font-medium transition-all">Abbrechen</button>
            <button onClick={onConfirm} className="flex-1 py-2.5 bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 rounded-xl text-red-300 text-sm font-medium transition-all">Löschen</button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Main ───────────────────────────────────────────────────────────────────

export default function DatasetUpload() {
  const { currentTheme } = useTheme();
  const { success, error, warning, info } = useNotification();
  const { setCurrentPageContent } = usePageContext();

  // Core state
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loading, setLoading] = useState(true);

  // Delete
  const [deleteTarget, setDeleteTarget] = useState<DatasetInfo | null>(null);

  // Import modal
  const [showImportModal, setShowImportModal] = useState(false);
  const [importMode, setImportMode] = useState<ImportMode>('local');

  // Local import
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [datasetName, setDatasetName] = useState('');
  const [dirInfo, setDirInfo] = useState<{ size: number; files: number } | null>(null);
  const [importing, setImporting] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  // HuggingFace
  const [hfQuery, setHfQuery] = useState('');
  const [hfResults, setHfResults] = useState<HuggingFaceDataset[]>([]);
  const [hfSearching, setHfSearching] = useState(false);
  const [selectedHfDataset, setSelectedHfDataset] = useState<HuggingFaceDataset | null>(null);
  const [hfDatasetName, setHfDatasetName] = useState('');
  const [downloading, setDownloading] = useState(false);
  const [downloadStatus, setDownloadStatus] = useState('');
  const downloadIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // HF Filters
  const [filterOptions, setFilterOptions] = useState<FilterOptions | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [filterTask, setFilterTask] = useState('');
  const [filterLanguage, setFilterLanguage] = useState('');
  const [filterSize, setFilterSize] = useState('');

  // Split modal
  const [showSplitModal, setShowSplitModal] = useState(false);
  const [datasetToSplit, setDatasetToSplit] = useState<DatasetInfo | null>(null);
  const [trainRatio, setTrainRatio] = useState(0.8);
  const [valRatio, setValRatio] = useState(0.1);
  const [testRatio, setTestRatio] = useState(0.1);
  const [splitting, setSplitting] = useState(false);

  // Halve modal
  const [showHalveModal, setShowHalveModal] = useState(false);
  const [datasetToHalve, setDatasetToHalve] = useState<DatasetInfo | null>(null);
  const [halving, setHalving] = useState(false);

  // File manager
  const [fileManagerDataset, setFileManagerDataset] = useState<DatasetInfo | null>(null);

  const searchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Init ──

  useEffect(() => { initLoad(); }, []);
  useEffect(() => { if (selectedModelId) loadDatasets(); }, [selectedModelId]);

  // AI coach context
  useEffect(() => {
    const selModel = models.find(m => m.id === selectedModelId);
    setCurrentPageContent([
      '=== FrameTrain Dataset-Manager ===',
      `Modell: ${selModel?.name ?? 'keins'}`,
      `Datasets: ${datasets.length}`,
      ...datasets.map(d =>
        `• ${d.name} | ${d.status === 'split' ? '✅ Split' : '⚠️ Kein Split'} | ${d.file_count} Dateien | ${formatBytes(d.size_bytes)}`
      ),
    ].join('\n'));
  }, [models, selectedModelId, datasets, setCurrentPageContent]);

  // Debounced HF search
  useEffect(() => {
    if (searchTimeoutRef.current) clearTimeout(searchTimeoutRef.current);
    if (hfQuery.trim().length < 2) { setHfResults([]); setHfSearching(false); return; }
    setHfSearching(true);
    searchTimeoutRef.current = setTimeout(async () => {
      try {
        const res = await invoke<HuggingFaceDataset[]>('search_huggingface_datasets', {
          query: hfQuery.trim(), limit: 15,
          filterTask: filterTask || null,
          filterLanguage: filterLanguage || null,
          filterSize: filterSize || null,
        });
        setHfResults(res);
      } catch { /* ignore */ } finally { setHfSearching(false); }
    }, 300);
    return () => { if (searchTimeoutRef.current) clearTimeout(searchTimeoutRef.current); };
  }, [hfQuery, filterTask, filterLanguage, filterSize]);

  // ── Load ──

  const initLoad = async () => {
    try {
      const list = await invoke<ModelInfo[]>('list_models');
      setModels(list);
      if (list.length > 0) setSelectedModelId(list[0].id);
      try {
        const opts = await invoke<FilterOptions>('get_dataset_filter_options');
        setFilterOptions(opts);
      } catch { /* optional */ }
    } catch (err: unknown) {
      error('Fehler beim Laden der Modelle', String(err));
    } finally {
      setLoading(false);
    }
  };

  const loadDatasets = async () => {
    if (!selectedModelId) return;
    try {
      const list = await invoke<DatasetInfo[]>('list_datasets_for_model', { modelId: selectedModelId });
      setDatasets(list);
    } catch (err: unknown) {
      error('Fehler beim Laden der Datasets', String(err));
    }
  };

  // ── Local Import ──

  const validateAndSetPath = async (path: string) => {
    setSelectedPath(path);
    setDatasetName(path.split(/[/\\]/).pop() ?? 'Dataset');
    try {
      const [size, files] = await invoke<[number, number]>('get_directory_size', { path });
      setDirInfo({ size, files });
    } catch {
      setDirInfo(null);
    }
  };

  const handleBrowseFolder = async () => {
    try {
      const sel = await open({ directory: true, multiple: false, title: 'Dataset-Ordner auswählen' });
      if (sel && typeof sel === 'string') await validateAndSetPath(sel);
    } catch (err: unknown) { error('Fehler', String(err)); }
  };

  const handleLocalImport = async () => {
    if (!selectedPath || !datasetName.trim() || !selectedModelId) {
      warning('Fehlende Angaben', 'Ordner, Name und Modell werden benötigt.');
      return;
    }
    setImporting(true);
    try {
      const ds = await invoke<DatasetInfo>('import_local_dataset', {
        sourcePath: selectedPath, datasetName: datasetName.trim(), modelId: selectedModelId,
      });
      success('Dataset importiert!', `„${ds.name}" wurde hinzugefügt.`);
      closeModal();
      await loadDatasets();
    } catch (err: unknown) {
      error('Import fehlgeschlagen', String(err));
    } finally {
      setImporting(false);
    }
  };

  // Drag & Drop
  const handleDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); }, []);
  const handleDragLeave = useCallback((e: React.DragEvent) => { e.preventDefault(); setIsDragging(false); }, []);
  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault(); setIsDragging(false);
    const file = e.dataTransfer.items?.[0]?.getAsFile?.();
    const path = file && (file as unknown as { path?: string }).path;
    if (path) await validateAndSetPath(path);
    else info('Drag & Drop', 'Bitte nutze den „Ordner durchsuchen"-Button.');
  }, []);

  // ── HuggingFace ──

  const handleHfSelect = (ds: HuggingFaceDataset) => {
    setSelectedHfDataset(ds);
    setHfDatasetName(ds.id.split('/').pop() ?? ds.id);
  };

  const handleHfDownload = async () => {
    if (!selectedHfDataset || !hfDatasetName.trim() || !selectedModelId) {
      warning('Fehlende Angaben', 'Dataset, Name und Modell werden benötigt.');
      return;
    }
    setDownloading(true);
    setDownloadStatus('Verbinde mit Hugging Face…');
    const steps = ['Lade Konfiguration…', 'Lade Trainingsdaten…', 'Lade Validierungsdaten…', 'Speichere Dateien…', 'Fast fertig…'];
    let si = 0;
    downloadIntervalRef.current = setInterval(() => {
      si = (si + 1) % steps.length;
      setDownloadStatus(steps[si]);
    }, 2000);

    try {
      const ds = await invoke<DatasetInfo>('download_huggingface_dataset', {
        repoId: selectedHfDataset.id, datasetName: hfDatasetName.trim(), modelId: selectedModelId,
      });
      clearInterval(downloadIntervalRef.current!);
      success('Download abgeschlossen!', `„${ds.name}" wurde heruntergeladen.`);
      closeModal();
      await loadDatasets();
    } catch (err: unknown) {
      clearInterval(downloadIntervalRef.current!);
      error('Download fehlgeschlagen', String(err));
      setDownloadStatus('');
    } finally {
      setDownloading(false);
    }
  };

  const handleCancelDownload = () => {
    setDownloading(false);
    setDownloadStatus('');
    if (downloadIntervalRef.current) clearInterval(downloadIntervalRef.current);
    info('Abgebrochen', 'Der Download wurde abgebrochen.');
  };

  // ── Split ──

  const openSplitModal = (ds: DatasetInfo) => {
    setDatasetToSplit(ds);
    setTrainRatio(0.8); setValRatio(0.1); setTestRatio(0.1);
    setShowSplitModal(true);
  };

  const handleSplit = async () => {
    if (!datasetToSplit || !selectedModelId) return;
    if (Math.abs(trainRatio + valRatio + testRatio - 1) > 0.01) {
      warning('Ungültige Aufteilung', 'Die Summe muss 100% ergeben.'); return;
    }
    setSplitting(true);
    try {
      await invoke('split_dataset', {
        datasetId: datasetToSplit.id, modelId: selectedModelId, trainRatio, valRatio, testRatio,
      });
      success('Aufgeteilt!', `„${datasetToSplit.name}" wurde in Train/Val/Test aufgeteilt.`);
      setShowSplitModal(false); setDatasetToSplit(null);
      await loadDatasets();
    } catch (err: unknown) {
      error('Split fehlgeschlagen', String(err));
    } finally {
      setSplitting(false);
    }
  };

  // ── Halve ──

  const handleHalve = async () => {
    if (!datasetToHalve || !selectedModelId) return;
    setHalving(true);
    try {
      const result = await invoke<{ dataset_a: DatasetInfo; dataset_b: DatasetInfo }>(
        'split_dataset_in_half', { datasetId: datasetToHalve.id, modelId: selectedModelId }
      );
      success('Geteilt!', `„${result.dataset_a.name}" und „${result.dataset_b.name}" erstellt.`);
      setShowHalveModal(false); setDatasetToHalve(null);
      await loadDatasets();
    } catch (err: unknown) {
      error('Halbieren fehlgeschlagen', String(err));
    } finally {
      setHalving(false);
    }
  };

  // ── Delete ──

  const handleDeleteConfirm = async () => {
    if (!deleteTarget) return;
    try {
      await invoke('delete_dataset', { datasetId: deleteTarget.id, modelId: deleteTarget.model_id });
      success('Gelöscht', `„${deleteTarget.name}" wurde entfernt.`);
      await loadDatasets();
    } catch (err: unknown) {
      error('Löschen fehlgeschlagen', String(err));
    } finally {
      setDeleteTarget(null);
    }
  };

  // ── Helpers ──

  const closeModal = () => {
    setShowImportModal(false);
    setSelectedPath(null); setDatasetName(''); setDirInfo(null);
    setSelectedHfDataset(null); setHfDatasetName('');
    setHfQuery(''); setHfResults([]); setDownloadStatus('');
    setFilterTask(''); setFilterLanguage(''); setFilterSize('');
  };

  const selectedModel = models.find(m => m.id === selectedModelId);

  // ── Early states ──

  if (loading) return (
    <div className="flex items-center justify-center py-24">
      <Loader2 className="w-8 h-8 text-gray-500 animate-spin" />
    </div>
  );

  if (models.length === 0) return (
    <div className="space-y-6">
      <div><h1 className="text-2xl font-bold text-white">Datasets</h1><p className="text-gray-400 mt-1">Verwalte deine Trainingsdaten</p></div>
      <div className="rounded-2xl border border-white/10 bg-white/5 p-16 text-center space-y-4">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-white/5 border border-white/10">
          <Layers className="w-8 h-8 text-gray-500" />
        </div>
        <div>
          <h3 className="text-white font-semibold text-lg">Kein Modell vorhanden</h3>
          <p className="text-gray-400 text-sm mt-1">Füge zuerst ein Modell hinzu, bevor du Datasets importierst.</p>
        </div>
      </div>
    </div>
  );

  // ── Main Render ──

  return (
    <div className="space-y-6">

      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Datasets</h1>
          <p className="text-gray-400 mt-1">Verwalte Trainingsdaten für deine Modelle</p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={loadDatasets} className="p-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white transition-all" title="Aktualisieren">
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowImportModal(true)}
            disabled={!selectedModelId}
            className={`flex items-center gap-2 px-4 py-2 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-xl text-white text-sm font-medium hover:opacity-90 transition-all disabled:opacity-40`}
          >
            <Upload className="w-4 h-4" /> Dataset hinzufügen
          </button>
        </div>
      </div>

      {/* ── Model Selector ── */}
      <div className="rounded-2xl border border-white/10 bg-white/5 p-4 space-y-2">
        <label className="block text-sm font-medium text-gray-300">Modell auswählen</label>
        <div className="relative">
          <select
            value={selectedModelId ?? ''}
            onChange={e => setSelectedModelId(e.target.value)}
            className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white text-sm appearance-none cursor-pointer focus:outline-none focus:border-white/20 transition-all"
          >
            {models.map(m => (
              <option key={m.id} value={m.id} className="bg-slate-900">
                {m.name} ({m.source === 'huggingface' ? 'HF' : 'Lokal'})
              </option>
            ))}
          </select>
          <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
        </div>
      </div>

      {/* ── Dataset Grid ── */}
      {datasets.length === 0 ? (
        <div className="rounded-2xl border border-white/10 bg-white/5 p-16 text-center space-y-4">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-white/5 border border-white/10">
            <Database className="w-8 h-8 text-gray-500" />
          </div>
          <div>
            <h3 className="text-white font-semibold text-lg">Keine Datasets</h3>
            <p className="text-gray-400 text-sm mt-1">Füge ein Dataset für „{selectedModel?.name}" hinzu.</p>
          </div>
          <button
            onClick={() => setShowImportModal(true)}
            className={`inline-flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-xl text-white text-sm font-medium hover:opacity-90 transition-all`}
          >
            <Upload className="w-4 h-4" /> Erstes Dataset hinzufügen
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {datasets.map(ds => (
            <DatasetCard
              key={ds.id}
              dataset={ds}
              gradientClass={currentTheme.colors.gradient}
              onDelete={() => setDeleteTarget(ds)}
              onSplit={() => openSplitModal(ds)}
              onHalve={() => { setDatasetToHalve(ds); setShowHalveModal(true); }}
              onFiles={() => setFileManagerDataset(ds)}
            />
          ))}
        </div>
      )}

      {/* ── Modals ── */}

      {/* Delete */}
      {deleteTarget && (
        <DeleteDialog name={deleteTarget.name} onConfirm={handleDeleteConfirm} onCancel={() => setDeleteTarget(null)} />
      )}

      {/* File Manager */}
      {fileManagerDataset && (
        <DatasetFileManager
          datasetId={fileManagerDataset.id}
          datasetName={fileManagerDataset.name}
          onClose={() => { setFileManagerDataset(null); loadDatasets(); }}
        />
      )}

      {/* Split Modal */}
      {showSplitModal && datasetToSplit && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-md">
            <div className="flex items-center justify-between px-6 py-5 border-b border-white/10">
              <div>
                <h2 className="text-xl font-bold text-white">Dataset aufteilen</h2>
                <p className="text-sm text-gray-400 mt-0.5">{datasetToSplit.name}</p>
              </div>
              <button onClick={() => { setShowSplitModal(false); setDatasetToSplit(null); }} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-6 space-y-6">
              <p className="text-gray-400 text-sm">{datasetToSplit.file_count} Dateien in Train / Val / Test aufteilen.</p>

              {/* Sliders */}
              {([
                { label: 'Training', color: '#3b82f6', ratio: trainRatio, set: (v: number) => {
                  const r = 1 - v; const vp = valRatio / (valRatio + testRatio) || 0.5;
                  setTrainRatio(v); setValRatio(r * vp); setTestRatio(r * (1 - vp));
                }},
                { label: 'Validierung', color: '#a855f7', ratio: valRatio, set: (v: number) => {
                  const r = 1 - v; const tp = trainRatio / (trainRatio + testRatio) || 0.5;
                  setValRatio(v); setTrainRatio(r * tp); setTestRatio(r * (1 - tp));
                }},
                { label: 'Test', color: '#10b981', ratio: testRatio, set: (v: number) => {
                  const r = 1 - v; const tp = trainRatio / (trainRatio + valRatio) || 0.5;
                  setTestRatio(v); setTrainRatio(r * tp); setValRatio(r * (1 - tp));
                }},
              ] as const).map(({ label, color, ratio, set }) => (
                <div key={label} className="space-y-1.5">
                  <div className="flex justify-between text-sm">
                    <span style={{ color }}>{label}</span>
                    <span className="text-white">{Math.round(ratio * 100)}%</span>
                  </div>
                  <input
                    type="range" min="0" max="100" value={Math.round(ratio * 100)}
                    onChange={e => set(parseInt(e.target.value) / 100)}
                    className="w-full"
                    style={{ accentColor: color }}
                  />
                </div>
              ))}

              {/* Preview */}
              <div className="grid grid-cols-3 gap-2 text-center text-sm">
                {[
                  { label: 'Train', color: 'blue', count: Math.round(datasetToSplit.file_count * trainRatio) },
                  { label: 'Val', color: 'purple', count: Math.round(datasetToSplit.file_count * valRatio) },
                  { label: 'Test', color: 'green', count: Math.round(datasetToSplit.file_count * testRatio) },
                ].map(({ label, color, count }) => (
                  <div key={label} className={`p-3 rounded-xl bg-${color}-500/10`}>
                    <div className={`text-${color}-400 font-bold text-lg`}>{count}</div>
                    <div className="text-gray-500 text-xs">{label}</div>
                  </div>
                ))}
              </div>

              <button
                onClick={handleSplit}
                disabled={splitting}
                className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r ${currentTheme.colors.gradient} text-white text-sm font-medium hover:opacity-90 transition-all disabled:opacity-50`}
              >
                {splitting ? <><Loader2 className="w-4 h-4 animate-spin" /> Teile auf…</> : <><Scissors className="w-4 h-4" /> Dataset aufteilen</>}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Halve Modal */}
      {showHalveModal && datasetToHalve && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-md">
            <div className="flex items-center justify-between px-6 py-5 border-b border-white/10">
              <div>
                <h2 className="text-xl font-bold text-white">Dataset halbieren</h2>
                <p className="text-sm text-gray-400 mt-0.5">{datasetToHalve.name}</p>
              </div>
              <button onClick={() => { setShowHalveModal(false); setDatasetToHalve(null); }} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-6 space-y-5">
              <div className="p-4 rounded-xl border border-amber-500/30 bg-amber-500/10">
                <p className="text-amber-300 text-sm font-medium">⚠️ Wofür ist das?</p>
                <p className="text-gray-300 text-sm mt-1">Wenn das Training mit einem RAM-Fehler abbricht, hilft es den Datensatz zu halbieren.</p>
              </div>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: 'Hälfte 1', count: Math.ceil(datasetToHalve.file_count / 2) },
                  { label: 'Hälfte 2', count: Math.floor(datasetToHalve.file_count / 2) },
                ].map(({ label, count }) => (
                  <div key={label} className="p-3 rounded-xl bg-white/5 text-center">
                    <div className="text-white font-bold text-lg">{count}</div>
                    <div className="text-gray-500 text-xs mt-0.5">Dateien ({label})</div>
                  </div>
                ))}
              </div>
              <p className="text-gray-500 text-xs">Das Original bleibt erhalten.</p>
              <button
                onClick={handleHalve}
                disabled={halving}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r from-orange-500 to-amber-500 text-white text-sm font-medium hover:opacity-90 transition-all disabled:opacity-50"
              >
                {halving ? <><Loader2 className="w-4 h-4 animate-spin" /> Teile auf…</> : <>½ In zwei Hälften teilen</>}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Import Modal */}
      {showImportModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">

            {/* Modal Header */}
            <div className="px-6 py-5 border-b border-white/10 flex items-start justify-between flex-shrink-0">
              <div>
                <h2 className="text-xl font-bold text-white">Dataset hinzufügen</h2>
                <p className="text-sm text-gray-400 mt-0.5">für: {selectedModel?.name}</p>
              </div>
              <button onClick={closeModal} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all">
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Tabs */}
            <div className="flex border-b border-white/10 flex-shrink-0">
              {([
                { mode: 'local' as ImportMode, icon: <HardDrive className="w-4 h-4" />, label: 'Lokaler Ordner' },
                { mode: 'huggingface' as ImportMode, icon: <Cloud className="w-4 h-4" />, label: 'Hugging Face' },
              ]).map(({ mode, icon, label }) => (
                <button
                  key={mode}
                  onClick={() => setImportMode(mode)}
                  className={`flex-1 flex items-center justify-center gap-2 px-4 py-3.5 text-sm font-medium transition-all border-b-2 ${
                    importMode === mode ? 'text-white' : 'text-gray-400 hover:text-white border-transparent'
                  }`}
                  style={importMode === mode ? { borderColor: currentTheme.colors.primary, color: currentTheme.colors.primary } : {}}
                >
                  {icon}{label}
                </button>
              ))}
            </div>

            {/* Body */}
            <div className="p-6 overflow-y-auto flex-1">
              {importMode === 'local' ? (
                // ── Local ──
                <div className="space-y-5">
                  {/* Supported formats */}
                  <div className="flex flex-wrap gap-2 text-xs">
                    {['.csv', '.jsonl', '.json', '.parquet', '.txt'].map(f => (
                      <span key={f} className="px-2 py-1 rounded-lg bg-violet-500/10 border border-violet-500/20 text-violet-300 font-mono">{f}</span>
                    ))}
                  </div>

                  {/* Drop Zone */}
                  <div
                    onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}
                    className={`border-2 border-dashed rounded-2xl p-10 text-center transition-all ${
                      isDragging ? 'border-violet-500 bg-violet-500/10' :
                      selectedPath ? 'border-emerald-500/50 bg-emerald-500/5' :
                      'border-white/15 hover:border-white/30'
                    }`}
                  >
                    {selectedPath ? (
                      <div className="space-y-3">
                        <CheckCircle className="w-12 h-12 text-emerald-400 mx-auto" />
                        <div>
                          <p className="text-white font-medium">Ordner ausgewählt</p>
                          <p className="text-gray-400 text-sm mt-0.5 break-all">{selectedPath}</p>
                        </div>
                        {dirInfo && <p className="text-gray-500 text-sm">{dirInfo.files} Dateien · {formatBytes(dirInfo.size)}</p>}
                        <button onClick={() => { setSelectedPath(null); setDatasetName(''); setDirInfo(null); }} className="text-sm text-gray-400 hover:text-white underline transition-colors">
                          Anderen Ordner wählen
                        </button>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-white/5 border border-white/10">
                          <Upload className="w-7 h-7 text-gray-400" />
                        </div>
                        <div>
                          <p className="text-white font-medium">{isDragging ? 'Ordner hier ablegen' : 'Dataset-Ordner hierher ziehen'}</p>
                          <p className="text-gray-500 text-sm mt-1">oder Ordner manuell auswählen</p>
                        </div>
                        <button onClick={handleBrowseFolder} className="inline-flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/15 rounded-xl text-white text-sm transition-all">
                          <FolderOpen className="w-4 h-4" /> Ordner durchsuchen
                        </button>
                      </div>
                    )}
                  </div>

                  {selectedPath && (
                    <>
                      <div className="space-y-1.5">
                        <label className="block text-sm font-medium text-gray-300">Dataset-Name</label>
                        <input
                          type="text" value={datasetName} onChange={e => setDatasetName(e.target.value)}
                          placeholder="z.&nbsp;B. xlm-roberta-keywords-v1"
                          className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-white/30 transition-all"
                        />
                      </div>
                      <button
                        onClick={handleLocalImport} disabled={importing || !datasetName.trim()}
                        className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r ${currentTheme.colors.gradient} text-white text-sm font-medium hover:opacity-90 transition-all disabled:opacity-50`}
                      >
                        {importing ? <><Loader2 className="w-4 h-4 animate-spin" /> Importiere…</> : <><Upload className="w-4 h-4" /> Dataset importieren</>}
                      </button>
                    </>
                  )}
                </div>
              ) : (
                // ── HuggingFace ──
                <div className="space-y-5">
                  {/* Filters */}
                  <div className="space-y-3">
                    <button
                      onClick={() => setShowFilters(!showFilters)}
                      className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-all"
                    >
                      <Filter className="w-4 h-4" />
                      Filter {showFilters ? 'ausblenden' : 'anzeigen'}
                      <ChevronDown className={`w-4 h-4 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
                    </button>
                    {showFilters && filterOptions && (
                      <div className="grid grid-cols-3 gap-3">
                        {([
                          { label: 'Task', val: filterTask, set: setFilterTask, opts: filterOptions.tasks },
                          { label: 'Sprache', val: filterLanguage, set: setFilterLanguage, opts: filterOptions.languages.map(l => l.toUpperCase()) },
                          { label: 'Größe', val: filterSize, set: setFilterSize, opts: filterOptions.sizes },
                        ] as const).map(({ label, val, set, opts }) => (
                          <div key={label} className="space-y-1">
                            <label className="block text-xs text-gray-500">{label}</label>
                            <select
                              value={val}
                              onChange={e => (set as (v: string) => void)(e.target.value)}
                              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none appearance-none"
                            >
                              <option value="" className="bg-slate-900">Alle</option>
                              {opts.map(o => <option key={o} value={o} className="bg-slate-900">{o}</option>)}
                            </select>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Search */}
                  <div className="space-y-1.5">
                    <label className="block text-sm font-medium text-gray-300">Dataset suchen</label>
                    <div className="relative">
                      <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                      <input
                        type="text" value={hfQuery} onChange={e => setHfQuery(e.target.value)}
                        placeholder="z.&nbsp;B. squad, imdb, common_voice…"
                        className="w-full pl-10 pr-10 py-3 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-white/30 transition-all"
                      />
                      {hfSearching && <Loader2 className="absolute right-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 animate-spin" />}
                    </div>
                    <p className="text-gray-600 text-xs">Mindestens 2 Zeichen eingeben</p>
                  </div>

                  {/* Results */}
                  {hfResults.length > 0 && (
                    <div className="space-y-1.5">
                      <p className="text-gray-500 text-xs">{hfResults.length} Datasets gefunden</p>
                      <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
                        {hfResults.map(ds => (
                          <button
                            key={ds.id} onClick={() => handleHfSelect(ds)}
                            className={`w-full flex items-center justify-between p-3 rounded-xl border text-left transition-all ${
                              selectedHfDataset?.id === ds.id
                                ? 'bg-violet-500/10 border-violet-500/40'
                                : 'bg-white/5 border-white/10 hover:bg-white/10'
                            }`}
                          >
                            <div className="min-w-0">
                              <p className="text-white text-sm font-medium truncate">{ds.id}</p>
                              <div className="flex items-center gap-2 mt-0.5 text-xs text-gray-500">
                                <span>↓ {formatDownloads(ds.downloads)}</span>
                                {ds.likes ? <span>♥ {formatDownloads(ds.likes)}</span> : null}
                              </div>
                            </div>
                            {selectedHfDataset?.id === ds.id && <CheckCircle className="w-4 h-4 text-violet-400 flex-shrink-0 ml-2" />}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Selected */}
                  {selectedHfDataset && (
                    <div className="space-y-4 p-4 rounded-2xl border border-white/10 bg-white/5">
                      <div className="flex items-center gap-3">
                        <Cloud className="w-5 h-5 text-gray-400" />
                        <div>
                          <p className="text-white font-medium text-sm">{selectedHfDataset.id}</p>
                          <a
                            href={`https://huggingface.co/datasets/${selectedHfDataset.id}`}
                            target="_blank" rel="noopener noreferrer"
                            className="text-xs text-gray-500 hover:text-gray-300 flex items-center gap-1 transition-colors"
                          >
                            Auf HuggingFace ansehen <ExternalLink className="w-3 h-3" />
                          </a>
                        </div>
                      </div>
                      <div className="space-y-1.5">
                        <label className="block text-sm font-medium text-gray-300">Lokaler Name</label>
                        <input
                          type="text" value={hfDatasetName} onChange={e => setHfDatasetName(e.target.value)}
                          placeholder="Name für lokale Speicherung"
                          className="w-full px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-white/30 transition-all"
                        />
                      </div>

                      <button
                        onClick={handleHfDownload} disabled={downloading || !hfDatasetName.trim()}
                        className={`w-full relative overflow-hidden rounded-xl text-white text-sm font-medium transition-all disabled:cursor-not-allowed ${
                          downloading ? 'bg-white/10' : `bg-gradient-to-r ${currentTheme.colors.gradient} hover:opacity-90`
                        }`}
                      >
                        {downloading && (
                          <div className="absolute inset-0">
                            <div className={`absolute inset-0 bg-gradient-to-r ${currentTheme.colors.gradient} opacity-25`} />
                            <div className={`absolute inset-y-0 w-1/3 bg-gradient-to-r ${currentTheme.colors.gradient} opacity-50 animate-[progress-slide_1.5s_ease-in-out_infinite]`} />
                          </div>
                        )}
                        <div className="relative flex flex-col items-center py-3">
                          {downloading ? (
                            <><div className="flex items-center gap-2"><Loader2 className="w-4 h-4 animate-spin" /> Lade herunter…</div>
                            {downloadStatus && <span className="text-xs text-white/60 mt-0.5">{downloadStatus}</span>}</>
                          ) : (
                            <div className="flex items-center gap-2"><Download className="w-4 h-4" /> Dataset herunterladen</div>
                          )}
                        </div>
                      </button>

                      {downloading ? (
                        <button onClick={handleCancelDownload} className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-white/5 hover:bg-red-500/10 border border-white/10 hover:border-red-500/30 text-gray-400 hover:text-red-400 text-sm transition-all">
                          <X className="w-4 h-4" /> Abbrechen
                        </button>
                      ) : (
                        <p className="text-xs text-gray-600 text-center">Download-Dauer hängt von der Datenmenge ab</p>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── DatasetCard ────────────────────────────────────────────────────────────

interface DatasetCardProps {
  dataset: DatasetInfo;
  gradientClass: string;
  onDelete: () => void;
  onSplit: () => void;
  onHalve: () => void;
  onFiles: () => void;
}

function DatasetCard({ dataset, gradientClass, onDelete, onSplit, onHalve, onFiles }: DatasetCardProps) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-5 hover:bg-white/[0.07] transition-all group flex flex-col gap-4">
      {/* Top row */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3 min-w-0">
          <div className={`p-2 rounded-xl bg-gradient-to-r ${gradientClass} flex-shrink-0`}>
            {dataset.source === 'huggingface' ? <Cloud className="w-4 h-4 text-white" /> : <HardDrive className="w-4 h-4 text-white" />}
          </div>
          <div className="min-w-0">
            <h3 className="font-semibold text-white truncate" title={dataset.name}>{dataset.name}</h3>
            <div className="flex items-center gap-2 mt-1 flex-wrap">
              <span className={`text-xs px-2 py-0.5 rounded-full ${
                dataset.status === 'split' ? 'bg-emerald-500/15 text-emerald-400' : 'bg-amber-500/15 text-amber-400'
              }`}>
                {dataset.status === 'split' ? '✅ Aufgeteilt' : '⚠️ Kein Split'}
              </span>
              {dataset.training_count > 0 && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-400">
                  {dataset.training_count}× benutzt
                </span>
              )}
            </div>
          </div>
        </div>
        <button
          onClick={onDelete}
          className="p-1.5 rounded-lg text-gray-600 hover:text-red-400 hover:bg-red-500/10 opacity-0 group-hover:opacity-100 transition-all flex-shrink-0"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {/* Meta */}
      <div className="space-y-1.5 text-sm">
        <div className="flex items-center justify-between text-gray-400">
          <span className="flex items-center gap-1.5"><Database className="w-3.5 h-3.5" />{dataset.file_count} Dateien</span>
          <span>{formatBytes(dataset.size_bytes)}</span>
        </div>
        <div className="flex items-center gap-1.5 text-gray-500 text-xs">
          <Calendar className="w-3 h-3" />{formatDate(dataset.created_at)}
        </div>
        {dataset.last_used_at && (
          <div className="flex items-center gap-1.5 text-gray-500 text-xs">
            <CheckCircle className="w-3 h-3 text-cyan-500/60" /> Zuletzt: {formatDate(dataset.last_used_at)}
          </div>
        )}
      </div>

      {/* Split preview */}
      {dataset.split_info && (
        <div className="grid grid-cols-3 gap-2 pt-1">
          {[
            { label: 'Train', color: 'blue', count: dataset.split_info.train_count },
            { label: 'Val', color: 'purple', count: dataset.split_info.val_count },
            { label: 'Test', color: 'green', count: dataset.split_info.test_count },
          ].map(({ label, color, count }) => (
            <div key={label} className={`p-2 rounded-xl bg-${color}-500/10 text-center`}>
              <div className={`text-${color}-400 font-semibold text-sm`}>{count}</div>
              <div className="text-gray-600 text-[11px]">{label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2 pt-1">
        <button onClick={onFiles} className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white text-xs transition-all">
          <FileText className="w-3.5 h-3.5" /> Dateien
        </button>
        {dataset.status === 'unused' && (
          <button onClick={onSplit} className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white text-xs transition-all">
            <Scissors className="w-3.5 h-3.5" /> Split
          </button>
        )}
        <button onClick={onHalve} className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl bg-white/5 hover:bg-amber-500/10 border border-white/10 hover:border-amber-500/20 text-gray-400 hover:text-amber-400 text-xs transition-all" title="Bei RAM-Problemen halbieren">
          ½ Halbieren
        </button>
      </div>
    </div>
  );
}
