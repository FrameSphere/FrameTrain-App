import { useState, useEffect, useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { 
  Upload, 
  FolderOpen, 
  Download, 
  Trash2, 
  Search,
  HardDrive,
  Cloud,
  CheckCircle,
  AlertCircle,
  Loader2,
  Database,
  Calendar,
  ExternalLink,
  X,
  RefreshCw,
  ChevronDown,
  Scissors,
  Filter,
  Layers,
  FileText
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';
import DatasetFileManager from './DatasetFileManager';

// ============ Types ============

interface ModelInfo {
  id: string;
  name: string;
  source: string;
}

interface DatasetInfo {
  id: string;
  name: string;
  model_id: string;
  source: string;
  source_path: string | null;
  size_bytes: number;
  file_count: number;
  created_at: string;
  status: 'unused' | 'split';
  split_info: SplitInfo | null;
}

interface SplitInfo {
  train_count: number;
  val_count: number;
  test_count: number;
  train_ratio: number;
  val_ratio: number;
  test_ratio: number;
}

interface HuggingFaceDataset {
  id: string;
  author?: string;
  downloads?: number;
  likes?: number;
  tags?: string[];
  lastModified?: string;
  description?: string;
}

interface FilterOptions {
  tasks: string[];
  languages: string[];
  sizes: string[];
}

type ImportMode = 'local' | 'huggingface';

// ============ Helper Functions ============

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString('de-DE', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}

function formatDownloads(num: number | undefined): string {
  if (!num) return '0';
  if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
  if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
  return num.toString();
}

// ============ Main Component ============

export default function DatasetUpload() {
  const { currentTheme } = useTheme();
  const { success, error, warning, info } = useNotification();

  // Models & Datasets State
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loading, setLoading] = useState(true);

  // Import Modal State
  const [showImportModal, setShowImportModal] = useState(false);
  const [importMode, setImportMode] = useState<ImportMode>('local');

  // Local Import State
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [datasetName, setDatasetName] = useState('');
  const [dirInfo, setDirInfo] = useState<{ size: number; files: number } | null>(null);
  const [importing, setImporting] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  // HuggingFace State
  const [hfSearchQuery, setHfSearchQuery] = useState('');
  const [hfResults, setHfResults] = useState<HuggingFaceDataset[]>([]);
  const [hfSearching, setHfSearching] = useState(false);
  const [selectedHfDataset, setSelectedHfDataset] = useState<HuggingFaceDataset | null>(null);
  const [hfDatasetName, setHfDatasetName] = useState('');
  const [downloading, setDownloading] = useState(false);
  const [downloadStatus, setDownloadStatus] = useState('');
  const [downloadCancelled, setDownloadCancelled] = useState(false);
  const downloadIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Filter State
  const [filterOptions, setFilterOptions] = useState<FilterOptions | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [filterTask, setFilterTask] = useState('');
  const [filterLanguage, setFilterLanguage] = useState('');
  const [filterSize, setFilterSize] = useState('');

  // Split Modal State
  const [showSplitModal, setShowSplitModal] = useState(false);
  const [datasetToSplit, setDatasetToSplit] = useState<DatasetInfo | null>(null);
  const [trainRatio, setTrainRatio] = useState(0.8);

  // File Manager State
  const [showFileManager, setShowFileManager] = useState(false);
  const [fileManagerDataset, setFileManagerDataset] = useState<DatasetInfo | null>(null);
  const [valRatio, setValRatio] = useState(0.1);
  const [testRatio, setTestRatio] = useState(0.1);
  const [splitting, setSplitting] = useState(false);

  // Debounce ref
  const searchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ============ Load Data ============

  useEffect(() => {
    loadModels();
    loadFilterOptions();
  }, []);

  useEffect(() => {
    if (selectedModelId) {
      loadDatasets();
    }
  }, [selectedModelId]);

  // Debounced HuggingFace search
  useEffect(() => {
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    if (hfSearchQuery.trim().length < 2) {
      setHfResults([]);
      setHfSearching(false);
      return;
    }

    setHfSearching(true);

    searchTimeoutRef.current = setTimeout(async () => {
      try {
        const results = await invoke<HuggingFaceDataset[]>('search_huggingface_datasets', {
          query: hfSearchQuery.trim(),
          limit: 15,
          filterTask: filterTask || null,
          filterLanguage: filterLanguage || null,
          filterSize: filterSize || null,
        });
        setHfResults(results);
      } catch (err: any) {
        console.error('HuggingFace search error:', err);
      } finally {
        setHfSearching(false);
      }
    }, 300);

    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, [hfSearchQuery, filterTask, filterLanguage, filterSize]);

  const loadModels = async () => {
    try {
      const modelList = await invoke<ModelInfo[]>('list_models');
      setModels(modelList);
      if (modelList.length > 0 && !selectedModelId) {
        setSelectedModelId(modelList[0].id);
      }
    } catch (err: any) {
      error('Fehler beim Laden der Modelle', String(err));
    } finally {
      setLoading(false);
    }
  };

  const loadDatasets = async () => {
    if (!selectedModelId) return;
    
    try {
      const datasetList = await invoke<DatasetInfo[]>('list_datasets_for_model', {
        modelId: selectedModelId
      });
      setDatasets(datasetList);
    } catch (err: any) {
      error('Fehler beim Laden der Datasets', String(err));
    }
  };

  const loadFilterOptions = async () => {
    try {
      const options = await invoke<FilterOptions>('get_dataset_filter_options');
      setFilterOptions(options);
    } catch (err: any) {
      console.error('Could not load filter options:', err);
    }
  };

  // ============ Local Import ============

  const handleBrowseFolder = async () => {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: 'Dataset-Ordner auswählen'
      });

      if (selected && typeof selected === 'string') {
        await validateAndSetPath(selected);
      }
    } catch (err: any) {
      console.error('Browse folder error:', err);
      error('Fehler beim Auswählen', String(err));
    }
  };

  const validateAndSetPath = async (path: string) => {
    setSelectedPath(path);
    const folderName = path.split(/[/\\\\]/).pop() || 'Unbekanntes Dataset';
    setDatasetName(folderName);

    try {
      const [size, files] = await invoke<[number, number]>('get_directory_size', { path });
      setDirInfo({ size, files });
    } catch (err: any) {
      error('Fehler beim Lesen', String(err));
      setDirInfo(null);
    }
  };

  const handleLocalImport = async () => {
    if (!selectedPath || !datasetName.trim() || !selectedModelId) {
      warning('Fehlende Angaben', 'Bitte wähle einen Ordner, gib einen Namen ein und wähle ein Modell.');
      return;
    }

    setImporting(true);
    try {
      const newDataset = await invoke<DatasetInfo>('import_local_dataset', {
        sourcePath: selectedPath,
        datasetName: datasetName.trim(),
        modelId: selectedModelId
      });

      success('Dataset importiert!', `"${newDataset.name}" wurde erfolgreich importiert.`);
      resetImportModal();
      await loadDatasets();
    } catch (err: any) {
      error('Import fehlgeschlagen', String(err));
    } finally {
      setImporting(false);
    }
  };

  // ============ Drag & Drop ============

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const items = e.dataTransfer.items;
    if (items && items.length > 0) {
      const item = items[0];
      if (item.kind === 'file') {
        const file = item.getAsFile();
        if (file) {
          const path = (file as any).path;
          if (path) {
            await validateAndSetPath(path);
          } else {
            info('Drag & Drop', 'Bitte nutze den "Ordner durchsuchen" Button.');
          }
        }
      }
    }
  }, []);

  // ============ HuggingFace Download ============

  const handleHfDatasetSelect = (dataset: HuggingFaceDataset) => {
    setSelectedHfDataset(dataset);
    setHfDatasetName(dataset.id.split('/').pop() || dataset.id);
  };

  const handleHfDownload = async () => {
    if (!selectedHfDataset || !hfDatasetName.trim() || !selectedModelId) {
      warning('Fehlende Angaben', 'Bitte wähle ein Dataset und gib einen Namen ein.');
      return;
    }

    setDownloading(true);
    setDownloadCancelled(false);
    setDownloadStatus('Verbinde mit Hugging Face...');

    try {
      downloadIntervalRef.current = setInterval(() => {
        setDownloadStatus(prev => {
          const statuses = [
            'Lade Dataset-Konfiguration...',
            'Lade Trainingsdaten...',
            'Lade Validierungsdaten...',
            'Speichere Dateien...',
            'Fast fertig...'
          ];
          const currentIndex = statuses.indexOf(prev);
          if (currentIndex < statuses.length - 1) {
            return statuses[currentIndex + 1] || statuses[0];
          }
          return prev;
        });
      }, 2000);

      const newDataset = await invoke<DatasetInfo>('download_huggingface_dataset', {
        repoId: selectedHfDataset.id,
        datasetName: hfDatasetName.trim(),
        modelId: selectedModelId
      });

      if (downloadIntervalRef.current) {
        clearInterval(downloadIntervalRef.current);
      }

      success('Download abgeschlossen!', `"${newDataset.name}" wurde heruntergeladen.`);
      resetImportModal();
      await loadDatasets();
    } catch (err: any) {
      if (downloadIntervalRef.current) {
        clearInterval(downloadIntervalRef.current);
      }
      if (!downloadCancelled) {
        error('Download fehlgeschlagen', String(err));
      }
      setDownloadStatus('');
    } finally {
      setDownloading(false);
    }
  };

  const handleCancelDownload = () => {
    setDownloadCancelled(true);
    setDownloading(false);
    setDownloadStatus('');
    if (downloadIntervalRef.current) {
      clearInterval(downloadIntervalRef.current);
    }
    info('Download abgebrochen', 'Der Download wurde abgebrochen.');
  };

  // ============ Split Dataset ============

  const openSplitModal = (dataset: DatasetInfo) => {
    setDatasetToSplit(dataset);
    setTrainRatio(0.8);
    setValRatio(0.1);
    setTestRatio(0.1);
    setShowSplitModal(true);
  };

  const handleSplit = async () => {
    if (!datasetToSplit || !selectedModelId) return;

    const total = trainRatio + valRatio + testRatio;
    if (Math.abs(total - 1.0) > 0.01) {
      warning('Ungültige Aufteilung', 'Die Summe muss 100% ergeben.');
      return;
    }

    setSplitting(true);
    try {
      await invoke('split_dataset', {
        datasetId: datasetToSplit.id,
        modelId: selectedModelId,
        trainRatio,
        valRatio,
        testRatio
      });

      success('Dataset aufgeteilt!', `"${datasetToSplit.name}" wurde in Train/Val/Test aufgeteilt.`);
      setShowSplitModal(false);
      setDatasetToSplit(null);
      await loadDatasets();
    } catch (err: any) {
      error('Split fehlgeschlagen', String(err));
    } finally {
      setSplitting(false);
    }
  };

  // ============ Delete Dataset ============

  const handleDeleteDataset = async (dataset: DatasetInfo) => {
    if (!confirm(`Möchtest du "${dataset.name}" wirklich löschen?`)) return;

    try {
      await invoke('delete_dataset', {
        datasetId: dataset.id,
        modelId: dataset.model_id
      });
      success('Dataset gelöscht', `"${dataset.name}" wurde entfernt.`);
      await loadDatasets();
    } catch (err: any) {
      error('Löschen fehlgeschlagen', String(err));
    }
  };

  // ============ Helpers ============

  const resetImportModal = () => {
    setShowImportModal(false);
    setSelectedPath(null);
    setDatasetName('');
    setDirInfo(null);
    setSelectedHfDataset(null);
    setHfDatasetName('');
    setHfSearchQuery('');
    setHfResults([]);
    setDownloadStatus('');
    setFilterTask('');
    setFilterLanguage('');
    setFilterSize('');
  };

  const selectedModel = models.find(m => m.id === selectedModelId);

  // ============ Render ============

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-8 h-8 text-gray-400 animate-spin" />
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-white">Datensätze</h1>
          <p className="text-gray-400 mt-1">Verwalte deine Trainingsdaten</p>
        </div>
        <div className="bg-white/5 rounded-2xl border border-white/10 p-12 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-white/5 mb-4">
            <Layers className="w-8 h-8 text-gray-400" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">Kein Modell vorhanden</h3>
          <p className="text-gray-400 mb-6">
            Füge zuerst ein Modell hinzu, bevor du Datensätze importieren kannst.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Datensätze</h1>
          <p className="text-gray-400 mt-1">Verwalte deine Trainingsdaten</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={loadDatasets}
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-all"
            title="Aktualisieren"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
          <button
            onClick={() => setShowImportModal(true)}
            disabled={!selectedModelId}
            className={`flex items-center gap-2 px-4 py-2 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white font-medium hover:opacity-90 transition-all disabled:opacity-50`}
          >
            <Upload className="w-5 h-5" />
            Dataset hinzufügen
          </button>
        </div>
      </div>

      {/* Model Selector */}
      <div className="bg-white/5 rounded-xl border border-white/10 p-4">
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Modell auswählen
        </label>
        <div className="relative">
          <select
            value={selectedModelId || ''}
            onChange={(e) => setSelectedModelId(e.target.value)}
            className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white appearance-none cursor-pointer focus:outline-none focus:ring-2 transition-all"
            style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
          >
            {models.map((model) => (
              <option key={model.id} value={model.id} className="bg-slate-800">
                {model.name} ({model.source === 'huggingface' ? 'HF' : 'Lokal'})
              </option>
            ))}
          </select>
          <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
        </div>
      </div>

      {/* Datasets Grid */}
      {datasets.length === 0 ? (
        <div className="bg-white/5 rounded-2xl border border-white/10 p-12 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-white/5 mb-4">
            <Database className="w-8 h-8 text-gray-400" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">Keine Datensätze</h3>
          <p className="text-gray-400 mb-6">
            Füge einen Datensatz für "{selectedModel?.name}" hinzu.
          </p>
          <button
            onClick={() => setShowImportModal(true)}
            className={`inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white font-medium hover:opacity-90 transition-all`}
          >
            <Upload className="w-5 h-5" />
            Erstes Dataset hinzufügen
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {datasets.map((dataset) => (
            <div
              key={dataset.id}
              className="bg-white/5 rounded-xl border border-white/10 p-5 hover:bg-white/[0.07] transition-all group"
            >
              {/* Dataset Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg bg-gradient-to-r ${currentTheme.colors.gradient}`}>
                    {dataset.source === 'huggingface' ? (
                      <Cloud className="w-5 h-5 text-white" />
                    ) : (
                      <HardDrive className="w-5 h-5 text-white" />
                    )}
                  </div>
                  <div>
                    <h3 className="font-semibold text-white truncate max-w-[180px]" title={dataset.name}>
                      {dataset.name}
                    </h3>
                    <span className={`text-xs px-2 py-0.5 rounded ${
                      dataset.status === 'split' 
                        ? 'bg-green-500/20 text-green-400' 
                        : 'bg-amber-500/20 text-amber-400'
                    }`}>
                      {dataset.status === 'split' ? 'Aufgeteilt' : 'Unbenutzt'}
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => handleDeleteDataset(dataset)}
                  className="p-1.5 rounded-lg text-gray-500 hover:text-red-400 hover:bg-red-500/10 opacity-0 group-hover:opacity-100 transition-all"
                  title="Löschen"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>

              {/* Dataset Info */}
              <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between text-gray-400">
                  <span className="flex items-center gap-2">
                    <Database className="w-4 h-4" />
                    {dataset.file_count} Dateien
                  </span>
                  <span>{formatBytes(dataset.size_bytes)}</span>
                </div>

                <div className="flex items-center gap-2 text-gray-500 text-xs">
                  <Calendar className="w-3 h-3" />
                  <span>{formatDate(dataset.created_at)}</span>
                </div>

                {/* Split Info */}
                {dataset.split_info && (
                  <div className="mt-3 pt-3 border-t border-white/10">
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div className="text-center p-2 bg-blue-500/10 rounded">
                        <div className="text-blue-400 font-medium">{dataset.split_info.train_count}</div>
                        <div className="text-gray-500">Train</div>
                      </div>
                      <div className="text-center p-2 bg-purple-500/10 rounded">
                        <div className="text-purple-400 font-medium">{dataset.split_info.val_count}</div>
                        <div className="text-gray-500">Val</div>
                      </div>
                      <div className="text-center p-2 bg-green-500/10 rounded">
                        <div className="text-green-400 font-medium">{dataset.split_info.test_count}</div>
                        <div className="text-gray-500">Test</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Action Buttons */}
                <div className="mt-3 flex gap-2">
                  <button
                    onClick={() => {
                      setFileManagerDataset(dataset);
                      setShowFileManager(true);
                    }}
                    className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-gray-300 hover:text-white text-sm transition-all"
                    title="Dateien anzeigen und verwalten"
                  >
                    <FileText className="w-4 h-4" />
                    Dateien
                  </button>
                  {dataset.status === 'unused' && (
                    <button
                      onClick={() => openSplitModal(dataset)}
                      className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-gray-300 hover:text-white text-sm transition-all"
                    >
                      <Scissors className="w-4 h-4" />
                      Split
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Import Modal */}
      {showImportModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-2xl max-h-[90vh] overflow-hidden">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-6 border-b border-white/10">
              <div>
                <h2 className="text-xl font-bold text-white">Dataset hinzufügen</h2>
                <p className="text-sm text-gray-400 mt-1">
                  Für: {selectedModel?.name}
                </p>
              </div>
              <button
                onClick={resetImportModal}
                className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-all"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Mode Tabs */}
            <div className="flex border-b border-white/10">
              <button
                onClick={() => setImportMode('local')}
                className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 font-medium transition-all ${
                  importMode === 'local' ? 'text-white border-b-2' : 'text-gray-400 hover:text-white'
                }`}
                style={importMode === 'local' ? { borderColor: currentTheme.colors.primary } : {}}
              >
                <HardDrive className="w-5 h-5" />
                Lokaler Ordner
              </button>
              <button
                onClick={() => setImportMode('huggingface')}
                className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 font-medium transition-all ${
                  importMode === 'huggingface' ? 'text-white border-b-2' : 'text-gray-400 hover:text-white'
                }`}
                style={importMode === 'huggingface' ? { borderColor: currentTheme.colors.primary } : {}}
              >
                <Cloud className="w-5 h-5" />
                Hugging Face
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6 overflow-y-auto max-h-[60vh]">
              {importMode === 'local' ? (
                // ============ Local Import ============
                <div className="space-y-6">
                  <div
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${
                      isDragging
                        ? 'border-purple-500 bg-purple-500/10'
                        : selectedPath
                        ? 'border-green-500/50 bg-green-500/5'
                        : 'border-white/20 hover:border-white/40'
                    }`}
                  >
                    {selectedPath ? (
                      <div className="space-y-3">
                        <CheckCircle className="w-12 h-12 text-green-400 mx-auto" />
                        <div>
                          <p className="text-white font-medium">Ordner ausgewählt</p>
                          <p className="text-gray-400 text-sm truncate max-w-md mx-auto" title={selectedPath}>
                            {selectedPath}
                          </p>
                        </div>
                        {dirInfo && (
                          <div className="flex items-center justify-center gap-4 text-sm text-gray-400">
                            <span>{dirInfo.files} Dateien</span>
                            <span>•</span>
                            <span>{formatBytes(dirInfo.size)}</span>
                          </div>
                        )}
                        <button
                          onClick={() => {
                            setSelectedPath(null);
                            setDatasetName('');
                            setDirInfo(null);
                          }}
                          className="text-sm text-gray-400 hover:text-white underline"
                        >
                          Anderen Ordner wählen
                        </button>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-white/5">
                          <Upload className="w-8 h-8 text-gray-400" />
                        </div>
                        <div>
                          <p className="text-white font-medium">
                            {isDragging ? 'Ordner hier ablegen' : 'Dataset-Ordner hierher ziehen'}
                          </p>
                          <p className="text-gray-400 text-sm mt-1">oder klicke um zu durchsuchen</p>
                        </div>
                        <button
                          onClick={handleBrowseFolder}
                          className="inline-flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/15 rounded-lg text-white transition-all"
                        >
                          <FolderOpen className="w-5 h-5" />
                          Ordner durchsuchen
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Dataset Name Input */}
                  {selectedPath && (
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Dataset-Name
                      </label>
                      <input
                        type="text"
                        value={datasetName}
                        onChange={(e) => setDatasetName(e.target.value)}
                        placeholder="z.B. Training-Bilder-v1"
                        className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 transition-all"
                        style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
                      />
                    </div>
                  )}

                  {/* Import Button */}
                  {selectedPath && (
                    <button
                      onClick={handleLocalImport}
                      disabled={importing || !datasetName.trim()}
                      className={`w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white font-medium hover:opacity-90 transition-all disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                      {importing ? (
                        <>
                          <Loader2 className="w-5 h-5 animate-spin" />
                          Importiere...
                        </>
                      ) : (
                        <>
                          <Upload className="w-5 h-5" />
                          Dataset importieren
                        </>
                      )}
                    </button>
                  )}
                </div>
              ) : (
                // ============ HuggingFace Import ============
                <div className="space-y-6">
                  {/* Filters */}
                  <div>
                    <button
                      onClick={() => setShowFilters(!showFilters)}
                      className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-all"
                    >
                      <Filter className="w-4 h-4" />
                      Filter {showFilters ? 'ausblenden' : 'anzeigen'}
                      <ChevronDown className={`w-4 h-4 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
                    </button>

                    {showFilters && filterOptions && (
                      <div className="grid grid-cols-3 gap-3 mt-3">
                        <div>
                          <label className="block text-xs text-gray-500 mb-1">Task</label>
                          <select
                            value={filterTask}
                            onChange={(e) => setFilterTask(e.target.value)}
                            className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none"
                          >
                            <option value="" className="bg-slate-800">Alle</option>
                            {filterOptions.tasks.map((task) => (
                              <option key={task} value={task} className="bg-slate-800">{task}</option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="block text-xs text-gray-500 mb-1">Sprache</label>
                          <select
                            value={filterLanguage}
                            onChange={(e) => setFilterLanguage(e.target.value)}
                            className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none"
                          >
                            <option value="" className="bg-slate-800">Alle</option>
                            {filterOptions.languages.map((lang) => (
                              <option key={lang} value={lang} className="bg-slate-800">{lang.toUpperCase()}</option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="block text-xs text-gray-500 mb-1">Größe</label>
                          <select
                            value={filterSize}
                            onChange={(e) => setFilterSize(e.target.value)}
                            className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none"
                          >
                            <option value="" className="bg-slate-800">Alle</option>
                            {filterOptions.sizes.map((size) => (
                              <option key={size} value={size} className="bg-slate-800">{size}</option>
                            ))}
                          </select>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Search Input */}
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Dataset suchen
                    </label>
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                      <input
                        type="text"
                        value={hfSearchQuery}
                        onChange={(e) => setHfSearchQuery(e.target.value)}
                        placeholder="z.B. squad, imdb, common_voice..."
                        className="w-full pl-10 pr-10 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 transition-all"
                        style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
                      />
                      {hfSearching && (
                        <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 animate-spin" />
                      )}
                    </div>
                    <p className="text-xs text-gray-500 mt-1">Mindestens 2 Zeichen eingeben</p>
                  </div>

                  {/* Search Results */}
                  {hfResults.length > 0 && (
                    <div className="space-y-2 max-h-60 overflow-y-auto">
                      <p className="text-sm text-gray-400">{hfResults.length} Datasets gefunden</p>
                      {hfResults.map((dataset) => (
                        <button
                          key={dataset.id}
                          onClick={() => handleHfDatasetSelect(dataset)}
                          className={`w-full flex items-center justify-between p-3 rounded-lg border transition-all text-left ${
                            selectedHfDataset?.id === dataset.id
                              ? 'bg-purple-500/20 border-purple-500/50'
                              : 'bg-white/5 border-white/10 hover:bg-white/10'
                          }`}
                        >
                          <div className="min-w-0 flex-1">
                            <p className="text-white font-medium truncate">{dataset.id}</p>
                            <div className="flex items-center gap-3 text-xs text-gray-400 mt-1">
                              <span>↓ {formatDownloads(dataset.downloads)}</span>
                              {dataset.likes && <span>♥ {formatDownloads(dataset.likes)}</span>}
                            </div>
                          </div>
                          {selectedHfDataset?.id === dataset.id && (
                            <CheckCircle className="w-5 h-5 text-purple-400 flex-shrink-0 ml-2" />
                          )}
                        </button>
                      ))}
                    </div>
                  )}

                  {/* Selected Dataset Details */}
                  {selectedHfDataset && (
                    <div className="space-y-4 p-4 bg-white/5 rounded-lg border border-white/10">
                      <div className="flex items-center gap-3">
                        <Cloud className="w-5 h-5 text-purple-400" />
                        <div>
                          <p className="text-white font-medium">{selectedHfDataset.id}</p>
                          <a
                            href={`https://huggingface.co/datasets/${selectedHfDataset.id}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs text-gray-400 hover:text-white flex items-center gap-1"
                          >
                            Auf Hugging Face ansehen <ExternalLink className="w-3 h-3" />
                          </a>
                        </div>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Lokaler Name
                        </label>
                        <input
                          type="text"
                          value={hfDatasetName}
                          onChange={(e) => setHfDatasetName(e.target.value)}
                          placeholder="Name für lokale Speicherung"
                          className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 transition-all"
                          style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
                        />
                      </div>

                      <button
                        onClick={handleHfDownload}
                        disabled={downloading || !hfDatasetName.trim()}
                        className={`w-full relative overflow-hidden rounded-lg text-white font-medium transition-all disabled:cursor-not-allowed ${
                          downloading ? 'bg-white/10' : `bg-gradient-to-r ${currentTheme.colors.gradient} hover:opacity-90`
                        }`}
                      >
                        {downloading && (
                          <div className="absolute inset-0 overflow-hidden">
                            <div className={`absolute inset-0 bg-gradient-to-r ${currentTheme.colors.gradient} opacity-30`} />
                            <div className={`absolute inset-y-0 w-1/3 bg-gradient-to-r ${currentTheme.colors.gradient} opacity-60 animate-progress-slide`} />
                          </div>
                        )}
                        <div className="relative flex flex-col items-center justify-center py-3 px-4">
                          {downloading ? (
                            <>
                              <div className="flex items-center gap-2">
                                <Loader2 className="w-5 h-5 animate-spin" />
                                <span>Lade herunter...</span>
                              </div>
                              {downloadStatus && (
                                <span className="text-xs text-white/70 mt-1">{downloadStatus}</span>
                              )}
                            </>
                          ) : (
                            <div className="flex items-center gap-2">
                              <Download className="w-5 h-5" />
                              <span>Dataset herunterladen</span>
                            </div>
                          )}
                        </div>
                      </button>

                      {downloading && (
                        <button
                          onClick={handleCancelDownload}
                          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-white/5 hover:bg-red-500/20 border border-white/10 hover:border-red-500/30 rounded-lg text-gray-400 hover:text-red-400 font-medium transition-all"
                        >
                          <X className="w-4 h-4" />
                          <span>Download abbrechen</span>
                        </button>
                      )}

                      {!downloading && (
                        <p className="text-xs text-gray-500 text-center">
                          Der Download kann je nach Größe einige Minuten dauern
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* File Manager Modal */}
      {showFileManager && fileManagerDataset && (
        <DatasetFileManager
          datasetId={fileManagerDataset.id}
          datasetName={fileManagerDataset.name}
          onClose={() => {
            setShowFileManager(false);
            setFileManagerDataset(null);
            loadDatasets();
          }}
        />
      )}

      {/* Split Modal */}
      {showSplitModal && datasetToSplit && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-md">
            <div className="flex items-center justify-between p-6 border-b border-white/10">
              <div>
                <h2 className="text-xl font-bold text-white">Dataset aufteilen</h2>
                <p className="text-sm text-gray-400 mt-1">{datasetToSplit.name}</p>
              </div>
              <button
                onClick={() => {
                  setShowSplitModal(false);
                  setDatasetToSplit(null);
                }}
                className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-all"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-6 space-y-6">
              <p className="text-gray-400 text-sm">
                Teile die {datasetToSplit.file_count} Dateien in Training-, Validierungs- und Test-Sets auf.
              </p>

              {/* Sliders */}
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-blue-400">Training</span>
                    <span className="text-white">{Math.round(trainRatio * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={trainRatio * 100}
                    onChange={(e) => {
                      const newTrain = parseInt(e.target.value) / 100;
                      const remaining = 1 - newTrain;
                      const valPart = valRatio / (valRatio + testRatio) || 0.5;
                      setTrainRatio(newTrain);
                      setValRatio(remaining * valPart);
                      setTestRatio(remaining * (1 - valPart));
                    }}
                    className="w-full accent-blue-500"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-purple-400">Validierung</span>
                    <span className="text-white">{Math.round(valRatio * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={valRatio * 100}
                    onChange={(e) => {
                      const newVal = parseInt(e.target.value) / 100;
                      const remaining = 1 - newVal;
                      const trainPart = trainRatio / (trainRatio + testRatio) || 0.5;
                      setValRatio(newVal);
                      setTrainRatio(remaining * trainPart);
                      setTestRatio(remaining * (1 - trainPart));
                    }}
                    className="w-full accent-purple-500"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-green-400">Test</span>
                    <span className="text-white">{Math.round(testRatio * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={testRatio * 100}
                    onChange={(e) => {
                      const newTest = parseInt(e.target.value) / 100;
                      const remaining = 1 - newTest;
                      const trainPart = trainRatio / (trainRatio + valRatio) || 0.5;
                      setTestRatio(newTest);
                      setTrainRatio(remaining * trainPart);
                      setValRatio(remaining * (1 - trainPart));
                    }}
                    className="w-full accent-green-500"
                  />
                </div>
              </div>

              {/* Preview */}
              <div className="grid grid-cols-3 gap-2 text-center text-sm">
                <div className="p-3 bg-blue-500/10 rounded-lg">
                  <div className="text-blue-400 font-bold">
                    {Math.round(datasetToSplit.file_count * trainRatio)}
                  </div>
                  <div className="text-gray-500 text-xs">Train</div>
                </div>
                <div className="p-3 bg-purple-500/10 rounded-lg">
                  <div className="text-purple-400 font-bold">
                    {Math.round(datasetToSplit.file_count * valRatio)}
                  </div>
                  <div className="text-gray-500 text-xs">Val</div>
                </div>
                <div className="p-3 bg-green-500/10 rounded-lg">
                  <div className="text-green-400 font-bold">
                    {Math.round(datasetToSplit.file_count * testRatio)}
                  </div>
                  <div className="text-gray-500 text-xs">Test</div>
                </div>
              </div>

              {/* Split Button */}
              <button
                onClick={handleSplit}
                disabled={splitting}
                className={`w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white font-medium hover:opacity-90 transition-all disabled:opacity-50`}
              >
                {splitting ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Teile auf...
                  </>
                ) : (
                  <>
                    <Scissors className="w-5 h-5" />
                    Dataset aufteilen
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}