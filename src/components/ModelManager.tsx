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
  FileBox,
  Cpu,
  Calendar,
  ExternalLink,
  X,
  RefreshCw
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';

// ============ Types ============

interface ModelInfo {
  id: string;
  name: string;
  source: string;
  source_path: string | null;
  size_bytes: number;
  file_count: number;
  created_at: string;
  model_type: string | null;
}

interface HuggingFaceModel {
  id: string;
  modelId?: string;
  author?: string;
  downloads?: number;
  likes?: number;
  tags?: string[];
  pipeline_tag?: string;
  lastModified?: string;
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

export default function ModelManager() {
  const { currentTheme } = useTheme();
  const { success, error, warning, info } = useNotification();

  // State
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [importMode, setImportMode] = useState<ImportMode>('local');
  const [showImportModal, setShowImportModal] = useState(false);
  
  // Local Import State
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [modelName, setModelName] = useState('');
  const [dirInfo, setDirInfo] = useState<{ size: number; files: number } | null>(null);
  const [isValidModel, setIsValidModel] = useState(false);
  const [importing, setImporting] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  // HuggingFace State
  const [hfSearchQuery, setHfSearchQuery] = useState('');
  const [hfResults, setHfResults] = useState<HuggingFaceModel[]>([]);
  const [hfSearching, setHfSearching] = useState(false);
  const [selectedHfModel, setSelectedHfModel] = useState<HuggingFaceModel | null>(null);
  const [hfModelName, setHfModelName] = useState('');
  const [downloading, setDownloading] = useState(false);
  const [downloadStatus, setDownloadStatus] = useState<string>('');
  const [downloadCancelled, setDownloadCancelled] = useState(false);
  const downloadIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  
  // Debounce ref for HuggingFace search
const searchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load models on mount
  useEffect(() => {
    loadModels();
  }, []);

  // Debounced HuggingFace search - triggers 300ms after user stops typing
  useEffect(() => {
    // Clear previous timeout
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    // Don't search if query is too short
    if (hfSearchQuery.trim().length < 2) {
      setHfResults([]);
      setHfSearching(false);
      return;
    }

    // Set searching state immediately for UI feedback
    setHfSearching(true);

    // Debounce the actual search
    searchTimeoutRef.current = setTimeout(async () => {
      try {
        const results = await invoke<HuggingFaceModel[]>('search_huggingface_models', {
          query: hfSearchQuery.trim(),
          limit: 15
        });
        setHfResults(results);
      } catch (err: any) {
        console.error('HuggingFace search error:', err);
        // Don't show error notification for every keystroke
      } finally {
        setHfSearching(false);
      }
    }, 300);

    // Cleanup on unmount
    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, [hfSearchQuery]);

  const loadModels = async () => {
    try {
      setLoading(true);
      const modelList = await invoke<ModelInfo[]>('list_models');
      setModels(modelList);
    } catch (err: any) {
      error('Fehler beim Laden der Modelle', err.message);
    } finally {
      setLoading(false);
    }
  };

  // ============ Local Import Functions ============

  const handleBrowseFolder = async () => {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: 'Modell-Ordner auswählen'
      });

      console.log('Dialog result:', selected);

      if (selected && typeof selected === 'string') {
        await validateAndSetPath(selected);
      } else if (selected === null) {
        // User cancelled - do nothing
        return;
      }
    } catch (err: any) {
      console.error('Browse folder error:', err);
      error('Fehler beim Auswählen', String(err));
    }
  };

  const validateAndSetPath = async (path: string) => {
    setSelectedPath(path);
    
    // Extrahiere Ordnernamen als Standard-Modellname
    const folderName = path.split(/[/\\\\]/).pop() || 'Unbekanntes Modell';
    setModelName(folderName);

    try {
      // Validiere ob es ein Modell ist
      const isValid = await invoke<boolean>('validate_model_directory', { path });
      setIsValidModel(isValid);

      if (!isValid) {
        warning('Kein gültiges Modell erkannt', 'Der Ordner enthält keine erkennbaren Modell-Dateien.');
      }

      // Hole Größe
      const [size, files] = await invoke<[number, number]>('get_directory_size', { path });
      setDirInfo({ size, files });
    } catch (err: any) {
      error('Validierungsfehler', err.message);
      setIsValidModel(false);
      setDirInfo(null);
    }
  };

  const handleLocalImport = async () => {
    if (!selectedPath || !modelName.trim()) {
      warning('Fehlende Angaben', 'Bitte wähle einen Ordner und gib einen Namen ein.');
      return;
    }

    setImporting(true);
    try {
      const newModel = await invoke<ModelInfo>('import_local_model', {
        sourcePath: selectedPath,
        modelName: modelName.trim()
      });

      success('Modell importiert!', `"${newModel.name}" wurde erfolgreich importiert.`);
      
      // Reset und schließen
      setSelectedPath(null);
      setModelName('');
      setDirInfo(null);
      setIsValidModel(false);
      setShowImportModal(false);
      
      // Neu laden
      await loadModels();
    } catch (err: any) {
      error('Import fehlgeschlagen', err.message);
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

    // In Tauri können wir den Pfad aus dem dataTransfer bekommen
    const items = e.dataTransfer.items;
    if (items && items.length > 0) {
      // Versuche den Pfad zu bekommen
      const item = items[0];
      if (item.kind === 'file') {
        const file = item.getAsFile();
        if (file) {
          // @ts-ignore - webkitRelativePath existiert
          const path = (file as any).path;
          if (path) {
            await validateAndSetPath(path);
          } else {
            // Fallback: Nutzer muss manuell auswählen
            info('Drag & Drop', 'Bitte nutze den "Ordner durchsuchen" Button.');
          }
        }
      }
    }
  }, []);

  // ============ HuggingFace Functions ============

  const handleHfModelSelect = (model: HuggingFaceModel) => {
    setSelectedHfModel(model);
    setHfModelName(model.id.split('/').pop() || model.id);
  };

  const handleHfDownload = async () => {
    if (!selectedHfModel || !hfModelName.trim()) {
      warning('Fehlende Angaben', 'Bitte wähle ein Modell und gib einen Namen ein.');
      return;
    }

    setDownloading(true);
    setDownloadCancelled(false);
    setDownloadStatus('Verbinde mit Hugging Face...');
    
    try {
      console.log('Downloading HF model:', selectedHfModel.id);
      
      // Progress-Updates während des Downloads
      downloadIntervalRef.current = setInterval(() => {
        setDownloadStatus(prev => {
          const statuses = [
            'Lade Modell-Konfiguration...',
            'Lade Tokenizer...',
            'Lade Modell-Gewichte...',
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

      const newModel = await invoke<ModelInfo>('download_huggingface_model', {
        repoId: selectedHfModel.id,
        modelName: hfModelName.trim()
      });

      if (downloadIntervalRef.current) {
        clearInterval(downloadIntervalRef.current);
      }
      
      console.log('Download complete:', newModel);
      success('Download abgeschlossen!', `"${newModel.name}" wurde heruntergeladen.`);
      
      // Reset
      setSelectedHfModel(null);
      setHfModelName('');
      setHfSearchQuery('');
      setHfResults([]);
      setShowImportModal(false);
      setDownloadStatus('');
      
      await loadModels();
    } catch (err: any) {
      if (downloadIntervalRef.current) {
        clearInterval(downloadIntervalRef.current);
      }
      
      if (!downloadCancelled) {
        console.error('HuggingFace download error:', err);
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

  // ============ Delete Model ============

  const handleDeleteModel = async (model: ModelInfo) => {
    if (!confirm(`Möchtest du "${model.name}" wirklich löschen? Diese Aktion kann nicht rückgängig gemacht werden.`)) {
      return;
    }

    try {
      await invoke('delete_model', { modelId: model.id });
      success('Modell gelöscht', `"${model.name}" wurde entfernt.`);
      await loadModels();
    } catch (err: any) {
      error('Löschen fehlgeschlagen', err.message);
    }
  };

  // ============ Render ============

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Modelle</h1>
          <p className="text-gray-400 mt-1">Verwalte deine lokalen ML-Modelle</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={loadModels}
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-all"
            title="Aktualisieren"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
          <button
            onClick={() => setShowImportModal(true)}
            className={`flex items-center gap-2 px-4 py-2 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white font-medium hover:opacity-90 transition-all`}
          >
            <Upload className="w-5 h-5" />
            Modell hinzufügen
          </button>
        </div>
      </div>

      {/* Models Grid */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 text-gray-400 animate-spin" />
        </div>
      ) : models.length === 0 ? (
        <div className="bg-white/5 rounded-2xl border border-white/10 p-12 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-white/5 mb-4">
            <FileBox className="w-8 h-8 text-gray-400" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">Keine Modelle vorhanden</h3>
          <p className="text-gray-400 mb-6">Füge dein erstes Modell hinzu, um mit dem Training zu beginnen.</p>
          <button
            onClick={() => setShowImportModal(true)}
            className={`inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white font-medium hover:opacity-90 transition-all`}
          >
            <Upload className="w-5 h-5" />
            Erstes Modell hinzufügen
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map((model) => (
            <div
              key={model.id}
              className="bg-white/5 rounded-xl border border-white/10 p-5 hover:bg-white/[0.07] transition-all group"
            >
              {/* Model Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg bg-gradient-to-r ${currentTheme.colors.gradient}`}>
                    {model.source === 'huggingface' ? (
                      <Cloud className="w-5 h-5 text-white" />
                    ) : (
                      <HardDrive className="w-5 h-5 text-white" />
                    )}
                  </div>
                  <div>
                    <h3 className="font-semibold text-white truncate max-w-[180px]" title={model.name}>
                      {model.name}
                    </h3>
                    <span className="text-xs text-gray-500 uppercase">
                      {model.source === 'huggingface' ? 'Hugging Face' : 'Lokal'}
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => handleDeleteModel(model)}
                  className="p-1.5 rounded-lg text-gray-500 hover:text-red-400 hover:bg-red-500/10 opacity-0 group-hover:opacity-100 transition-all"
                  title="Löschen"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>

              {/* Model Info */}
              <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between text-gray-400">
                  <span className="flex items-center gap-2">
                    <FileBox className="w-4 h-4" />
                    {model.file_count} Dateien
                  </span>
                  <span>{formatBytes(model.size_bytes)}</span>
                </div>

                {model.model_type && (
                  <div className="flex items-center gap-2 text-gray-400">
                    <Cpu className="w-4 h-4" />
                    <span className="capitalize">{model.model_type}</span>
                  </div>
                )}

                <div className="flex items-center gap-2 text-gray-500 text-xs">
                  <Calendar className="w-3 h-3" />
                  <span>{formatDate(model.created_at)}</span>
                </div>

                {model.source_path && (
                  <div className="text-xs text-gray-600 truncate" title={model.source_path}>
                    {model.source === 'huggingface' ? (
                      <a
                        href={`https://huggingface.co/${model.source_path}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-1 hover:text-gray-400 transition-colors"
                      >
                        <ExternalLink className="w-3 h-3" />
                        {model.source_path}
                      </a>
                    ) : (
                      <span className="truncate">{model.source_path}</span>
                    )}
                  </div>
                )}
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
              <h2 className="text-xl font-bold text-white">Modell hinzufügen</h2>
              <button
                onClick={() => {
                  setShowImportModal(false);
                  setSelectedPath(null);
                  setModelName('');
                  setDirInfo(null);
                  setSelectedHfModel(null);
                  setHfModelName('');
                  setHfSearchQuery('');
                  setHfResults([]);
                }}
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
                  importMode === 'local'
                    ? `text-white border-b-2`
                    : 'text-gray-400 hover:text-white'
                }`}
                style={importMode === 'local' ? { borderColor: currentTheme.colors.primary } : {}}
              >
                <HardDrive className="w-5 h-5" />
                Lokaler Ordner
              </button>
              <button
                onClick={() => setImportMode('huggingface')}
                className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 font-medium transition-all ${
                  importMode === 'huggingface'
                    ? `text-white border-b-2`
                    : 'text-gray-400 hover:text-white'
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
                  {/* Drag & Drop Zone */}
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
                        {!isValidModel && (
                          <div className="flex items-center justify-center gap-2 text-amber-400 text-sm">
                            <AlertCircle className="w-4 h-4" />
                            <span>Keine Standard-Modelldateien erkannt</span>
                          </div>
                        )}
                        <button
                          onClick={() => {
                            setSelectedPath(null);
                            setModelName('');
                            setDirInfo(null);
                            setIsValidModel(false);
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
                            {isDragging ? 'Ordner hier ablegen' : 'Modell-Ordner hierher ziehen'}
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

                  {/* Model Name Input */}
                  {selectedPath && (
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Modellname
                      </label>
                      <input
                        type="text"
                        value={modelName}
                        onChange={(e) => setModelName(e.target.value)}
                        placeholder="z.B. Llama-2-7B"
                        className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 transition-all"
                        style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
                      />
                    </div>
                  )}

                  {/* Import Button */}
                  {selectedPath && (
                    <button
                      onClick={handleLocalImport}
                      disabled={importing || !modelName.trim()}
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
                          Modell importieren
                        </>
                      )}
                    </button>
                  )}
                </div>
              ) : (
                // ============ HuggingFace Import ============
                <div className="space-y-6">
                  {/* Search Input */}
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Modell suchen
                    </label>
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                      <input
                        type="text"
                        value={hfSearchQuery}
                        onChange={(e) => setHfSearchQuery(e.target.value)}
                        placeholder="z.B. llama, mistral, stable-diffusion..."
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
                      <p className="text-sm text-gray-400">{hfResults.length} Modelle gefunden</p>
                      {hfResults.map((model) => (
                        <button
                          key={model.id}
                          onClick={() => handleHfModelSelect(model)}
                          className={`w-full flex items-center justify-between p-3 rounded-lg border transition-all text-left ${
                            selectedHfModel?.id === model.id
                              ? 'bg-purple-500/20 border-purple-500/50'
                              : 'bg-white/5 border-white/10 hover:bg-white/10'
                          }`}
                        >
                          <div className="min-w-0 flex-1">
                            <p className="text-white font-medium truncate">{model.id}</p>
                            <div className="flex items-center gap-3 text-xs text-gray-400 mt-1">
                              {model.pipeline_tag && (
                                <span className="px-2 py-0.5 bg-white/10 rounded">{model.pipeline_tag}</span>
                              )}
                              <span>↓ {formatDownloads(model.downloads)}</span>
                              {model.likes && <span>♥ {formatDownloads(model.likes)}</span>}
                            </div>
                          </div>
                          {selectedHfModel?.id === model.id && (
                            <CheckCircle className="w-5 h-5 text-purple-400 flex-shrink-0 ml-2" />
                          )}
                        </button>
                      ))}
                    </div>
                  )}

                  {/* Selected Model Details */}
                  {selectedHfModel && (
                    <div className="space-y-4 p-4 bg-white/5 rounded-lg border border-white/10">
                      <div className="flex items-center gap-3">
                        <Cloud className="w-5 h-5 text-purple-400" />
                        <div>
                          <p className="text-white font-medium">{selectedHfModel.id}</p>
                          <a
                            href={`https://huggingface.co/${selectedHfModel.id}`}
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
                          value={hfModelName}
                          onChange={(e) => setHfModelName(e.target.value)}
                          placeholder="Name für lokale Speicherung"
                          className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 transition-all"
                          style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
                        />
                      </div>

                      <button
                        onClick={handleHfDownload}
                        disabled={downloading || !hfModelName.trim()}
                        className={`w-full relative overflow-hidden rounded-lg text-white font-medium transition-all disabled:cursor-not-allowed ${
                          downloading ? 'bg-white/10' : `bg-gradient-to-r ${currentTheme.colors.gradient} hover:opacity-90`
                        }`}
                      >
                        {/* Animated Progress Bar Background */}
                        {downloading && (
                          <div className="absolute inset-0 overflow-hidden">
                            <div 
                              className={`absolute inset-0 bg-gradient-to-r ${currentTheme.colors.gradient} opacity-30`}
                            />
                            <div 
                              className={`absolute inset-y-0 w-1/3 bg-gradient-to-r ${currentTheme.colors.gradient} opacity-60 animate-progress-slide`}
                            />
                          </div>
                        )}
                        
                        {/* Button Content */}
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
                              <span>Modell herunterladen</span>
                            </div>
                          )}
                        </div>
                      </button>

                      {/* Abbruch Button */}
                      {downloading && (
                        <button
                          onClick={handleCancelDownload}
                          className="w-full mt-2 flex items-center justify-center gap-2 px-4 py-2 bg-white/5 hover:bg-red-500/20 border border-white/10 hover:border-red-500/30 rounded-lg text-gray-400 hover:text-red-400 font-medium transition-all"
                        >
                          <X className="w-4 h-4" />
                          <span>Download abbrechen</span>
                        </button>
                      )}

                      {/* Download Hinweis */}
                      {!downloading && (
                        <p className="text-xs text-gray-500 text-center mt-2">
                          Der Download kann je nach Modellgröße einige Minuten dauern
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
    </div>
  );
}