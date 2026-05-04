// ModelManager.tsx – Modell-Verwaltung mit Plugin-Erkennung
// Portiert & erweitert aus desktop-app2

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
  RefreshCw,
  Puzzle,
  Ban,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';
import { usePageContext } from '../contexts/PageContext';
import { detectPlugin } from '../plugins/registry';
import type { ModelConfig } from '../plugins/types';

// ============ Types ============

interface ModelInfo {
  id: string;
  name: string;
  source: 'local' | 'huggingface';
  source_path: string | null;
  size_bytes: number;
  file_count: number;
  created_at: string;
  model_type: string | null;
}

interface HuggingFaceModel {
  id: string;
  author?: string;
  downloads?: number;
  likes?: number;
  pipeline_tag?: string;
}

type ImportMode = 'local' | 'huggingface';

// ============ Helpers ============

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString: string): string {
  return new Date(dateString).toLocaleDateString('de-DE', {
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

// ============ Plugin-Badge ============

function PluginBadge({ modelNameOrPath, configJson }: { modelNameOrPath: string; configJson?: ModelConfig }) {
  const result = detectPlugin(modelNameOrPath, configJson);
  if (result.supported) {
    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-500/15 border border-emerald-500/30 text-emerald-400">
        <Puzzle className="w-3 h-3" />
        {result.plugin.name}
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium bg-white/5 border border-white/10 text-gray-500">
      <Ban className="w-3 h-3" />
      Kein Plugin
    </span>
  );
}

// ============ Delete Confirm Dialog ============

interface DeleteDialogProps {
  modelName: string;
  onConfirm: () => void;
  onCancel: () => void;
}

function DeleteConfirmDialog({ modelName, onConfirm, onCancel }: DeleteDialogProps) {
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
              <h2 className="text-white font-semibold text-lg">Modell löschen?</h2>
              <p className="text-gray-400 text-sm mt-1.5 leading-relaxed">
                <span className="text-white">„{modelName}"</span> und alle zugehörigen Versionen, Datensätze und Trainingsdaten werden unwiderruflich gelöscht.
              </p>
            </div>
          </div>
          <div className="flex gap-3">
            <button onClick={onCancel} className="flex-1 py-2.5 px-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-white text-sm font-medium transition-all">
              Abbrechen
            </button>
            <button onClick={onConfirm} className="flex-1 py-2.5 px-4 bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 rounded-xl text-red-300 text-sm font-medium transition-all">
              Löschen
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============ Main Component ============

export default function ModelManager() {
  const { currentTheme } = useTheme();
  const { success, error, warning, info } = useNotification();
  const { setCurrentPageContent } = usePageContext();

  // Models
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);

  // Delete dialog
  const [deleteTarget, setDeleteTarget] = useState<ModelInfo | null>(null);

  // Import modal
  const [showImportModal, setShowImportModal] = useState(false);
  const [importMode, setImportMode] = useState<ImportMode>('local');

  // ── Local import ──
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [modelName, setModelName] = useState('');
  const [dirInfo, setDirInfo] = useState<{ size: number; files: number } | null>(null);
  const [isValidModel, setIsValidModel] = useState(false);
  const [importing, setImporting] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  // ── HuggingFace import ──
  const [hfQuery, setHfQuery] = useState('');
  const [hfResults, setHfResults] = useState<HuggingFaceModel[]>([]);
  const [hfSearching, setHfSearching] = useState(false);
  const [selectedHfModel, setSelectedHfModel] = useState<HuggingFaceModel | null>(null);
  const [hfModelName, setHfModelName] = useState('');
  const [downloading, setDownloading] = useState(false);
  const [downloadStatus, setDownloadStatus] = useState('');
  const downloadIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const searchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Load ──
  useEffect(() => { loadModels(); }, []);

  // ── AI-Coach Context ──
  useEffect(() => {
    const lines = [
      '=== FrameTrain Modell-Manager ===',
      '',
      `Geladene Modelle: ${models.length}`,
      ...(models.length === 0
        ? ['(keine Modelle vorhanden)']
        : models.map(m =>
            `• ${m.name} | ${m.source === 'huggingface' ? 'HuggingFace' : 'Lokal'} | ${m.model_type ?? 'unbekannter Typ'} | ${formatBytes(m.size_bytes)}`
          )),
    ];
    setCurrentPageContent(lines.join('\n'));
  }, [models, setCurrentPageContent]);

  // ── Debounced HuggingFace search ──
  useEffect(() => {
    if (searchTimeoutRef.current) clearTimeout(searchTimeoutRef.current);
    if (hfQuery.trim().length < 2) { setHfResults([]); setHfSearching(false); return; }

    setHfSearching(true);
    searchTimeoutRef.current = setTimeout(async () => {
      try {
        const results = await invoke<HuggingFaceModel[]>('search_huggingface_models', {
          query: hfQuery.trim(), limit: 15,
        });
        setHfResults(results);
      } catch { /* still ok */ } finally { setHfSearching(false); }
    }, 300);

    return () => { if (searchTimeoutRef.current) clearTimeout(searchTimeoutRef.current); };
  }, [hfQuery]);

  // ──────────────────────────────────────────
  // Load
  // ──────────────────────────────────────────
  const loadModels = async () => {
    try {
      setLoading(true);
      const list = await invoke<ModelInfo[]>('list_models');
      setModels(list);
    } catch (err: unknown) {
      error('Fehler beim Laden', err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  // ──────────────────────────────────────────
  // Local Import
  // ──────────────────────────────────────────
  const validateAndSetPath = async (path: string) => {
    setSelectedPath(path);
    setModelName(path.split(/[/\\]/).pop() ?? 'Modell');
    try {
      const isValid = await invoke<boolean>('validate_model_directory', { path });
      setIsValidModel(isValid);
      if (!isValid) warning('Kein gültiges Modell', 'Der Ordner enthält keine erkennbaren Modell-Dateien.');
      const [size, files] = await invoke<[number, number]>('get_directory_size', { path });
      setDirInfo({ size, files });
    } catch (err: unknown) {
      error('Validierungsfehler', err instanceof Error ? err.message : String(err));
      setIsValidModel(false);
      setDirInfo(null);
    }
  };

  const handleBrowseFolder = async () => {
    try {
      const selected = await open({ directory: true, multiple: false, title: 'Modell-Ordner auswählen' });
      if (selected && typeof selected === 'string') await validateAndSetPath(selected);
    } catch (err: unknown) {
      error('Fehler', String(err));
    }
  };

  const handleLocalImport = async () => {
    if (!selectedPath || !modelName.trim()) { warning('Fehlende Angaben', 'Ordner und Name werden benötigt.'); return; }
    setImporting(true);
    try {
      const newModel = await invoke<ModelInfo>('import_local_model', {
        sourcePath: selectedPath, modelName: modelName.trim(),
      });
      success('Modell importiert!', `„${newModel.name}" wurde erfolgreich hinzugefügt.`);
      resetLocalImport();
      setShowImportModal(false);
      await loadModels();
    } catch (err: unknown) {
      error('Import fehlgeschlagen', err instanceof Error ? err.message : String(err));
    } finally {
      setImporting(false);
    }
  };

  const resetLocalImport = () => {
    setSelectedPath(null); setModelName(''); setDirInfo(null); setIsValidModel(false);
  };

  // ── Drag & Drop ──
  const handleDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); }, []);
  const handleDragLeave = useCallback((e: React.DragEvent) => { e.preventDefault(); setIsDragging(false); }, []);
  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault(); setIsDragging(false);
    const file = e.dataTransfer.items?.[0]?.getAsFile?.();
    const path = file && (file as unknown as { path?: string }).path;
    if (path) await validateAndSetPath(path);
    else info('Drag & Drop', 'Bitte nutze den „Ordner durchsuchen"-Button.');
  }, []);

  // ──────────────────────────────────────────
  // HuggingFace Import
  // ──────────────────────────────────────────
  const handleHfSelect = (m: HuggingFaceModel) => {
    setSelectedHfModel(m);
    setHfModelName(m.id.split('/').pop() ?? m.id);
  };

  const handleHfDownload = async () => {
    if (!selectedHfModel || !hfModelName.trim()) { warning('Fehlende Angaben', 'Modell und Name werden benötigt.'); return; }
    setDownloading(true);
    setDownloadStatus('Verbinde mit Hugging Face…');

    const statusSteps = ['Lade Konfiguration…', 'Lade Tokenizer…', 'Lade Gewichte…', 'Speichere Dateien…', 'Fast fertig…'];
    let stepIdx = 0;
    downloadIntervalRef.current = setInterval(() => {
      stepIdx = (stepIdx + 1) % statusSteps.length;
      setDownloadStatus(statusSteps[stepIdx]);
    }, 2000);

    try {
      const newModel = await invoke<ModelInfo>('download_huggingface_model', {
        repoId: selectedHfModel.id, modelName: hfModelName.trim(),
      });
      clearInterval(downloadIntervalRef.current!);
      success('Download abgeschlossen!', `„${newModel.name}" wurde heruntergeladen.`);
      setSelectedHfModel(null); setHfModelName(''); setHfQuery(''); setHfResults([]);
      setShowImportModal(false); setDownloadStatus('');
      await loadModels();
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

  // ──────────────────────────────────────────
  // Delete
  // ──────────────────────────────────────────
  const handleDeleteConfirm = async () => {
    if (!deleteTarget) return;
    try {
      await invoke('delete_model', { modelId: deleteTarget.id });
      success('Gelöscht', `„${deleteTarget.name}" wurde entfernt.`);
      await loadModels();
    } catch (err: unknown) {
      error('Löschen fehlgeschlagen', err instanceof Error ? err.message : String(err));
    } finally {
      setDeleteTarget(null);
    }
  };

  // ──────────────────────────────────────────
  // Modal close helper
  // ──────────────────────────────────────────
  const closeModal = () => {
    setShowImportModal(false);
    resetLocalImport();
    setSelectedHfModel(null); setHfModelName(''); setHfQuery(''); setHfResults([]);
  };

  // ──────────────────────────────────────────
  // Render
  // ──────────────────────────────────────────
  return (
    <div className="space-y-6">

      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Modelle</h1>
          <p className="text-gray-400 mt-1">Verwalte lokale und HuggingFace-Modelle</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={loadModels}
            className="p-2 rounded-xl bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white border border-white/10 transition-all"
            title="Aktualisieren"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowImportModal(true)}
            className={`flex items-center gap-2 px-4 py-2 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-xl text-white text-sm font-medium hover:opacity-90 transition-all`}
          >
            <Upload className="w-4 h-4" />
            Modell hinzufügen
          </button>
        </div>
      </div>

      {/* ── Model Grid ── */}
      {loading ? (
        <div className="flex items-center justify-center py-24">
          <Loader2 className="w-8 h-8 text-gray-500 animate-spin" />
        </div>
      ) : models.length === 0 ? (
        <div className="rounded-2xl border border-white/10 bg-white/5 p-16 text-center space-y-4">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-white/5 border border-white/10">
            <FileBox className="w-8 h-8 text-gray-500" />
          </div>
          <div>
            <h3 className="text-white font-semibold text-lg">Keine Modelle vorhanden</h3>
            <p className="text-gray-400 text-sm mt-1">Füge dein erstes Modell hinzu, um mit dem Training zu beginnen.</p>
          </div>
          <button
            onClick={() => setShowImportModal(true)}
            className={`inline-flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-xl text-white text-sm font-medium hover:opacity-90 transition-all`}
          >
            <Upload className="w-4 h-4" />
            Erstes Modell hinzufügen
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map((model) => (
            <ModelCard
              key={model.id}
              model={model}
              onDelete={() => setDeleteTarget(model)}
              gradientClass={currentTheme.colors.gradient}
            />
          ))}
        </div>
      )}

      {/* ── Delete Dialog ── */}
      {deleteTarget && (
        <DeleteConfirmDialog
          modelName={deleteTarget.name}
          onConfirm={handleDeleteConfirm}
          onCancel={() => setDeleteTarget(null)}
        />
      )}

      {/* ── Import Modal ── */}
      {showImportModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">

            {/* Modal Header */}
            <div className="flex items-center justify-between px-6 py-5 border-b border-white/10 flex-shrink-0">
              <h2 className="text-xl font-bold text-white">Modell hinzufügen</h2>
              <button onClick={closeModal} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all">
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Tabs */}
            <div className="flex border-b border-white/10 flex-shrink-0">
              {(['local', 'huggingface'] as ImportMode[]).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setImportMode(mode)}
                  className={`flex-1 flex items-center justify-center gap-2 px-4 py-3.5 text-sm font-medium transition-all border-b-2 ${
                    importMode === mode
                      ? 'text-white border-current'
                      : 'text-gray-400 hover:text-white border-transparent'
                  }`}
                  style={importMode === mode ? { borderColor: currentTheme.colors.primary, color: currentTheme.colors.primary } : {}}
                >
                  {mode === 'local' ? <HardDrive className="w-4 h-4" /> : <Cloud className="w-4 h-4" />}
                  {mode === 'local' ? 'Lokaler Ordner' : 'Hugging Face'}
                </button>
              ))}
            </div>

            {/* Modal Body */}
            <div className="p-6 overflow-y-auto flex-1">
              {importMode === 'local' ? (
                <LocalImportPanel
                  isDragging={isDragging}
                  selectedPath={selectedPath}
                  dirInfo={dirInfo}
                  isValidModel={isValidModel}
                  modelName={modelName}
                  importing={importing}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onBrowse={handleBrowseFolder}
                  onReset={resetLocalImport}
                  onNameChange={setModelName}
                  onImport={handleLocalImport}
                  gradientClass={currentTheme.colors.gradient}
                />
              ) : (
                <HuggingFaceImportPanel
                  query={hfQuery}
                  results={hfResults}
                  searching={hfSearching}
                  selected={selectedHfModel}
                  localName={hfModelName}
                  downloading={downloading}
                  downloadStatus={downloadStatus}
                  onQueryChange={setHfQuery}
                  onSelect={handleHfSelect}
                  onNameChange={setHfModelName}
                  onDownload={handleHfDownload}
                  onCancel={handleCancelDownload}
                  gradientClass={currentTheme.colors.gradient}
                />
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ──────────────────────────────────────────
// ModelCard
// ──────────────────────────────────────────

interface ModelCardProps {
  model: ModelInfo;
  onDelete: () => void;
  gradientClass: string;
}

function ModelCard({ model, onDelete, gradientClass }: ModelCardProps) {
  // Determine the identifier to use for plugin detection
  const detectionKey = model.source === 'huggingface' && model.source_path
    ? model.source_path
    : model.name;

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-5 hover:bg-white/[0.07] transition-all group flex flex-col gap-4">
      {/* Top row */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3 min-w-0">
          <div className={`p-2 rounded-xl bg-gradient-to-r ${gradientClass} flex-shrink-0`}>
            {model.source === 'huggingface'
              ? <Cloud className="w-4 h-4 text-white" />
              : <HardDrive className="w-4 h-4 text-white" />
            }
          </div>
          <div className="min-w-0">
            <h3 className="font-semibold text-white truncate" title={model.name}>{model.name}</h3>
            <span className="text-xs text-gray-500">{model.source === 'huggingface' ? 'Hugging Face' : 'Lokal'}</span>
          </div>
        </div>
        <button
          onClick={onDelete}
          className="p-1.5 rounded-lg text-gray-600 hover:text-red-400 hover:bg-red-500/10 opacity-0 group-hover:opacity-100 transition-all flex-shrink-0"
          title="Löschen"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {/* Plugin Badge */}
      <div>
        <PluginBadge
          modelNameOrPath={detectionKey}
          configJson={model.model_type ? { model_type: model.model_type } : undefined}
        />
      </div>

      {/* Meta */}
      <div className="space-y-1.5 text-sm">
        <div className="flex items-center justify-between text-gray-400">
          <span className="flex items-center gap-1.5"><FileBox className="w-3.5 h-3.5" />{model.file_count} Dateien</span>
          <span>{formatBytes(model.size_bytes)}</span>
        </div>

        {model.model_type && (
          <div className="flex items-center gap-1.5 text-gray-400">
            <Cpu className="w-3.5 h-3.5" />
            <span className="capitalize text-xs">{model.model_type}</span>
          </div>
        )}

        <div className="flex items-center gap-1.5 text-gray-500 text-xs">
          <Calendar className="w-3 h-3" />
          {formatDate(model.created_at)}
        </div>

        {model.source_path && (
          <div className="text-xs text-gray-600 truncate pt-0.5">
            {model.source === 'huggingface' ? (
              <a
                href={`https://huggingface.co/${model.source_path}`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 hover:text-gray-400 transition-colors"
              >
                <ExternalLink className="w-3 h-3 flex-shrink-0" />
                <span className="truncate">{model.source_path}</span>
              </a>
            ) : (
              <span className="truncate" title={model.source_path}>{model.source_path}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ──────────────────────────────────────────
// LocalImportPanel
// ──────────────────────────────────────────

interface LocalImportPanelProps {
  isDragging: boolean;
  selectedPath: string | null;
  dirInfo: { size: number; files: number } | null;
  isValidModel: boolean;
  modelName: string;
  importing: boolean;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent) => void;
  onBrowse: () => void;
  onReset: () => void;
  onNameChange: (v: string) => void;
  onImport: () => void;
  gradientClass: string;
}

function LocalImportPanel({
  isDragging, selectedPath, dirInfo, isValidModel, modelName, importing,
  onDragOver, onDragLeave, onDrop, onBrowse, onReset, onNameChange, onImport,
  gradientClass,
}: LocalImportPanelProps) {
  return (
    <div className="space-y-5">
      {/* Drop Zone */}
      <div
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={`border-2 border-dashed rounded-2xl p-10 text-center transition-all ${
          isDragging ? 'border-purple-500 bg-purple-500/10' :
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
            {dirInfo && (
              <p className="text-gray-500 text-sm">{dirInfo.files} Dateien · {formatBytes(dirInfo.size)}</p>
            )}
            {!isValidModel && (
              <div className="inline-flex items-center gap-2 text-amber-400 text-sm">
                <AlertCircle className="w-4 h-4" />
                Keine Standard-Modelldateien erkannt
              </div>
            )}
            <button onClick={onReset} className="text-sm text-gray-400 hover:text-white underline transition-colors">
              Anderen Ordner wählen
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-white/5 border border-white/10">
              <Upload className="w-7 h-7 text-gray-400" />
            </div>
            <div>
              <p className="text-white font-medium">{isDragging ? 'Ordner hier ablegen' : 'Modell-Ordner hierher ziehen'}</p>
              <p className="text-gray-500 text-sm mt-1">oder Ordner manuell auswählen</p>
            </div>
            <button
              onClick={onBrowse}
              className="inline-flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/15 rounded-xl text-white text-sm transition-all"
            >
              <FolderOpen className="w-4 h-4" />
              Ordner durchsuchen
            </button>
          </div>
        )}
      </div>

      {/* Name Input */}
      {selectedPath && (
        <div className="space-y-1.5">
          <label className="block text-sm font-medium text-gray-300">Modellname</label>
          <input
            type="text"
            value={modelName}
            onChange={(e) => onNameChange(e.target.value)}
            placeholder="z.&nbsp;B. xlm-roberta-base"
            className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-white/30 transition-all"
          />
        </div>
      )}

      {/* Import Button */}
      {selectedPath && (
        <button
          onClick={onImport}
          disabled={importing || !modelName.trim()}
          className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r ${gradientClass} text-white text-sm font-medium hover:opacity-90 transition-all disabled:opacity-50 disabled:cursor-not-allowed`}
        >
          {importing ? <><Loader2 className="w-4 h-4 animate-spin" /> Importiere…</> : <><Upload className="w-4 h-4" /> Modell importieren</>}
        </button>
      )}
    </div>
  );
}

// ──────────────────────────────────────────
// HuggingFaceImportPanel
// ──────────────────────────────────────────

interface HuggingFaceImportPanelProps {
  query: string;
  results: HuggingFaceModel[];
  searching: boolean;
  selected: HuggingFaceModel | null;
  localName: string;
  downloading: boolean;
  downloadStatus: string;
  onQueryChange: (v: string) => void;
  onSelect: (m: HuggingFaceModel) => void;
  onNameChange: (v: string) => void;
  onDownload: () => void;
  onCancel: () => void;
  gradientClass: string;
}

function HuggingFaceImportPanel({
  query, results, searching, selected, localName, downloading, downloadStatus,
  onQueryChange, onSelect, onNameChange, onDownload, onCancel, gradientClass,
}: HuggingFaceImportPanelProps) {
  return (
    <div className="space-y-5">
      {/* Search */}
      <div className="space-y-1.5">
        <label className="block text-sm font-medium text-gray-300">Modell suchen</label>
        <div className="relative">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder="z.&nbsp;B. xlm-roberta, bert, mistral…"
            className="w-full pl-10 pr-10 py-3 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-white/30 transition-all"
          />
          {searching && <Loader2 className="absolute right-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 animate-spin" />}
        </div>
        <p className="text-gray-600 text-xs">Mindestens 2 Zeichen eingeben</p>
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-2">
          <p className="text-gray-500 text-xs">{results.length} Modelle gefunden</p>
          <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
            {results.map((m) => (
              <button
                key={m.id}
                onClick={() => onSelect(m)}
                className={`w-full flex items-center justify-between p-3 rounded-xl border text-left transition-all ${
                  selected?.id === m.id
                    ? 'bg-emerald-500/10 border-emerald-500/40'
                    : 'bg-white/5 border-white/10 hover:bg-white/10'
                }`}
              >
                <div className="min-w-0">
                  <p className="text-white text-sm font-medium truncate">{m.id}</p>
                  <div className="flex items-center gap-2 mt-0.5 text-xs text-gray-500">
                    {m.pipeline_tag && (
                      <span className="px-1.5 py-0.5 bg-white/10 rounded-md">{m.pipeline_tag}</span>
                    )}
                    <span>↓ {formatDownloads(m.downloads)}</span>
                    {m.likes ? <span>♥ {formatDownloads(m.likes)}</span> : null}
                  </div>
                </div>
                {selected?.id === m.id && <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0 ml-2" />}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Selected model details */}
      {selected && (
        <div className="space-y-4 p-4 rounded-2xl border border-white/10 bg-white/5">
          <div className="flex items-center gap-3">
            <Cloud className="w-5 h-5 text-gray-400" />
            <div>
              <p className="text-white font-medium text-sm">{selected.id}</p>
              <a
                href={`https://huggingface.co/${selected.id}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-gray-500 hover:text-gray-300 flex items-center gap-1 transition-colors"
              >
                Auf HuggingFace ansehen <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>

          {/* Plugin Preview */}
          <div className="flex items-center gap-2">
            <span className="text-gray-500 text-xs">Plugin-Erkennung:</span>
            <PluginBadge modelNameOrPath={selected.id} />
          </div>

          {/* Local name */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium text-gray-300">Lokaler Name</label>
            <input
              type="text"
              value={localName}
              onChange={(e) => onNameChange(e.target.value)}
              placeholder="Name für die lokale Speicherung"
              className="w-full px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-white/30 transition-all"
            />
          </div>

          {/* Download button */}
          <button
            onClick={onDownload}
            disabled={downloading || !localName.trim()}
            className={`w-full relative overflow-hidden rounded-xl text-white text-sm font-medium transition-all disabled:cursor-not-allowed ${
              downloading ? 'bg-white/10' : `bg-gradient-to-r ${gradientClass} hover:opacity-90`
            }`}
          >
            {downloading && (
              <div className="absolute inset-0">
                <div className={`absolute inset-0 bg-gradient-to-r ${gradientClass} opacity-25`} />
                <div className={`absolute inset-y-0 w-1/3 bg-gradient-to-r ${gradientClass} opacity-50 animate-[progress-slide_1.5s_ease-in-out_infinite]`} />
              </div>
            )}
            <div className="relative flex flex-col items-center py-3">
              {downloading ? (
                <>
                  <div className="flex items-center gap-2"><Loader2 className="w-4 h-4 animate-spin" /> Lade herunter…</div>
                  {downloadStatus && <span className="text-xs text-white/60 mt-0.5">{downloadStatus}</span>}
                </>
              ) : (
                <div className="flex items-center gap-2"><Download className="w-4 h-4" /> Modell herunterladen</div>
              )}
            </div>
          </button>

          {downloading ? (
            <button
              onClick={onCancel}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-white/5 hover:bg-red-500/10 border border-white/10 hover:border-red-500/30 text-gray-400 hover:text-red-400 text-sm transition-all"
            >
              <X className="w-4 h-4" /> Download abbrechen
            </button>
          ) : (
            <p className="text-xs text-gray-600 text-center">
              Download-Dauer hängt von der Modellgröße ab
            </p>
          )}
        </div>
      )}
    </div>
  );
}
