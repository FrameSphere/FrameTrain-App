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
  Heart,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';
import { usePageContext } from '../contexts/PageContext';
import { useLanguage } from '../contexts/LanguageContext';
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
  const { t } = useLanguage();
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
      {t('modelManager.noPlugin')}
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
  const { t } = useLanguage();
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
              <h2 className="text-white font-semibold text-lg">{t('modelManager.deleteDialog.title')}</h2>
              <p className="text-gray-400 text-sm mt-1.5 leading-relaxed">
                <span className="text-white">{t('modelManager.deleteDialog.description').replace('{name}', modelName)}</span><br />
              </p>
            </div>
          </div>
          <div className="flex gap-3">
            <button onClick={onCancel} className="flex-1 py-2.5 px-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-white text-sm font-medium transition-all">
            {t('common.cancel')}
            </button>
            <button onClick={onConfirm} className="flex-1 py-2.5 px-4 bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 rounded-xl text-red-300 text-sm font-medium transition-all">
            {t('common.delete')}
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
  const { t } = useLanguage();

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
  const [downloadProgress, setDownloadProgress] = useState<{
    status: string;
    currentFile: string;
    currentFileIndex: number;
    totalFiles: number;
    downloadedBytes: number;
    totalBytes: number;
    progressPercent: number;
    speedMbs: number;
    elapsedSecs: number;
    etaSecs: number;
    message: string;
  } | null>(null);
  const downloadIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const searchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Load ──
  useEffect(() => { loadModels(); }, []);

  // ── AI-Coach Context (dynamisch) ──
  useEffect(() => {
    const lines: string[] = [
      '=== FrameTrain Modell-Manager ===',
      '',
      '--- AKTUELLE SEITE ---',
      'Zentrale Verwaltung für lokale und HuggingFace-Modelle.',
      'Modelle sind die Grundlage für das Training und die Tests.',
      '',
      '--- VERFÜGBARE MODELLE ---',
    ];

    if (loading) {
      lines.push('Status: Modelle werden geladen...');
    } else if (models.length === 0) {
      lines.push('Status: Keine Modelle vorhanden');
      lines.push('Nächster Schritt: Füge dein erstes Modell hinzu (Button "Modell hinzufügen")');
    } else {
      lines.push(`Status: ${models.length} Modell${models.length !== 1 ? 'e' : ''} verfügbar`);
      lines.push('');
      models.forEach(m => {
        const plugin = detectPlugin(
          m.source === 'huggingface' && m.source_path ? m.source_path : m.name,
          m.model_type ? { model_type: m.model_type } : undefined
        );
        const pluginInfo = plugin.supported ? `[✓ ${plugin.plugin.name}]` : '[⚠ Kein Plugin]';
        lines.push(
          `• **${m.name}** (${m.source === 'huggingface' ? '☁️ HF' : '💾 Lokal'}) · ${pluginInfo}`
        );
        lines.push(`  Type: ${m.model_type ?? '?'} | Size: ${formatBytes(m.size_bytes)} | Files: ${m.file_count}`);
        if (m.source === 'huggingface' && m.source_path) {
          lines.push(`  Source: huggingface.co/${m.source_path}`);
        }
      });
    }

    lines.push('');
    lines.push('--- AKTIVE AKTIONEN ---');

    if (importing) {
      lines.push('🔄 Importiere Modell von lokalem Ordner...');
      lines.push(`   • Ordner: ${selectedPath}`);
      lines.push(`   • Name: ${modelName}`);
      lines.push('   → Bitte warten Sie, bis der Import abgeschlossen ist');
    } else if (hfSearching) {
      lines.push('🔍 Suche HuggingFace-Modelle...');
      lines.push(`   • Suchbegriff: "${hfQuery}"`);
    } else if (downloading) {
      lines.push('📥 Download läuft...');
      if (downloadProgress) {
        lines.push(`   • Modell: ${selectedHfModel?.id}`);
        lines.push(`   • Fortschritt: ${downloadProgress.progressPercent}%`);
        lines.push(`   • Download: ${formatBytes(downloadProgress.downloadedBytes)} / ${formatBytes(downloadProgress.totalBytes)}`);
        lines.push(`   • Geschwindigkeit: ${downloadProgress.speedMbs.toFixed(1)} MB/s`);
        lines.push(`   • Verbleibend: ~${Math.ceil(downloadProgress.etaSecs)}s`);
        lines.push(`   • Datei: ${downloadProgress.currentFileIndex}/${downloadProgress.totalFiles}`);
      }
      lines.push('   → Bitte warten Sie, bis der Download abgeschlossen ist');
    } else {
      lines.push('(Keine laufenden Aktionen)');
    }

    lines.push('');
    lines.push('--- MODAL-STATUS ---');

    if (showImportModal) {
      lines.push(`✓ Import-Modal ist offen`);
      lines.push(`  • Aktueller Tab: ${importMode === 'local' ? 'Lokaler Ordner' : 'HuggingFace'}`);

      if (importMode === 'local') {
        lines.push('  • Local Mode:');
        if (selectedPath) {
          lines.push(`    - Ordner: ${selectedPath}`);
          lines.push(`    - Name: ${modelName || '(nicht gesetzt)'}`);
          lines.push(`    - Valid: ${isValidModel ? '✓ Ja' : '⚠ Nein'}`);
          if (dirInfo) {
            lines.push(`    - Files: ${dirInfo.files}, Size: ${formatBytes(dirInfo.size)}`);
          }
        } else {
          lines.push('    - Kein Ordner gewählt → Drag & Drop oder "Ordner durchsuchen"');
        }
      } else {
        lines.push('  • HuggingFace Mode:');
        if (hfResults.length > 0) {
          lines.push(`    - ${hfResults.length} Modelle gefunden`);
          if (selectedHfModel) {
            lines.push(`    - Ausgewählt: ${selectedHfModel.id}`);
            lines.push(`    - Lokaler Name: ${hfModelName || '(nicht gesetzt)'}`);
          } else {
            lines.push('    - Wähle ein Modell aus der Liste');
          }
        } else if (hfQuery.trim().length > 0) {
          lines.push(`    - Suche: "${hfQuery}" (mind. 2 Zeichen)`);
        } else {
          lines.push('    - Gib Suchbegriff ein (z.B. bert, mistral, xlm-roberta)');
        }
      }
    } else {
      lines.push('(Modal ist geschlossen)');
    }

    if (deleteTarget) {
      lines.push('');
      lines.push('⚠️ Lösch-Bestätigung ausstehend');
      lines.push(`   • Zu löschen: "${deleteTarget.name}"`);
      lines.push('   • Bestätigung: (Modal am Bildschirm)');
    }

    lines.push('');
    lines.push('--- VERFÜGBARE AKTIONEN ---');
    lines.push('1. **Modell hinzufügen** (oben rechts)');
    lines.push('   → Lokal: Ordner mit Model-Dateien hinzufügen');
    lines.push('   → HuggingFace: Von HF herunterladen (z.B. bert, mistral)');
    lines.push('2. **Modell ansehen** → Plugin-Support, Größe, Typ');
    lines.push('3. **Modell löschen** → Mit Bestätigung (hover auf Karte)');
    lines.push('4. **Aktualisieren** (oben links) → Modell-Liste neu laden');
    lines.push('5. **Mit anderen Seiten arbeiten:**');
    lines.push('   → Training: Wähle Modell zum Trainieren');
    lines.push('   → Tests: Wähle Modell zum Testen');
    lines.push('   → Laboratory: Teste Samples mit Modell');

    lines.push('');
    lines.push('--- TIPPS FÜR DEN AI-COACH ---');
    lines.push('• KI kann dir helfen, das richtige Modell zu wählen');
    lines.push('• KI kann Plugins erkennen und kompatible Konfigurationen vorschlagen');
    lines.push('• Bei Download-Fehlern: KI kann dir weitere Schritte empfehlen');
    lines.push('• Bei Speicherplatz-Problemen: Frag den Coach um Hilfe');

    setCurrentPageContent(lines.join('\n'));
  }, [
    models,
    loading,
    showImportModal,
    importMode,
    selectedPath,
    modelName,
    dirInfo,
    isValidModel,
    hfQuery,
    hfResults,
    hfSearching,
    selectedHfModel,
    hfModelName,
    importing,
    downloading,
    downloadProgress,
    deleteTarget,
    setCurrentPageContent,
  ]);

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

  // ── Download Progress Listener ──
  useEffect(() => {
    let unlisten: (() => void) | null = null;

    const setupListener = async () => {
      try {
        const { listen } = await import('@tauri-apps/api/event');
        unlisten = await listen<{
          status: string;
          current_file: string;
          current_file_index: number;
          total_files: number;
          downloaded_bytes: number;
          total_bytes: number;
          progress_percent: number;
          speed_mbs: number;
          elapsed_secs: number;
          eta_secs: number;
          message: string;
        }>('model-download-progress', (event) => {
          const progress = event.payload;
          setDownloadProgress({
            status: progress.status,
            currentFile: progress.current_file,
            currentFileIndex: progress.current_file_index,
            totalFiles: progress.total_files,
            downloadedBytes: progress.downloaded_bytes,
            totalBytes: progress.total_bytes,
            progressPercent: progress.progress_percent,
            speedMbs: progress.speed_mbs,
            elapsedSecs: progress.elapsed_secs,
            etaSecs: progress.eta_secs,
            message: progress.message,
          });

          if (progress.status === 'complete') {
            setDownloading(false);
          } else if (progress.status === 'error') {
            setDownloading(false);
          }
        });
      } catch {
        // Fallback: keine Events verfügbar
      }
    };

    setupListener();

    return () => {
      if (unlisten) {
        unlisten();
      }
    };
  }, []);

  // ──────────────────────────────────────────
  // Load
  // ──────────────────────────────────────────
  const loadModels = async () => {
    try {
      setLoading(true);
      const list = await invoke<ModelInfo[]>('list_models');
      setModels(list);
    } catch (err: unknown) {
      error(t('modelManager.notifications.loadError'), err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  // ──────────────────────────────────────────
  // Local Import
  // ──────────────────────────────────────────
  const validateAndSetPath = async (path: string) => {
    setSelectedPath(path);
    setModelName(path.split(/[/\\]/).pop() ?? t('modelManager.defaultModelName'));
    try {
      const isValid = await invoke<boolean>('validate_model_directory', { path });
      setIsValidModel(isValid);
      if (!isValid) warning(t('modelManager.notifications.validationWarning'), t('modelManager.notifications.validationWarningDetail'));
      const [size, files] = await invoke<[number, number]>('get_directory_size', { path });
      setDirInfo({ size, files });
    } catch (err: unknown) {
      error(t('modelManager.notifications.validationError'), err instanceof Error ? err.message : String(err));
      setIsValidModel(false);
      setDirInfo(null);
    }
  };

  const handleBrowseFolder = async () => {
    try {
      const selected = await open({ directory: true, multiple: false, title: t('modelManager.importModal.local.browseFolderTitle') });
      if (selected && typeof selected === 'string') await validateAndSetPath(selected);
    } catch (err: unknown) {
      error(t('common.error'), String(err));
    }
  };

  const handleLocalImport = async () => {
    if (!selectedPath || !modelName.trim()) { warning(t('modelManager.notifications.missingFields'), t('modelManager.notifications.missingFieldsLocal')); return; }
    setImporting(true);
    try {
      const newModel = await invoke<ModelInfo>('import_local_model', {
        sourcePath: selectedPath, modelName: modelName.trim(),
      });
      success(t('modelManager.notifications.importSuccess'), t('modelManager.notifications.importSuccessDetail').replace('{name}', newModel.name));
      resetLocalImport();
      setShowImportModal(false);
      await loadModels();
    } catch (err: unknown) {
      error(t('modelManager.notifications.importError'), err instanceof Error ? err.message : String(err));
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
    else info(t('modelManager.notifications.dragDropInfo'), t('modelManager.notifications.dragDropInfoDetail'));
  }, []);

  // ──────────────────────────────────────────
  // HuggingFace Import
  // ──────────────────────────────────────────
  const handleHfSelect = (m: HuggingFaceModel) => {
    setSelectedHfModel(m);
    setHfModelName(m.id.split('/').pop() ?? m.id);
  };

  const handleHfDownload = async () => {
    if (!selectedHfModel || !hfModelName.trim()) { warning(t('modelManager.notifications.missingFields'), t('modelManager.notifications.missingFieldsHF')); return; }
    setDownloading(true);
    setDownloadProgress(null);

    // Speichere die model_id für potentiellen Cleanup bei Abbruch
    const modelIdForCleanup = `hf_${Date.now().toString().slice(-12)}`;

    try {
      const newModel = await invoke<ModelInfo>('download_huggingface_model', {
        repoId: selectedHfModel.id, modelName: hfModelName.trim(),
      });
      success(t('modelManager.notifications.downloadSuccess'), t('modelManager.notifications.downloadSuccessDetail').replace('{name}', newModel.name));
      setSelectedHfModel(null); setHfModelName(''); setHfQuery(''); setHfResults([]);
      setShowImportModal(false);
      setDownloadProgress(null);
      await loadModels();
    } catch (err: unknown) {
      error(t('modelManager.notifications.downloadError'), String(err));
    } finally {
      setDownloading(false);
    }
  };

  const handleCancelDownload = async () => {
    setDownloading(false);
    setDownloadProgress(null);
    
    // Versuche die unvollständigen Dateien zu löschen
    // Der model_id ist leider nicht einfach verfügbar, aber der Cleanup passiert auch automatisch bei Errors
    info(t('modelManager.notifications.cancelInfo'), t('modelManager.notifications.cancelInfoDetail'));
  };

  // ──────────────────────────────────────────
  // Delete
  // ──────────────────────────────────────────
  const handleDeleteConfirm = async () => {
    if (!deleteTarget) return;
    try {
      await invoke('delete_model', { modelId: deleteTarget.id });
      success(t('modelManager.notifications.deleteSuccess'), t('modelManager.notifications.deleteSuccessDetail').replace('{name}', deleteTarget.name));
      await loadModels();
    } catch (err: unknown) {
      error(t('modelManager.notifications.deleteFailed'), err instanceof Error ? err.message : String(err));
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
    setDownloadProgress(null);
  };

  // ──────────────────────────────────────────
  // Render
  // ──────────────────────────────────────────
  return (
    <div className="space-y-6">

      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">{t('modelManager.title')}</h1>
          <p className="text-gray-400 mt-1">{t('modelManager.subtitle')}</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={loadModels}
            className="p-2 rounded-xl bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white border border-white/10 transition-all"
            title={t('common.refresh')}
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowImportModal(true)}
            className={`flex items-center gap-2 px-4 py-2 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-xl text-white text-sm font-medium hover:opacity-90 transition-all`}
          >
            <Upload className="w-4 h-4" />
            {t('common.import')}
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
            <h3 className="text-white font-semibold text-lg">{t('modelManager.noModels')}</h3>
            <p className="text-gray-400 text-sm mt-1">{t('modelManager.noModelsHint')}</p>
          </div>
          <button
            onClick={() => setShowImportModal(true)}
            className={`inline-flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-xl text-white text-sm font-medium hover:opacity-90 transition-all`}
          >
            <Upload className="w-4 h-4" />
            {t('common.import')}
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
              <h2 className="text-xl font-bold text-white">{t('common.import')}</h2>
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
                  {mode === 'local' ? t('modelManager.local') : t('modelManager.huggingface')}
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
                  downloadProgress={downloadProgress}
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
  const { t } = useLanguage();
  // Determine the identifier to use for plugin detection
  const detectionKey = model.source === 'huggingface' && model.source_path
    ? model.source_path
    : model.name;

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-5 hover:bg-white/[0.07] transition-all group flex flex-col gap-4">
      {/* Top row */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3 min-w-0">
          <div className={`p-2 rounded-xl ${model.source === 'huggingface' || model.id?.startsWith('canvas_') ? `bg-gradient-to-r ${gradientClass}` : `bg-gradient-to-r ${gradientClass}`} flex-shrink-0`}>
            {model.id?.startsWith('canvas_')
              ? <span style={{ fontSize: 14, color: 'white' }}>◈</span>
              : model.source === 'huggingface'
                ? <Cloud className="w-4 h-4 text-white" />
                : <HardDrive className="w-4 h-4 text-white" />
            }
          </div>
          <div className="min-w-0">
            <h3 className="font-semibold text-white truncate" title={model.name}>{model.name}</h3>
            <span className="text-xs text-gray-500">
              {model.id?.startsWith('canvas_')
                ? t('modelManager.card.sourceSynapse')
                : model.source === 'huggingface'
                  ? t('modelManager.card.sourceHuggingFace')
                  : t('modelManager.card.sourceLocal')}
            </span>
          </div>
        </div>
        <button
          onClick={onDelete}
          className="p-1.5 rounded-lg text-gray-600 hover:text-red-400 hover:bg-red-500/10 opacity-0 group-hover:opacity-100 transition-all flex-shrink-0"
          title={t('modelManager.card.deleteTooltip')}
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
          <span className="flex items-center gap-1.5">
            <FileBox className="w-3.5 h-3.5" />
            {t('modelManager.card.filesLabel').replace('{count}', String(model.file_count))}
          </span>
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
  const { t } = useLanguage();
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
              <p className="text-white font-medium">{t('modelManager.importModal.local.folderSelectedTitle')}</p>
              <p className="text-gray-400 text-sm mt-0.5 break-all">{selectedPath}</p>
            </div>
            {dirInfo && (
              <p className="text-gray-500 text-sm">
                {t('modelManager.importModal.local.folderFilesSummary')
                  .replace('{files}', String(dirInfo.files))
                  .replace('{size}', formatBytes(dirInfo.size))}
              </p>
            )}
            {!isValidModel && (
              <div className="inline-flex items-center gap-2 text-amber-400 text-sm">
                <AlertCircle className="w-4 h-4" />
                {t('modelManager.importModal.local.noModelFilesWarning')}
              </div>
            )}
            <button onClick={onReset} className="text-sm text-gray-400 hover:text-white underline transition-colors">
              {t('modelManager.importModal.local.changeFolderLink')}
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-white/5 border border-white/10">
              <Upload className="w-7 h-7 text-gray-400" />
            </div>
            <div>
              <p className="text-white font-medium">{isDragging ? t('modelManager.importModal.local.dropzoneDragging') : t('modelManager.importModal.local.dropzoneIdle')}</p>
              <p className="text-gray-500 text-sm mt-1">{t('modelManager.importModal.local.dropzoneSubtitle')}</p>
            </div>
            <button
              onClick={onBrowse}
              className="inline-flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/15 rounded-xl text-white text-sm transition-all"
            >
              <FolderOpen className="w-4 h-4" />
              {t('modelManager.importModal.local.browseButton')}
            </button>
          </div>
        )}
      </div>

      {/* Name Input */}
      {selectedPath && (
        <div className="space-y-1.5">
          <label className="block text-sm font-medium text-gray-300">{t('modelManager.importModal.local.nameLabel')}</label>
          <input
            type="text"
            value={modelName}
            onChange={(e) => onNameChange(e.target.value)}
            placeholder={t('modelManager.importModal.local.namePlaceholder')}
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
          {importing ? t('modelManager.importModal.local.importingButton') : t('modelManager.importModal.local.importButton')}
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
  downloadProgress: {
    status: string;
    currentFile: string;
    currentFileIndex: number;
    totalFiles: number;
    downloadedBytes: number;
    totalBytes: number;
    progressPercent: number;
    speedMbs: number;
    elapsedSecs: number;
    etaSecs: number;
    message: string;
  } | null;
  onQueryChange: (v: string) => void;
  onSelect: (m: HuggingFaceModel) => void;
  onNameChange: (v: string) => void;
  onDownload: () => void;
  onCancel: () => void;
  gradientClass: string;
}

function HuggingFaceImportPanel({
  query, results, searching, selected, localName, downloading, downloadProgress,
  onQueryChange, onSelect, onNameChange, onDownload, onCancel, gradientClass,
}: HuggingFaceImportPanelProps) {
  const { t } = useLanguage();
  // Formatiere Zeit von Sekunden zu "mm:ss" oder "h:mm:ss"
  const formatTime = (seconds: number): string => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  };

  return (
    <div className="space-y-5">
      {/* Search */}
      <div className="space-y-1.5">
        <label className="block text-sm font-medium text-gray-300">{t('common.search')}</label>
        <div className="relative">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder={t('modelManager.importModal.hf.searchPlaceholder')}
            disabled={downloading}
            className="w-full pl-10 pr-10 py-3 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-white/30 transition-all disabled:opacity-50"
          />
          {searching && <Loader2 className="absolute right-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 animate-spin" />}
        </div>
        <p className="text-gray-600 text-xs">{t('modelManager.importModal.hf.searchHint')}</p>
      </div>

      {/* Results */}
      {results.length > 0 && !downloading && (
        <div className="space-y-2">
          <p className="text-gray-500 text-xs">{t('modelManager.importModal.hf.resultsCount').replace('{count}', results.length.toString())}</p>
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
                    {m.likes ? (
                      <span className="inline-flex items-center gap-1">
                        <Heart className="w-3.5 h-3.5" />
                        {formatDownloads(m.likes)}
                      </span>
                    ) : null}
                  </div>
                </div>
                {selected?.id === m.id && <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0 ml-2" />}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Selected model details & Download */}
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
                {t('modelManager.importModal.hf.viewOnHF')} <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>

          {/* Plugin Preview */}
          <div className="flex items-center gap-2">
            <span className="text-gray-500 text-xs">{t('modelManager.card.pluginLabel')}</span>
            <PluginBadge modelNameOrPath={selected.id} />
          </div>

          {/* Download Progress Display */}
          {downloading && downloadProgress ? (
            <div className="space-y-3 p-3 rounded-xl bg-black/30 border border-white/5">
              {/* Status & Percentage */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 text-white animate-spin" />
                  <span className="text-white text-sm font-medium">{downloadProgress.progressPercent}%</span>
                </div>
                <span className="text-xs text-gray-400">{formatBytes(downloadProgress.downloadedBytes)} / {formatBytes(downloadProgress.totalBytes)}</span>
              </div>

              {/* Progress Bar */}
              <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                <div
                  className={`h-full bg-gradient-to-r ${gradientClass} transition-all duration-300`}
                  style={{ width: `${downloadProgress.progressPercent}%` }}
                />
              </div>

              {/* File Info */}
              <div className="flex items-center justify-between">
                <div className="min-w-0 flex-1">
                  <p className="text-white text-xs truncate" title={downloadProgress.currentFile}>
                    {downloadProgress.currentFile || t('modelManager.importModal.hf.preparingFile')}
                  </p>
                  <p className="text-gray-500 text-xs">
                    {t('modelManager.importModal.hf.fileProgress').replace('{current}', downloadProgress.currentFileIndex.toString()).replace('{total}', downloadProgress.totalFiles.toString())}
                  </p>
                </div>
              </div>

              {/* Speed & ETA */}
              <div className="grid grid-cols-3 gap-2">
                <div className="text-center">
                  <p className="text-gray-500 text-xs">{t('modelManager.importModal.hf.speedLabel')}</p>
                  <p className="text-white text-sm font-medium">{downloadProgress.speedMbs.toFixed(1)} MB/s</p>
                </div>
                <div className="text-center">
                  <p className="text-gray-500 text-xs">{t('modelManager.importModal.hf.elapsedLabel')}</p>
                  <p className="text-white text-sm font-medium">{formatTime(downloadProgress.elapsedSecs)}</p>
                </div>
                <div className="text-center">
                  <p className="text-gray-500 text-xs">{t('modelManager.importModal.hf.remainingLabel')}</p>
                  <p className="text-white text-sm font-medium">{formatTime(downloadProgress.etaSecs)}</p>
                </div>
              </div>

              {/* Cancel Button */}
              <button
                onClick={onCancel}
                className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg bg-white/5 hover:bg-red-500/10 border border-white/10 hover:border-red-500/30 text-gray-400 hover:text-red-400 text-sm transition-all"
              >
                <X className="w-4 h-4" /> {t('common.cancel')}
              </button>
            </div>
          ) : (
            <>
              {/* Local name input */}
              <div className="space-y-1.5">
                <label className="block text-sm font-medium text-gray-300">{t('modelManager.importModal.hf.localNameLabel')}</label>
                <input
                  type="text"
                  value={localName}
                  onChange={(e) => onNameChange(e.target.value)}
                  placeholder={t('modelManager.importModal.hf.localNamePlaceholder')}
                  className="w-full px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-white/30 transition-all"
                />
              </div>

              {/* Download button */}
              <button
                onClick={onDownload}
                disabled={!localName.trim()}
                className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl text-white text-sm font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed ${
                  !localName.trim() ? 'bg-white/5' : `bg-gradient-to-r ${gradientClass} hover:opacity-90`
                }`}
              >
                <Download className="w-4 h-4" /> {t('modelManager.hfDownload')}
              </button>

              <p className="text-xs text-gray-600 text-center">
                {t('modelManager.importModal.hf.downloadNote')}
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
