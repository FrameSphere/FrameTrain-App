import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { 
  GitBranch,
  Trash2,
  Edit2,
  Calendar,
  HardDrive,
  Cpu,
  FileBox,
  Loader2,
  X,
  Check,
  RefreshCw,
  FolderOpen,
  Clock,
  Star,
  TrendingUp,
  Download,
  Upload,
  FolderDown
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';

// ============ Types ============

interface ModelWithVersions {
  id: string;
  name: string;
  root_path: string;
  version_count: number;
  total_size: number;
  model_type: string | null;
  last_updated: string;
}

interface ModelVersion {
  id: string;
  model_id: string;
  version_name: string;
  version_number: number;
  path: string;
  size_bytes: number;
  file_count: number;
  created_at: string;
  is_root: boolean;
  parent_version_id: string | null;
  training_metrics: TrainingMetrics | null;
}

interface TrainingMetrics {
  final_train_loss: number;
  final_val_loss: number | null;
  total_epochs: number;
  total_steps: number;
  best_epoch: number | null;
  training_duration_seconds: number | null;
}

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

function formatDuration(seconds: number | null): string {
  if (!seconds) return '-';
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

// ============ Versions Modal ============

interface VersionsModalProps {
  model: ModelWithVersions;
  versions: ModelVersion[];
  onClose: () => void;
  onDelete: (versionId: string) => Promise<void>;
  onRename: (versionId: string, newName: string) => Promise<void>;
  onExport: (versionId: string, versionName: string) => Promise<void>;
  onRefresh: () => Promise<void>;
  gradient: string;
  primaryColor: string;
}

function VersionsModal({ 
  model, 
  versions, 
  onClose, 
  onDelete, 
  onRename,
  onExport,
  onRefresh,
  gradient,
  primaryColor 
}: VersionsModalProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [loading, setLoading] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);
  const [exportingVersion, setExportingVersion] = useState<ModelVersion | null>(null);

  const handleStartEdit = (version: ModelVersion) => {
    setEditingId(version.id);
    setEditName(version.version_name);
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditName('');
  };

  const handleSaveEdit = async (versionId: string) => {
    if (!editName.trim()) return;
    setLoading(true);
    try {
      await onRename(versionId, editName.trim());
      setEditingId(null);
      setEditName('');
    } catch (err) {
      // Error is handled by parent
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (versionId: string, versionName: string) => {
    if (!confirm(`Möchtest du Version "${versionName}" wirklich löschen?`)) return;
    setLoading(true);
    try {
      await onDelete(versionId);
    } catch (err) {
      // Error is handled by parent
    } finally {
      setLoading(false);
    }
  };

  const handleExportClick = (version: ModelVersion) => {
    setExportingVersion(version);
    setShowExportModal(true);
  };

  const handleExportLocal = async () => {
    if (!exportingVersion) return;
    setLoading(true);
    try {
      await onExport(exportingVersion.id, exportingVersion.version_name);
      setShowExportModal(false);
      setExportingVersion(null);
    } catch (err) {
      // Error is handled by parent
    } finally {
      setLoading(false);
    }
  };

  // Sort: root first, then by version number desc
  const sortedVersions = [...versions].sort((a, b) => {
    if (a.is_root) return -1;
    if (b.is_root) return 1;
    return b.version_number - a.version_number;
  });

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-4xl max-h-[85vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-white/10">
          <div className="flex items-center gap-3">
            <GitBranch className="w-6 h-6 text-purple-400" />
            <div>
              <h2 className="text-xl font-bold text-white">{model.name}</h2>
              <p className="text-sm text-gray-400">{versions.length} Version{versions.length !== 1 ? 'en' : ''}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={onRefresh}
              disabled={loading}
              className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-all disabled:opacity-50"
              title="Aktualisieren"
            >
              <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-all"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Versions List */}
        <div className="p-6 overflow-y-auto max-h-[65vh]">
          <div className="space-y-3">
            {sortedVersions.map((version) => (
              <div
                key={version.id}
                className={`bg-white/5 rounded-xl border p-4 transition-all ${
                  version.is_root 
                    ? 'border-yellow-500/30 bg-yellow-500/5' 
                    : 'border-white/10 hover:bg-white/[0.07]'
                }`}
              >
                <div className="flex items-start justify-between gap-4">
                  {/* Version Info */}
                  <div className="flex-1 min-w-0">
                    {/* Name / Edit */}
                    {editingId === version.id ? (
                      <div className="flex items-center gap-2 mb-2">
                        <input
                          type="text"
                          value={editName}
                          onChange={(e) => setEditName(e.target.value)}
                          className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2"
                          style={{ '--tw-ring-color': primaryColor } as React.CSSProperties}
                          autoFocus
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleSaveEdit(version.id);
                            if (e.key === 'Escape') handleCancelEdit();
                          }}
                        />
                        <button
                          onClick={() => handleSaveEdit(version.id)}
                          disabled={loading}
                          className="p-2 rounded-lg bg-green-500/20 hover:bg-green-500/30 text-green-400 transition-all disabled:opacity-50"
                        >
                          <Check className="w-4 h-4" />
                        </button>
                        <button
                          onClick={handleCancelEdit}
                          disabled={loading}
                          className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 transition-all disabled:opacity-50"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 mb-2">
                        <h3 className="font-semibold text-white">{version.version_name}</h3>
                        {version.is_root && (
                          <span className="px-2 py-0.5 bg-yellow-500/20 text-yellow-400 text-xs rounded-full flex items-center gap-1">
                            <Star className="w-3 h-3" />
                            Original
                          </span>
                        )}
                      </div>
                    )}

                    {/* Stats Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                      <div className="flex items-center gap-2 text-gray-400">
                        <FileBox className="w-4 h-4" />
                        <span>{version.file_count} Dateien</span>
                      </div>
                      <div className="flex items-center gap-2 text-gray-400">
                        <HardDrive className="w-4 h-4" />
                        <span>{formatBytes(version.size_bytes)}</span>
                      </div>
                      <div className="flex items-center gap-2 text-gray-400">
                        <Calendar className="w-4 h-4" />
                        <span className="text-xs">{formatDate(version.created_at)}</span>
                      </div>
                      {!version.is_root && (
                        <div className="flex items-center gap-2 text-gray-400">
                          <GitBranch className="w-4 h-4" />
                          <span className="text-xs">v{version.version_number}</span>
                        </div>
                      )}
                    </div>

                    {/* Training Metrics */}
                    {version.training_metrics && (
                      <div className="mt-3 pt-3 border-t border-white/10">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                          <div>
                            <div className="text-gray-500 mb-1">Train Loss</div>
                            <div className="text-white font-medium flex items-center gap-1">
                              <TrendingUp className="w-3 h-3 text-blue-400" />
                              {version.training_metrics.final_train_loss.toFixed(4)}
                            </div>
                          </div>
                          {version.training_metrics.final_val_loss !== null && (
                            <div>
                              <div className="text-gray-500 mb-1">Val Loss</div>
                              <div className="text-white font-medium">
                                {version.training_metrics.final_val_loss.toFixed(4)}
                              </div>
                            </div>
                          )}
                          <div>
                            <div className="text-gray-500 mb-1">Epochen</div>
                            <div className="text-white font-medium">
                              {version.training_metrics.total_epochs}
                            </div>
                          </div>
                          <div>
                            <div className="text-gray-500 mb-1">Dauer</div>
                            <div className="text-white font-medium flex items-center gap-1">
                              <Clock className="w-3 h-3 text-purple-400" />
                              {formatDuration(version.training_metrics.training_duration_seconds)}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Path */}
                    <div className="mt-2 text-xs text-gray-600 truncate" title={version.path}>
                      {version.path}
                    </div>
                  </div>

                  {/* Actions */}
                  {editingId !== version.id && (
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleExportClick(version)}
                        disabled={loading}
                        className="p-2 rounded-lg text-gray-500 hover:text-green-400 hover:bg-green-500/10 transition-all disabled:opacity-50"
                        title="Exportieren"
                      >
                        <FolderDown className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleStartEdit(version)}
                        disabled={loading}
                        className="p-2 rounded-lg text-gray-500 hover:text-blue-400 hover:bg-blue-500/10 transition-all disabled:opacity-50"
                        title="Umbenennen"
                      >
                        <Edit2 className="w-4 h-4" />
                      </button>
                      {!version.is_root && (
                        <button
                          onClick={() => handleDelete(version.id, version.version_name)}
                          disabled={loading}
                          className="p-2 rounded-lg text-gray-500 hover:text-red-400 hover:bg-red-500/10 transition-all disabled:opacity-50"
                          title="Löschen"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-white/10">
          <button
            onClick={onClose}
            className={`w-full py-3 bg-gradient-to-r ${gradient} rounded-lg text-white font-medium hover:opacity-90 transition-all`}
          >
            Schließen
          </button>
        </div>
      </div>

      {/* Export Modal */}
      {showExportModal && exportingVersion && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-[60] flex items-center justify-center p-4">
          <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-md">
            {/* Export Modal Header */}
            <div className="flex items-center justify-between p-6 border-b border-white/10">
              <div className="flex items-center gap-3">
                <FolderDown className="w-6 h-6 text-green-400" />
                <div>
                  <h3 className="text-lg font-bold text-white">Version exportieren</h3>
                  <p className="text-sm text-gray-400">{exportingVersion.version_name}</p>
                </div>
              </div>
              <button
                onClick={() => {
                  setShowExportModal(false);
                  setExportingVersion(null);
                }}
                disabled={loading}
                className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-all disabled:opacity-50"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Export Options */}
            <div className="p-6 space-y-4">
              <p className="text-gray-400 text-sm">
                Wähle eine Export-Option für diese Version:
              </p>

              {/* Local Download */}
              <button
                onClick={handleExportLocal}
                disabled={loading}
                className="w-full flex items-center gap-4 p-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl transition-all disabled:opacity-50 group"
              >
                <div className="p-3 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg group-hover:scale-110 transition-transform">
                  <Download className="w-6 h-6 text-white" />
                </div>
                <div className="flex-1 text-left">
                  <div className="font-semibold text-white">Lokal herunterladen</div>
                  <div className="text-sm text-gray-400">Exportiere in den Downloads-Ordner</div>
                </div>
                {loading && <Loader2 className="w-5 h-5 text-gray-400 animate-spin" />}
              </button>

              {/* SphereNet Upload - Disabled */}
              <button
                disabled
                className="w-full flex items-center gap-4 p-4 bg-white/5 border border-white/10 rounded-xl opacity-50 cursor-not-allowed"
              >
                <div className="p-3 bg-gradient-to-r from-purple-500 to-blue-500 rounded-lg">
                  <Upload className="w-6 h-6 text-white" />
                </div>
                <div className="flex-1 text-left">
                  <div className="font-semibold text-white flex items-center gap-2">
                    Auf SphereNet hochladen
                    <span className="px-2 py-0.5 bg-amber-500/20 text-amber-400 text-xs rounded-full">
                      Bald verfügbar
                    </span>
                  </div>
                  <div className="text-sm text-gray-400">Teile dein Modell mit der Community</div>
                </div>
              </button>
            </div>

            {/* Cancel Button */}
            <div className="p-6 border-t border-white/10">
              <button
                onClick={() => {
                  setShowExportModal(false);
                  setExportingVersion(null);
                }}
                disabled={loading}
                className="w-full py-2.5 bg-white/5 hover:bg-white/10 rounded-lg text-white transition-all disabled:opacity-50"
              >
                Abbrechen
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ============ Main Component ============

export default function VersionManager() {
  const { currentTheme } = useTheme();
  const { success, error, warning } = useNotification();

  const [models, setModels] = useState<ModelWithVersions[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState<ModelWithVersions | null>(null);
  const [modelVersions, setModelVersions] = useState<ModelVersion[]>([]);
  const [showVersionsModal, setShowVersionsModal] = useState(false);

  useEffect(() => {
    loadModels();
  }, []);

  // Auto-refresh when training completes
  useEffect(() => {
    // Listen for training complete events
    const unlistenComplete = listen('training-complete', () => {
      console.log('[VersionManager] Training complete detected, refreshing...');
      loadModels();
    });

    const unlistenFinished = listen('training-finished', () => {
      console.log('[VersionManager] Training finished detected, refreshing...');
      loadModels();
    });
    
    return () => {
      unlistenComplete.then(f => f());
      unlistenFinished.then(f => f());
    };
  }, []);


  const loadModels = async () => {
    try {
      setLoading(true);
      const modelList = await invoke<ModelWithVersions[]>('list_models_with_versions');
      setModels(modelList);
    } catch (err: any) {
      console.error('Error loading models:', err);
      error('Fehler beim Laden', String(err));
    } finally {
      setLoading(false);
    }
  };

  const loadVersions = async (modelId: string) => {
    try {
      const versions = await invoke<ModelVersion[]>('list_model_versions', { modelId });
      setModelVersions(versions);
    } catch (err: any) {
      console.error('Error loading versions:', err);
      error('Fehler beim Laden der Versionen', String(err));
      throw err;
    }
  };

  const handleShowVersions = async (model: ModelWithVersions) => {
    setSelectedModel(model);
    try {
      await loadVersions(model.id);
      setShowVersionsModal(true);
    } catch (err) {
      // Error already handled
    }
  };

  const handleCloseVersions = () => {
    setShowVersionsModal(false);
    setSelectedModel(null);
    setModelVersions([]);
  };

  const handleDeleteVersion = async (versionId: string) => {
    try {
      await invoke('delete_model_version', { versionId });
      success('Version gelöscht', 'Die Version wurde erfolgreich entfernt.');
      if (selectedModel) {
        await loadVersions(selectedModel.id);
        await loadModels(); // Refresh model list to update counts
      }
    } catch (err: any) {
      error('Löschen fehlgeschlagen', String(err));
      throw err;
    }
  };

  const handleRenameVersion = async (versionId: string, newName: string) => {
    try {
      await invoke('rename_model_version', { versionId, newName });
      success('Version umbenannt', 'Der Name wurde erfolgreich geändert.');
      if (selectedModel) {
        await loadVersions(selectedModel.id);
      }
    } catch (err: any) {
      error('Umbenennen fehlgeschlagen', String(err));
      throw err;
    }
  };

  const handleExportVersion = async (versionId: string, versionName: string) => {
    try {
      const exportPath = await invoke<string>('export_model_version', { versionId });
      success('Export erfolgreich', `Version "${versionName}" wurde nach ${exportPath} exportiert.`);
    } catch (err: any) {
      error('Export fehlgeschlagen', String(err));
      throw err;
    }
  };

  const handleRefreshVersions = async () => {
    if (selectedModel) {
      try {
        await loadVersions(selectedModel.id);
        await loadModels();
      } catch (err) {
        // Error already handled
      }
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Versionen</h1>
          <p className="text-gray-400 mt-1">Verwalte Modell-Versionen und Training-Verläufe</p>
        </div>
        <button
          onClick={loadModels}
          className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-all"
          title="Aktualisieren"
        >
          <RefreshCw className="w-5 h-5" />
        </button>
      </div>

      {/* Models List */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 text-gray-400 animate-spin" />
        </div>
      ) : models.length === 0 ? (
        <div className="bg-white/5 rounded-2xl border border-white/10 p-12 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-white/5 mb-4">
            <GitBranch className="w-8 h-8 text-gray-400" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">Keine Modelle vorhanden</h3>
          <p className="text-gray-400">Füge zuerst Modelle auf der Modelle-Seite hinzu.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {models.map((model) => (
            <div
              key={model.id}
              className="bg-white/5 rounded-xl border border-white/10 p-5 hover:bg-white/[0.07] transition-all"
            >
              {/* Model Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  <div className={`p-2 rounded-lg bg-gradient-to-r ${currentTheme.colors.gradient} flex-shrink-0`}>
                    <FolderOpen className="w-5 h-5 text-white" />
                  </div>
                  <div className="min-w-0">
                    <h3 className="font-semibold text-white truncate" title={model.name}>
                      {model.name}
                    </h3>
                    <p className="text-xs text-gray-500 truncate" title={model.root_path}>
                      {model.root_path}
                    </p>
                  </div>
                </div>
              </div>

              {/* Model Stats */}
              <div className="grid grid-cols-2 gap-3 mb-4 text-sm">
                <div className="flex items-center gap-2 text-gray-400">
                  <GitBranch className="w-4 h-4" />
                  <span>{model.version_count} Version{model.version_count !== 1 ? 'en' : ''}</span>
                </div>
                <div className="flex items-center gap-2 text-gray-400">
                  <HardDrive className="w-4 h-4" />
                  <span>{formatBytes(model.total_size)}</span>
                </div>
                {model.model_type && (
                  <div className="flex items-center gap-2 text-gray-400">
                    <Cpu className="w-4 h-4" />
                    <span className="capitalize">{model.model_type}</span>
                  </div>
                )}
                <div className="flex items-center gap-2 text-gray-500 text-xs">
                  <Calendar className="w-3 h-3" />
                  <span>{formatDate(model.last_updated)}</span>
                </div>
              </div>

              {/* Versionen Button */}
              <button
                onClick={() => handleShowVersions(model)}
                className={`w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white font-medium hover:opacity-90 transition-all`}
              >
                <GitBranch className="w-4 h-4" />
                Versionen anzeigen
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Versions Modal */}
      {showVersionsModal && selectedModel && (
        <VersionsModal
          model={selectedModel}
          versions={modelVersions}
          onClose={handleCloseVersions}
          onDelete={handleDeleteVersion}
          onRename={handleRenameVersion}
          onExport={handleExportVersion}
          onRefresh={handleRefreshVersions}
          gradient={currentTheme.colors.gradient}
          primaryColor={currentTheme.colors.primary}
        />
      )}
    </div>
  );
}
