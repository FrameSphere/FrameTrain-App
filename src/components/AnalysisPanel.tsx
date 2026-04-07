import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  BarChart3,
  TrendingDown,
  TrendingUp,
  Activity,
  Zap,
  Target,
  Clock,
  Layers,
  RefreshCw,
  Loader2,
  ChevronDown,
  AlertCircle,
  Info,
  Download,
  GitBranch,
  CheckCircle,
  XCircle,
  FileText,
  Award,
  Brain,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';

// ============ Types ============

interface ModelInfo {
  id: string;
  name: string;
  source: string;
}

interface ModelWithVersionTree {
  id: string;
  name: string;
  versions: VersionTreeItem[];
}

interface VersionTreeItem {
  id: string;
  name: string;
  is_root: boolean;
  version_number: number;
}

interface TrainingMetrics {
  id: string;
  version_id: string;
  final_train_loss: number;
  final_val_loss: number | null;
  total_epochs: number;
  total_steps: number;
  best_epoch: number | null;
  training_duration_seconds: number | null;
  created_at: string;
}

interface LogEntry {
  epoch: number;
  step: number;
  train_loss: number;
  val_loss: number | null;
  learning_rate: number;
  timestamp: string;
}

interface VersionDetails {
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
}

// ============ Helper Functions ============

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDuration(seconds: number | null): string {
  if (!seconds) return '-';
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  } else {
    return `${secs}s`;
  }
}

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString('de-DE', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

// ============ Chart Components ============

interface LineChartProps {
  data: { x: number; y: number }[];
  label: string;
  color: string;
  yLabel: string;
  xLabel: string;
}

function LineChart({ data, label, color, yLabel, xLabel }: LineChartProps) {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        <div className="text-center">
          <BarChart3 className="w-12 h-12 mx-auto mb-2 opacity-50" />
          <p>Keine Daten verfügbar</p>
        </div>
      </div>
    );
  }

  const maxY = Math.max(...data.map(d => d.y));
  const minY = Math.min(...data.map(d => d.y));
  const rangeY = maxY - minY || 1;
  const maxX = Math.max(...data.map(d => d.x));
  const minX = Math.min(...data.map(d => d.x));
  const rangeX = maxX - minX || 1;

  const getX = (x: number) => ((x - minX) / rangeX) * 100;
  const getY = (y: number) => 100 - (((y - minY) / rangeY) * 80 + 10);

  const points = data.map(d => `${getX(d.x)},${getY(d.y)}`).join(' ');

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-gray-300">{label}</h4>
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <div className="w-3 h-0.5" style={{ backgroundColor: color }} />
          <span>{yLabel}</span>
        </div>
      </div>
      
      <div className="bg-white/5 rounded-xl p-4 border border-white/10">
        <svg viewBox="0 0 100 100" className="w-full h-48" preserveAspectRatio="none">
          {/* Grid lines */}
          <line x1="0" y1="10" x2="100" y2="10" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" />
          <line x1="0" y1="32.5" x2="100" y2="32.5" stroke="rgba(255,255,255,0.05)" strokeWidth="0.5" />
          <line x1="0" y1="55" x2="100" y2="55" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" />
          <line x1="0" y1="77.5" x2="100" y2="77.5" stroke="rgba(255,255,255,0.05)" strokeWidth="0.5" />
          <line x1="0" y1="90" x2="100" y2="90" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" />

          {/* Data line */}
          <polyline
            points={points}
            fill="none"
            stroke={color}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Data points */}
          {data.map((d, i) => (
            <circle
              key={i}
              cx={getX(d.x)}
              cy={getY(d.y)}
              r="1.5"
              fill={color}
            />
          ))}
        </svg>
        
        <div className="flex justify-between text-xs text-gray-500 mt-2">
          <span>{xLabel}: {minX.toFixed(0)}</span>
          <span>{xLabel}: {maxX.toFixed(0)}</span>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-3 text-xs">
        <div className="text-center p-2 bg-white/5 rounded-lg">
          <div className="text-gray-400">Min</div>
          <div className="text-white font-medium">{minY.toFixed(4)}</div>
        </div>
        <div className="text-center p-2 bg-white/5 rounded-lg">
          <div className="text-gray-400">Max</div>
          <div className="text-white font-medium">{maxY.toFixed(4)}</div>
        </div>
        <div className="text-center p-2 bg-white/5 rounded-lg">
          <div className="text-gray-400">Final</div>
          <div className="text-white font-medium">{data[data.length - 1]?.y.toFixed(4) || '-'}</div>
        </div>
      </div>
    </div>
  );
}

interface DualLineChartProps {
  trainData: { x: number; y: number }[];
  valData: { x: number; y: number }[];
  label: string;
  trainColor: string;
  valColor: string;
  yLabel: string;
  xLabel: string;
}

function DualLineChart({ trainData, valData, label, trainColor, valColor, yLabel, xLabel }: DualLineChartProps) {
  if (trainData.length === 0 && valData.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        <div className="text-center">
          <BarChart3 className="w-12 h-12 mx-auto mb-2 opacity-50" />
          <p>Keine Daten verfügbar</p>
        </div>
      </div>
    );
  }

  const allData = [...trainData, ...valData];
  const maxY = Math.max(...allData.map(d => d.y));
  const minY = Math.min(...allData.map(d => d.y));
  const rangeY = maxY - minY || 1;
  const maxX = Math.max(...allData.map(d => d.x));
  const minX = Math.min(...allData.map(d => d.x));
  const rangeX = maxX - minX || 1;

  const getX = (x: number) => ((x - minX) / rangeX) * 100;
  const getY = (y: number) => 100 - (((y - minY) / rangeY) * 80 + 10);

  const trainPoints = trainData.map(d => `${getX(d.x)},${getY(d.y)}`).join(' ');
  const valPoints = valData.map(d => `${getX(d.x)},${getY(d.y)}`).join(' ');

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-gray-300">{label}</h4>
        <div className="flex items-center gap-4 text-xs text-gray-400">
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5" style={{ backgroundColor: trainColor }} />
            <span>Train</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5" style={{ backgroundColor: valColor }} />
            <span>Validation</span>
          </div>
        </div>
      </div>
      
      <div className="bg-white/5 rounded-xl p-4 border border-white/10">
        <svg viewBox="0 0 100 100" className="w-full h-48" preserveAspectRatio="none">
          {/* Grid lines */}
          <line x1="0" y1="10" x2="100" y2="10" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" />
          <line x1="0" y1="32.5" x2="100" y2="32.5" stroke="rgba(255,255,255,0.05)" strokeWidth="0.5" />
          <line x1="0" y1="55" x2="100" y2="55" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" />
          <line x1="0" y1="77.5" x2="100" y2="77.5" stroke="rgba(255,255,255,0.05)" strokeWidth="0.5" />
          <line x1="0" y1="90" x2="100" y2="90" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" />

          {/* Train line */}
          {trainPoints && (
            <polyline
              points={trainPoints}
              fill="none"
              stroke={trainColor}
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}

          {/* Val line */}
          {valPoints && (
            <polyline
              points={valPoints}
              fill="none"
              stroke={valColor}
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}

          {/* Train points */}
          {trainData.map((d, i) => (
            <circle
              key={`train-${i}`}
              cx={getX(d.x)}
              cy={getY(d.y)}
              r="1.5"
              fill={trainColor}
            />
          ))}

          {/* Val points */}
          {valData.map((d, i) => (
            <circle
              key={`val-${i}`}
              cx={getX(d.x)}
              cy={getY(d.y)}
              r="1.5"
              fill={valColor}
            />
          ))}
        </svg>
        
        <div className="flex justify-between text-xs text-gray-500 mt-2">
          <span>{xLabel}: {minX.toFixed(0)}</span>
          <span>{xLabel}: {maxX.toFixed(0)}</span>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-2">
          <div className="text-xs text-gray-400 font-medium">Train Loss</div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="text-center p-2 bg-white/5 rounded-lg">
              <div className="text-gray-400">Final</div>
              <div className="text-white font-medium">{trainData[trainData.length - 1]?.y.toFixed(4) || '-'}</div>
            </div>
            <div className="text-center p-2 bg-white/5 rounded-lg">
              <div className="text-gray-400">Best</div>
              <div className="text-white font-medium">{Math.min(...trainData.map(d => d.y)).toFixed(4)}</div>
            </div>
          </div>
        </div>
        <div className="space-y-2">
          <div className="text-xs text-gray-400 font-medium">Val Loss</div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="text-center p-2 bg-white/5 rounded-lg">
              <div className="text-gray-400">Final</div>
              <div className="text-white font-medium">{valData[valData.length - 1]?.y.toFixed(4) || '-'}</div>
            </div>
            <div className="text-center p-2 bg-white/5 rounded-lg">
              <div className="text-gray-400">Best</div>
              <div className="text-white font-medium">{valData.length > 0 ? Math.min(...valData.map(d => d.y)).toFixed(4) : '-'}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============ Main Component ============

export default function AnalysisPanel() {
  const { currentTheme } = useTheme();
  const { success, error, info } = useNotification();

  // Data State
  const [modelsWithVersions, setModelsWithVersions] = useState<ModelWithVersionTree[]>([]);
  const [loading, setLoading] = useState(true);

  // Selection State
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);

  // Analysis Data State
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [versionDetails, setVersionDetails] = useState<VersionDetails | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);

  // ============ Load Data ============
  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    if (selectedModelId) {
      const model = modelsWithVersions.find(m => m.id === selectedModelId);

      if (model && model.versions.length > 0) {
        // ✨ Neue Logik: Neueste Version automatisch wählen
        const sortedVersions = [...model.versions].sort(
          (a, b) => b.version_number - a.version_number
        );

        const newestVersion = sortedVersions[0];
        setSelectedVersionId(newestVersion?.id || null);
      } else {
        setSelectedVersionId(null);
      }

    } else {
      setSelectedVersionId(null);
    }
  }, [selectedModelId, modelsWithVersions]);

  useEffect(() => {
    if (selectedVersionId) {
      loadAnalysisData();
    } else {
      setMetrics(null);
      setVersionDetails(null);
      setLogs([]);
    }
  }, [selectedVersionId]);

  const loadModels = async () => {
    try {
      setLoading(true);
      const modelList = await invoke<ModelWithVersionTree[]>('list_models_with_version_tree');
      setModelsWithVersions(modelList);
      
      if (modelList.length > 0) {
        setSelectedModelId(modelList[0].id);
      }
    } catch (err: any) {
      console.error('Error loading models:', err);
      error('Fehler beim Laden', String(err));
    } finally {
      setLoading(false);
    }
  };

  const loadAnalysisData = async () => {
    if (!selectedVersionId) return;

    try {
      setLoadingAnalysis(true);

      // Load metrics
      try {
        const metricsData = await invoke<TrainingMetrics>('get_training_metrics', {
          versionId: selectedVersionId,
        });
        setMetrics(metricsData);
      } catch (err) {
        console.log('No metrics found for this version');
        setMetrics(null);
      }

      // Load version details
      try {
        const details = await invoke<VersionDetails>('get_version_details', {
          versionId: selectedVersionId,
        });
        setVersionDetails(details);
      } catch (err) {
        console.error('Error loading version details:', err);
      }

      // Load logs (try to read from file system)
      try {
        const logsData = await invoke<LogEntry[]>('get_training_logs', {
          versionId: selectedVersionId,
        });
        setLogs(logsData);
      } catch (err) {
        console.log('No logs found for this version');
        setLogs([]);
      }

    } catch (err: any) {
      console.error('Error loading analysis data:', err);
      error('Fehler beim Laden der Analyse', String(err));
    } finally {
      setLoadingAnalysis(false);
    }
  };

  const handleExportReport = async () => {
    if (!metrics || !selectedVersionId) return;

    try {
      // Create report data
      const report = {
        version_id: selectedVersionId,
        version_name: versionDetails?.version_name,
        model_name: modelsWithVersions.find(m => m.id === selectedModelId)?.name,
        metrics: metrics,
        logs: logs,
        generated_at: new Date().toISOString(),
      };

      // Convert to JSON
      const jsonContent = JSON.stringify(report, null, 2);
      
      // Create blob and download
      const blob = new Blob([jsonContent], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `training_report_${versionDetails?.version_name || selectedVersionId}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      success('Report exportiert', 'Die Datei wurde in deinem Downloads-Ordner gespeichert.');
    } catch (err: any) {
      error('Export fehlgeschlagen', String(err));
    }
  };

  // ============ Render ============

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-8 h-8 text-gray-400 animate-spin" />
      </div>
    );
  }

  if (modelsWithVersions.length === 0) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-white">Analyse</h1>
          <p className="text-gray-400 mt-1">Analysiere deine Modell-Performance</p>
        </div>
        <div className="bg-white/5 rounded-2xl border border-white/10 p-12 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-white/5 mb-4">
            <Layers className="w-8 h-8 text-gray-400" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">Kein Modell vorhanden</h3>
          <p className="text-gray-400">
            Füge zuerst ein Modell hinzu und trainiere es, um die Analyse zu nutzen.
          </p>
        </div>
      </div>
    );
  }

  const selectedModel = modelsWithVersions.find(m => m.id === selectedModelId);
  const selectedVersion = selectedModel?.versions.find(v => v.id === selectedVersionId);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Analyse</h1>
          <p className="text-gray-400 mt-1">Detaillierte Trainings-Metriken und Performance-Analyse</p>
        </div>
        <div className="flex items-center gap-3">
          {/* Export Button */}
          {metrics && (
            <button
              onClick={handleExportReport}
              className={`flex items-center gap-2 px-4 py-2 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white font-medium hover:opacity-90 transition-all`}
            >
              <Download className="w-4 h-4" />
              Report exportieren
            </button>
          )}

          {/* Refresh Button */}
          <button
            onClick={loadAnalysisData}
            disabled={!selectedVersionId || loadingAnalysis}
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-all disabled:opacity-50"
            title="Aktualisieren"
          >
            <RefreshCw className={`w-5 h-5 ${loadingAnalysis ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Model & Version Selection */}
      <div className="bg-white/5 rounded-xl border border-white/10 p-6">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Layers className="w-5 h-5" />
          Modell & Version auswählen
        </h2>
        <div className="grid grid-cols-2 gap-4">
          {/* Model Selector */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">Root-Modell</label>
            <div className="relative">
              <select
                value={selectedModelId || ''}
                onChange={(e) => setSelectedModelId(e.target.value)}
                className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white appearance-none cursor-pointer focus:outline-none focus:ring-2 transition-all"
                style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
              >
                {modelsWithVersions.map((model) => (
                  <option key={model.id} value={model.id} className="bg-slate-800">
                    {model.name}
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
            </div>
          </div>

          {/* Version Selector */}
          <div>
            <label className="block text-sm text-gray-400 mb-2 flex items-center gap-2">
              <GitBranch className="w-4 h-4" />
              Modell-Version
            </label>
            <div className="relative">
              <select
                value={selectedVersionId || ''}
                onChange={(e) => setSelectedVersionId(e.target.value)}
                disabled={!selectedModel || selectedModel.versions.length === 0}
                className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white appearance-none cursor-pointer focus:outline-none focus:ring-2 disabled:opacity-50 transition-all"
                style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
              >
                {selectedModel && selectedModel.versions.length > 0 ? (
                  selectedModel.versions.map((version) => (
                    <option key={version.id} value={version.id} className="bg-slate-800">
                      {version.is_root ? '⭐ ' : ''}{version.name}
                      {version.is_root ? ' (Original)' : ` (v${version.version_number})`}
                    </option>
                  ))
                ) : (
                  <option value="" className="bg-slate-800">Keine Versionen verfügbar</option>
                )}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
            </div>
          </div>
        </div>
      </div>

      {/* Loading State */}
      {loadingAnalysis && (
        <div className="flex items-center justify-center py-12">
          <div className="text-center">
            <Loader2 className="w-12 h-12 text-purple-400 animate-spin mx-auto mb-4" />
            <p className="text-gray-400">Lade Analyse-Daten...</p>
          </div>
        </div>
      )}

      {/* No Data State */}
      {!loadingAnalysis && selectedVersionId && !metrics && (
        <div className="bg-white/5 rounded-2xl border border-white/10 p-12 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-amber-500/20 mb-4">
            <AlertCircle className="w-8 h-8 text-amber-400" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">Keine Trainings-Daten</h3>
          <p className="text-gray-400 mb-4">
            Diese Version wurde noch nicht trainiert oder die Metriken sind nicht verfügbar.
          </p>
          <p className="text-sm text-gray-500">
            Starte ein Training auf der Training-Seite, um Metriken zu sammeln.
          </p>
        </div>
      )}

      {/* Analysis Content */}
      {!loadingAnalysis && metrics && (
        <div className="space-y-6">
          {/* Overview Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Training Status */}
            <div className="bg-white/5 rounded-xl border border-white/10 p-5">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-gray-400">Status</h3>
                <CheckCircle className="w-5 h-5 text-green-400" />
              </div>
              <div className="text-2xl font-bold text-white mb-1">Abgeschlossen</div>
              <div className="text-xs text-gray-500">{formatDate(metrics.created_at)}</div>
            </div>

            {/* Final Loss */}
            <div className="bg-white/5 rounded-xl border border-white/10 p-5">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-gray-400">Final Loss</h3>
                <TrendingDown className="w-5 h-5 text-blue-400" />
              </div>
              <div className="text-2xl font-bold text-white mb-1">
                {metrics.final_train_loss.toFixed(4)}
              </div>
              <div className="text-xs text-gray-500">
                Train • Val: {metrics.final_val_loss?.toFixed(4) || 'N/A'}
              </div>
            </div>

            {/* Total Epochs */}
            <div className="bg-white/5 rounded-xl border border-white/10 p-5">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-gray-400">Epochen</h3>
                <Activity className="w-5 h-5 text-purple-400" />
              </div>
              <div className="text-2xl font-bold text-white mb-1">{metrics.total_epochs}</div>
              <div className="text-xs text-gray-500">
                {metrics.total_steps.toLocaleString()} Steps
              </div>
            </div>

            {/* Duration */}
            <div className="bg-white/5 rounded-xl border border-white/10 p-5">
              <div className="flex items-center justify-between mb-3">
              <div className="text-xs text-gray-500"><h3>Dauer</h3>
                <Clock className="w-5 h-5 text-yellow-400" />
                </div>
              </div>
              <div className="text-2xl font-bold text-white mb-1">
                {formatDuration(metrics.training_duration_seconds)}
              </div>
              <div className="text-xs text-gray-500">
                {metrics.best_epoch ? `Best: Epoch ${metrics.best_epoch}` : 'Training completed'}
              </div>
            </div>
          </div>

          {/* Version Info Card */}
          {versionDetails && (
            <div className="bg-white/5 rounded-xl border border-white/10 p-6">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Info className="w-5 h-5" />
                Versions-Informationen
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <div className="text-gray-400 mb-1">Version</div>
                  <div className="text-white font-medium">{versionDetails.version_name}</div>
                </div>
                <div>
                  <div className="text-gray-400 mb-1">Größe</div>
                  <div className="text-white font-medium">{formatBytes(versionDetails.size_bytes)}</div>
                </div>
                <div>
                  <div className="text-gray-400 mb-1">Dateien</div>
                  <div className="text-white font-medium">{versionDetails.file_count}</div>
                </div>
                <div>
                  <div className="text-gray-400 mb-1">Erstellt</div>
                  <div className="text-white font-medium">{formatDate(versionDetails.created_at)}</div>
                </div>
              </div>
              <div className="mt-4 p-3 bg-white/5 rounded-lg">
                <div className="text-xs text-gray-400 mb-1">Pfad</div>
                <div className="text-xs text-gray-300 font-mono truncate" title={versionDetails.path}>
                  {versionDetails.path}
                </div>
              </div>
            </div>
          )}

          {/* Loss Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Loss over Epochs */}
            {logs.length > 0 && (
              <div className="bg-white/5 rounded-xl border border-white/10 p-6">
                <DualLineChart
                  trainData={logs.map(l => ({ x: l.epoch, y: l.train_loss }))}
                  valData={logs.filter(l => l.val_loss !== null).map(l => ({ x: l.epoch, y: l.val_loss! }))}
                  label="Loss über Epochen"
                  trainColor="#3b82f6"
                  valColor="#10b981"
                  yLabel="Loss"
                  xLabel="Epoche"
                />
              </div>
            )}

            {/* Loss over Steps */}
            {logs.length > 0 && (
              <div className="bg-white/5 rounded-xl border border-white/10 p-6">
                <DualLineChart
                  trainData={logs.map(l => ({ x: l.step, y: l.train_loss }))}
                  valData={logs.filter(l => l.val_loss !== null).map(l => ({ x: l.step, y: l.val_loss! }))}
                  label="Loss über Steps"
                  trainColor="#8b5cf6"
                  valColor="#ec4899"
                  yLabel="Loss"
                  xLabel="Step"
                />
              </div>
            )}
          </div>

          {/* Learning Rate Schedule */}
          {logs.length > 0 && logs.some(l => l.learning_rate > 0) && (
            <div className="bg-white/5 rounded-xl border border-white/10 p-6">
              <LineChart
                data={logs.map(l => ({ x: l.step, y: l.learning_rate }))}
                label="Learning Rate Schedule"
                color="#f59e0b"
                yLabel="Learning Rate"
                xLabel="Step"
              />
            </div>
          )}

          {/* Performance Metrics */}
          <div className="bg-white/5 rounded-xl border border-white/10 p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Target className="w-5 h-5" />
              Performance Metriken
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {/* Training Loss Improvement */}
              {logs.length > 0 && (
                <div className="p-4 bg-white/5 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-sm text-gray-400">Loss Verbesserung</div>
                    <TrendingDown className="w-4 h-4 text-green-400" />
                  </div>
                  <div className="text-2xl font-bold text-white">
                    {(
                      ((logs[0].train_loss - logs[logs.length - 1].train_loss) / logs[0].train_loss) *
                      100
                    ).toFixed(1)}
                    %
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {logs[0].train_loss.toFixed(4)} → {logs[logs.length - 1].train_loss.toFixed(4)}
                  </div>
                </div>
              )}

              {/* Convergence Speed */}
              {logs.length > 0 && (
                <div className="p-4 bg-white/5 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-sm text-gray-400">Konvergenz</div>
                    <Zap className="w-4 h-4 text-yellow-400" />
                  </div>
                  <div className="text-2xl font-bold text-white">
                    {(() => {
                      // Find epoch where loss dropped below 50% of initial loss
                      const halfLoss = logs[0].train_loss / 2;
                      const convergenceEpoch = logs.find(l => l.train_loss < halfLoss);
                      return convergenceEpoch ? convergenceEpoch.epoch : logs.length;
                    })()}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">Epochen bis 50% Loss</div>
                </div>
              )}

              {/* Overfitting Indicator */}
              {logs.filter(l => l.val_loss !== null).length > 0 && (
                <div className="p-4 bg-white/5 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-sm text-gray-400">Overfitting</div>
                    <Activity className="w-4 h-4 text-blue-400" />
                  </div>
                  <div className="text-2xl font-bold text-white">
                    {(() => {
                      const finalTrain = logs[logs.length - 1].train_loss;
                      const finalVal = logs.filter(l => l.val_loss !== null).slice(-1)[0]?.val_loss;
                      if (!finalVal) return 'N/A';
                      const gap = ((finalVal - finalTrain) / finalTrain) * 100;
                      return gap > 0 ? `+${gap.toFixed(1)}%` : `${gap.toFixed(1)}%`;
                    })()}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">Val-Train Gap</div>
                </div>
              )}

              {/* Steps per Second */}
              {metrics.training_duration_seconds && metrics.total_steps && (
                <div className="p-4 bg-white/5 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-sm text-gray-400">Geschwindigkeit</div>
                    <Clock className="w-4 h-4 text-purple-400" />
                  </div>
                  <div className="text-2xl font-bold text-white">
                    {(metrics.total_steps / metrics.training_duration_seconds).toFixed(2)}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">Steps/Sekunde</div>
                </div>
              )}
            </div>
          </div>

          {/* Training Summary */}
          <div className="bg-white/5 rounded-xl border border-white/10 p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Brain className="w-5 h-5" />
              Training Zusammenfassung
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Left Column: Metrics */}
              <div className="space-y-3">
                <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                  <span className="text-gray-400">Final Train Loss</span>
                  <span className="text-white font-medium">{metrics.final_train_loss.toFixed(4)}</span>
                </div>
                {metrics.final_val_loss && (
                  <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                    <span className="text-gray-400">Final Val Loss</span>
                    <span className="text-white font-medium">{metrics.final_val_loss.toFixed(4)}</span>
                  </div>
                )}
                <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                  <span className="text-gray-400">Total Epochen</span>
                  <span className="text-white font-medium">{metrics.total_epochs}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                  <span className="text-gray-400">Total Steps</span>
                  <span className="text-white font-medium">{metrics.total_steps.toLocaleString()}</span>
                </div>
              </div>

              {/* Right Column: Status */}
              <div className="space-y-3">
                {metrics.best_epoch && (
                  <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                    <span className="text-gray-400">Best Epoch</span>
                    <span className="text-white font-medium">{metrics.best_epoch}</span>
                  </div>
                )}
                {metrics.training_duration_seconds && (
                  <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                    <span className="text-gray-400">Training Dauer</span>
                    <span className="text-white font-medium">
                      {formatDuration(metrics.training_duration_seconds)}
                    </span>
                  </div>
                )}
                <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                  <span className="text-gray-400">Training Status</span>
                  <span className="flex items-center gap-2 text-green-400 font-medium">
                    <CheckCircle className="w-4 h-4" />
                    Abgeschlossen
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                  <span className="text-gray-400">Version</span>
                  <span className="text-white font-medium">{versionDetails?.version_name || 'N/A'}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Recommendations */}
          <div className="bg-gradient-to-br from-purple-500/10 to-blue-500/10 rounded-xl border border-purple-500/20 p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Award className="w-5 h-5 text-purple-400" />
              Empfehlungen
            </h3>
            <div className="space-y-3">
              {/* Overfitting Check */}
              {(() => {
                if (logs.filter(l => l.val_loss !== null).length === 0) return null;
                const finalTrain = logs[logs.length - 1].train_loss;
                const finalVal = logs.filter(l => l.val_loss !== null).slice(-1)[0]?.val_loss;
                if (!finalVal) return null;
                const gap = ((finalVal - finalTrain) / finalTrain) * 100;
                
                if (gap > 20) {
                  return (
                    <div className="flex items-start gap-3 p-4 bg-amber-500/10 border border-amber-500/20 rounded-lg">
                      <AlertCircle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                      <div>
                        <div className="text-amber-400 font-medium mb-1">Overfitting erkannt</div>
                        <div className="text-sm text-gray-300">
                          Validation Loss ist {gap.toFixed(1)}% höher als Training Loss. 
                          Erwäge mehr Regularisierung (Dropout, Weight Decay) oder einen kleineren Learning Rate.
                        </div>
                      </div>
                    </div>
                  );
                }
              })()}

              {/* Loss Plateau Check */}
              {(() => {
                if (logs.length < 10) return null;
                const lastTen = logs.slice(-10);
                const lossVariance = lastTen.reduce((sum, l) => sum + Math.abs(l.train_loss - metrics.final_train_loss), 0) / 10;
                
                if (lossVariance < 0.001) {
                  return (
                    <div className="flex items-start gap-3 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                      <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                      <div>
                        <div className="text-blue-400 font-medium mb-1">Loss-Plateau erreicht</div>
                        <div className="text-sm text-gray-300">
                          Der Loss hat sich in den letzten Epochen stabilisiert. 
                          Das Modell hat wahrscheinlich konvergiert oder benötigt eine Learning Rate Anpassung.
                        </div>
                      </div>
                    </div>
                  );
                }
              })()}

              {/* Good Training */}
              {(() => {
                const finalTrain = logs[logs.length - 1]?.train_loss;
                const initialTrain = logs[0]?.train_loss;
                if (!finalTrain || !initialTrain) return null;
                
                const improvement = ((initialTrain - finalTrain) / initialTrain) * 100;
                const valLogs = logs.filter(l => l.val_loss !== null);
                const hasVal = valLogs.length > 0;
                const finalVal = valLogs.slice(-1)[0]?.val_loss;
                const gap = hasVal && finalVal ? ((finalVal - finalTrain) / finalTrain) * 100 : 0;
                
                if (improvement > 50 && (!hasVal || gap < 10)) {
                  return (
                    <div className="flex items-start gap-3 p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                      <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                      <div>
                        <div className="text-green-400 font-medium mb-1">Gutes Training</div>
                        <div className="text-sm text-gray-300">
                          Das Modell zeigt eine starke Verbesserung ({improvement.toFixed(1)}% Loss-Reduktion)
                          {hasVal && ' und keine Anzeichen von Overfitting'}. 
                          Das Training war erfolgreich!
                        </div>
                      </div>
                    </div>
                  );
                }
              })()}

              {/* No Recommendations */}
              {(() => {
                // Check if any recommendation was shown
                const finalTrain = logs[logs.length - 1]?.train_loss;
                const initialTrain = logs[0]?.train_loss;
                if (!finalTrain || !initialTrain) return null;
                
                const valLogs = logs.filter(l => l.val_loss !== null);
                const hasVal = valLogs.length > 0;
                const finalVal = valLogs.slice(-1)[0]?.val_loss;
                const gap = hasVal && finalVal ? ((finalVal - finalTrain) / finalTrain) * 100 : 0;
                const improvement = ((initialTrain - finalTrain) / initialTrain) * 100;
                
                // If no other recommendation was shown
                if (improvement <= 50 && gap <= 20 && gap > -5) {
                  return (
                    <div className="flex items-start gap-3 p-4 bg-white/5 border border-white/10 rounded-lg">
                      <Info className="w-5 h-5 text-gray-400 flex-shrink-0 mt-0.5" />
                      <div>
                        <div className="text-gray-300 font-medium mb-1">Standard Training</div>
                        <div className="text-sm text-gray-400">
                          Das Training verlief normal. Teste das Modell auf der Test-Seite, um die praktische Performance zu evaluieren.
                        </div>
                      </div>
                    </div>
                  );
                }
              })()}
            </div>
          </div>

          {/* Training Logs Table (collapsed by default) */}
          <details className="bg-white/5 rounded-xl border border-white/10">
            <summary className="p-6 cursor-pointer hover:bg-white/5 transition-colors">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <FileText className="w-5 h-5" />
                  Training Logs ({logs.length} Einträge)
                </h3>
                <ChevronDown className="w-5 h-5 text-gray-400" />
              </div>
            </summary>
            <div className="p-6 pt-0 max-h-96 overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-slate-900">
                  <tr className="text-left text-gray-400 border-b border-white/10">
                    <th className="pb-3 pr-4">Epoch</th>
                    <th className="pb-3 pr-4">Step</th>
                    <th className="pb-3 pr-4">Train Loss</th>
                    <th className="pb-3 pr-4">Val Loss</th>
                    <th className="pb-3 pr-4">LR</th>
                    <th className="pb-3">Zeit</th>
                  </tr>
                </thead>
                <tbody>
                  {logs.map((log, i) => (
                    <tr key={i} className="border-b border-white/5 hover:bg-white/5">
                      <td className="py-2 pr-4 text-white">{log.epoch}</td>
                      <td className="py-2 pr-4 text-gray-300">{log.step}</td>
                      <td className="py-2 pr-4 text-blue-400 font-medium">{log.train_loss.toFixed(4)}</td>
                      <td className="py-2 pr-4 text-green-400 font-medium">
                        {log.val_loss ? log.val_loss.toFixed(4) : '-'}
                      </td>
                      <td className="py-2 pr-4 text-yellow-400 font-mono text-xs">
                        {log.learning_rate.toExponential(2)}
                      </td>
                      <td className="py-2 text-gray-500 text-xs">
                        {new Date(log.timestamp).toLocaleTimeString('de-DE')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>

          {/* Test Results Section */}
          <TestResultsSection versionId={selectedVersionId} />
        </div>
      )}
    </div>
  );
}

// ============ Test Results Section Component ============

interface TestResultsSectionProps {
  versionId: string;
}

function TestResultsSection({ versionId }: TestResultsSectionProps) {
  const [testResults, setTestResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadTestResults();
  }, [versionId]);

  const loadTestResults = async () => {
    try {
      setLoading(true);
      const results = await invoke<any[]>('get_test_results_for_version', {
        versionId,
      });
      setTestResults(results);
    } catch (err) {
      console.error('Error loading test results:', err);
      setTestResults([]);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-white/5 rounded-xl border border-white/10 p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Target className="w-5 h-5" />
          Test-Ergebnisse
        </h3>
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-6 h-6 text-gray-400 animate-spin" />
        </div>
      </div>
    );
  }

  if (testResults.length === 0) {
    return (
      <div className="bg-white/5 rounded-xl border border-white/10 p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Target className="w-5 h-5" />
          Test-Ergebnisse
        </h3>
        <div className="text-center py-8">
          <Info className="w-12 h-12 text-gray-400 mx-auto mb-3 opacity-50" />
          <p className="text-gray-400">Noch keine Test-Ergebnisse vorhanden.</p>
          <p className="text-sm text-gray-500 mt-1">
            Teste das Modell auf der Test-Seite, um Ergebnisse zu sehen.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white/5 rounded-xl border border-white/10 p-6">
      <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <Target className="w-5 h-5" />
        Test-Ergebnisse ({testResults.length})
      </h3>

      <div className="space-y-4">
        {testResults.map((result, idx) => (
          <div
            key={idx}
            className="p-4 bg-white/5 rounded-lg border border-white/10"
          >
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-green-400">
                  {result.accuracy.toFixed(2)}%
                </div>
                <div className="text-xs text-gray-400 mt-1">Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-white">
                  {result.total_samples}
                </div>
                <div className="text-xs text-gray-400 mt-1">Samples</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-400">
                  {result.average_loss.toFixed(4)}
                </div>
                <div className="text-xs text-gray-400 mt-1">Avg Loss</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-400">
                  {(result.average_inference_time * 1000).toFixed(0)}ms
                </div>
                <div className="text-xs text-gray-400 mt-1">Inferenz</div>
              </div>
            </div>

            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span className="text-gray-300">
                    {result.correct_predictions} korrekt
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <XCircle className="w-4 h-4 text-red-400" />
                  <span className="text-gray-300">
                    {result.incorrect_predictions} falsch
                  </span>
                </div>
              </div>
              {result.metrics?.total_time && (
                <span className="text-gray-400">
                  Gesamt: {formatDuration(result.metrics.total_time)}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}