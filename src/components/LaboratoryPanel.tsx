import { useState, useEffect, useRef, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  FlaskConical,
  ChevronDown,
  Play,
  SkipForward,
  RefreshCw,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Loader2,
  Layers,
  Database,
  GitBranch,
  MessageSquare,
  Trash2,
  Download,
  BarChart3,
  Clock,
  ChevronRight,
  Zap,
  Eye,
  Tag,
  Image as ImageIcon,
  FileText,
  Hash,
  ThumbsUp,
  ThumbsDown,
  Minus,
  Info,
  X,
  Save,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';

// ============ Types ============

interface ModelInfo { id: string; name: string; source: string; }
interface ModelWithVersionTree { id: string; name: string; versions: VersionItem[]; }
interface VersionItem { id: string; name: string; is_root: boolean; version_number: number; }
interface DatasetInfo { id: string; name: string; model_id: string; status: string; file_count: number; size_bytes: number; }

interface LabSample {
  path: string;
  filename: string;
  sample_type: string;
  content: string | null;
  dataset_id: string;
  sample_index: number;
  total_samples: number;
}

interface LabelResult { label: string; score: number; }
interface BoundingBox { label: string; score: number; x: number; y: number; width: number; height: number; }
interface TextSpan { start: number; end: number; label: string; score: number; color: string; }

interface RenderedOutput {
  primary_label: string | null;
  confidence: number | null;
  labels: LabelResult[];
  bounding_boxes: BoundingBox[];
  highlighted_spans: TextSpan[];
  generated_text: string | null;
  key_values: [string, string][];
}

interface LabInferenceResult {
  sample_path: string;
  model_output_type: string;
  raw_output: any;
  rendered: RenderedOutput;
  inference_time_ms: number;
  error: string | null;
}

interface LabFeedback {
  rating: 'correct' | 'partial' | 'incorrect';
  comment: string;
  corrected_label: string | null;
}

interface LabSession {
  id: string;
  model_id: string;
  model_name: string;
  version_id: string;
  version_name: string;
  dataset_id: string;
  dataset_name: string;
  sample: LabSample;
  inference_result: LabInferenceResult;
  feedback: LabFeedback;
  created_at: string;
  added_to_dataset: boolean;
}

interface LabStats {
  total_sessions: number;
  correct: number;
  partial: number;
  incorrect: number;
  accuracy_rate: number;
  exported_samples: number;
}

// ============ Helpers ============

function sampleTypeIcon(type: string) {
  switch (type) {
    case 'image': return <ImageIcon className="w-4 h-4" />;
    case 'text': return <FileText className="w-4 h-4" />;
    case 'json': return <Hash className="w-4 h-4" />;
    case 'csv': return <BarChart3 className="w-4 h-4" />;
    default: return <FileText className="w-4 h-4" />;
  }
}

function confidenceColor(score: number): string {
  if (score >= 0.8) return 'text-green-400';
  if (score >= 0.5) return 'text-amber-400';
  return 'text-red-400';
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

// ============ Image Canvas – zeigt Bounding Boxes ============

function ImageCanvas({ samplePath, boxes }: { samplePath: string; boxes: BoundingBox[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });

  const draw = useCallback((img: HTMLImageElement) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    setImgSize({ w: img.naturalWidth, h: img.naturalHeight });

    ctx.drawImage(img, 0, 0);

    const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];
    boxes.forEach((box, i) => {
      const color = colors[i % colors.length];
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(box.x, box.y, box.width, box.height);

      const label = `${box.label} ${(box.score * 100).toFixed(0)}%`;
      ctx.font = 'bold 14px sans-serif';
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = color;
      ctx.fillRect(box.x, box.y - 22, tw + 8, 22);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, box.x + 4, box.y - 6);
    });
  }, [boxes]);

  useEffect(() => {
    const img = new Image();
    // Convert file path to tauri asset URL
    const url = samplePath.startsWith('http') ? samplePath : `asset://${samplePath}`;
    img.src = url;
    img.onload = () => draw(img);
    img.onerror = () => {
      // Try without asset protocol
      const img2 = new Image();
      img2.src = samplePath;
      img2.onload = () => draw(img2);
    };
  }, [samplePath, draw]);

  return (
    <div className="relative w-full overflow-hidden rounded-xl bg-black/30 flex items-center justify-center min-h-48">
      <canvas
        ref={canvasRef}
        className="max-w-full max-h-[400px] object-contain"
        style={{ imageRendering: 'crisp-edges' }}
      />
      {imgSize.w === 0 && (
        <div className="absolute inset-0 flex items-center justify-center text-gray-500">
          <ImageIcon className="w-12 h-12 opacity-30" />
        </div>
      )}
    </div>
  );
}

// ============ Text mit NER-Highlights ============

function HighlightedText({ text, spans }: { text: string; spans: TextSpan[] }) {
  if (!text || spans.length === 0) {
    return (
      <div className="p-4 bg-white/5 rounded-xl border border-white/10 text-gray-300 text-sm leading-relaxed whitespace-pre-wrap font-mono max-h-64 overflow-y-auto">
        {text}
      </div>
    );
  }

  const sorted = [...spans].sort((a, b) => a.start - b.start);
  const parts: JSX.Element[] = [];
  let cursor = 0;

  sorted.forEach((span, i) => {
    if (span.start > cursor) {
      parts.push(<span key={`t${i}`}>{text.slice(cursor, span.start)}</span>);
    }
    parts.push(
      <span
        key={`s${i}`}
        title={`${span.label}: ${(span.score * 100).toFixed(1)}%`}
        className="rounded px-0.5 py-0 cursor-help"
        style={{ backgroundColor: span.color + '40', borderBottom: `2px solid ${span.color}`, color: span.color }}
      >
        {text.slice(span.start, span.end)}
        <sup className="text-[9px] ml-0.5 opacity-70">{span.label}</sup>
      </span>
    );
    cursor = span.end;
  });

  if (cursor < text.length) {
    parts.push(<span key="tail">{text.slice(cursor)}</span>);
  }

  return (
    <div className="p-4 bg-white/5 rounded-xl border border-white/10 text-gray-200 text-sm leading-relaxed max-h-64 overflow-y-auto">
      {parts}
    </div>
  );
}

// ============ Error Message Helper ============

function getDetailedErrorMessage(errorMsg: string): string {
  // Check for common library import errors
  if (errorMsg.includes("No module named 'torch'")) {
    return `⚠️ Fehlende Bibliothek: torch\n\nBitte installiere die erforderlichen Pakete:\n\n# Versuche je nach Python-Version:\npip install torch transformers pillow\n  ODER\npip3 install torch transformers pillow\n\nNutzer-Hinweise:\n• Stelle sicher, dass du die richtige Python-Umgebung verwendest\n• Prüfe: python --version oder python3 --version\n• Bei virtuellen Umgebungen: Aktiviere die Umgebung zuerst\n• Nutze pip3 wenn du mehrere Python-Versionen hast\n• Bei macOS Silicon: Möglicherweise conda statt pip besser`;
  }
  if (errorMsg.includes("No module named 'transformers'")) {
    return `⚠️ Fehlende Bibliothek: transformers\n\nBitte installiere:\npip install transformers torch\n  ODER\npip3 install transformers torch\n\nSiehe Dokumentation für weitere Hilfe.`;
  }
  if (errorMsg.includes("No module named 'PIL'") || errorMsg.includes("No module named 'pillow'")) {
    return `⚠️ Fehlende Bibliothek: Pillow (PIL)\n\nBitte installiere:\npip install pillow\n  ODER\npip3 install pillow`;
  }
  if (errorMsg.includes("CUDA") || errorMsg.includes("gpu")) {
    return `⚠️ GPU/CUDA Problem:\n${errorMsg}\n\nVersuche mit CPU-Fallback oder prüfe deine GPU-Treiber.`;
  }
  // Return original if no match
  return errorMsg;
}

// ============ Hauptkomponente ============

export default function LaboratoryPanel() {
  const { currentTheme } = useTheme();
  const { success, error, warning, info } = useNotification();

  // Data
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsWithVersions, setModelsWithVersions] = useState<ModelWithVersionTree[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [sessions, setSessions] = useState<LabSession[]>([]);
  const [stats, setStats] = useState<LabStats | null>(null);
  const [loading, setLoading] = useState(true);

  // Selection
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [selectedVersionId, setSelectedVersionId] = useState<string>('');
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');

  // Lab State
  const [currentSample, setCurrentSample] = useState<LabSample | null>(null);
  const [inferenceResult, setInferenceResult] = useState<LabInferenceResult | null>(null);
  const [runningInference, setRunningInference] = useState(false);
  const [loadingSample, setLoadingSample] = useState(false);
  const [sampleIndex, setSampleIndex] = useState(0);

  // Feedback
  const [feedbackRating, setFeedbackRating] = useState<'correct' | 'partial' | 'incorrect' | null>(null);
  const [feedbackComment, setFeedbackComment] = useState('');
  const [correctedLabel, setCorrectedLabel] = useState('');
  const [savingSession, setSavingSession] = useState(false);

  // UI
  const [showSessions, setShowSessions] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);
  const [exportName, setExportName] = useState('');
  const [exportOnlyIncorrect, setExportOnlyIncorrect] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [showRaw, setShowRaw] = useState(false);

  const selectedModel = models.find(m => m.id === selectedModelId);
  const selectedDataset = datasets.find(d => d.id === selectedDatasetId);
  const selectedModelWithVersions = modelsWithVersions.find(m => m.id === selectedModelId);

  // ============ Load ============

  useEffect(() => { loadData(); }, []);

  useEffect(() => {
    if (selectedModelId) {
      loadDatasets();
      loadSessions();
      loadStats();
    }
  }, [selectedModelId]);

  useEffect(() => {
    if (selectedModelId && selectedModelWithVersions) {
      const sorted = [...(selectedModelWithVersions.versions || [])].sort(
        (a, b) => b.version_number - a.version_number
      );
      setSelectedVersionId(sorted[0]?.id || '');
    }
  }, [selectedModelId, modelsWithVersions]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [modelList, modelsWithVer] = await Promise.all([
        invoke<ModelInfo[]>('list_models'),
        invoke<ModelWithVersionTree[]>('list_models_with_version_tree'),
      ]);
      setModels(modelList);
      setModelsWithVersions(modelsWithVer);
      if (modelList.length > 0) {
        setSelectedModelId(modelList[0].id);
      }
    } catch (err: any) {
      error('Fehler beim Laden', String(err));
    } finally {
      setLoading(false);
    }
  };

  const loadDatasets = async () => {
    if (!selectedModelId) return;
    try {
      const list = await invoke<DatasetInfo[]>('list_all_datasets');
      const filtered = list.filter(d => d.model_id === selectedModelId);
      setDatasets(filtered);
      if (filtered.length > 0 && !selectedDatasetId) {
        setSelectedDatasetId(filtered[0].id);
      }
    } catch (err: any) {
      console.error('Datasets laden:', err);
    }
  };

  const loadSessions = async () => {
    try {
      const list = await invoke<LabSession[]>('lab_get_sessions', { modelId: selectedModelId || null });
      setSessions(list);
    } catch (err: any) {
      console.error('Sessions laden:', err);
    }
  };

  const loadStats = async () => {
    try {
      const s = await invoke<LabStats>('lab_get_stats', { modelId: selectedModelId || null });
      setStats(s);
    } catch (err: any) {
      console.error('Stats laden:', err);
    }
  };

  // ============ Lab Actions ============

  const loadSample = async (index?: number) => {
    if (!selectedModelId || !selectedDatasetId) {
      warning('Auswahl fehlt', 'Bitte wähle ein Modell und ein Dataset.');
      return;
    }
    setLoadingSample(true);
    setInferenceResult(null);
    setFeedbackRating(null);
    setFeedbackComment('');
    setCorrectedLabel('');
    setShowRaw(false);
    try {
      const sample = await invoke<LabSample>('lab_load_sample', {
        modelId: selectedModelId,
        datasetId: selectedDatasetId,
        sampleIndex: index ?? sampleIndex,
      });
      setCurrentSample(sample);
      setSampleIndex(sample.sample_index);
    } catch (err: any) {
      error('Sample laden fehlgeschlagen', String(err));
    } finally {
      setLoadingSample(false);
    }
  };

  const runInference = async () => {
    if (!currentSample || !selectedModelId) return;
    setRunningInference(true);
    setInferenceResult(null);
    setFeedbackRating(null);
    setFeedbackComment('');
    setCorrectedLabel('');
    try {
      const result = await invoke<LabInferenceResult>('lab_run_inference', {
        modelId: selectedModelId,
        versionId: selectedVersionId || '',
        samplePath: currentSample.path,
        taskType: null,
      });
      setInferenceResult(result);
      if (result.error) {
        const detailedMsg = getDetailedErrorMessage(result.error);
        warning('Inferenz-Warnung', detailedMsg);
      }
    } catch (err: any) {
      error('Inferenz fehlgeschlagen', String(err));
    } finally {
      setRunningInference(false);
    }
  };

  const nextSample = async () => {
    const next = currentSample ? currentSample.sample_index + 1 : 0;
    await loadSample(next);
  };

  const saveSession = async () => {
    if (!currentSample || !inferenceResult || !feedbackRating || !selectedModel || !selectedDataset) return;
    setSavingSession(true);
    try {
      const version = selectedModelWithVersions?.versions.find(v => v.id === selectedVersionId);
      await invoke<string>('lab_save_session', {
        modelId: selectedModelId,
        modelName: selectedModel.name,
        versionId: selectedVersionId || '',
        versionName: version?.name || 'Original',
        datasetId: selectedDatasetId,
        datasetName: selectedDataset.name,
        sample: currentSample,
        inferenceResult,
        feedback: {
          rating: feedbackRating,
          comment: feedbackComment,
          corrected_label: correctedLabel || null,
        },
      });
      success('Feedback gespeichert', 'Die Session wurde gespeichert.');
      await loadSessions();
      await loadStats();
      // Auto-weiter
      await nextSample();
    } catch (err: any) {
      error('Speichern fehlgeschlagen', String(err));
    } finally {
      setSavingSession(false);
    }
  };

  const deleteSession = async (id: string) => {
    try {
      await invoke('lab_delete_session', { sessionId: id });
      await loadSessions();
      await loadStats();
    } catch (err: any) {
      error('Löschen fehlgeschlagen', String(err));
    }
  };

  const exportDataset = async () => {
    if (!exportName.trim()) { warning('Name fehlt', 'Bitte gib einen Dataset-Namen ein.'); return; }
    setExporting(true);
    try {
      const id = await invoke<string>('lab_export_as_dataset', {
        modelId: selectedModelId,
        datasetName: exportName.trim(),
        onlyIncorrect: exportOnlyIncorrect,
      });
      success('Dataset erstellt', `Dataset "${exportName}" wurde als JSONL exportiert.`);
      setShowExportModal(false);
      setExportName('');
    } catch (err: any) {
      error('Export fehlgeschlagen', String(err));
    } finally {
      setExporting(false);
    }
  };

  // ============ Render Inference Result ============

  function renderInferenceOutput() {
    if (!inferenceResult) return null;
    const r = inferenceResult.rendered;
    const type = inferenceResult.model_output_type;

    return (
      <div className="space-y-4">
        {/* Primäres Ergebnis */}
        {r.primary_label && (
          <div className="flex items-center justify-between p-4 bg-white/5 rounded-xl border border-white/10">
            <div>
              <div className="text-xs text-gray-400 mb-1">Hauptergebnis</div>
              <div className="text-xl font-bold text-white">{r.primary_label}</div>
            </div>
            {r.confidence !== null && r.confidence !== undefined && (
              <div className="text-right">
                <div className="text-xs text-gray-400 mb-1">Konfidenz</div>
                <div className={`text-2xl font-bold ${confidenceColor(r.confidence)}`}>
                  {(r.confidence * 100).toFixed(1)}%
                </div>
              </div>
            )}
          </div>
        )}

        {/* Klassifikation – Top Labels */}
        {r.labels.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs text-gray-400 font-medium">Top Vorhersagen</div>
            {r.labels.map((lbl, i) => (
              <div key={i} className="flex items-center gap-3">
                <div className="w-24 text-sm text-gray-300 truncate">{lbl.label}</div>
                <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full bg-gradient-to-r ${currentTheme.colors.gradient}`}
                    style={{ width: `${lbl.score * 100}%` }}
                  />
                </div>
                <div className="w-12 text-right text-xs text-gray-400">{(lbl.score * 100).toFixed(1)}%</div>
              </div>
            ))}
          </div>
        )}

        {/* NER – Highlighted Text */}
        {r.highlighted_spans.length > 0 && currentSample?.content && (
          <div>
            <div className="text-xs text-gray-400 font-medium mb-2">Erkannte Entitäten</div>
            <HighlightedText text={currentSample.content} spans={r.highlighted_spans} />
            <div className="mt-2 flex flex-wrap gap-2">
              {[...new Set(r.highlighted_spans.map(s => s.label))].map(lbl => {
                const span = r.highlighted_spans.find(s => s.label === lbl)!;
                return (
                  <span key={lbl} className="px-2 py-0.5 rounded text-xs font-medium"
                    style={{ backgroundColor: span.color + '30', color: span.color, border: `1px solid ${span.color}60` }}>
                    {lbl}
                  </span>
                );
              })}
            </div>
          </div>
        )}

        {/* Generierter Text */}
        {r.generated_text && (
          <div>
            <div className="text-xs text-gray-400 font-medium mb-2">Generierter Text</div>
            <div className="p-4 bg-white/5 rounded-xl border border-white/10 text-gray-200 text-sm leading-relaxed max-h-48 overflow-y-auto whitespace-pre-wrap">
              {r.generated_text}
            </div>
          </div>
        )}

        {/* Key-Value Ausgabe */}
        {r.key_values.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs text-gray-400 font-medium">Ausgabe</div>
            {r.key_values.map(([k, v], i) => (
              <div key={i} className="flex gap-3 p-3 bg-white/5 rounded-lg text-sm">
                <span className="text-gray-400 w-24 shrink-0">{k}</span>
                <span className="text-white break-all">{v}</span>
              </div>
            ))}
          </div>
        )}

        {/* Raw JSON toggle */}
        <button
          onClick={() => setShowRaw(v => !v)}
          className="text-xs text-gray-500 hover:text-gray-300 transition-colors flex items-center gap-1"
        >
          <ChevronRight className={`w-3 h-3 transition-transform ${showRaw ? 'rotate-90' : ''}`} />
          Rohe Ausgabe
        </button>
        {showRaw && (
          <pre className="p-3 bg-black/40 rounded-lg text-xs text-gray-400 overflow-auto max-h-40">
            {JSON.stringify(inferenceResult.raw_output, null, 2)}
          </pre>
        )}
      </div>
    );
  }

  // ============ Loading / Empty ============

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
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <FlaskConical className="w-8 h-8" />
            Laboratory
          </h1>
          <p className="text-gray-400 mt-1">Interaktives Lernen durch Echtzeit-Feedback</p>
        </div>
        <div className="bg-white/5 rounded-2xl border border-white/10 p-12 text-center">
          <FlaskConical className="w-12 h-12 text-gray-500 mx-auto mb-4 opacity-40" />
          <h3 className="text-xl font-semibold text-white mb-2">Kein Modell vorhanden</h3>
          <p className="text-gray-400">Füge zuerst ein Modell und ein Dataset hinzu.</p>
        </div>
      </div>
    );
  }

  // ============ Main Render ============

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <FlaskConical className="w-8 h-8" style={{ color: currentTheme.colors.primary }} />
            Laboratory
          </h1>
          <p className="text-gray-400 mt-1">
            Lass das Modell Samples verarbeiten – gib Feedback – generiere neue Trainingsdaten
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Stats Badge */}
          {stats && stats.total_sessions > 0 && (
            <div className="flex items-center gap-4 px-4 py-2 bg-white/5 rounded-lg border border-white/10 text-sm">
              <div className="flex items-center gap-1.5 text-green-400">
                <ThumbsUp className="w-3.5 h-3.5" />
                {stats.correct}
              </div>
              <div className="flex items-center gap-1.5 text-amber-400">
                <Minus className="w-3.5 h-3.5" />
                {stats.partial}
              </div>
              <div className="flex items-center gap-1.5 text-red-400">
                <ThumbsDown className="w-3.5 h-3.5" />
                {stats.incorrect}
              </div>
              <div className="text-gray-400">|</div>
              <div className="text-white font-medium">{(stats.accuracy_rate * 100).toFixed(0)}% korrekt</div>
            </div>
          )}

          <button
            onClick={() => { setShowSessions(v => !v); loadSessions(); }}
            className={`p-2 rounded-lg transition-all ${showSessions ? 'bg-white/10 text-white' : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-white'}`}
            title="Session-Verlauf"
          >
            <Clock className="w-5 h-5" />
          </button>

          {sessions.length > 0 && (
            <button
              onClick={() => setShowExportModal(true)}
              className="flex items-center gap-2 px-3 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-gray-300 hover:text-white transition-all text-sm"
            >
              <Download className="w-4 h-4" />
              Exportieren
            </button>
          )}

          <button onClick={loadData} className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-all">
            <RefreshCw className="w-5 h-5" />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* LEFT – Konfiguration + Sample */}
        <div className="lg:col-span-2 space-y-5">

          {/* Modell & Dataset Auswahl */}
          <div className="bg-white/5 rounded-xl border border-white/10 p-6">
            <h2 className="text-base font-semibold text-white mb-4 flex items-center gap-2">
              <Layers className="w-4 h-4" />
              Modell & Dataset
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Modell */}
              <div>
                <label className="block text-xs text-gray-400 mb-1.5">Modell</label>
                <div className="relative">
                  <select
                    value={selectedModelId}
                    onChange={e => { setSelectedModelId(e.target.value); setCurrentSample(null); setInferenceResult(null); }}
                    className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-lg text-white text-sm appearance-none cursor-pointer focus:outline-none focus:ring-2"
                    style={{ '--tw-ring-color': currentTheme.colors.primary } as any}
                  >
                    {models.map(m => (
                      <option key={m.id} value={m.id} className="bg-slate-800">{m.name}</option>
                    ))}
                  </select>
                  <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
                </div>
              </div>

              {/* Version */}
              <div>
                <label className="block text-xs text-gray-400 mb-1.5 flex items-center gap-1">
                  <GitBranch className="w-3 h-3" /> Version
                </label>
                <div className="relative">
                  <select
                    value={selectedVersionId}
                    onChange={e => setSelectedVersionId(e.target.value)}
                    className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-lg text-white text-sm appearance-none cursor-pointer focus:outline-none focus:ring-2"
                    style={{ '--tw-ring-color': currentTheme.colors.primary } as any}
                  >
                    {(selectedModelWithVersions?.versions || []).map(v => (
                      <option key={v.id} value={v.id} className="bg-slate-800">
                        {v.is_root ? '⭐ ' : ''}{v.name}
                      </option>
                    ))}
                  </select>
                  <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
                </div>
              </div>

              {/* Dataset */}
              <div>
                <label className="block text-xs text-gray-400 mb-1.5 flex items-center gap-1">
                  <Database className="w-3 h-3" /> Dataset
                </label>
                <div className="relative">
                  <select
                    value={selectedDatasetId}
                    onChange={e => { setSelectedDatasetId(e.target.value); setCurrentSample(null); setInferenceResult(null); }}
                    disabled={datasets.length === 0}
                    className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-lg text-white text-sm appearance-none cursor-pointer focus:outline-none focus:ring-2 disabled:opacity-40"
                    style={{ '--tw-ring-color': currentTheme.colors.primary } as any}
                  >
                    {datasets.length === 0
                      ? <option value="" className="bg-slate-800">Kein Dataset</option>
                      : datasets.map(d => (
                          <option key={d.id} value={d.id} className="bg-slate-800">{d.name}</option>
                        ))
                    }
                  </select>
                  <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
                </div>
              </div>
            </div>

            {/* Load Buttons */}
            <div className="flex gap-3 mt-4">
              <button
                onClick={() => loadSample(0)}
                disabled={!selectedModelId || !selectedDatasetId || loadingSample}
                className={`flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white text-sm font-medium hover:opacity-90 transition-all disabled:opacity-40 disabled:cursor-not-allowed`}
              >
                {loadingSample ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                Sample laden
              </button>
              {currentSample && (
                <button
                  onClick={nextSample}
                  disabled={loadingSample}
                  className="flex items-center gap-2 px-4 py-2.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-gray-300 text-sm font-medium transition-all disabled:opacity-40"
                >
                  <SkipForward className="w-4 h-4" />
                  Nächstes
                </button>
              )}
            </div>
          </div>

          {/* Sample Anzeige */}
          {currentSample && (
            <div className="bg-white/5 rounded-xl border border-white/10 p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-base font-semibold text-white flex items-center gap-2">
                  {sampleTypeIcon(currentSample.sample_type)}
                  Sample
                  <span className="px-2 py-0.5 bg-white/10 rounded text-xs text-gray-400 font-normal">
                    {currentSample.sample_index + 1} / {currentSample.total_samples}
                  </span>
                </h2>
                <span className="text-xs text-gray-500 truncate max-w-xs" title={currentSample.filename}>
                  {currentSample.filename}
                </span>
              </div>

              {/* Bild */}
              {currentSample.sample_type === 'image' && (
                <ImageCanvas
                  samplePath={currentSample.path}
                  boxes={inferenceResult?.rendered.bounding_boxes || []}
                />
              )}

              {/* Text / JSON / CSV */}
              {(currentSample.sample_type === 'text' || currentSample.sample_type === 'json' || currentSample.sample_type === 'csv') && currentSample.content && (
                inferenceResult && inferenceResult.rendered.highlighted_spans.length > 0
                  ? <HighlightedText text={currentSample.content} spans={inferenceResult.rendered.highlighted_spans} />
                  : (
                    <div className="p-4 bg-white/5 rounded-xl border border-white/10 text-gray-300 text-sm leading-relaxed whitespace-pre-wrap font-mono max-h-64 overflow-y-auto">
                      {currentSample.content}
                    </div>
                  )
              )}

              {/* Audio */}
              {currentSample.sample_type === 'audio' && (
                <div className="p-4 bg-white/5 rounded-xl border border-white/10 flex items-center gap-4">
                  <div className="w-12 h-12 rounded-full bg-white/10 flex items-center justify-center">
                    <Zap className="w-6 h-6 text-purple-400" />
                  </div>
                  <div>
                    <div className="text-white text-sm font-medium">{currentSample.filename}</div>
                    <div className="text-gray-400 text-xs mt-1">Audio-Sample</div>
                  </div>
                </div>
              )}

              {/* Inferenz starten */}
              <button
                onClick={runInference}
                disabled={runningInference}
                className={`mt-4 w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
                  inferenceResult
                    ? 'bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300'
                    : `bg-gradient-to-r ${currentTheme.colors.gradient} text-white hover:opacity-90`
                } disabled:opacity-40 disabled:cursor-not-allowed`}
              >
                {runningInference
                  ? <><Loader2 className="w-4 h-4 animate-spin" /> Modell läuft...</>
                  : inferenceResult
                    ? <><RefreshCw className="w-4 h-4" /> Erneut ausführen</>
                    : <><Zap className="w-4 h-4" /> Modell ausführen</>
                }
              </button>
            </div>
          )}

          {/* Modell-Ausgabe */}
          {inferenceResult && (
            <div className="bg-white/5 rounded-xl border border-white/10 p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-base font-semibold text-white flex items-center gap-2">
                  <Eye className="w-4 h-4" />
                  Modell-Ausgabe
                  <span className="px-2 py-0.5 bg-white/10 rounded text-xs text-gray-400 font-normal capitalize">
                    {inferenceResult.model_output_type}
                  </span>
                </h2>
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <Clock className="w-3 h-3" />
                  {formatMs(inferenceResult.inference_time_ms)}
                </div>
              </div>

              {inferenceResult.error && (
                <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-300 text-sm flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                  <span className="whitespace-pre-wrap">{getDetailedErrorMessage(inferenceResult.error)}</span>
                </div>
              )}

              {renderInferenceOutput()}
            </div>
          )}

          {/* Feedback */}
          {inferenceResult && (
            <div className="bg-white/5 rounded-xl border border-white/10 p-6">
              <h2 className="text-base font-semibold text-white mb-4 flex items-center gap-2">
                <MessageSquare className="w-4 h-4" />
                Dein Feedback
              </h2>

              {/* Rating Buttons */}
              <div className="grid grid-cols-3 gap-3 mb-4">
                <button
                  onClick={() => setFeedbackRating('correct')}
                  className={`flex flex-col items-center gap-2 p-4 rounded-xl border transition-all ${
                    feedbackRating === 'correct'
                      ? 'bg-green-500/20 border-green-500/50 text-green-300'
                      : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                  }`}
                >
                  <CheckCircle className="w-6 h-6" />
                  <span className="text-sm font-medium">Richtig</span>
                  <span className="text-xs opacity-60">Modell lag richtig</span>
                </button>
                <button
                  onClick={() => setFeedbackRating('partial')}
                  className={`flex flex-col items-center gap-2 p-4 rounded-xl border transition-all ${
                    feedbackRating === 'partial'
                      ? 'bg-amber-500/20 border-amber-500/50 text-amber-300'
                      : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                  }`}
                >
                  <AlertTriangle className="w-6 h-6" />
                  <span className="text-sm font-medium">Teilweise</span>
                  <span className="text-xs opacity-60">Teilweise korrekt</span>
                </button>
                <button
                  onClick={() => setFeedbackRating('incorrect')}
                  className={`flex flex-col items-center gap-2 p-4 rounded-xl border transition-all ${
                    feedbackRating === 'incorrect'
                      ? 'bg-red-500/20 border-red-500/50 text-red-300'
                      : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                  }`}
                >
                  <XCircle className="w-6 h-6" />
                  <span className="text-sm font-medium">Falsch</span>
                  <span className="text-xs opacity-60">Modell lag falsch</span>
                </button>
              </div>

              {/* Korrektur & Kommentar */}
              {(feedbackRating === 'partial' || feedbackRating === 'incorrect') && (
                <div className="space-y-3 mb-4">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1.5">Korrekte Antwort / Label</label>
                    <input
                      type="text"
                      value={correctedLabel}
                      onChange={e => setCorrectedLabel(e.target.value)}
                      placeholder={
                        inferenceResult.model_output_type === 'detection'
                          ? 'z.B. Hund, Katze, Auto ...'
                          : inferenceResult.model_output_type === 'generation'
                            ? 'Korrekte Antwort oder Text ...'
                            : 'Korrektes Label eingeben ...'
                      }
                      className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none focus:ring-2"
                      style={{ '--tw-ring-color': currentTheme.colors.primary } as any}
                    />
                  </div>
                </div>
              )}

              <div className="mb-4">
                <label className="block text-xs text-gray-400 mb-1.5">Kommentar (optional)</label>
                <textarea
                  value={feedbackComment}
                  onChange={e => setFeedbackComment(e.target.value)}
                  rows={2}
                  placeholder="Was war falsch? Was hätte das Modell tun sollen?"
                  className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-lg text-white text-sm placeholder-gray-500 resize-none focus:outline-none focus:ring-2"
                  style={{ '--tw-ring-color': currentTheme.colors.primary } as any}
                />
              </div>

              <button
                onClick={saveSession}
                disabled={!feedbackRating || savingSession}
                className={`w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white font-medium hover:opacity-90 transition-all disabled:opacity-40 disabled:cursor-not-allowed`}
              >
                {savingSession
                  ? <><Loader2 className="w-4 h-4 animate-spin" /> Speichern...</>
                  : <><Save className="w-4 h-4" /> Feedback speichern &amp; weiter</>
                }
              </button>

              {!feedbackRating && (
                <p className="text-xs text-gray-500 text-center mt-2">
                  Wähle eine Bewertung um fortzufahren
                </p>
              )}
            </div>
          )}
        </div>

        {/* RIGHT – Sidebar */}
        <div className="space-y-5">
          {/* Info Box */}
          <div className="bg-white/5 rounded-xl border border-white/10 p-5">
            <div className="flex items-center gap-2 mb-3">
              <Info className="w-4 h-4 text-blue-400" />
              <h3 className="text-sm font-medium text-gray-300">Wie es funktioniert</h3>
            </div>
            <ol className="space-y-2.5">
              {[
                ['1', 'Modell, Version & Dataset wählen'],
                ['2', 'Sample laden – das Modell verarbeitet es'],
                ['3', 'Ergebnis anschauen & bewerten'],
                ['4', 'Feedback speichern – automatisch nächstes Sample'],
                ['5', 'Sessions als neues Dataset exportieren'],
              ].map(([n, text]) => (
                <li key={n} className="flex items-start gap-2.5 text-xs text-gray-400">
                  <span
                    className="w-5 h-5 rounded-full flex items-center justify-center text-white text-xs font-bold shrink-0 mt-0.5"
                    style={{ background: `linear-gradient(135deg, ${currentTheme.colors.primary}, ${currentTheme.colors.primary}aa)` }}
                  >
                    {n}
                  </span>
                  {text}
                </li>
              ))}
            </ol>
          </div>

          {/* Stats */}
          {stats && (
            <div className="bg-white/5 rounded-xl border border-white/10 p-5">
              <h3 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-2">
                <BarChart3 className="w-4 h-4" />
                Statistiken
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-white/5 rounded-lg text-center">
                  <div className="text-2xl font-bold text-white">{stats.total_sessions}</div>
                  <div className="text-xs text-gray-400 mt-0.5">Sessions</div>
                </div>
                <div className="p-3 bg-white/5 rounded-lg text-center">
                  <div className="text-2xl font-bold text-white">{(stats.accuracy_rate * 100).toFixed(0)}%</div>
                  <div className="text-xs text-gray-400 mt-0.5">Genauigkeit</div>
                </div>
                <div className="p-3 bg-green-500/10 border border-green-500/20 rounded-lg text-center">
                  <div className="text-xl font-bold text-green-400">{stats.correct}</div>
                  <div className="text-xs text-green-400/70 mt-0.5">Richtig</div>
                </div>
                <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg text-center">
                  <div className="text-xl font-bold text-amber-400">{stats.partial}</div>
                  <div className="text-xs text-amber-400/70 mt-0.5">Teilweise</div>
                </div>
                <div className="col-span-2 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-center">
                  <div className="text-xl font-bold text-red-400">{stats.incorrect}</div>
                  <div className="text-xs text-red-400/70 mt-0.5">Falsch → Trainingsdata</div>
                </div>
              </div>

              {/* Accuracy Bar */}
              {stats.total_sessions > 0 && (
                <div className="mt-4">
                  <div className="h-2 w-full bg-white/10 rounded-full overflow-hidden flex">
                    <div className="bg-green-500 h-full transition-all" style={{ width: `${(stats.correct / stats.total_sessions) * 100}%` }} />
                    <div className="bg-amber-500 h-full transition-all" style={{ width: `${(stats.partial / stats.total_sessions) * 100}%` }} />
                    <div className="bg-red-500 h-full transition-all" style={{ width: `${(stats.incorrect / stats.total_sessions) * 100}%` }} />
                  </div>
                </div>
              )}

              {stats.incorrect > 0 && (
                <button
                  onClick={() => setShowExportModal(true)}
                  className="mt-4 w-full flex items-center justify-center gap-2 px-3 py-2.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-gray-300 text-sm transition-all"
                >
                  <Download className="w-4 h-4" />
                  {stats.incorrect} falsche Samples exportieren
                </button>
              )}
            </div>
          )}

          {/* Session Verlauf */}
          {showSessions && (
            <div className="bg-white/5 rounded-xl border border-white/10 p-5">
              <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center justify-between">
                <span className="flex items-center gap-2"><Clock className="w-4 h-4" /> Verlauf</span>
                <span className="text-xs text-gray-500">{sessions.length} Sessions</span>
              </h3>
              <div className="space-y-2 max-h-80 overflow-y-auto">
                {sessions.length === 0 && (
                  <div className="text-center py-6 text-gray-500 text-xs">Noch keine Sessions</div>
                )}
                {sessions.map(s => (
                  <div key={s.id} className="flex items-start justify-between gap-2 p-3 bg-white/5 rounded-lg hover:bg-white/[0.08] transition-all">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2 mb-0.5">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                          s.feedback.rating === 'correct' ? 'bg-green-500/20 text-green-400'
                          : s.feedback.rating === 'partial' ? 'bg-amber-500/20 text-amber-400'
                          : 'bg-red-500/20 text-red-400'
                        }`}>
                          {s.feedback.rating === 'correct' ? '✓' : s.feedback.rating === 'partial' ? '~' : '✗'}
                        </span>
                        <span className="text-xs text-white truncate">{s.sample.filename}</span>
                      </div>
                      {s.inference_result.rendered.primary_label && (
                        <div className="text-[10px] text-gray-500 truncate">
                          → {s.inference_result.rendered.primary_label}
                        </div>
                      )}
                      {s.feedback.comment && (
                        <div className="text-[10px] text-gray-500 truncate mt-0.5 italic">
                          "{s.feedback.comment}"
                        </div>
                      )}
                    </div>
                    <button
                      onClick={() => deleteSession(s.id)}
                      className="p-1 text-gray-600 hover:text-red-400 transition-colors shrink-0"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Export Modal */}
      {showExportModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-md">
            <div className="flex items-center justify-between p-6 border-b border-white/10">
              <div className="flex items-center gap-3">
                <Download className="w-5 h-5 text-purple-400" />
                <h2 className="text-lg font-bold text-white">Als Dataset exportieren</h2>
              </div>
              <button onClick={() => setShowExportModal(false)} className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-all">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-6 space-y-4">
              <p className="text-sm text-gray-400">
                Erstellt aus deinen Feedback-Sessions ein neues JSONL-Dataset, das du direkt für weiteres Training verwenden kannst.
              </p>
              <div>
                <label className="block text-xs text-gray-400 mb-1.5">Dataset-Name</label>
                <input
                  type="text"
                  value={exportName}
                  onChange={e => setExportName(e.target.value)}
                  placeholder="z.B. Lab-Feedback-Korrektur-v1"
                  className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none focus:ring-2"
                  style={{ '--tw-ring-color': currentTheme.colors.primary } as any}
                />
              </div>

              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <div>
                  <div className="text-sm text-white">Nur falsche/teilweise Samples</div>
                  <div className="text-xs text-gray-400">Empfohlen: Fokus auf Fehler</div>
                </div>
                <button
                  onClick={() => setExportOnlyIncorrect(v => !v)}
                  className={`w-11 h-6 rounded-full transition-all ${exportOnlyIncorrect ? `bg-gradient-to-r ${currentTheme.colors.gradient}` : 'bg-white/10'}`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white shadow transform transition-transform ${exportOnlyIncorrect ? 'translate-x-5' : 'translate-x-0.5'}`} />
                </button>
              </div>

              {stats && (
                <div className="text-xs text-gray-500 text-center">
                  {exportOnlyIncorrect
                    ? `${stats.incorrect + stats.partial} Samples werden exportiert`
                    : `${stats.total_sessions} Samples werden exportiert`}
                </div>
              )}
            </div>
            <div className="p-6 border-t border-white/10 flex gap-3">
              <button onClick={() => setShowExportModal(false)} className="flex-1 py-2.5 bg-white/5 hover:bg-white/10 rounded-lg text-white text-sm transition-all">
                Abbrechen
              </button>
              <button
                onClick={exportDataset}
                disabled={exporting || !exportName.trim()}
                className={`flex-1 flex items-center justify-center gap-2 py-2.5 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white text-sm font-medium hover:opacity-90 transition-all disabled:opacity-40`}
              >
                {exporting ? <><Loader2 className="w-4 h-4 animate-spin" /> Exportiere...</> : <><Download className="w-4 h-4" /> Exportieren</>}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
