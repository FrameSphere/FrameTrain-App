import { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import {
  Play, Square, Download, Filter, CheckCircle, XCircle, Clock, Zap,
  Target, TrendingUp, Loader2, ChevronDown, AlertTriangle, FileText,
  Layers, GitBranch, MessageSquare, Image, Mic, Table2, Eye,
  Send, RotateCcw, ChevronRight, BarChart3, HardDrive, AlertCircle,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';

// ============ Types ============

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
interface Dataset {
  id: string;
  name: string;
  path: string;
  file_count: number;
  size_bytes: number;
  type: string;
}
interface TestJob {
  id: string;
  model_id: string;
  model_name: string;
  version_id: string;
  version_name: string;
  dataset_id: string;
  dataset_name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  progress: TestProgress;
  results: TestResults | null;
  error: string | null;
  task_type: string;
  mode: string;
}
interface TestProgress {
  current_sample: number;
  total_samples: number;
  progress_percent: number;
  samples_per_second: number;
  estimated_time_remaining: number | null;
}
interface TestResults {
  total_samples: number;
  correct_predictions: number | null;
  incorrect_predictions: number | null;
  accuracy: number | null;
  average_loss: number | null;
  average_inference_time: number;
  predictions: PredictionResult[];
  metrics: Record<string, any>;
  task_type: string;
  hard_examples_file: string | null;
}
interface PredictionResult {
  sample_id: number;
  input_text: string;
  input_path: string | null;
  expected_output: string | null;
  predicted_output: string;
  is_correct: boolean;
  loss: number | null;
  confidence: number | null;
  inference_time: number;
  error_type: string | null;
  top_predictions?: Array<{ label: string; confidence: number }>;
  detections?: Array<{ label: string; confidence: number; bbox?: any }>;
  wer?: number | null;
}

// Single test result (modality-agnostic)
interface SingleResult {
  task_type: string;
  input: string;
  input_type: string;
  result: Record<string, any>;
}

type TestMode = 'dataset' | 'single';

// ============ Helpers ============

function formatBytes(b: number): string {
  if (b === 0) return '0 B';
  const k = 1024, s = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(b) / Math.log(k));
  return `${parseFloat((b / Math.pow(k, i)).toFixed(2))} ${s[i]}`;
}
function formatDuration(s: number): string {
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}
function getModalityIcon(taskType: string) {
  if (!taskType) return <FileText className="w-4 h-4" />;
  const t = taskType.toLowerCase();
  if (t.includes('vision') || t.includes('image')) return <Image className="w-4 h-4" />;
  if (t.includes('audio') || t.includes('speech') || t.includes('asr')) return <Mic className="w-4 h-4" />;
  if (t.includes('detection') || t.includes('yolo')) return <Eye className="w-4 h-4" />;
  if (t.includes('tabular') || t.includes('regression') || t.includes('classification')) return <Table2 className="w-4 h-4" />;
  return <MessageSquare className="w-4 h-4" />;
}
function getModalityLabel(taskType: string): string {
  if (!taskType || taskType === 'auto') return 'Auto-Detect';
  const t = taskType.toLowerCase();
  if (t.includes('vision') || t.includes('image_class')) return 'Vision (Bildklassifikation)';
  if (t.includes('detection') || t.includes('yolo')) return 'Detection (Objekterkennung)';
  if (t.includes('audio') || t.includes('asr') || t.includes('speech')) return 'Audio / ASR';
  if (t.includes('tabular')) return 'Tabular (Strukturiert)';
  if (t.includes('nlp') || t.includes('causal') || t.includes('seq2seq') || t.includes('text') || t.includes('lm')) return 'NLP / Text';
  return taskType;
}
function getSingleInputPlaceholder(taskType: string, inputType: string): string {
  if (inputType === 'image_path') return '/absoluter/pfad/zum/bild.jpg';
  if (inputType === 'audio_path') return '/absoluter/pfad/zur/audio.wav';
  if (inputType === 'json') return '{"feature1": 1.0, "feature2": "A"}';
  const t = taskType.toLowerCase();
  if (t.includes('summarization')) return 'Langer Text der zusammengefasst werden soll…';
  if (t.includes('translation')) return 'Text der übersetzt werden soll…';
  if (t.includes('question')) return 'Frage an das Modell…';
  return 'Eingabe für das Modell…';
}

// ============ Single Result Renderer ============

function SingleResultView({ result, taskType }: { result: Record<string, any>; taskType: string }) {
  const t = taskType.toLowerCase();

  if (t.includes('audio') || result.transcript !== undefined) {
    return (
      <div className="space-y-3">
        {result.mode === 'asr' || result.transcript ? (
          <div>
            <p className="text-xs text-gray-400 mb-1">Transkript</p>
            <p className="text-white bg-white/5 rounded-lg p-3 text-sm leading-relaxed">
              {result.transcript || result.output || '(leer)'}
            </p>
          </div>
        ) : (
          <div>
            <p className="text-xs text-gray-400 mb-1">Vorhergesagte Klasse</p>
            <p className="text-white font-medium text-lg">{result.predicted_label}</p>
          </div>
        )}
        {result.top_predictions && (
          <div>
            <p className="text-xs text-gray-400 mb-2">Top Vorhersagen</p>
            <div className="space-y-1">
              {result.top_predictions.slice(0, 5).map((p: any, i: number) => (
                <div key={i} className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">{p.label}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-24 h-1.5 bg-white/10 rounded-full overflow-hidden">
                      <div className="h-full bg-blue-500 rounded-full" style={{ width: `${(p.score || p.confidence || 0) * 100}%` }} />
                    </div>
                    <span className="text-xs text-gray-400 w-12 text-right">
                      {((p.score || p.confidence || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  if (t.includes('detection') || result.detections !== undefined) {
    const dets: any[] = result.detections || [];
    return (
      <div className="space-y-3">
        <p className="text-xs text-gray-400">{dets.length} Objekt(e) erkannt</p>
        {dets.length === 0 ? (
          <p className="text-gray-500 text-sm italic">Keine Objekte gefunden</p>
        ) : (
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {dets.map((d: any, i: number) => (
              <div key={i} className="flex items-center justify-between bg-white/5 rounded-lg px-3 py-2">
                <span className="text-white text-sm font-medium">{d.label}</span>
                <span className="text-blue-400 text-sm">{((d.confidence || 0) * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  if (t.includes('vision') || result.top_predictions !== undefined) {
    const top: any[] = result.top_predictions || [];
    return (
      <div className="space-y-3">
        <div>
          <p className="text-xs text-gray-400 mb-1">Vorhergesagte Klasse</p>
          <p className="text-white font-semibold text-xl">{result.predicted_label || top[0]?.label}</p>
          <p className="text-gray-400 text-sm">{((result.confidence || top[0]?.confidence || 0) * 100).toFixed(2)}% Konfidenz</p>
        </div>
        {top.length > 1 && (
          <div>
            <p className="text-xs text-gray-400 mb-2">Top-{top.length} Vorhersagen</p>
            <div className="space-y-1">
              {top.map((p: any, i: number) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-xs text-gray-500 w-4">{i + 1}.</span>
                  <div className="flex-1">
                    <div className="flex justify-between text-xs mb-0.5">
                      <span className="text-gray-300">{p.label}</span>
                      <span className="text-gray-400">{((p.confidence || 0) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1 bg-white/10 rounded-full overflow-hidden">
                      <div className="h-full bg-purple-500 rounded-full transition-all" style={{ width: `${(p.confidence || 0) * 100}%` }} />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  if (t.includes('tabular') || result.prediction !== undefined) {
    const probs: number[] = result.probabilities || [];
    return (
      <div className="space-y-3">
        <div>
          <p className="text-xs text-gray-400 mb-1">Vorhersage</p>
          <p className="text-white font-semibold text-2xl">{result.prediction}</p>
          {result.confidence && (
            <p className="text-gray-400 text-sm mt-1">{(result.confidence * 100).toFixed(2)}% Konfidenz</p>
          )}
        </div>
        {probs.length > 0 && (
          <div>
            <p className="text-xs text-gray-400 mb-2">Klassenwahrscheinlichkeiten</p>
            <div className="space-y-1">
              {probs.slice(0, 8).map((p: number, i: number) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-xs text-gray-500 w-6">{i}</span>
                  <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                    <div className="h-full bg-green-500 rounded-full" style={{ width: `${p * 100}%` }} />
                  </div>
                  <span className="text-xs text-gray-400 w-14 text-right">{(p * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  // NLP / default
  return (
    <div className="space-y-3">
      <div>
        <p className="text-xs text-gray-400 mb-1">Ausgabe</p>
        <p className="text-white bg-white/5 rounded-lg p-3 text-sm leading-relaxed whitespace-pre-wrap">
          {result.output || result.transcript || result.predicted_label || JSON.stringify(result, null, 2)}
        </p>
      </div>
      {result.confidence !== undefined && result.confidence !== null && (
        <p className="text-gray-400 text-xs">Konfidenz: {(result.confidence * 100).toFixed(2)}%</p>
      )}
      {result.model_class && (
        <p className="text-gray-500 text-xs">Modell-Klasse: {result.model_class}</p>
      )}
    </div>
  );
}

// ============ Dataset Metrics View ============

function MetricsGrid({ results, taskType }: { results: TestResults; taskType: string }) {
  const t = taskType.toLowerCase();
  const m = results.metrics || {};

  const cards: Array<{ label: string; value: string; sub?: string; icon: React.ReactNode; color: string }> = [];

  // Accuracy / Haupt-Metrik
  if (results.accuracy !== null && results.accuracy !== undefined) {
    cards.push({
      label: 'Accuracy',
      value: `${results.accuracy.toFixed(2)}%`,
      sub: results.correct_predictions !== null
        ? `${results.correct_predictions} / ${results.total_samples} korrekt`
        : `${results.total_samples} Samples`,
      icon: <Target className="w-5 h-5 text-green-400" />,
      color: 'text-green-400',
    });
  } else if (m.average_wer !== undefined && m.average_wer !== null) {
    cards.push({
      label: 'WER',
      value: `${(m.average_wer * 100).toFixed(2)}%`,
      sub: `CER: ${m.average_cer !== null ? (m.average_cer * 100).toFixed(2) + '%' : 'N/A'}`,
      icon: <BarChart3 className="w-5 h-5 text-orange-400" />,
      color: 'text-orange-400',
    });
  } else if (m.precision !== undefined) {
    cards.push({
      label: 'Precision',
      value: m.precision !== null ? `${(m.precision * 100).toFixed(2)}%` : 'N/A',
      sub: m.recall !== null ? `Recall: ${(m.recall * 100).toFixed(2)}%` : '',
      icon: <Target className="w-5 h-5 text-blue-400" />,
      color: 'text-blue-400',
    });
  } else if (m.r2_score !== undefined) {
    cards.push({
      label: 'R² Score',
      value: m.r2_score !== null ? m.r2_score.toFixed(4) : 'N/A',
      sub: m.mae !== undefined ? `MAE: ${m.mae?.toFixed(4)}` : '',
      icon: <TrendingUp className="w-5 h-5 text-purple-400" />,
      color: 'text-purple-400',
    });
  }

  // F1 (Klassifikation)
  if (m.f1_macro !== undefined) {
    cards.push({
      label: 'F1 Macro',
      value: m.f1_macro !== null ? `${(m.f1_macro * 100).toFixed(2)}%` : 'N/A',
      sub: 'Macro-Average',
      icon: <BarChart3 className="w-5 h-5 text-blue-400" />,
      color: 'text-blue-400',
    });
  }

  // Top-5 (Vision)
  if (m.top5_accuracy !== undefined && m.top5_accuracy !== null) {
    cards.push({
      label: 'Top-5 Accuracy',
      value: `${m.top5_accuracy.toFixed(2)}%`,
      sub: 'Unter Top-5',
      icon: <CheckCircle className="w-5 h-5 text-teal-400" />,
      color: 'text-teal-400',
    });
  }

  // Avg Confidence
  if (m.average_confidence !== undefined && m.average_confidence !== null) {
    cards.push({
      label: 'Avg. Konfidenz',
      value: `${(m.average_confidence * 100).toFixed(2)}%`,
      sub: 'Durchschnitt',
      icon: <Zap className="w-5 h-5 text-yellow-400" />,
      color: 'text-yellow-400',
    });
  }

  // Avg Loss
  if (results.average_loss !== null && results.average_loss !== undefined) {
    cards.push({
      label: 'Avg. Loss',
      value: results.average_loss.toFixed(4),
      sub: 'Cross-Entropy',
      icon: <TrendingUp className="w-5 h-5 text-blue-400" />,
      color: 'text-blue-400',
    });
  }

  // Inferenz-Zeit
  cards.push({
    label: 'Inferenz-Zeit',
    value: `${(results.average_inference_time * 1000).toFixed(0)}ms`,
    sub: 'Pro Sample',
    icon: <Clock className="w-5 h-5 text-purple-400" />,
    color: 'text-white',
  });

  // Samples/s
  if (m.samples_per_second !== undefined) {
    cards.push({
      label: 'Geschwindigkeit',
      value: `${(m.samples_per_second || 0).toFixed(1)}`,
      sub: 'Samples/Sekunde',
      icon: <Zap className="w-5 h-5 text-yellow-400" />,
      color: 'text-white',
    });
  }

  // Detections (Detection-Modus)
  if (m.average_detections !== undefined) {
    cards.push({
      label: 'Avg. Detektionen',
      value: `${(m.average_detections || 0).toFixed(1)}`,
      sub: 'Pro Bild',
      icon: <Eye className="w-5 h-5 text-cyan-400" />,
      color: 'text-cyan-400',
    });
  }

  if (cards.length === 0) {
    cards.push({
      label: 'Samples',
      value: String(results.total_samples),
      sub: 'Getestet',
      icon: <Target className="w-5 h-5 text-gray-400" />,
      color: 'text-white',
    });
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
      {cards.map((card, i) => (
        <div key={i} className="bg-white/5 rounded-xl border border-white/10 p-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-400">{card.label}</h3>
            {card.icon}
          </div>
          <div className={`text-3xl font-bold mb-1 ${card.color}`}>{card.value}</div>
          {card.sub && <div className="text-xs text-gray-500">{card.sub}</div>}
        </div>
      ))}
    </div>
  );
}

// ============ Main Component ============

export default function TestPanel() {
  const { currentTheme } = useTheme();
  const { success, error: notifyError, info } = useNotification();

  // Data
  const [modelsWithVersions, setModelsWithVersions] = useState<ModelWithVersionTree[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);

  // Mode
  const [testMode, setTestMode] = useState<TestMode>('dataset');

  // Dataset test state
  const [currentTest, setCurrentTest] = useState<TestJob | null>(null);
  const [testResults, setTestResults] = useState<TestResults | null>(null);
  const [loading, setLoading] = useState(true);

  // Single test state
  const [singleInput, setSingleInput] = useState('');
  const [singleInputType, setSingleInputType] = useState<'text' | 'image_path' | 'audio_path' | 'json'>('text');
  const [singleRunning, setSingleRunning] = useState(false);
  const [singleResult, setSingleResult] = useState<SingleResult | null>(null);
  const [singleHistory, setSingleHistory] = useState<SingleResult[]>([]);

  // Filters (dataset mode)
  const [showOnlyIncorrect, setShowOnlyIncorrect] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const versionIdRef = useRef<string | null>(null);

  // ============ Init ============

  useEffect(() => {
    loadInitialData();
    setupEventListeners();
  }, []);

  useEffect(() => {
    if (selectedModelId) {
      loadDatasets(selectedModelId);
      const model = modelsWithVersions.find(m => m.id === selectedModelId);
      if (model && model.versions.length > 0) {
        const sorted = [...model.versions].sort((a, b) => b.version_number - a.version_number);
        setSelectedVersionId(sorted[0]?.id || null);
      }
    }
  }, [selectedModelId, modelsWithVersions]);

  useEffect(() => {
    versionIdRef.current = selectedVersionId;
  }, [selectedVersionId]);

  // Auto-detect best input type from task_type
  useEffect(() => {
    if (!selectedVersionId) return;
    const model = modelsWithVersions.find(m => m.id === selectedModelId);
    // We don't know task_type until test starts, keep text as default for now
  }, [selectedVersionId]);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      const models = await invoke<ModelWithVersionTree[]>('list_models_with_version_tree');
      setModelsWithVersions(models);
      if (models.length > 0) setSelectedModelId(models[0].id);
    } catch (err: any) {
      notifyError('Fehler beim Laden', String(err));
    } finally {
      setLoading(false);
    }
  };

  const loadDatasets = async (modelId: string) => {
    try {
      const ds = await invoke<Dataset[]>('list_test_datasets_for_model', { modelId });
      setDatasets(ds);
      if (ds.length > 0) setSelectedDatasetId(ds[0].id);
      else setSelectedDatasetId(null);
    } catch (err) {
      console.error('Datasets laden:', err);
    }
  };

  const setupEventListeners = () => {
    const unlisteners: Promise<() => void>[] = [];

    unlisteners.push(listen('test-progress', (event: any) => {
      const prog = event.payload.data;
      setCurrentTest(prev => {
        if (!prev) return null;
        return { ...prev, status: 'running', progress: prog };
      });
    }));

    unlisteners.push(listen('test-status', (event: any) => {
      const d = event.payload.data;
      console.log('[Test] Status:', d);
    }));

    unlisteners.push(listen('test-complete', async (event: any) => {
      const data = event.payload.data;
      const vid = event.payload.version_id || versionIdRef.current;

      let fullResults: TestResults = {
        total_samples: data?.total_samples || 0,
        correct_predictions: data?.correct_predictions ?? null,
        incorrect_predictions: data?.incorrect_predictions ?? null,
        accuracy: data?.accuracy ?? null,
        average_loss: data?.average_loss ?? null,
        average_inference_time: data?.average_inference_time || 0,
        predictions: [],
        metrics: {
          samples_per_second: data?.samples_per_second || 0,
          ...(data?.metrics || {}),
        },
        task_type: data?.task_type || '',
        hard_examples_file: data?.hard_examples_file || null,
      };

      // Predictions aus DB laden
      if (vid) {
        try {
          const dbResults = await invoke<TestResults[]>('get_test_results_for_version', { versionId: vid });
          if (dbResults && dbResults.length > 0) {
            const latest = dbResults[0];
            fullResults = {
              ...fullResults,
              ...latest,
              metrics: { ...fullResults.metrics, ...latest.metrics },
            };
          }
        } catch (e) {
          console.warn('[Test] Predictions aus DB nicht geladen:', e);
        }
      }

      setTestResults(fullResults);
      setCurrentTest(prev => prev ? { ...prev, status: 'completed', results: fullResults } : null);
      success('Test abgeschlossen', fullResults.accuracy !== null ? `Accuracy: ${fullResults.accuracy.toFixed(2)}%` : 'Fertig');
    }));

    unlisteners.push(listen('test-single-complete', (event: any) => {
      const data = event.payload.data;
      const result: SingleResult = {
        task_type: data?.task_type || '',
        input: data?.input || singleInput,
        input_type: data?.input_type || singleInputType,
        result: data?.result || {},
      };
      setSingleResult(result);
      setSingleHistory(prev => [result, ...prev.slice(0, 9)]);
      setSingleRunning(false);
      success('Inferenz abgeschlossen', `${(result.result.inference_time * 1000).toFixed(0)}ms`);
    }));

    unlisteners.push(listen('test-error', (event: any) => {
      const err = event.payload.error || event.payload.data?.error || 'Unbekannter Fehler';
      setCurrentTest(prev => prev ? { ...prev, status: 'failed', error: err } : null);
      setSingleRunning(false);
      notifyError('Test fehlgeschlagen', err);
    }));

    unlisteners.push(listen('test-finished', () => {
      setCurrentTest(prev => {
        if (!prev) return null;
        if (prev.status === 'running' || prev.status === 'pending') {
          return { ...prev, status: 'completed', completed_at: new Date().toISOString() };
        }
        return prev;
      });
      setSingleRunning(false);
    }));

    unlisteners.push(listen('test-done', () => {
      setSingleRunning(false);
    }));

    return () => { unlisteners.forEach(u => u.then(fn => fn())); };
  };

  // ============ Actions ============

  const startDatasetTest = async () => {
    if (!selectedModelId || !selectedVersionId || !selectedDatasetId) {
      notifyError('Auswahl unvollständig', 'Bitte Modell, Version und Dataset wählen');
      return;
    }
    try {
      const model = modelsWithVersions.find(m => m.id === selectedModelId);
      const version = model?.versions.find(v => v.id === selectedVersionId);
      const dataset = datasets.find(d => d.id === selectedDatasetId);

      const job = await invoke<TestJob>('start_test', {
        modelId: selectedModelId,
        modelName: model?.name || '',
        versionId: selectedVersionId,
        versionName: version?.name || '',
        datasetId: selectedDatasetId,
        datasetName: dataset?.name || '',
        batchSize: 8,
        maxSamples: null,
      });

      setCurrentTest(job);
      setTestResults(null);
      info('Test gestartet', 'Dataset wird verarbeitet…');
    } catch (err: any) {
      notifyError('Fehler beim Starten', String(err));
    }
  };

  const stopTest = async () => {
    try {
      await invoke('stop_test');
      setCurrentTest(prev => prev ? { ...prev, status: 'stopped', completed_at: new Date().toISOString() } : null);
      success('Test gestoppt', '');
    } catch (err: any) {
      notifyError('Fehler beim Stoppen', String(err));
    }
  };

  const runSingleTest = async () => {
    if (!selectedVersionId || !singleInput.trim()) {
      notifyError('Eingabe fehlt', 'Bitte Version auswählen und Input eingeben');
      return;
    }
    setSingleRunning(true);
    setSingleResult(null);
    try {
      await invoke('test_single_input', {
        versionId: selectedVersionId,
        singleInput: singleInput.trim(),
        singleInputType: singleInputType,
      });
    } catch (err: any) {
      setSingleRunning(false);
      notifyError('Fehler', String(err));
    }
  };

  const exportHardExamples = async (format: string) => {
    if (!testResults) return;
    const hard = testResults.predictions.filter(p => !p.is_correct);
    if (hard.length === 0) { info('Keine Hard-Examples', 'Alle Predictions waren korrekt'); return; }
    try {
      const path = await invoke<string>('export_hard_examples', { predictions: hard, format });
      success('Exportiert', `${hard.length} Hard-Examples → ${path}`);
    } catch (err: any) {
      notifyError('Export fehlgeschlagen', String(err));
    }
  };

  // ============ Derived ============

  const selectedModel = modelsWithVersions.find(m => m.id === selectedModelId);
  const selectedVersion = selectedModel?.versions.find(v => v.id === selectedVersionId);
  const isRunning = currentTest?.status === 'running' || currentTest?.status === 'pending';

  const filteredPredictions = testResults?.predictions.filter(p => {
    if (showOnlyIncorrect && p.is_correct) return false;
    const q = searchQuery.toLowerCase();
    if (q && !p.input_text.toLowerCase().includes(q) && !(p.predicted_output || '').toLowerCase().includes(q)) return false;
    return true;
  }) || [];

  const taskType = currentTest?.task_type || testResults?.task_type || '';

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-8 h-8 text-gray-400 animate-spin" />
      </div>
    );
  }

  // ============ Render ============

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white">Test</h1>
        <p className="text-gray-400 mt-1">Teste jedes KI-Modell — Text, Bild, Audio, Detection, Tabular</p>
      </div>

      {/* Mode Tabs */}
      <div className="flex gap-2 bg-white/5 rounded-xl p-1 w-fit border border-white/10">
        {([
          { key: 'dataset', label: 'Dataset-Test', icon: <HardDrive className="w-4 h-4" /> },
          { key: 'single',  label: 'Einzeltest',   icon: <Send className="w-4 h-4" /> },
        ] as const).map(tab => (
          <button
            key={tab.key}
            onClick={() => setTestMode(tab.key)}
            className={`flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-medium transition-all ${
              testMode === tab.key
                ? `bg-gradient-to-r ${currentTheme.colors.gradient} text-white shadow`
                : 'text-gray-400 hover:text-white'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Model & Version Selection (always visible) */}
      <div className="bg-white/5 rounded-xl border border-white/10 p-6">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Layers className="w-5 h-5" />
          Modell & Version
        </h2>
        <div className="grid grid-cols-2 gap-4">
          {/* Model */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">Root-Modell</label>
            <div className="relative">
              <select
                value={selectedModelId || ''}
                onChange={e => setSelectedModelId(e.target.value)}
                disabled={isRunning || singleRunning}
                className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white appearance-none cursor-pointer focus:outline-none disabled:opacity-50"
              >
                {modelsWithVersions.map(m => (
                  <option key={m.id} value={m.id} className="bg-slate-800">{m.name}</option>
                ))}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
            </div>
          </div>

          {/* Version */}
          <div>
            <label className="block text-sm text-gray-400 mb-2 flex items-center gap-1">
              <GitBranch className="w-4 h-4" />
              Version
            </label>
            <div className="relative">
              <select
                value={selectedVersionId || ''}
                onChange={e => setSelectedVersionId(e.target.value)}
                disabled={!selectedModel || isRunning || singleRunning}
                className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white appearance-none cursor-pointer focus:outline-none disabled:opacity-50"
              >
                {selectedModel?.versions.map(v => (
                  <option key={v.id} value={v.id} className="bg-slate-800">
                    {v.is_root ? '⭐ ' : ''}{v.name}
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
            </div>
          </div>
        </div>
      </div>

      {/* ==================== DATASET MODE ==================== */}
      {testMode === 'dataset' && (
        <div className="space-y-6">
          {/* Dataset auswählen */}
          <div className="bg-white/5 rounded-xl border border-white/10 p-6">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <HardDrive className="w-5 h-5" />
              Test-Dataset
            </h2>
            <div className="relative">
              <select
                value={selectedDatasetId || ''}
                onChange={e => setSelectedDatasetId(e.target.value)}
                disabled={datasets.length === 0 || isRunning}
                className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white appearance-none cursor-pointer focus:outline-none disabled:opacity-50"
              >
                {datasets.map(ds => (
                  <option key={ds.id} value={ds.id} className="bg-slate-800">
                    {ds.name} ({ds.file_count} Dateien, {formatBytes(ds.size_bytes)})
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
            </div>
            {datasets.length === 0 && (
              <div className="mt-3 p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-amber-200">
                  Kein Test-Dataset verfügbar. Splitte einen Datensatz mit Test-Anteil &gt; 0%.
                </p>
              </div>
            )}

            {/* Start/Stop */}
            <div className="mt-5 flex items-center gap-4">
              {!isRunning ? (
                <button
                  onClick={startDatasetTest}
                  disabled={!selectedModelId || !selectedVersionId || !selectedDatasetId}
                  className={`flex items-center gap-2 px-6 py-3 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white font-medium hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all`}
                >
                  <Play className="w-5 h-5" />
                  Test starten
                </button>
              ) : (
                <button
                  onClick={stopTest}
                  className="flex items-center gap-2 px-6 py-3 bg-red-500 hover:bg-red-600 rounded-lg text-white font-medium transition-all"
                >
                  <Square className="w-5 h-5" />
                  Stoppen
                </button>
              )}
              {currentTest && (
                <span className="text-sm text-gray-400">
                  Status: <span className="text-white font-medium capitalize">{currentTest.status}</span>
                  {taskType && (
                    <span className="ml-3 flex items-center gap-1 inline-flex">
                      {getModalityIcon(taskType)}
                      <span className="text-gray-300">{getModalityLabel(taskType)}</span>
                    </span>
                  )}
                </span>
              )}
            </div>
          </div>

          {/* Progress */}
          {isRunning && currentTest?.progress && (
            <div className="bg-gradient-to-br from-purple-500/10 to-blue-500/10 rounded-xl border border-purple-500/30 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <Loader2 className="w-5 h-5 animate-spin text-purple-400" />
                  {currentTest.status === 'pending' ? 'Wird vorbereitet…' : 'Test läuft…'}
                </h3>
                <span className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                  {(currentTest.progress.progress_percent || 0).toFixed(1)}%
                </span>
              </div>

              <div className="relative h-4 bg-white/10 rounded-full overflow-hidden mb-4">
                <div
                  className="h-full bg-gradient-to-r from-purple-500 via-pink-500 to-blue-500 transition-all duration-300"
                  style={{ width: `${Math.min(currentTest.progress.progress_percent || 0, 100)}%` }}
                />
                <div className="absolute inset-0 flex items-center justify-center text-xs font-medium text-white">
                  {currentTest.progress.current_sample} / {currentTest.progress.total_samples || '?'} Samples
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 text-sm">
                {[
                  { label: 'Samples', value: `${currentTest.progress.current_sample} / ${currentTest.progress.total_samples || '?'}` },
                  { label: 'Geschwindigkeit', value: `${(currentTest.progress.samples_per_second || 0).toFixed(2)} S/s` },
                  { label: 'Verbleibend', value: currentTest.progress.estimated_time_remaining ? formatDuration(currentTest.progress.estimated_time_remaining) : 'Berechne…' },
                ].map((s, i) => (
                  <div key={i} className="bg-white/5 rounded-lg p-3">
                    <div className="text-gray-400 mb-1 text-xs">{s.label}</div>
                    <div className="text-white font-medium">{s.value}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Ergebnisse */}
          {testResults && (
            <div className="space-y-6">
              {/* Metriken */}
              <MetricsGrid results={testResults} taskType={testResults.task_type} />

              {/* Hard-Examples Hinweis */}
              {testResults.hard_examples_file && (
                <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-4 flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-amber-200 font-medium text-sm">Hard-Examples gespeichert</p>
                    <p className="text-amber-300/60 text-xs mt-0.5 break-all">{testResults.hard_examples_file}</p>
                  </div>
                </div>
              )}

              {/* Predictions-Tabelle */}
              {testResults.predictions.length > 0 && (
                <div className="bg-white/5 rounded-xl border border-white/10 p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-white">Predictions</h3>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => exportHardExamples('json')}
                        className="flex items-center gap-1.5 px-3 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-white transition-all"
                      >
                        <Download className="w-4 h-4" /> JSON
                      </button>
                      <button
                        onClick={() => exportHardExamples('csv')}
                        className="flex items-center gap-1.5 px-3 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-white transition-all"
                      >
                        <Download className="w-4 h-4" /> CSV
                      </button>
                      <button
                        onClick={() => exportHardExamples('txt')}
                        className="flex items-center gap-1.5 px-3 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-white transition-all"
                      >
                        <Download className="w-4 h-4" /> TXT
                      </button>
                    </div>
                  </div>

                  {/* Filter */}
                  <div className="flex items-center gap-4 mb-4 flex-wrap">
                    <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={showOnlyIncorrect}
                        onChange={e => setShowOnlyIncorrect(e.target.checked)}
                        className="rounded"
                      />
                      Nur falsche Predictions
                    </label>
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={e => setSearchQuery(e.target.value)}
                      placeholder="Suchen…"
                      className="flex-1 min-w-40 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 text-sm"
                    />
                    <span className="text-sm text-gray-400">
                      {filteredPredictions.length} / {testResults.predictions.length}
                    </span>
                  </div>

                  {/* Liste */}
                  <div className="space-y-3 max-h-[500px] overflow-y-auto pr-1">
                    {filteredPredictions.map(pred => (
                      <div
                        key={pred.sample_id}
                        className={`p-4 rounded-lg border ${
                          pred.is_correct
                            ? 'bg-green-500/5 border-green-500/20'
                            : 'bg-red-500/5 border-red-500/20'
                        }`}
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center gap-2">
                            {pred.is_correct
                              ? <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0" />
                              : <XCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
                            }
                            <span className="text-xs text-gray-500">#{pred.sample_id}</span>
                          </div>
                          <div className="flex gap-3 text-xs text-gray-500">
                            {pred.loss !== null && pred.loss !== undefined && (
                              <span>Loss: {pred.loss.toFixed(4)}</span>
                            )}
                            {pred.confidence !== null && pred.confidence !== undefined && (
                              <span>Conf: {(pred.confidence * 100).toFixed(1)}%</span>
                            )}
                            {pred.wer !== null && pred.wer !== undefined && (
                              <span>WER: {(pred.wer * 100).toFixed(1)}%</span>
                            )}
                            <span>{(pred.inference_time * 1000).toFixed(0)}ms</span>
                          </div>
                        </div>

                        <div className="space-y-1.5 text-sm">
                          {pred.input_text && (
                            <div>
                              <span className="text-gray-400">Input: </span>
                              <span className="text-white">{pred.input_text.slice(0, 200)}</span>
                            </div>
                          )}
                          {pred.input_path && (
                            <div>
                              <span className="text-gray-400">Datei: </span>
                              <span className="text-gray-300 text-xs">{pred.input_path.split('/').pop()}</span>
                            </div>
                          )}
                          {pred.expected_output !== null && pred.expected_output !== undefined && (
                            <div>
                              <span className="text-gray-400">Erwartet: </span>
                              <span className="text-white">{String(pred.expected_output).slice(0, 200)}</span>
                            </div>
                          )}
                          <div>
                            <span className="text-gray-400">Vorhersage: </span>
                            <span className={pred.is_correct ? 'text-green-400' : 'text-red-400'}>
                              {pred.predicted_output?.slice(0, 200) || '—'}
                            </span>
                          </div>
                          {pred.error_type && (
                            <div className="text-red-400 text-xs">Fehler: {pred.error_type}</div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Empty state */}
          {!currentTest && !testResults && (
            <div className="bg-white/5 rounded-2xl border border-white/10 p-12 text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-white/5 mb-4">
                <HardDrive className="w-8 h-8 text-gray-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Kein Dataset-Test aktiv</h3>
              <p className="text-gray-400 text-sm max-w-sm mx-auto">
                Wähle Modell, Version und Dataset aus und starte den Test um Metriken und Hard-Examples zu erhalten.
              </p>
            </div>
          )}
        </div>
      )}

      {/* ==================== SINGLE MODE ==================== */}
      {testMode === 'single' && (
        <div className="space-y-6">
          {/* Input Card */}
          <div className="bg-white/5 rounded-xl border border-white/10 p-6">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Send className="w-5 h-5" />
              Einzeltest-Eingabe
            </h2>

            {/* Input Typ */}
            <div className="flex gap-2 mb-4 flex-wrap">
              {([
                { key: 'text',        label: 'Text',        icon: <MessageSquare className="w-3.5 h-3.5" /> },
                { key: 'image_path',  label: 'Bildpfad',    icon: <Image className="w-3.5 h-3.5" /> },
                { key: 'audio_path',  label: 'Audiopfad',   icon: <Mic className="w-3.5 h-3.5" /> },
                { key: 'json',        label: 'JSON',        icon: <Table2 className="w-3.5 h-3.5" /> },
              ] as const).map(t => (
                <button
                  key={t.key}
                  onClick={() => setSingleInputType(t.key)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
                    singleInputType === t.key
                      ? `bg-gradient-to-r ${currentTheme.colors.gradient} text-white border-transparent`
                      : 'text-gray-400 border-white/10 hover:text-white hover:border-white/20'
                  }`}
                >
                  {t.icon}
                  {t.label}
                </button>
              ))}
            </div>

            {/* Eingabe-Feld */}
            <textarea
              value={singleInput}
              onChange={e => setSingleInput(e.target.value)}
              placeholder={getSingleInputPlaceholder(taskType, singleInputType)}
              disabled={singleRunning}
              rows={singleInputType === 'text' ? 4 : 2}
              className="w-full px-4 py-3 bg-black/20 border border-white/10 rounded-lg text-white placeholder-gray-500 text-sm resize-none focus:outline-none focus:border-white/20 disabled:opacity-50 font-mono"
              onKeyDown={e => {
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && !singleRunning) {
                  runSingleTest();
                }
              }}
            />
            <p className="text-xs text-gray-600 mt-1">Cmd/Ctrl+Enter zum Ausführen</p>

            {/* Run Button */}
            <div className="mt-4 flex items-center gap-3">
              <button
                onClick={runSingleTest}
                disabled={singleRunning || !selectedVersionId || !singleInput.trim()}
                className={`flex items-center gap-2 px-6 py-3 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white font-medium hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all`}
              >
                {singleRunning
                  ? <><Loader2 className="w-5 h-5 animate-spin" /> Läuft…</>
                  : <><Send className="w-5 h-5" /> Testen</>
                }
              </button>
              {singleResult && (
                <button
                  onClick={() => { setSingleResult(null); setSingleInput(''); }}
                  className="flex items-center gap-2 px-4 py-3 bg-white/5 hover:bg-white/10 rounded-lg text-gray-400 hover:text-white text-sm transition-all"
                >
                  <RotateCcw className="w-4 h-4" /> Zurücksetzen
                </button>
              )}
            </div>
          </div>

          {/* Ergebnis */}
          {singleResult && (
            <div className="bg-gradient-to-br from-green-500/5 to-blue-500/5 rounded-xl border border-green-500/20 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-400" />
                  Ergebnis
                  <span className="ml-2 px-2 py-0.5 rounded-full bg-white/10 text-xs text-gray-300 flex items-center gap-1">
                    {getModalityIcon(singleResult.task_type)}
                    {getModalityLabel(singleResult.task_type)}
                  </span>
                </h3>
                <span className="text-xs text-gray-500">
                  {singleResult.result.inference_time
                    ? `${(singleResult.result.inference_time * 1000).toFixed(0)}ms`
                    : ''}
                </span>
              </div>
              <SingleResultView result={singleResult.result} taskType={singleResult.task_type} />
            </div>
          )}

          {/* Historie */}
          {singleHistory.length > 1 && (
            <div className="bg-white/5 rounded-xl border border-white/10 p-6">
              <h3 className="text-sm font-semibold text-gray-400 mb-3">Verlauf (letzte {singleHistory.length - 1})</h3>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {singleHistory.slice(1).map((r, i) => (
                  <button
                    key={i}
                    onClick={() => { setSingleResult(r); setSingleInput(r.input); }}
                    className="w-full text-left p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-all"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-300 truncate max-w-xs">
                        {r.input.slice(0, 60)}{r.input.length > 60 ? '…' : ''}
                      </span>
                      <ChevronRight className="w-4 h-4 text-gray-500 flex-shrink-0" />
                    </div>
                    <div className="text-xs text-gray-500 mt-0.5 flex items-center gap-1">
                      {getModalityIcon(r.task_type)}
                      {getModalityLabel(r.task_type)}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Empty state */}
          {!singleRunning && !singleResult && (
            <div className="bg-white/5 rounded-2xl border border-white/10 p-10 text-center">
              <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-white/5 mb-4">
                <Send className="w-7 h-7 text-gray-400" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Bereit zum Testen</h3>
              <p className="text-gray-400 text-sm">
                Gib Text, Bildpfad, Audiopfad oder JSON-Features ein und teste das Modell direkt.
              </p>
              <div className="mt-4 flex flex-wrap gap-2 justify-center">
                {['NLP / Text', 'Vision', 'Audio / ASR', 'Detection', 'Tabular'].map(label => (
                  <span key={label} className="px-3 py-1 bg-white/5 border border-white/10 rounded-full text-xs text-gray-400">
                    {label}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
