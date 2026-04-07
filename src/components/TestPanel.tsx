import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import {
  Play,
  Square,
  Download,
  Filter,
  CheckCircle,
  XCircle,
  Clock,
  Zap,
  Target,
  TrendingUp,
  Loader2,
  ChevronDown,
  AlertTriangle,
  FileText,
  Layers,
  GitBranch,
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
  correct_predictions: number;
  incorrect_predictions: number;
  accuracy: number;
  average_loss: number;
  average_inference_time: number;
  predictions: PredictionResult[];
  metrics: Record<string, number>;
}

interface PredictionResult {
  sample_id: number;
  input_text: string;
  expected_output: string | null;
  predicted_output: string;
  is_correct: boolean;
  loss: number | null;
  confidence: number | null;
  inference_time: number;
  error_type: string | null;
}

// ============ Helper Functions ============

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
}

// ============ Main Component ============

export default function TestPanel() {
  const { currentTheme } = useTheme();
  const { success, error: notifyError, info } = useNotification();

  // Selection State
  const [modelsWithVersions, setModelsWithVersions] = useState<ModelWithVersionTree[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);

  // Test State
  const [currentTest, setCurrentTest] = useState<TestJob | null>(null);
  const [testResults, setTestResults] = useState<TestResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [testStartTime, setTestStartTime] = useState<number | null>(null);

  // Filter State
  const [showOnlyIncorrect, setShowOnlyIncorrect] = useState(false);
  const [minLoss, setMinLoss] = useState<number>(0);
  const [searchQuery, setSearchQuery] = useState('');

    // ============ Load Data ============
  useEffect(() => {
    loadInitialData();
    checkForActiveTest();
    setupEventListeners();
  }, []);

  useEffect(() => {
    if (selectedModelId) {
      loadDatasets(selectedModelId);
      
      const model = modelsWithVersions.find(m => m.id === selectedModelId);

      if (model && model.versions.length > 0) {

        // ✨ Neue Logik: neueste Version anhand version_number auswählen
        const sortedVersions = [...model.versions].sort(
          (a, b) => b.version_number - a.version_number
        );
        const newestVersion = sortedVersions[0];

        setSelectedVersionId(newestVersion?.id || null);
      }
    }
  }, [selectedModelId, modelsWithVersions]);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      const models = await invoke<ModelWithVersionTree[]>('list_models_with_version_tree');
      setModelsWithVersions(models);
      
      if (models.length > 0) {
        setSelectedModelId(models[0].id);
      }
    } catch (err: any) {
      notifyError('Fehler beim Laden', String(err));
    } finally {
      setLoading(false);
    }
  };

  const checkForActiveTest = async () => {
    try {
      // Check if there's an active test job
      const activeTest = await invoke<TestJob | null>('get_active_test_job');
      if (activeTest) {
        setCurrentTest(activeTest);
        if (activeTest.started_at) {
          setTestStartTime(new Date(activeTest.started_at).getTime());
        }
        if (activeTest.results) {
          setTestResults(activeTest.results);
        }
        info('Laufender Test gefunden', 'Der Test wird fortgesetzt...');
      }
    } catch (err: any) {
      // No active test or error - that's fine
      console.log('No active test found:', err);
    }
  };

  const loadDatasets = async (modelId: string) => {
    try {
      const ds = await invoke<Dataset[]>('list_test_datasets_for_model', { modelId });
      setDatasets(ds);
      if (ds.length > 0) {
        setSelectedDatasetId(ds[0].id);
      }
    } catch (err: any) {
      console.error('Error loading datasets:', err);
    }
  };

  const setupEventListeners = () => {
    const unlistenProgress = listen('test-progress', (event: any) => {
      const progressData = event.payload.data;
      setCurrentTest(prev => {
        if (!prev) return null;
        
        // Calculate estimated time remaining if not provided
        let estimatedTimeRemaining = progressData.estimated_time_remaining;
        
        if (!estimatedTimeRemaining && testStartTime && progressData.samples_per_second > 0) {
          const remaining = progressData.total_samples - progressData.current_sample;
          estimatedTimeRemaining = remaining / progressData.samples_per_second;
        }
        
        return {
          ...prev,
          progress: {
            ...progressData,
            estimated_time_remaining: estimatedTimeRemaining
          }
        };
      });
    });

    const unlistenComplete = listen('test-complete', async (event: any) => {
      console.log('[TestPanel] ========== TEST COMPLETE EVENT ==========');
      console.log('[TestPanel] Full event payload:', event.payload);
      const resultsSummary = event.payload.data;
      console.log('[TestPanel] Test complete event received', resultsSummary);
      console.log('[TestPanel] currentTest at event time:', currentTest);
      console.log('[TestPanel] event.payload.version_id:', event.payload.version_id);
      
      // CRITICAL FIX: Load full results with predictions from DB
      let fullResults: TestResults = {
        total_samples: resultsSummary.total_samples || 0,
        correct_predictions: resultsSummary.correct_predictions || 0,
        incorrect_predictions: resultsSummary.incorrect_predictions || 0,
        accuracy: resultsSummary.accuracy || 0,
        average_loss: resultsSummary.average_loss || 0,
        average_inference_time: resultsSummary.average_inference_time || 0,
        predictions: [],
        metrics: {
          samples_per_second: resultsSummary.samples_per_second || 0,
          ...resultsSummary.metrics
        }
      };
      
      // CRITICAL FIX: Get version_id from multiple sources
      let versionIdToUse = currentTest?.version_id;
      
      // Fallback: Try to get from event or state
      if (!versionIdToUse) {
        console.warn('[TestPanel] ⚠️ currentTest.version_id is null, checking alternatives...');
        // Check if version_id is in the event payload
        versionIdToUse = event.payload.version_id || event.payload.data?.version_id;
        console.log('[TestPanel] version_id from event:', versionIdToUse);
      }
      
      // Load predictions from database if we have a version_id
      if (versionIdToUse) {
        try {
          console.log('[TestPanel] Loading predictions from database for version:', versionIdToUse);
          const dbResults = await invoke<string[]>('get_test_results_for_version', {
            versionId: versionIdToUse
          });
          
          console.log('[TestPanel] DB query returned', dbResults?.length || 0, 'results');
          
          // Get the most recent result (first in array)
          if (dbResults && dbResults.length > 0) {
            const latestResultJson = dbResults[0];
            console.log('[TestPanel] Latest result JSON length:', latestResultJson?.length || 0);
            
            try {
              const parsedResult = JSON.parse(latestResultJson);
              console.log('[TestPanel] Parsed result:', {
                hasPredictions: !!parsedResult.predictions,
                predictionsCount: parsedResult.predictions?.length || 0,
                keys: Object.keys(parsedResult)
              });
              
              if (parsedResult.predictions && parsedResult.predictions.length > 0) {
                console.log('[TestPanel] ✅ Successfully loaded', parsedResult.predictions.length, 'predictions from DB');
                fullResults = {
                  ...fullResults,
                  predictions: parsedResult.predictions,
                  // Also update metrics if they exist in DB
                  accuracy: parsedResult.accuracy || fullResults.accuracy,
                  average_loss: parsedResult.average_loss || fullResults.average_loss,
                  average_inference_time: parsedResult.average_inference_time || fullResults.average_inference_time,
                  total_samples: parsedResult.total_samples || fullResults.total_samples,
                  correct_predictions: parsedResult.correct_predictions || fullResults.correct_predictions,
                  incorrect_predictions: parsedResult.incorrect_predictions || fullResults.incorrect_predictions
                };
              } else {
                console.warn('[TestPanel] ⚠️ DB result has no predictions array');
              }
            } catch (parseErr) {
              console.error('[TestPanel] Failed to parse DB result JSON:', parseErr);
            }
          } else {
            console.warn('[TestPanel] ⚠️ No results found in database');
          }
        } catch (err) {
          console.error('[TestPanel] Failed to load predictions from DB:', err);
        }
      } else {
        console.error('[TestPanel] ❌ CRITICAL: No version_id available from any source!');
        console.error('[TestPanel] currentTest:', currentTest);
        console.error('[TestPanel] event.payload:', event.payload);
      }
      
      console.log('[TestPanel] Final results:', {
        totalSamples: fullResults.total_samples,
        predictionsCount: fullResults.predictions.length,
        accuracy: fullResults.accuracy
      });
      
      setTestResults(fullResults);
      setCurrentTest(prev => prev ? {
        ...prev,
        status: 'completed',
        completed_at: new Date().toISOString(),
        results: fullResults
      } : null);
      
      // CRITICAL: Reset test start time to unlock UI
      setTestStartTime(null);
      
      success('Test abgeschlossen', `Accuracy: ${fullResults.accuracy.toFixed(2)}%`);
    });

    const unlistenError = listen('test-error', (event: any) => {
      setCurrentTest(prev => prev ? {
        ...prev,
        status: 'failed',
        error: event.payload.error || event.payload.data?.error
      } : null);
      setTestStartTime(null); // CRITICAL: Reset test start time
      notifyError('Test fehlgeschlagen', event.payload.error || 'Unbekannter Fehler');
    });

    // CRITICAL: Listen for test-finished event to ensure UI is fully unlocked
    const unlistenFinished = listen('test-finished', (event: any) => {
      console.log('[TestPanel] Test finished event received:', event.payload);
      // Ensure test is marked as completed/stopped
      setCurrentTest(prev => {
        if (!prev) return null;
        // Only update if status is still running/pending
        if (prev.status === 'running' || prev.status === 'pending') {
          return {
            ...prev,
            status: event.payload.success ? 'completed' : 'failed',
            completed_at: new Date().toISOString()
          };
        }
        return prev;
      });
      setTestStartTime(null);
    });

    // CRITICAL: Listen for done event - this is the final event that unlocks everything
    const unlistenDone = listen('test-done', (event: any) => {
      console.log('[TestPanel] Test DONE event received - UI fully unlocked');
      // Final state cleanup
      setTestStartTime(null);
      setCurrentTest(prev => {
        if (!prev) return null;
        if (prev.status === 'running' || prev.status === 'pending') {
          return {
            ...prev,
            status: 'completed',
            completed_at: new Date().toISOString()
          };
        }
        return prev;
      });
    });

    return () => {
      unlistenProgress.then(fn => fn());
      unlistenComplete.then(fn => fn());
      unlistenError.then(fn => fn());
      unlistenFinished.then(fn => fn());
      unlistenDone.then(fn => fn());
    };
  };

  // ============ Actions ============

  const startTest = async () => {
    if (!selectedModelId || !selectedVersionId || !selectedDatasetId) {
      notifyError('Auswahl unvollständig', 'Bitte wähle Modell, Version und Dataset');
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
      setTestStartTime(Date.now());
      info('Test gestartet', 'Model wird getestet...');
    } catch (err: any) {
      notifyError('Fehler beim Starten', String(err));
    }
  };

  const stopTest = async () => {
    try {
      await invoke('stop_test');
      // CRITICAL: Reset state immediately to unlock UI
      setTestStartTime(null);
      setCurrentTest(prev => prev ? {
        ...prev,
        status: 'stopped',
        completed_at: new Date().toISOString()
      } : null);
      success('Test gestoppt', 'Der Test wurde abgebrochen');
    } catch (err: any) {
      notifyError('Fehler beim Stoppen', String(err));
    }
  };

  const exportHardExamples = async (format: string) => {
    if (!testResults) return;

    // Filter hard examples
    const hardExamples = testResults.predictions.filter(p => 
      !p.is_correct || (p.loss && p.loss > minLoss)
    );

    if (hardExamples.length === 0) {
      info('Keine Beispiele', 'Keine schwierigen Beispiele gefunden');
      return;
    }

    try {
      const path = await invoke<string>('export_hard_examples', {
        predictions: hardExamples,
        format,
      });
      success('Exportiert', `${hardExamples.length} Beispiele gespeichert unter:\n${path}`);
    } catch (err: any) {
      notifyError('Export fehlgeschlagen', String(err));
    }
  };

  // ============ Filtered Predictions ============

  const filteredPredictions = testResults?.predictions.filter(p => {
    if (showOnlyIncorrect && p.is_correct) return false;
    if (minLoss > 0 && (!p.loss || p.loss < minLoss)) return false;
    if (searchQuery && !p.input_text.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  }) || [];

  // ============ Render ============

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-8 h-8 text-gray-400 animate-spin" />
      </div>
    );
  }

  const selectedModel = modelsWithVersions.find(m => m.id === selectedModelId);
  const selectedVersion = selectedModel?.versions.find(v => v.id === selectedVersionId);
  const selectedDataset = datasets.find(d => d.id === selectedDatasetId);

  const isRunning = currentTest?.status === 'running' || currentTest?.status === 'pending';

  // DEBUG: Log state changes
  console.log('[TestPanel] Render state:', {
    isRunning,
    currentTestStatus: currentTest?.status,
    hasResults: !!testResults,
    testStartTime
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white">Test</h1>
        <p className="text-gray-400 mt-1">Teste deine Modelle auf Evaluations-Datensätzen</p>
      </div>

      {/* Model & Version & Dataset Selection */}
      <div className="bg-white/5 rounded-xl border border-white/10 p-6">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Layers className="w-5 h-5" />
          Modell, Version & Dataset auswählen
        </h2>
        <div className="grid grid-cols-3 gap-4">
          {/* Model */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">Root-Modell</label>
            <div className="relative">
              <select
                value={selectedModelId || ''}
                onChange={(e) => setSelectedModelId(e.target.value)}
                disabled={isRunning}
                className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white appearance-none cursor-pointer focus:outline-none focus:ring-2 disabled:opacity-50 transition-all"
                style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
              >
                {modelsWithVersions.map(model => (
                  <option key={model.id} value={model.id} className="bg-slate-800">
                    {model.name}
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
            </div>
          </div>

          {/* Version */}
          <div>
            <label className="block text-sm text-gray-400 mb-2 flex items-center gap-2">
              <GitBranch className="w-4 h-4" />
              Version
            </label>
            <div className="relative">
              <select
                value={selectedVersionId || ''}
                onChange={(e) => setSelectedVersionId(e.target.value)}
                disabled={!selectedModel || selectedModel.versions.length === 0 || isRunning}
                className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white appearance-none cursor-pointer focus:outline-none focus:ring-2 disabled:opacity-50 transition-all"
                style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
              >
                {selectedModel?.versions.map(version => (
                  <option key={version.id} value={version.id} className="bg-slate-800">
                    {version.is_root ? '⭐ ' : ''}{version.name}
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
            </div>
          </div>

          {/* Dataset */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">Test-Dataset</label>
            <div className="relative">
              <select
                value={selectedDatasetId || ''}
                onChange={(e) => setSelectedDatasetId(e.target.value)}
                disabled={datasets.length === 0 || isRunning}
                className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white appearance-none cursor-pointer focus:outline-none focus:ring-2 disabled:opacity-50 transition-all"
                style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
              >
                {datasets.map(ds => (
                  <option key={ds.id} value={ds.id} className="bg-slate-800">
                    {ds.name} ({ds.file_count} files, {formatBytes(ds.size_bytes)})
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
            </div>
            {datasets.length === 0 && selectedModel && (
              <div className="mt-2 p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-amber-200">
                  Keine Test-Datasets verfügbar. Bitte splitte ein Dataset mit einem Test-Anteil {'>'} 0%.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Start/Stop Button */}
        <div className="mt-6 flex items-center gap-4">
          {!isRunning ? (
            <button
              onClick={startTest}
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
              Test stoppen
            </button>
          )}

          {currentTest && (
            <div className="text-sm text-gray-400">
              Status: <span className="text-white font-medium capitalize">{currentTest.status}</span>
            </div>
          )}
        </div>
      </div>

      {/* Progress - Enhanced with overlay effect - ONLY show when actually running */}
     {isRunning && currentTest && ['running', 'pending'].includes(currentTest.status) && currentTest.progress && (
        <div className="bg-gradient-to-br from-purple-500/10 to-blue-500/10 rounded-xl border border-purple-500/30 p-6 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              <Loader2 className="w-5 h-5 animate-spin text-purple-400" />
              {currentTest.status === 'pending' ? 'Test wird vorbereitet...' : 'Test läuft...'}
            </h3>
            <span className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
              {currentTest.progress?.progress_percent?.toFixed(1) || '0.0'}%
            </span>
          </div>

          {/* Enhanced Progress Bar */}
          <div className="relative h-4 bg-white/10 rounded-full overflow-hidden mb-4">
            <div
              className="h-full bg-gradient-to-r from-purple-500 via-pink-500 to-blue-500 transition-all duration-300 relative"
              style={{ width: `${Math.min(currentTest.progress?.progress_percent || 0, 100)}%` }}
            >
              {/* Shimmer effect - ONLY show when not at 100% */}
              {currentTest.progress?.progress_percent < 100 && (
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
              )}
            </div>
            {/* Progress text overlay */}
            <div className="absolute inset-0 flex items-center justify-center text-xs font-medium text-white">
              {currentTest.progress?.current_sample || 0} / {currentTest.progress?.total_samples || '?'} Samples
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4 text-sm">
            <div className="bg-white/5 rounded-lg p-3">
              <div className="text-gray-400 mb-1">Samples</div>
              <div className="text-white font-medium text-lg">
                {currentTest.progress?.current_sample || 0} / {currentTest.progress?.total_samples || '?'}
              </div>
            </div>
            <div className="bg-white/5 rounded-lg p-3">
              <div className="text-gray-400 mb-1 flex items-center gap-1">
                <Zap className="w-3 h-3" />
                Geschwindigkeit
              </div>
              <div className="text-white font-medium text-lg">
                {currentTest.progress?.samples_per_second?.toFixed(2) || '0.00'} samples/s
              </div>
            </div>
            <div className="bg-white/5 rounded-lg p-3">
              <div className="text-gray-400 mb-1 flex items-center gap-1">
                <Clock className="w-3 h-3" />
                Verbleibend
              </div>
              <div className="text-white font-medium text-lg">
                {currentTest.progress?.estimated_time_remaining
                  ? formatDuration(currentTest.progress.estimated_time_remaining)
                  : 'Berechne...'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {testResults && (
        <div className="space-y-6">
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-white/5 rounded-xl border border-white/10 p-5">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-gray-400">Accuracy</h3>
                <Target className="w-5 h-5 text-green-400" />
              </div>
              <div className="text-3xl font-bold text-white mb-1">
                {testResults.accuracy.toFixed(2)}%
              </div>
              <div className="text-xs text-gray-500">
                {testResults.correct_predictions} / {testResults.total_samples} korrekt
              </div>
            </div>

            <div className="bg-white/5 rounded-xl border border-white/10 p-5">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-gray-400">Avg. Loss</h3>
                <TrendingUp className="w-5 h-5 text-blue-400" />
              </div>
              <div className="text-3xl font-bold text-white mb-1">
                {testResults.average_loss.toFixed(4)}
              </div>
              <div className="text-xs text-gray-500">Durchschnittlicher Loss</div>
            </div>

            <div className="bg-white/5 rounded-xl border border-white/10 p-5">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-gray-400">Inferenz-Zeit</h3>
                <Clock className="w-5 h-5 text-purple-400" />
              </div>
              <div className="text-3xl font-bold text-white mb-1">
                {(testResults.average_inference_time * 1000).toFixed(0)}ms
              </div>
              <div className="text-xs text-gray-500">Pro Sample</div>
            </div>

            <div className="bg-white/5 rounded-xl border border-white/10 p-5">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-gray-400">Geschwindigkeit</h3>
                <Zap className="w-5 h-5 text-yellow-400" />
              </div>
              <div className="text-3xl font-bold text-white mb-1">
                {testResults.metrics.samples_per_second?.toFixed(1) || '0'}
              </div>
              <div className="text-xs text-gray-500">Samples/Sekunde</div>
            </div>
          </div>

          {/* Export & Filter Controls */}
          <div className="bg-white/5 rounded-xl border border-white/10 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Predictions</h3>
              <div className="flex items-center gap-3">
                {/* Export Buttons - Disabled during test */}
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => exportHardExamples('json')}
                    disabled={isRunning}
                    className="flex items-center gap-2 px-3 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-white transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Download className="w-4 h-4" />
                    JSON
                  </button>
                  <button
                    onClick={() => exportHardExamples('csv')}
                    disabled={isRunning}
                    className="flex items-center gap-2 px-3 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-white transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Download className="w-4 h-4" />
                    CSV
                  </button>
                  <button
                    onClick={() => exportHardExamples('txt')}
                    disabled={isRunning}
                    className="flex items-center gap-2 px-3 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-white transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Download className="w-4 h-4" />
                    TXT
                  </button>
                </div>
              </div>
            </div>

            {/* Filters - Disabled during test */}
            <div className="flex items-center gap-4 mb-4">
              <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showOnlyIncorrect}
                  onChange={(e) => setShowOnlyIncorrect(e.target.checked)}
                  disabled={isRunning}
                  className="rounded disabled:opacity-50 disabled:cursor-not-allowed"
                />
                <span className={isRunning ? 'opacity-50' : ''}>Nur falsche Predictions</span>
              </label>

              <div className="flex items-center gap-2">
                <label className={`text-sm text-gray-400 ${isRunning ? 'opacity-50' : ''}`}>Min Loss:</label>
                <input
                  type="number"
                  value={minLoss}
                  onChange={(e) => setMinLoss(parseFloat(e.target.value) || 0)}
                  step="0.1"
                  min="0"
                  disabled={isRunning}
                  className="w-24 px-3 py-1 bg-white/5 border border-white/10 rounded-lg text-white text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                />
              </div>

              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Suche in Input..."
                disabled={isRunning}
                className="flex-1 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 text-sm disabled:opacity-50 disabled:cursor-not-allowed"
              />

              <span className="text-sm text-gray-400">
                {filteredPredictions.length} / {testResults.predictions.length}
              </span>
            </div>

            {/* Predictions List */}
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {filteredPredictions.map((pred) => (
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
                      {pred.is_correct ? (
                        <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0" />
                      ) : (
                        <XCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                      )}
                      <span className="text-sm text-gray-400">Sample #{pred.sample_id}</span>
                    </div>
                    <div className="flex items-center gap-3 text-xs text-gray-500">
                      {pred.loss && (
                        <span>Loss: {pred.loss.toFixed(4)}</span>
                      )}
                      <span>{(pred.inference_time * 1000).toFixed(0)}ms</span>
                    </div>
                  </div>

                  <div className="space-y-2 text-sm">
                    <div>
                      <span className="text-gray-400">Input: </span>
                      <span className="text-white">{pred.input_text}</span>
                    </div>
                    {pred.expected_output && (
                      <div>
                        <span className="text-gray-400">Expected: </span>
                        <span className="text-white">{pred.expected_output}</span>
                      </div>
                    )}
                    <div>
                      <span className="text-gray-400">Predicted: </span>
                      <span className={pred.is_correct ? 'text-green-400' : 'text-red-400'}>
                        {pred.predicted_output}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Empty State */}
      {!currentTest && !testResults && (
        <div className="bg-white/5 rounded-2xl border border-white/10 p-12 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-white/5 mb-4">
            <FileText className="w-8 h-8 text-gray-400" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">Kein Test aktiv</h3>
          <p className="text-gray-400">
            Wähle ein Modell, eine Version und einen Dataset aus und starte einen Test.
          </p>
        </div>
      )}
    </div>
  );
}
