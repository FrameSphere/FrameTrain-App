// TestPanel.tsx – Plugin-basiertes Testing mit nativer Modellauswahl + Dev Test Mode

import { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Loader2, CheckCircle, AlertTriangle, Layers, Play, Code2 } from 'lucide-react';
import { detectPlugin } from '../plugins/registry';
import type { ModelPlugin, DatasetInfo } from '../plugins/types';
import DevTestPanel from './DevTestPanel';

// ── Types (analog zu TrainingPanel) ───────────────────────────────────────

interface ModelInfo {
  id: string; name: string; source: string;
  source_path: string | null; local_path: string;
  model_type: string | null; size_bytes?: number;
}

interface VersionTreeItem {
  id: string; name: string; is_root: boolean; version_number: number;
}

interface ModelWithVersionTree {
  id: string; name: string; versions: VersionTreeItem[];
}

// ── Panel-State ────────────────────────────────────────────────────────────

type ReadyState =
  | { phase: 'idle' }
  | { phase: 'unsupported'; reason: string }
  | { phase: 'ready'; plugin: ModelPlugin };

// ── Hauptkomponente ────────────────────────────────────────────────────────

export default function TestPanel() {
  const [loadingData, setLoadingData] = useState(true);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsWithVersions, setModelsWithVersions] = useState<ModelWithVersionTree[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);

  const [selectedModelId, setSelectedModelId]   = useState<string | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [selectedVersionPath, setSelectedVersionPath] = useState<string>('');

  const [panelState, setPanelState] = useState<ReadyState>({ phase: 'idle' });

  // ── Mode Toggle ─────────────────────────────────────────────────────────

  const [mode, setMode] = useState<'test' | 'dev'>('test');

  // ── Initialer Load ──────────────────────────────────────────────────────

  useEffect(() => {
    (async () => {
      setLoadingData(true);
      try {
        const [list, listWithVersions] = await Promise.all([
          invoke<ModelInfo[]>('list_models'),
          invoke<ModelWithVersionTree[]>('list_models_with_version_tree'),
        ]);
        setModels(list);
        setModelsWithVersions(listWithVersions);
        if (listWithVersions.length > 0) setSelectedModelId(listWithVersions[0].id);
      } catch (e) {
        console.error('[TestPanel] initLoad:', e);
      } finally {
        setLoadingData(false);
      }
    })();
  }, []);

  // ── Datasets laden wenn Modell wechselt ────────────────────────────────

  useEffect(() => {
    if (!selectedModelId) { setDatasets([]); return; }
    invoke<DatasetInfo[]>('list_datasets_for_model', { modelId: selectedModelId })
      .then(setDatasets)
      .catch(() => setDatasets([]));
  }, [selectedModelId]);

  // ── Versions-Sync ───────────────────────────────────────────────────────

  useEffect(() => {
    if (!selectedModelId) { setSelectedVersionId(null); return; }
    const m = modelsWithVersions.find(x => x.id === selectedModelId);
    if (!m?.versions.length) { setSelectedVersionId(null); return; }
    setSelectedVersionId(
      [...m.versions].sort((a, b) => b.version_number - a.version_number)[0].id
    );
    setPanelState({ phase: 'idle' });
  }, [selectedModelId, modelsWithVersions]);

  // Load version path when selectedVersionId changes (wie TrainingPanel)
  useEffect(() => {
    if (!selectedVersionId) { setSelectedVersionPath(''); return; }
    invoke<string>('get_version_path_for_ui', { versionId: selectedVersionId })
      .then(path => setSelectedVersionPath(path))
      .catch(() => setSelectedVersionPath(''));
  }, [selectedVersionId]);

  // ── Abgeleitete Werte ───────────────────────────────────────────────────

  const selectedModel      = models.find(m => m.id === selectedModelId);
  const selectedModelTree  = modelsWithVersions.find(m => m.id === selectedModelId);
  const selectedVersionTree = selectedModelTree?.versions.find(v => v.id === selectedVersionId);

  const detectedPlugin: ModelPlugin | null = (() => {
    if (!selectedModel) return null;
    const key = selectedModel.source_path ?? selectedModel.name;
    const r = detectPlugin(key);
    return r.supported ? r.plugin : null;
  })();

  // ── Test starten ────────────────────────────────────────────────────────

  const handleStartTest = useCallback(() => {
    if (!selectedModel || !detectedPlugin) return;
    setPanelState({ phase: 'ready', plugin: detectedPlugin });
  }, [selectedModel, detectedPlugin]);

  // ── Render ──────────────────────────────────────────────────────────────

  if (loadingData) {
    return (
      <div className="flex items-center justify-center py-24">
        <Loader2 className="w-8 h-8 text-gray-500 animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">

      {/* ── Header mit Mode-Toggle ── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Tests</h1>
          <p className="text-gray-400 mt-1">
            {mode === 'test'
              ? 'Wähle ein Modell – das passende Test-Plugin wird automatisch geladen'
              : 'Eigenes Python-Skript für Inference und Evaluation'}
          </p>
        </div>

        {/* Mode Toggle – identisch zum TrainingPanel */}
        <div className="flex items-center gap-1 p-1 rounded-xl bg-white/5 border border-white/10">
          {(['test', 'dev'] as const).map(m => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                mode === m
                  ? m === 'test'
                    ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                    : 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {m === 'test'
                ? <><Play className="w-3.5 h-3.5" /> Test Engine</>
                : <><Code2 className="w-3.5 h-3.5" /> Dev Test</>}
            </button>
          ))}
        </div>
      </div>

      {/* Kein Modell vorhanden */}
      {models.length === 0 ? (
        <div className="rounded-2xl border border-white/10 bg-white/5 p-12 text-center space-y-3">
          <Layers className="w-10 h-10 text-gray-500 mx-auto" />
          <p className="text-white font-medium">Kein Modell vorhanden</p>
          <p className="text-gray-500 text-sm">Füge zuerst ein Modell im Model-Manager hinzu.</p>
        </div>
      ) : (
        <>
          {/* ── Modellauswahl-Block (immer sichtbar) ── */}
          <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-4">

            {/* Modell + Version */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="block text-sm font-medium text-white">Modell</label>
                <select
                  value={selectedModelId ?? ''}
                  onChange={e => { setSelectedModelId(e.target.value); setPanelState({ phase: 'idle' }); }}
                  className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-amber-500/50 appearance-none transition-all"
                >
                  {modelsWithVersions.map(m => (
                    <option key={m.id} value={m.id} className="bg-slate-900">{m.name}</option>
                  ))}
                </select>
              </div>

              <div className="space-y-1.5">
                <label className="block text-sm font-medium text-white">Version</label>
                <select
                  value={selectedVersionId ?? ''}
                  onChange={e => { setSelectedVersionId(e.target.value); setPanelState({ phase: 'idle' }); }}
                  className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-amber-500/50 appearance-none transition-all"
                >
                  {selectedModelTree?.versions?.length
                    ? [...selectedModelTree.versions]
                        .sort((a, b) => b.version_number - a.version_number)
                        .map((v, idx) => (
                          <option key={v.id} value={v.id} className="bg-slate-900">
                            {v.name}{idx === 0 ? ' (neueste)' : ''}
                          </option>
                        ))
                    : <option value="">Keine Versionen</option>
                  }
                </select>
              </div>
            </div>

            {/* Support-Badge – nur im Test Engine Mode relevant */}
            {mode === 'test' && selectedModel && (
              detectedPlugin ? (
                <div className="flex items-center gap-2 px-3 py-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20">
                  <CheckCircle className="w-4 h-4 text-amber-400 flex-shrink-0" />
                  <div>
                    <span className="text-amber-300 text-xs font-medium">{detectedPlugin.name}</span>
                    <span className="text-gray-500 text-xs"> – {detectedPlugin.description}</span>
                  </div>
                </div>
              ) : (
                <div className="flex items-center gap-2 px-3 py-2.5 rounded-xl bg-red-500/10 border border-red-500/20">
                  <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <span className="text-red-300 text-xs font-medium">Modell wird für Tests noch nicht unterstützt</span>
                    <span className="text-gray-500 text-xs ml-2">→</span>
                    <button
                      onClick={() => setMode('dev')}
                      className="ml-2 text-blue-300 text-xs font-medium hover:underline"
                    >
                      Dev Test Mode →
                    </button>
                  </div>
                </div>
              )
            )}

            {/* Start / Reset – nur im Test Engine Mode */}
            {mode === 'test' && (
              panelState.phase !== 'ready' ? (
                <button
                  onClick={handleStartTest}
                  disabled={!selectedModel || !detectedPlugin || !selectedVersionId}
                  className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-300 font-semibold text-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Test starten →
                </button>
              ) : (
                <button
                  onClick={() => setPanelState({ phase: 'idle' })}
                  className="w-full flex items-center justify-center gap-2 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 text-xs transition-all"
                >
                  ← Anderes Modell / Version wählen
                </button>
              )
            )}
          </div>

          {/* ── Dev Test Mode ── */}
          {mode === 'dev' && (
            <DevTestPanel
              modelInfo={selectedModel ?? null}
              selectedVersionPath={selectedVersionPath}
              datasets={datasets}
            />
          )}

          {/* ── Test Engine: Nicht unterstützt ── */}
          {mode === 'test' && panelState.phase === 'unsupported' && (
            <div className="flex items-start gap-4 p-5 rounded-2xl border border-red-500/30 bg-red-500/10">
              <span className="text-3xl mt-0.5">🚫</span>
              <div className="space-y-1">
                <p className="text-red-300 font-semibold">Modell wird noch nicht unterstützt</p>
                <p className="text-gray-500 text-xs mt-2">
                  Aktuell wird Testing nur für{' '}
                  <span className="text-white">XLM-RoBERTa</span>{' '}
                  unterstützt. Nutze den <button onClick={() => setMode('dev')} className="text-blue-300 font-medium hover:underline">Dev Test Mode</button> für eigene Skripte.
                </p>
              </div>
            </div>
          )}

          {/* ── Test Engine: Plugin geladen → Test-Interface ── */}
          {mode === 'test' && panelState.phase === 'ready' && selectedModel && selectedVersionId && selectedVersionTree && (
            <div className="space-y-5">
              {/* Plugin-Banner */}
              <div className="flex items-center justify-between px-4 py-2.5 rounded-xl bg-white/5 border border-white/10">
                <div className="flex items-center gap-2 text-sm">
                  <span className="text-gray-400">Plugin:</span>
                  <span className="text-white font-medium">{panelState.plugin.name}</span>
                  <span className="text-gray-600">·</span>
                  <span className="text-gray-400 text-xs">{selectedVersionTree.name}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                  <span className="text-amber-300 text-xs font-medium">Aktiv</span>
                </div>
              </div>

              {/* Plugin-Komponente */}
              <panelState.plugin.TestComponent
                modelPath={selectedModel.local_path ?? selectedModel.source_path ?? selectedModel.name}
                versionId={selectedVersionId}
                modelId={selectedModel.id}
                modelName={selectedModel.name}
                versionName={selectedVersionTree.name}
                datasets={datasets}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}
