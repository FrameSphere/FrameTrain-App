// TestPanel.tsx – Plugin-basiertes Testing mit nativer Modellauswahl + Dev Test Mode

import { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Loader2, CheckCircle, AlertTriangle, Layers, Play, Code2, Ban } from 'lucide-react';
import { detectPlugin } from '../plugins/registry';
import type { ModelPlugin, DatasetInfo } from '../plugins/types';
import DevTestPanel from './DevTestPanel';
import { usePageContext } from '../contexts/PageContext';
import { useLanguage } from '../contexts/LanguageContext';

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

export default function TestPanel({ userData }: { userData?: { userId: string; email: string; apiKey: string; password: string } }) {
  const { setCurrentPageContent } = usePageContext();
  const { t } = useLanguage();
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

  // ── AI Coach Seitenkontext ────────────────────────────────────────────────
  useEffect(() => {
    const selectedModel = models.find(m => m.id === selectedModelId);
    const selectedTree  = modelsWithVersions.find(m => m.id === selectedModelId);
    const selectedVer   = selectedTree?.versions.find(v => v.id === selectedVersionId);
    const detectedPlugin = selectedModel
      ? detectPlugin(selectedModel.source_path ?? selectedModel.name, selectedModel.model_type ? { model_type: selectedModel.model_type } : undefined)
      : null;
    const pluginName = detectedPlugin?.supported ? detectedPlugin.plugin.name : null;

    const lines: string[] = [
      '=== FrameTrain – Tests-Seite ===',
      '',
      'Zweck: Fertig trainierte Modell-Versionen mit echten Eingaben testen.',
      'Zwei Modi: Test Engine (Plugin-basiert, direkte Inference) oder Dev Test Mode (eigenes Python-Skript).',
      '',
      '--- Aktueller Zustand ---',
      `Modus: ${mode === 'test' ? 'Test Engine' : 'Dev Test Mode'}`,
    ];

    if (selectedModel) {
      lines.push(`Gewähltes Modell: ${selectedModel.name} (Typ: ${selectedModel.model_type ?? 'unbekannt'}, Quelle: ${selectedModel.source})`);
    } else {
      lines.push('Gewähltes Modell: Noch keins ausgewählt');
    }

    if (selectedVer) {
      lines.push(`Gewählte Version: ${selectedVer.name} (v${selectedVer.version_number})`);
    } else {
      lines.push('Gewählte Version: Noch keine ausgewählt');
    }

    if (mode === 'test') {
      if (pluginName) {
        lines.push(`Erkanntes Plugin: ${pluginName}`);
        lines.push(`Plugin-Status: ${panelState.phase === 'ready' ? 'Geladen und aktiv' : panelState.phase === 'unsupported' ? 'Nicht unterstützt' : 'Noch nicht gestartet'}`);
      } else if (selectedModel) {
        lines.push('Plugin-Status: Kein Plugin erkannt – Modell nicht unterstützt für Test Engine');
      }
    }

    lines.push('');
    lines.push('--- TEST MODE ---');
    lines.push(`Mode: ${mode === 'test' ? '🚀 Test Engine' : '🐍 Dev (Custom-Skript)'}`);

    lines.push('');
    lines.push('--- UI LAYOUT ---');
    if (mode === 'test') {
      lines.push('**OBEN (Header):**');
      lines.push('  • [Modell Dropdown] (linke Seite)');
      lines.push('  • [Version Dropdown] (daneben)');
      lines.push('  • [Mode Toggle: Test/Dev] (rechts)');
      lines.push('');
      lines.push('**LINKS:**');
      lines.push('  • [Eingabefeld] (großes Text Input für Testtext)');
      lines.push('  • 🧪 [Test Button] (unter Input, grün wenn Model ready)');
      lines.push('');
      lines.push('**RECHTS:**');
      lines.push('  • Prediction Results: Top-5 Klassen mit Confidence %');
      lines.push('  • Confidence Bar Visualisierung');
      lines.push('  • Copy/Export Results Button');
    } else {
      lines.push('**LINKS:**');
      lines.push('  • Python-Skript Editor (große Fläche)');
      lines.push('  • Input-Feld (unten links)');
      lines.push('  • 🧪 [Ausführen Button]');
      lines.push('');
      lines.push('**RECHTS:**');
      lines.push('  • Output/Error Logs');
      lines.push('  • JSON Response Display');
    }

    lines.push('');
    lines.push('--- Verfügbare Modelle ---');
    if (models.length === 0) {
      lines.push('Noch keine Modelle vorhanden.');
    } else {
      models.slice(0, 5).forEach(m => lines.push(`• ${m.name} (${m.model_type ?? 'unbekannt'})${m.id === selectedModelId ? ' ← ausgewählt' : ''}`));
      if (models.length > 5) lines.push(`… und ${models.length - 5} weitere Modelle`);
    }

    lines.push('');
    lines.push('--- VERFÜGBARE AKTIONEN ---');
    if (mode === 'test') {
      lines.push('1. Öffne [Modell Dropdown] oben → wähle Modell');
      lines.push('2. Öffne [Version Dropdown] → wähle trainierte Version');
      lines.push('3. Tippe Text in [Eingabefeld] links');
      lines.push('4. Klick 🧪 [Test Button] oder Enter');
      lines.push('5. Lies Predictions rechts (Top-5 Klassen mit %)');
    } else {
      lines.push('1. Öffne [Modell Dropdown] + [Version Dropdown]');
      lines.push('2. Schreib Python-Code in linkem Editor (Input via stdin)');
      lines.push('3. Gib Testdaten in Input-Feld unten links');
      lines.push('4. Klick 🧪 [Ausführen Button]');
      lines.push('5. Sehe JSON Output rechts');
    }

    lines.push('');
    lines.push('--- Test Engine Hinweise ---');
    lines.push('Unterstützte Modelle: XLM-RoBERTa (Klassifikation).');
    lines.push('Für nicht unterstützte Modelle: Dev Test Mode mit eigenem Python-Skript verwenden.');
    lines.push('Im Dev Test Mode: Skript empfängt Eingabe via stdin, Ausgabe als JSON auf stdout.');

    setCurrentPageContent(lines.join('\n'));
  }, [selectedModelId, selectedVersionId, panelState, mode, models, modelsWithVersions]);

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
    const r = detectPlugin(key, selectedModel.model_type ? { model_type: selectedModel.model_type } : undefined);
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
          <h1 className="text-2xl font-bold text-white">{t('testPanel.title')}</h1>
          <p className="text-gray-400 mt-1">
            {mode === 'test'
              ? t('testPanel.subtitleEngine')
              : t('testPanel.subtitleDev')}
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
                ? <><Play className="w-3.5 h-3.5" /> {t('testPanel.modeEngine')}</>
                : <><Code2 className="w-3.5 h-3.5" /> {t('testPanel.modeDev')}</>}
            </button>
          ))}
        </div>
      </div>

      {/* Kein Modell vorhanden */}
      {models.length === 0 ? (
        <div className="rounded-2xl border border-white/10 bg-white/5 p-12 text-center space-y-3">
          <Layers className="w-10 h-10 text-gray-500 mx-auto" />
          <p className="text-white font-medium">{t('testPanel.noModel.title')}</p>
          <p className="text-gray-500 text-sm">{t('testPanel.noModel.description')}</p>
        </div>
      ) : (
        <>
          {/* ── Modellauswahl-Block (immer sichtbar) ── */}
          <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-4">

            {/* Modell + Version */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="block text-sm font-medium text-white">{t('testPanel.modelSelector.modelLabel')}</label>
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
                <label className="block text-sm font-medium text-white">{t('testPanel.modelSelector.versionLabel')}</label>
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
                            {v.name}{idx === 0 ? ' ' + t('testPanel.modelSelector.versionLatest') : ''}
                          </option>
                        ))
                    : <option value="">{t('testPanel.modelSelector.noVersions')}</option>
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
                    <span className="text-red-300 text-xs font-medium">{t('testPanel.pluginBadge.unsupportedTitle')}</span>
                    <span className="text-gray-500 text-xs ml-2">→</span>
                    <button
                      onClick={() => setMode('dev')}
                      className="ml-2 text-blue-300 text-xs font-medium hover:underline"
                    >
                      {t('testPanel.pluginBadge.devModeLink')}
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
                  {t('testPanel.startButton')}
                </button>
              ) : (
                <button
                  onClick={() => setPanelState({ phase: 'idle' })}
                  className="w-full flex items-center justify-center gap-2 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 text-xs transition-all"
                >
                  {t('testPanel.changeModelButton')}
                </button>
              )
            )}
          </div>

          {/* ── Dev Test Mode ── */}
          {mode === 'dev' && (
            <DevTestPanel
              userData={userData}
              modelInfo={selectedModel ?? null}
              selectedVersionPath={selectedVersionPath}
              datasets={datasets}
            />
          )}

          {/* ── Test Engine: Nicht unterstützt ── */}
          {mode === 'test' && panelState.phase === 'unsupported' && (
            <div className="flex items-start gap-4 p-5 rounded-2xl border border-red-500/30 bg-red-500/10">
              <Ban className="w-8 h-8 text-red-300 mt-0.5" />
              <div className="space-y-1">
                <p className="text-red-300 font-semibold">{t('testPanel.unsupported.title')}</p>
                <p className="text-gray-500 text-xs mt-2">
                  {(() => {
                    const desc = t('testPanel.unsupported.description');
                    const link = t('testPanel.unsupported.devModeLink');
                    const [before, after] = desc.split(link);
                    return <>{before}<button onClick={() => setMode('dev')} className="text-blue-300 font-medium hover:underline">{link}</button>{after}</>;
                  })()}
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
                  <span className="text-gray-400">{t('testPanel.pluginBanner.pluginLabel')}</span>
                  <span className="text-white font-medium">{panelState.plugin.name}</span>
                  <span className="text-gray-600">·</span>
                  <span className="text-gray-400 text-xs">{selectedVersionTree.name}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                  <span className="text-amber-300 text-xs font-medium">{t('testPanel.pluginBanner.activeLabel')}</span>
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
