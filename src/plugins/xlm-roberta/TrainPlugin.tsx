// XLM-RoBERTa – Training Plugin UI

import { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import type { TrainPluginProps } from '../types';

export default function XLMRobertaTrainPlugin({ modelPath, onNavigateToAnalysis }: TrainPluginProps) {
  const [datasetPath, setDatasetPath] = useState('');
  const [epochs, setEpochs] = useState(3);
  const [learningRate, setLearningRate] = useState('2e-5');
  const [batchSize, setBatchSize] = useState(16);
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleStartTraining = async () => {
    if (!datasetPath.trim()) {
      setError('Bitte wähle ein Dataset aus.');
      return;
    }
    setError(null);
    setIsStarting(true);
    try {
      const jobId = await invoke<string>('start_training', {
        modelPath,
        datasetPath,
        epochs,
        learningRate: parseFloat(learningRate),
        batchSize,
        modelType: 'xlm-roberta-sequence-classification',
      });
      onNavigateToAnalysis(jobId);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      setIsStarting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3 p-4 rounded-2xl border border-emerald-500/30 bg-emerald-500/10">
        <div className="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center text-xl">
          🏋️
        </div>
        <div>
          <p className="text-emerald-300 text-sm font-medium">XLM-RoBERTa · Keyword Recognition</p>
          <p className="text-gray-400 text-xs font-mono truncate max-w-md">{modelPath}</p>
        </div>
      </div>

      {/* Fehlermeldung */}
      {error && (
        <div className="p-4 rounded-xl border border-red-500/40 bg-red-500/10 text-red-300 text-sm">
          ⚠️ {error}
        </div>
      )}

      {/* Konfiguration */}
      <div className="grid gap-4">
        {/* Dataset */}
        <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-3">
          <label className="block text-white text-sm font-medium">Dataset-Pfad</label>
          <input
            type="text"
            value={datasetPath}
            onChange={(e) => setDatasetPath(e.target.value)}
            placeholder="/pfad/zu/dataset.csv  oder  hf://datasets/..."
            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-emerald-500/50"
          />
          <p className="text-gray-500 text-xs">
            CSV mit Spalten <code className="text-gray-400">text</code> und <code className="text-gray-400">label</code>
          </p>
        </div>

        {/* Hyperparameter */}
        <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-4">
          <p className="text-white text-sm font-medium">Hyperparameter</p>

          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-1.5">
              <label className="text-gray-400 text-xs">Epochs</label>
              <input
                type="number"
                min={1}
                max={100}
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
                className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-white text-sm focus:outline-none focus:border-emerald-500/50"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-gray-400 text-xs">Learning Rate</label>
              <input
                type="text"
                value={learningRate}
                onChange={(e) => setLearningRate(e.target.value)}
                className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-white text-sm focus:outline-none focus:border-emerald-500/50"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-gray-400 text-xs">Batch Size</label>
              <input
                type="number"
                min={1}
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-white text-sm focus:outline-none focus:border-emerald-500/50"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Start Button */}
      <button
        onClick={handleStartTraining}
        disabled={isStarting}
        className="w-full py-3 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/40 text-emerald-300 font-medium text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isStarting ? '⏳ Starte Training…' : '🚀 Training starten'}
      </button>
    </div>
  );
}
