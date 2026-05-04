// ModelSelector – Eingabe + Modell-Erkennung
//
// Zeigt ein Eingabefeld für den Modellpfad / HuggingFace-ID.
// Führt die Plugin-Erkennung durch und gibt das Ergebnis zurück.

import { useState, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { detectPlugin, type DetectionResult } from '../plugins/registry';
import type { ModelConfig } from '../plugins/types';

interface ModelSelectorProps {
  onDetected: (result: DetectionResult, modelPath: string) => void;
  accentColor?: 'emerald' | 'amber';
}

type State =
  | { status: 'idle' }
  | { status: 'checking' }
  | { status: 'done'; result: DetectionResult; path: string };

export default function ModelSelector({ onDetected, accentColor = 'emerald' }: ModelSelectorProps) {
  const [inputValue, setInputValue] = useState('');
  const [state, setState] = useState<State>({ status: 'idle' });

  const accent = accentColor === 'amber'
    ? { border: 'border-amber-500/30', bg: 'bg-amber-500/10', text: 'text-amber-300', focusBorder: 'focus:border-amber-500/50', btn: 'bg-amber-500/20 hover:bg-amber-500/30 border-amber-500/40 text-amber-300' }
    : { border: 'border-emerald-500/30', bg: 'bg-emerald-500/10', text: 'text-emerald-300', focusBorder: 'focus:border-emerald-500/50', btn: 'bg-emerald-500/20 hover:bg-emerald-500/30 border-emerald-500/40 text-emerald-300' };

  const handleCheck = useCallback(async () => {
    const path = inputValue.trim();
    if (!path) return;

    setState({ status: 'checking' });

    // Versuche config.json zu lesen falls lokaler Pfad
    let configJson: ModelConfig | undefined;
    const isLocal = path.startsWith('/') || path.startsWith('~') || /^[A-Za-z]:\\/.test(path);
    if (isLocal) {
      try {
        const raw = await invoke<string>('read_model_config', { modelPath: path });
        configJson = JSON.parse(raw) as ModelConfig;
      } catch {
        // config.json nicht gefunden – Erkennung nur per Pfadname
      }
    }

    const result = detectPlugin(path, configJson);
    setState({ status: 'done', result, path });
    onDetected(result, path);
  }, [inputValue, onDetected]);

  const handleReset = () => {
    setInputValue('');
    setState({ status: 'idle' });
  };

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-4">
      <div className="space-y-1">
        <label className="block text-white text-sm font-medium">Modell</label>
        <p className="text-gray-500 text-xs">
          HuggingFace Model-ID (z.&nbsp;B.&nbsp;<code className="text-gray-400">xlm-roberta-base</code>) oder lokaler Pfad
        </p>
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => { setInputValue(e.target.value); setState({ status: 'idle' }); }}
          onKeyDown={(e) => { if (e.key === 'Enter') handleCheck(); }}
          placeholder="xlm-roberta-base  oder  /pfad/zum/modell"
          disabled={state.status === 'checking'}
          className={`flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-white text-sm placeholder:text-gray-600 focus:outline-none ${accent.focusBorder} disabled:opacity-50`}
        />
        {state.status === 'done' ? (
          <button
            onClick={handleReset}
            className="px-4 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 text-sm transition-all"
          >
            Ändern
          </button>
        ) : (
          <button
            onClick={handleCheck}
            disabled={!inputValue.trim() || state.status === 'checking'}
            className={`px-4 py-2.5 rounded-xl border ${accent.btn} text-sm font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed`}
          >
            {state.status === 'checking' ? '⏳' : 'Prüfen →'}
          </button>
        )}
      </div>

      {/* Ergebnis-Badge */}
      {state.status === 'done' && (() => {
        const result = state.result;
        if (result.supported) {
          return (
            <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border ${accent.border} ${accent.bg}`}>
              <span className="text-lg">✅</span>
              <div>
                <p className={`text-sm font-medium ${accent.text}`}>
                  {result.plugin.name} erkannt
                </p>
                <p className="text-gray-400 text-xs">{result.plugin.description}</p>
              </div>
            </div>
          );
        } else {
          const unsupported = result as { supported: false; reason: string };
          return (
            <div className="flex items-start gap-3 px-4 py-3 rounded-xl border border-red-500/30 bg-red-500/10">
              <span className="text-lg mt-0.5">🚫</span>
              <div>
                <p className="text-red-300 text-sm font-medium">Modell wird nicht unterstützt</p>
                <p className="text-gray-400 text-xs mt-0.5">{unsupported.reason}</p>
              </div>
            </div>
          );
        }
      })()}
    </div>
  );
}
