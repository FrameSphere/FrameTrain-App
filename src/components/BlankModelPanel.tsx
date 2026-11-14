// "Neu" im Import-Dialog: leeres Modell fuer Aufgaben ohne Basismodell.
//
// Wie bei Synapse-Netzen legt Rust (create_task_model) einen Modellordner mit
// config.json und Bibliothekseintrag an. Danach laeuft alles ueber den
// normalen Weg: Training waehlt das Modell, das Plugin baut das Netz aus den
// Daten, die trainierte Version wird getestet.

import { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { LineChart, Loader2, Table2 } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

export type BlankTask = 'time_series' | 'tabular';

export const BLANK_TASKS: Array<{ task: BlankTask; labelKey: string; hintKey: string; defaultName: string }> = [
  { task: 'time_series', labelKey: 'modelManager.blankModel.timeSeries', hintKey: 'modelManager.blankModel.timeSeriesHint', defaultName: 'PatchTST' },
  { task: 'tabular', labelKey: 'modelManager.blankModel.tabular', hintKey: 'modelManager.blankModel.tabularHint', defaultName: 'Tabellenmodell' },
];

export default function BlankModelPanel({ gradientClass, onCreated, onError }: {
  gradientClass: string;
  onCreated: (model: { id: string; name: string }) => void;
  onError: (message: string) => void;
}) {
  const { t } = useLanguage();
  const [task, setTask] = useState<BlankTask>('time_series');
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);

  const create = async () => {
    setBusy(true);
    try {
      const fallback = BLANK_TASKS.find(b => b.task === task)?.defaultName ?? task;
      const model = await invoke<{ id: string; name: string }>('create_task_model', {
        task, modelName: name.trim() || fallback,
      });
      onCreated(model);
    } catch (e) {
      onError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      <p className="text-gray-400 text-sm">{t('modelManager.blankModel.intro')}</p>
      <div className="grid gap-2" role="radiogroup">
        {BLANK_TASKS.map(b => {
          const Icon = b.task === 'time_series' ? LineChart : Table2;
          const active = task === b.task;
          return (
            <button
              key={b.task}
              role="radio"
              aria-checked={active}
              onClick={() => setTask(b.task)}
              className={`flex items-start gap-3 text-left p-4 rounded-xl border transition-all ${
                active ? 'border-white/40 bg-white/10' : 'border-white/10 bg-white/5 hover:bg-white/10'}`}
            >
              <Icon className="w-5 h-5 text-gray-300 flex-shrink-0 mt-0.5" />
              <span>
                <span className="block text-white text-sm font-medium">{t(b.labelKey)}</span>
                <span className="block text-gray-400 text-xs mt-0.5">{t(b.hintKey)}</span>
              </span>
            </button>
          );
        })}
      </div>
      <div className="space-y-1.5">
        <label className="block text-sm font-medium text-gray-300" htmlFor="blank-model-name">{t('modelManager.blankModel.nameLabel')}</label>
        <input
          id="blank-model-name"
          value={name}
          onChange={e => setName(e.target.value)}
          placeholder={BLANK_TASKS.find(b => b.task === task)?.defaultName}
          className="w-full px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-white/30"
        />
      </div>
      <button
        onClick={create}
        disabled={busy}
        className={`w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r ${gradientClass} rounded-xl text-white font-medium disabled:opacity-50`}
      >
        {busy && <Loader2 className="w-4 h-4 animate-spin" />}
        {busy ? t('modelManager.blankModel.creating') : t('modelManager.blankModel.create')}
      </button>
    </div>
  );
}
