// HF Encoder – Training Plugin UI (Fallback)
//
// Hinweis: Das "normale" Training läuft über das zentrale TrainingPanel.
// Dieses Plugin existiert v.a. für Konsistenz im Plugin-Typ (ModelPlugin).

import type { TrainPluginProps } from '../types';

export default function HFEncoderTrainPlugin(_props: TrainPluginProps) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
      <p className="text-white font-medium">Training über das Training-Panel</p>
      <p className="text-gray-400 text-sm mt-1">
        Dieses Modell wird vom Backend unterstützt (Sequence Classification). Starte das Training im Training-Tab.
      </p>
    </div>
  );
}

