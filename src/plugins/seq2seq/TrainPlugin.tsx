import type { TrainPluginProps } from '../types';
import { ArrowLeftRight } from 'lucide-react';

export default function Seq2SeqTrainPlugin(_props: TrainPluginProps) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-6 space-y-3">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center">
          <ArrowLeftRight className="w-5 h-5 text-cyan-300" />
        </div>
        <div>
          <p className="text-white font-medium">Seq2Seq</p>
          <p className="text-gray-400 text-sm">T5 · mT5 · BART · Pegasus · Marian</p>
        </div>
      </div>
      <p className="text-gray-400 text-sm">
        Zusammenfassung, Übersetzung, Textumformung. Das Dataset braucht zwei
        Spalten: Eingabetext und Zieltext.
      </p>
    </div>
  );
}
