import type { TrainPluginProps } from '../types';
import { AudioLines } from 'lucide-react';

export default function AudioTrainPlugin(_props: TrainPluginProps) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-6 space-y-3">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-orange-500/20 border border-orange-500/30 flex items-center justify-center">
          <AudioLines className="w-5 h-5 text-orange-300" />
        </div>
        <div>
          <p className="text-white font-medium">Audio Classification</p>
          <p className="text-gray-400 text-sm">Wav2Vec2 · HuBERT · WavLM · AST · Whisper-Encoder</p>
        </div>
      </div>
      <p className="text-gray-400 text-sm">
        Klassifiziert Audio: Kommandos, Sprecher, Geräusche, Stimmungen.
        Dataset als Ordner pro Klasse mit .wav/.mp3/.flac.
      </p>
      <p className="text-amber-300/80 text-xs">
        Für Transkription (Sprache zu Text) reicht das nicht — das wäre ein eigener Aufgabenbereich.
      </p>
    </div>
  );
}
