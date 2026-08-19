import type { TrainPluginProps } from '../types';
import { ImageIcon } from 'lucide-react';

export default function HFImageTrainPlugin(_props: TrainPluginProps) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-6 space-y-3">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center">
          <ImageIcon className="w-5 h-5 text-emerald-300" />
        </div>
        <div>
          <p className="text-white font-medium">Image Classification (HuggingFace)</p>
          <p className="text-gray-400 text-sm">ViT · DeiT · ConvNeXt · Swin · ResNet · BEiT</p>
        </div>
      </div>
      <p className="text-gray-400 text-sm">
        Trainiert das heruntergeladene Modell selbst. Dataset im Format
        &quot;Ordner pro Klasse&quot; wählen, optional in train/ und val/ aufgeteilt.
      </p>
    </div>
  );
}
