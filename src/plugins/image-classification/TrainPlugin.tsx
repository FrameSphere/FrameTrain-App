import type { TrainPluginProps } from '../types';
import { ImageIcon } from 'lucide-react';

export default function ImageClassificationTrainPlugin(_props: TrainPluginProps) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-6 space-y-3">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-blue-500/20 border border-blue-500/30 flex items-center justify-center">
          <ImageIcon className="w-5 h-5 text-blue-300" />
        </div>
        <div>
          <p className="text-white font-medium">Image Classification</p>
          <p className="text-gray-400 text-sm">ResNet / EfficientNet / ViT / MobileNet</p>
        </div>
      </div>
      <p className="text-gray-400 text-sm">
        Training wird über das Training-Panel gestartet. Wähle ein Dataset im Format
        &quot;Ordner pro Klasse&quot; (folder_class) und starte das Training dort.
      </p>
      <p className="text-amber-300/80 text-xs">
        Hinweis: Trainiert wird ein torchvision-Backbone mit ImageNet-Gewichten
        (Architektur im Training-Panel wählbar). Die heruntergeladenen
        HuggingFace-Gewichte werden dabei nicht verwendet.
      </p>
    </div>
  );
}
