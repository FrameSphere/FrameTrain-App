import type { TestPluginProps } from '../types';
import { ImageIcon } from 'lucide-react';

export default function HFImageTestPlugin(_props: TestPluginProps) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-6 flex items-center gap-3">
      <ImageIcon className="w-5 h-5 text-emerald-300" />
      <p className="text-gray-400 text-sm">Bild auswählen und klassifizieren lassen.</p>
    </div>
  );
}
