import type { TestPluginProps } from '../types';
import { AudioLines } from 'lucide-react';

export default function AudioTestPlugin(_props: TestPluginProps) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-6 flex items-center gap-3">
      <AudioLines className="w-5 h-5 text-orange-300" />
      <p className="text-gray-400 text-sm">Audiodatei auswählen und klassifizieren lassen.</p>
    </div>
  );
}
