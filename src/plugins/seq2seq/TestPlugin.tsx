import type { TestPluginProps } from '../types';
import { ArrowLeftRight } from 'lucide-react';

export default function Seq2SeqTestPlugin(_props: TestPluginProps) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-6 flex items-center gap-3">
      <ArrowLeftRight className="w-5 h-5 text-cyan-300" />
      <p className="text-gray-400 text-sm">Text eingeben und umformen lassen.</p>
    </div>
  );
}
