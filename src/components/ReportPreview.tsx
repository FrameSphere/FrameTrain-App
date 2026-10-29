// Vorschau eines Fehler-Reports vor dem Senden.
//
// Explizite Reports (Training, Dev Train) enthalten Skript, Logs und
// Konfiguration — darin stecken oft Dateipfade und Datensatznamen. Der Nutzer
// sieht deshalb vor dem Absenden, welche Teile rausgehen, und kann den
// vollstaendigen Body aufklappen, genau so, wie er verschickt wird.

import { useState } from 'react';
import { Send } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import { buildAppErrorPayload, type AppErrorReport } from '../utils/errorReport';

const PARTS = ['message', 'details', 'logs', 'script_full', 'config'] as const;

function lineCount(value: unknown): number {
  const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2) ?? '';
  return text.split('\n').length;
}

interface Props {
  report: AppErrorReport;
  onConfirm: () => void;
  onCancel: () => void;
}

export default function ReportPreview({ report, onConfirm, onCancel }: Props) {
  const { t } = useLanguage();
  const [showRaw, setShowRaw] = useState(false);
  const present = PARTS.filter(k => {
    const v = report[k];
    return v !== undefined && v !== null && v !== '' && !(typeof v === 'object' && Object.keys(v).length === 0);
  });

  return (
    <div className="rounded-xl border border-blue-500/25 bg-blue-500/[0.06] p-3 space-y-2 text-left">
      <p className="text-xs font-medium text-blue-300">{t('reportPreview.title')}</p>
      <p className="text-[11px] text-gray-400 leading-relaxed">{t('reportPreview.intro')}</p>
      <ul className="text-[11px] text-gray-300 space-y-0.5">
        {present.map(k => (
          <li key={k}>
            ● {t(`reportPreview.parts.${k}`)}{' '}
            <span className="text-gray-500">({t('reportPreview.lines').replace('{n}', String(lineCount(report[k])))})</span>
          </li>
        ))}
        <li>● {t('reportPreview.parts.meta')}</li>
      </ul>
      <button
        type="button"
        onClick={() => setShowRaw(v => !v)}
        className="text-[11px] text-gray-400 underline underline-offset-2 hover:text-white"
      >
        {showRaw ? t('reportPreview.hideRaw') : t('reportPreview.showRaw')}
      </button>
      {showRaw && (
        <pre className="text-[10px] font-mono text-gray-400 bg-black/30 rounded-lg p-2 max-h-56 overflow-auto whitespace-pre-wrap break-all">
          {JSON.stringify(buildAppErrorPayload(report), null, 2)}
        </pre>
      )}
      <div className="flex gap-2 pt-1">
        <button
          type="button"
          onClick={onConfirm}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/30 text-xs text-blue-200 transition-all"
        >
          <Send className="w-3.5 h-3.5" /> {t('reportPreview.confirm')}
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-xs text-gray-300 transition-all"
        >
          {t('reportPreview.cancel')}
        </button>
      </div>
    </div>
  );
}
