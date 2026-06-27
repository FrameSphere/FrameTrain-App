// DatasetCompatBadge – zeigt an ob ein Dataset für ein Modell geeignet ist
// Unterstützt alte API (extensions) und neue API (DatasetAnalysis)

import { useState } from 'react';
import { ChevronDown, ChevronUp, CheckCircle, Info, AlertTriangle, Ban } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import {
  checkDatasetCompat, LEVEL_META,
  type DatasetCompatResult, type DatasetAnalysis,
} from '../plugins/datasetCompat';

interface DatasetCompatBadgeProps {
  /** Plugin-ID des Modells, z.B. "xlm-roberta" */
  modelPluginId: string | null;
  /**
   * Dateiendungen im Dataset (lowercase, mit Punkt).
   * Wenn null → Ladeanimation.
   */
  extensions: string[] | null;
  /**
   * Optional: vollständige DatasetAnalysis vom Backend.
   * Wenn vorhanden, wird die neue checkDataset-API bevorzugt.
   */
  analysis?: DatasetAnalysis | null;
  /** Kompakte Ansicht (nur Badge, kein Detail-Dropdown) */
  compact?: boolean;
}

export default function DatasetCompatBadge({
  modelPluginId,
  extensions,
  analysis,
  compact = false,
}: DatasetCompatBadgeProps) {
  const [expanded, setExpanded] = useState(false);
  const { t } = useLanguage();

  if (!modelPluginId) {
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs bg-white/5 border border-white/10 text-gray-500">
        {t('datasetCompat.noModelSelected')}
      </span>
    );
  }

  if (extensions === null) {
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs bg-white/5 border border-white/10 text-gray-500 animate-pulse">
        {t('datasetCompat.checkingDataset')}
      </span>
    );
  }

  const result: DatasetCompatResult = checkDatasetCompat(modelPluginId, extensions, analysis);
  const meta = LEVEL_META[result.overallLevel];
  const levelIcon = (() => {
    const cls = 'w-4 h-4';
    switch (meta.icon) {
      case 'check': return <CheckCircle className={`${cls} ${meta.color}`} />;
      case 'info': return <Info className={`${cls} ${meta.color}`} />;
      case 'alert': return <AlertTriangle className={`${cls} ${meta.color}`} />;
      case 'ban': return <Ban className={`${cls} ${meta.color}`} />;
      default: return <Info className={`${cls} ${meta.color}`} />;
    }
  })();

  if (compact) {
    return (
      <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${meta.bg} ${meta.border} border ${meta.color}`}>
        {levelIcon} {meta.label}
      </span>
    );
  }

  return (
    <div className={`rounded-xl border ${meta.border} ${meta.bg} overflow-hidden`}>
      {/* Header – immer sichtbar */}
      <div className="flex items-center justify-between px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="text-base">{levelIcon}</span>
          <div>
            <span className={`text-sm font-semibold ${meta.color}`}>{meta.label}</span>
            <p className="text-gray-400 text-xs mt-0.5 leading-snug">{result.summary}</p>
          </div>
        </div>
        {result.fileResults.length > 0 && (
          <button
            onClick={() => setExpanded(e => !e)}
            className="p-1.5 rounded-lg hover:bg-white/10 text-gray-500 hover:text-white transition-all flex-shrink-0"
          >
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        )}
      </div>

      {/* Hint */}
      {result.hint && (
        <div className="px-4 pb-2">
          <p className="text-gray-500 text-xs italic">{result.hint}</p>
        </div>
      )}

      {/* Detail pro Dateiformat */}
      {expanded && result.fileResults.length > 0 && (
        <div className="px-4 pb-3 space-y-1.5 border-t border-white/10 pt-3">
          <p className="text-xs text-gray-500 font-medium mb-2">{t('datasetCompat.fileDetailsTitle')}</p>
          {result.fileResults.map(fr => {
            const fm = LEVEL_META[fr.level];
            return (
              <div key={fr.extension} className="flex items-start gap-2">
                <span className={`text-xs font-mono font-semibold min-w-[60px] mt-0.5 ${fm.color}`}>
                  {fr.extension}
                </span>
                <span className="text-gray-400 text-xs leading-snug">{fr.reason}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
