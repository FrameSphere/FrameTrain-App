// Was ein Export ergeben hat: Verteilung, Aufteilung, Dubletten, Herkunft.
//
// Frueher kam nach dem Export nur "Datensatz angelegt". Ob eine Klasse
// fast leer war oder Kopien ueber Train und Val verteilt lagen, sah man erst
// am Ergebnis des Trainings. Dieselben Zahlen stehen als EXPORT_REPORT.md
// neben dem Datensatz.

import { AlertTriangle, CheckCircle2 } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import type { ExportReport } from './studioTypes';

export default function ExportReportView({ report }: { report: ExportReport }) {
  const { t } = useLanguage();
  const zeile = (label: string, wert: string | number, warn = false) => (
    <div className="flex items-center justify-between text-xs">
      <span className="text-gray-400">{label}</span>
      <span className={`tabular-nums ${warn ? 'text-amber-300' : 'text-gray-200'}`}>{wert}</span>
    </div>
  );
  const groesste = Math.max(1, ...report.per_class.map(([, n]) => n));

  return (
    <div className="space-y-3" data-testid="export-report">
      <div className="rounded-lg bg-white/[0.04] border border-white/10 p-3 space-y-1.5">
        {zeile(t('studio.report.total'), report.total)}
        {report.per_split.map(([teil, n]) => (
          <div key={teil}>{zeile(t('studio.report.split', { split: teil }), n)}</div>
        ))}
        {report.per_split.length > 0 && zeile(t('studio.report.groupLeaks'), report.group_leaks.length, report.group_leaks.length > 0)}
        {zeile(t('studio.report.nearDuplicates'),
          report.near_checked ? report.near_duplicates : t('studio.report.notChecked'), report.near_duplicates_across > 0)}
        {report.near_checked && report.per_split.length > 0 &&
          zeile(t('studio.report.nearAcross'), report.near_duplicates_across, report.near_duplicates_across > 0)}
        {report.without_license > 0 && zeile(t('studio.report.withoutLicense'), report.without_license, true)}
      </div>

      {report.per_class.length > 0 && (
        <div className="rounded-lg bg-white/[0.04] border border-white/10 p-3 space-y-1.5">
          <p className="text-gray-500 text-[11px]">{t('studio.report.classes')}</p>
          {report.per_class.map(([klasse, n]) => (
            <div key={klasse} className="flex items-center gap-2 text-xs">
              <span className="text-gray-300 w-28 truncate">{klasse}</span>
              <div className="flex-1 h-1.5 rounded-full bg-white/5 overflow-hidden">
                <div className="h-full bg-emerald-400/60" style={{ width: `${Math.round((n / groesste) * 100)}%` }} />
              </div>
              <span className="text-gray-400 tabular-nums w-10 text-right">{n}</span>
            </div>
          ))}
        </div>
      )}

      {report.warnings.length > 0 ? (
        <div className="rounded-lg bg-amber-500/10 border border-amber-500/25 p-3 space-y-1">
          {report.warnings.map(w => (
            <p key={w} className="text-amber-200/90 text-xs flex items-start gap-1.5">
              <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" /> {w}
            </p>
          ))}
        </div>
      ) : (
        <p className="text-emerald-300/80 text-xs flex items-center gap-1.5">
          <CheckCircle2 className="w-3.5 h-3.5" /> {t('studio.report.allGood')}
        </p>
      )}
      <p className="text-gray-600 text-[11px]">{t('studio.report.fileHint')}</p>
    </div>
  );
}
