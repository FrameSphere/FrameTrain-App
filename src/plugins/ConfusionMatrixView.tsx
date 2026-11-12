// Konfusionsmatrix und Werte je Klasse aus einem Klassifikations-Test
// (metrics.confusion_matrix / metrics.per_class der Test-Engine,
// siehe src-tauri/python/test_engine/plugins/_shared_classify.py).
//
// Bei vielen Klassen passt keine volle Matrix in die Ansicht: gezeigt werden
// die Klassen, die am haeufigsten verwechselt werden.

import { useLanguage } from '../contexts/LanguageContext';

export interface ConfusionMatrixData {
  labels: string[];
  /** Zeile = erwartete Klasse, Spalte = vorhergesagte Klasse. */
  matrix: number[][];
  total_classes?: number;
  truncated?: boolean;
}

export interface PerClassScores { precision: number; recall: number; f1: number; support: number }

/** Hoechstens so viele Klassen zeigt die Matrix in der Test-Ansicht. */
export const MATRIX_UI_MAX = 10;

export function readConfusion(metrics: Record<string, unknown> | undefined): ConfusionMatrixData | null {
  const cm = metrics?.confusion_matrix as ConfusionMatrixData | undefined;
  if (!cm || !Array.isArray(cm.labels) || !Array.isArray(cm.matrix)) return null;
  if (cm.matrix.length !== cm.labels.length || cm.labels.length === 0) return null;
  return cm;
}

/**
 * Auf die am haeufigsten verwechselten Klassen begrenzen (Fehler als
 * erwartete plus als vorhergesagte Klasse), Reihenfolge bleibt erhalten.
 */
export function limitConfusion(cm: ConfusionMatrixData, max = MATRIX_UI_MAX): ConfusionMatrixData & { hidden: number } {
  const n = cm.labels.length;
  const total = cm.total_classes ?? n;
  if (n <= max) return { ...cm, hidden: total - n };
  const off = cm.labels.map((_, i) => {
    let s = 0;
    for (let j = 0; j < n; j++) if (j !== i) s += cm.matrix[i][j] + cm.matrix[j][i];
    return s;
  });
  const support = cm.matrix.map(row => row.reduce((a, b) => a + b, 0));
  const keep = cm.labels.map((_, i) => i)
    .sort((a, b) => off[b] - off[a] || support[b] - support[a] || cm.labels[a].localeCompare(cm.labels[b]))
    .slice(0, max)
    .sort((a, b) => a - b);
  return {
    labels: keep.map(i => cm.labels[i]),
    matrix: keep.map(i => keep.map(j => cm.matrix[i][j])),
    total_classes: total,
    truncated: true,
    hidden: total - keep.length,
  };
}

const pct = (v: number) => `${(v * 100).toFixed(1)} %`;

export default function ConfusionMatrixView({ metrics }: { metrics: Record<string, unknown> }) {
  const { t } = useLanguage();
  const full = readConfusion(metrics);
  const perClass = metrics.per_class as Record<string, PerClassScores> | undefined;
  if (!full && !perClass) return null;
  const cm = full ? limitConfusion(full) : null;
  const max = cm ? Math.max(1, ...cm.matrix.flat()) : 1;

  const classRows = perClass
    ? Object.entries(perClass).sort((a, b) => a[1].f1 - b[1].f1 || b[1].support - a[1].support)
    : [];

  return (
    <div className="pt-2 space-y-3" data-testid="confusion-view">
      {cm && (
        <div className="space-y-1">
          <p className="text-gray-400 text-[11px]">{t('testPlugins.confusion.title')}</p>
          <p className="text-gray-500 text-[10px]">{t('testPlugins.confusion.axes')}</p>
          <div className="overflow-x-auto">
            <table className="border-separate border-spacing-0.5 text-[10px]" aria-label={t('testPlugins.confusion.title')}>
              <thead>
                <tr>
                  <th />
                  {cm.labels.map(l => (
                    <th key={l} title={l} className="px-1 font-normal text-gray-400 max-w-[4.5rem] truncate">{l}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cm.matrix.map((row, i) => (
                  <tr key={cm.labels[i]}>
                    <th scope="row" title={cm.labels[i]} className="pr-2 font-normal text-gray-400 text-right max-w-[7rem] truncate">{cm.labels[i]}</th>
                    {row.map((v, j) => {
                      const diag = i === j;
                      const a = v === 0 ? 0 : 0.15 + 0.65 * (v / max);
                      return (
                        <td
                          key={j}
                          title={`${cm.labels[i]} → ${cm.labels[j]}: ${v}`}
                          className="w-9 h-7 text-center tabular-nums rounded text-white"
                          style={{ backgroundColor: v === 0 ? 'rgba(255,255,255,0.04)' : diag ? `rgba(16,185,129,${a})` : `rgba(239,68,68,${a})` }}
                        >
                          {v || ''}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {cm.hidden > 0 && (
            <p className="text-gray-500 text-[10px]">
              {t('testPlugins.confusion.limited', { shown: cm.labels.length, total: cm.total_classes ?? cm.labels.length })}
            </p>
          )}
        </div>
      )}

      {classRows.length > 0 && (
        <div className="space-y-1">
          <p className="text-gray-400 text-[11px]">{t('testPlugins.confusion.perClass')}</p>
          <div className="max-h-56 overflow-y-auto">
            <table className="w-full text-[11px]">
              <thead>
                <tr className="text-gray-500">
                  <th className="text-left font-normal">{t('testPlugins.confusion.class')}</th>
                  <th className="text-right font-normal">{t('testPlugins.confusion.precision')}</th>
                  <th className="text-right font-normal">{t('testPlugins.confusion.recall')}</th>
                  <th className="text-right font-normal">F1</th>
                  <th className="text-right font-normal">{t('testPlugins.confusion.support')}</th>
                </tr>
              </thead>
              <tbody>
                {classRows.map(([name, s]) => (
                  <tr key={name} className="text-gray-300">
                    <td className="truncate max-w-[10rem]" title={name}>{name}</td>
                    <td className="text-right tabular-nums">{pct(s.precision)}</td>
                    <td className="text-right tabular-nums">{pct(s.recall)}</td>
                    <td className="text-right tabular-nums text-white">{pct(s.f1)}</td>
                    <td className="text-right tabular-nums">{s.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
