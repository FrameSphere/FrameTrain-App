import { useEffect, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import type { TestPluginProps } from '../types';
import { useLanguage } from '../../contexts/LanguageContext';
import GenericTestPanel from '../GenericTestPanel';
import { isForecastCurve, linePath, valueRange, type ForecastCurve } from './forecastChart';

const W = 600;
const H = 180;

export function ForecastChart({ curve }: { curve: ForecastCurve }) {
  const { t } = useLanguage();
  const [min, max] = valueRange(curve.actual, curve.forecast, curve.naive);
  const n = curve.actual.length;
  const splitX = n > 1 ? (curve.forecast_start / (n - 1)) * W : 0;
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-2" data-testid="forecast-chart">
      <p className="text-white font-medium text-sm">{t('testPlugins.timeSeries.chartTitle')}</p>
      <p className="text-gray-500 text-[11px]">
        {t('testPlugins.timeSeries.chartSubtitle', { series: curve.series_id ?? '', column: curve.channel ?? '' })}
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-44" preserveAspectRatio="none" role="img"
        aria-label={t('testPlugins.timeSeries.chartTitle')}>
        <line x1={splitX} x2={splitX} y1={0} y2={H} stroke="currentColor" className="text-white/20" strokeDasharray="4 4" />
        <path d={linePath(curve.actual, W, H, min, max)} fill="none" stroke="#cbd5e1" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
        {curve.naive && (
          <path d={linePath(curve.naive, W, H, min, max)} fill="none" stroke="#f59e0b" strokeWidth={1} strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />
        )}
        <path d={linePath(curve.forecast, W, H, min, max)} fill="none" stroke="#10b981" strokeWidth={2} vectorEffect="non-scaling-stroke" />
      </svg>
      <div className="flex flex-wrap gap-4 text-[11px] text-gray-400">
        <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-0.5 bg-slate-300" />{t('testPlugins.timeSeries.legendActual')}</span>
        <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-0.5 bg-emerald-500" />{t('testPlugins.timeSeries.legendForecast')}</span>
        {curve.naive && <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-0.5 bg-amber-500" />{t('testPlugins.timeSeries.legendNaive')}</span>}
      </div>
    </div>
  );
}

export default function TimeSeriesTestPlugin(props: TestPluginProps) {
  const { t } = useLanguage();
  const [curve, setCurve] = useState<ForecastCurve | null>(null);

  // GenericTestPanel zeigt die Kennzahlen; die Kurve kommt aus demselben
  // test-complete-Ereignis (Feld forecast_curve, nur dieses Plugin sendet es).
  useEffect(() => {
    let off: (() => void) | undefined;
    let alive = true;
    listen<{ version_id?: string; data?: { forecast_curve?: unknown } }>('test-complete', e => {
      if (e.payload.version_id && e.payload.version_id !== props.versionId) return;
      const c = e.payload.data?.forecast_curve;
      if (isForecastCurve(c)) setCurve(c);
    }).then(fn => { if (alive) off = fn; else fn(); }).catch(() => {});
    return () => { alive = false; off?.(); };
  }, [props.versionId]);

  return (
    <div className="space-y-4">
      <GenericTestPanel
        {...props}
        taskType="time_series"
        inputKind="text"
        singleLabel={t('testPlugins.timeSeries.singleLabel')}
        singlePlaceholder={t('testPlugins.timeSeries.singlePlaceholder')}
        resultLabel={t('testPlugins.timeSeries.result')}
        showConfidence={false}
      />
      {curve && <ForecastChart curve={curve} />}
    </div>
  );
}
