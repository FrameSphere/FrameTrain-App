// BenchmarkView.tsx – Ansicht "Vergleich" der Startseite.
//
// Wie die Benchmark-Grafiken der Modellanbieter: je Test-Dataset eine Gruppe,
// je Modell ein Balken in fester Farbe; mehrere Versionen eines Modells stehen
// als eigene Balken daneben, aeltere heller. Anders als dort beginnt die Achse bei
// 0, und nebeneinander steht nur, was auf demselben Test-Dataset gemessen wurde.
// Darunter Genauigkeit gegen Geschwindigkeit und die Rangliste je Dataset.

import { useState } from 'react';
import { AlertTriangle, ChevronRight } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import { navigateToModel } from '../ui/navigationEvents';
import type { BenchDataset, BenchResult, BenchSeries, BenchTask } from './homeBenchmarks';
import type { ModelRow } from './homePipeline';

type Format = {
  date: (ms: number | null | undefined) => string;
  ago: (ms: number | null | undefined) => string;
  number: (value: number, digits: number) => string;
};

const seriesColor = (n: number) => `var(--ft-series-${n})`;
const seriesFill = (s: BenchSeries) => s.shade === 1
  ? seriesColor(s.color)
  : `color-mix(in srgb, ${seriesColor(s.color)} ${Math.round(s.shade * 100)}%, var(--ft-panel))`;
const seriesName = (s: BenchSeries) => (s.multi && s.version ? `${s.model} · ${s.version}` : s.model);

function valueText(r: BenchResult, f: Format): string {
  return r.metric.fraction !== null ? `${f.number(r.metric.fraction * 100, 1)}%` : f.number(r.metric.value, 3);
}

function speedText(ms: number | null, f: Format): string {
  if (ms === null) return '–';
  return ms >= 100 ? `${f.number(ms, 0)} ms` : `${f.number(ms, ms < 10 ? 1 : 0)} ms`;
}

/** Gruppierte Balken: x = Test-Datasets, je Modell ein Balken, Achse 0–100 %. */
function BenchBars({ task, f }: { task: BenchTask; f: Format }) {
  const { t } = useLanguage();
  const sets = task.datasets.filter(d => d.results.some(r => r.metric.fraction !== null));
  const W = 720, H = 220, L = 36, B = 40, T = 16;
  const groupW = (W - L) / sets.length;
  // Mit nur einem Modell bekommt jede Gruppe einen freien, gestrichelten Platz:
  // dort steht spaeter das zweite Modell.
  const slots = task.series.length + (task.modelCount === 1 && task.series.length === 1 ? 1 : 0);
  const barW = Math.min(36, (groupW - 24) / slots - 2);
  const free = (x: number, key: string, label: string) => (
    <rect key={key} x={x + 0.5} y={y(0.55)} width={barW - 1} height={H - B - y(0.55)} rx={4} className="ft-chart-free">
      <title>{label}</title>
    </rect>
  );
  const y = (frac: number) => T + (H - T - B) * (1 - frac);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="ft-chart" role="img" aria-label={t('home.pipeline.bench.chartTitle')}>
      {[0, 0.25, 0.5, 0.75, 1].map(v => (
        <g key={v}>
          <line x1={L} x2={W} y1={y(v)} y2={y(v)} className="ft-chart-grid" />
          <text x={L - 8} y={y(v) + 4} textAnchor="end" className="ft-chart-tick">{Math.round(v * 100)}</text>
        </g>
      ))}
      {sets.map((set, i) => {
        const start = L + i * groupW + (groupW - slots * (barW + 2)) / 2;
        const top = Math.max(...set.results.map(r => r.metric.fraction ?? 0));
        return (
          <g key={set.name}>
            {task.series.map((m, j) => {
              const r = set.results.find(res => res.key === m.key);
              const x = start + j * (barW + 2);
              if (!r || r.metric.fraction === null) {
                return free(x, m.key, t('home.pipeline.bench.notOnDataset', { model: seriesName(m) }));
              }
              const yTop = y(r.metric.fraction);
              const rad = Math.min(4, barW / 2, H - B - yTop);
              return (
                <g key={m.key}>
                  <path
                    className="ft-chart-bar"
                    style={{ animationDelay: `${(i * task.series.length + j) * 40}ms` }}
                    d={`M${x},${H - B} V${yTop + rad} Q${x},${yTop} ${x + rad},${yTop} H${x + barW - rad} Q${x + barW},${yTop} ${x + barW},${yTop + rad} V${H - B} Z`}
                    fill={seriesFill(m)}
                  >
                    <title>{`${seriesName(m)} · ${set.name}: ${valueText(r, f)}${r.samples !== null ? ` (${r.samples} ${t('home.pipeline.bench.samples').toLowerCase()})` : ''}`}</title>
                  </path>
                  {(set.ranked ? r.metric.fraction === top : set.results.length === 1) && (
                    <text x={x + barW / 2} y={yTop - 6} textAnchor="middle" className="ft-chart-label ft-fade-late">{f.number(r.metric.fraction * 100, 1)}</text>
                  )}
                </g>
              );
            })}
            {slots > task.series.length && free(start + barW + 2, 'free', t('home.pipeline.bench.freeSlot'))}
            <text x={L + i * groupW + groupW / 2} y={H - 20} textAnchor="middle" className="ft-chart-axis">{set.name}</text>
          </g>
        );
      })}
    </svg>
  );
}

/** Genauigkeit gegen Zeit pro Beispiel. Nur drei Modelle farbig (mehr sind fuer Farbenblinde nicht unterscheidbar), die anderen grau und beschriftet. */
function BenchScatter({ task, set, f }: { task: BenchTask; set: BenchDataset; f: Format }) {
  const { t } = useLanguage();
  const points = set.results.filter(r => r.metric.fraction !== null && r.msPerSample !== null);
  const W = 720, H = 200, L = 36, B = 34, T = 14, R = 150;
  const maxMs = Math.max(...points.map(p => p.msPerSample!)) * 1.15;
  const fracs = points.map(p => p.metric.fraction!);
  const lo = Math.max(0, Math.floor((Math.min(...fracs) - 0.05) * 20) / 20);
  const hi = Math.min(1, Math.ceil((Math.max(...fracs) + 0.02) * 20) / 20);
  const x = (ms: number) => L + (W - L - R) * (ms / maxMs);
  const y = (v: number) => T + (H - T - B) * (1 - (v - lo) / (hi - lo || 1));
  const ticks = [lo, (lo + hi) / 2, hi];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="ft-chart" role="img" aria-label={t('home.pipeline.bench.scatterTitle', { dataset: set.name })}>
      {ticks.map(v => (
        <g key={v}>
          <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} className="ft-chart-grid" />
          <text x={L - 8} y={y(v) + 4} textAnchor="end" className="ft-chart-tick">{Math.round(v * 100)}</text>
        </g>
      ))}
      <text x={L} y={H - 8} className="ft-chart-tick">{`0 ms  ← ${t('home.pipeline.bench.faster')}`}</text>
      <text x={W - R} y={H - 8} textAnchor="end" className="ft-chart-tick">{`${speedText(maxMs, f)} ${t('home.pipeline.bench.perSample')}`}</text>
      {points.map(p => {
        const s = task.series.find(m => m.key === p.key);
        const colored = s && s.color <= 3;
        const label = s ? seriesName(s) : p.model;
        return (
          <g key={p.key}>
            <circle cx={x(p.msPerSample!)} cy={y(p.metric.fraction!)} r={6} fill={colored ? seriesFill(s) : 'var(--ft-faint)'} className="ft-chart-dot ft-pop">
              <title>{`${label}: ${valueText(p, f)}, ${speedText(p.msPerSample, f)}`}</title>
            </circle>
            <text x={x(p.msPerSample!) + 11} y={y(p.metric.fraction!) + 4} className="ft-chart-axis">{label}</text>
          </g>
        );
      })}
    </svg>
  );
}

function ResultRow({ r, series, f }: { r: BenchResult; series: BenchSeries | undefined; f: Format }) {
  const { t } = useLanguage();
  const flags: string[] = [];
  if (r.sameData) flags.push(t('home.pipeline.compare.sameData'));
  if (r.fewSamples && r.samples !== null) flags.push(t('home.pipeline.bench.fewSamples', { n: String(r.samples) }));
  return (
    <div className="ft-bench ft-row">
      <span className={`ft-rank ${r.rank === 1 ? 'is-first' : ''}`}>{r.rank ?? '–'}</span>
      <button className="ft-name ft-bench-name" onClick={() => navigateToModel('tests', { modelId: r.modelId, versionId: r.versionId })} title={r.model}>
        {series && <i className="ft-swatch" style={{ background: seriesFill(series) }} aria-hidden="true" />}
        <span className="truncate">{r.model}</span>
        {r.version && r.version !== r.model && <small>{r.version}</small>}
      </button>
      <div className="ft-res">
        <span className="ft-v">{valueText(r, f)}</span>
        {r.metric.fraction !== null
          ? <span className={`ft-bar ${r.rank === 1 ? 'is-best' : ''}`} aria-hidden="true"><i style={{ width: `${Math.min(100, r.metric.fraction * 100)}%` }} /></span>
          : <span className="ft-res-label">{r.metric.label}</span>}
      </div>
      <span className="ft-t">{r.samples ?? '–'}</span>
      <span className="ft-t">{speedText(r.msPerSample, f)}</span>
      <span className="ft-t" title={f.date(r.at)}>{f.ago(r.at)}</span>
      {flags.map(text => (
        <p key={text} className="ft-cnote"><AlertTriangle className="w-3.5 h-3.5" aria-hidden="true" />{text}</p>
      ))}
      {r.msBefore !== null && r.msPerSample !== null && (
        <p className="ft-cnote is-quiet">{t('home.pipeline.bench.speedChange', { before: speedText(r.msBefore, f), after: speedText(r.msPerSample, f) })}</p>
      )}
    </div>
  );
}

function PendingRow({ row, f }: { row: ModelRow; f: Format }) {
  const { t } = useLanguage();
  const state = row.test.state;
  return (
    <div className="ft-bench ft-row">
      <span className="ft-rank">–</span>
      <span className="ft-name ft-bench-name"><span className="truncate">{row.name}</span></span>
      <div className="ft-cell">
        {state === 'running'
          ? <span className="ft-st is-live">{t('home.pipeline.state.running')}</span>
          : <span className={`ft-st ${state === 'failed' ? 'is-bad' : 'is-warn'}`}>{t(`home.pipeline.state.${state === 'failed' ? 'failed' : 'stopped'}`)}</span>}
        {row.next === 'retryTest' && (
          <button className="ft-go" onClick={() => navigateToModel('tests', { modelId: row.id, datasetId: row.data.id, versionId: row.test.versionId })}>{t('home.pipeline.next.retryTest')}</button>
        )}
      </div>
      <span className="ft-t">–</span>
      <span className="ft-t">–</span>
      <span className="ft-t" title={row.test.at ? f.date(row.test.at) : undefined}>{row.test.at ? f.ago(row.test.at) : '–'}</span>
    </div>
  );
}

export function BenchmarkView({ tasks, withoutResult, taskLabel, f }: {
  tasks: BenchTask[];
  withoutResult: ModelRow[];
  taskLabel: (key: string) => string;
  f: Format;
}) {
  const { t } = useLanguage();
  const [showAll, setShowAll] = useState(false);
  if (tasks.length === 0) return <p className="ft-intro mt-5">{t('home.pipeline.compare.empty')}</p>;
  return (
    <>
      <p className="ft-chart-note ft-bench-intro">{t('home.pipeline.bench.chartNote')}</p>
      {tasks.map(task => {
        const ranked = task.datasets.filter(d => d.ranked);
        const scatterSet = ranked.find(d => d.results.filter(r => r.msPerSample !== null).length >= 2);
        const seriesOf = (key: string) => task.series.find(m => m.key === key);
        return (
          <section key={task.key} className="ft-bench-task">
            <h2 className="ft-gh">{taskLabel(task.key)}<b>{task.modelCount + task.pending.length}</b></h2>
            {task.datasets.some(d => d.results.some(r => r.metric.fraction !== null)) && (
              <>
                <div className="ft-legend">
                  {task.series.map(m => (
                    <span key={m.key}><i className="ft-swatch" style={{ background: seriesFill(m) }} aria-hidden="true" />{seriesName(m)}</span>
                  ))}
                </div>
                <BenchBars task={task} f={f} />
                {ranked.length === 0 && (
                  <p className="ft-chart-note">{t('home.pipeline.bench.noComparisonHint', { dataset: task.datasets[0].name })}</p>
                )}
                {scatterSet && (
                  <>
                    <h3 className="ft-chart-h">{t('home.pipeline.bench.scatterTitle', { dataset: scatterSet.name })}</h3>
                    <p className="ft-chart-note">{t('home.pipeline.bench.scatterNote')}</p>
                    <BenchScatter task={task} set={scatterSet} f={f} />
                  </>
                )}
              </>
            )}
            {task.datasets.map(set => (
              <div key={set.name} className="ft-bench-set">
                <p className="ft-bench-ds">
                  {t('home.pipeline.bench.testDataset')} <code>{set.name}</code>
                  {set.ranked && <span> · {t('home.pipeline.bench.ranked')}</span>}
                </p>
                <div className="ft-bench ft-heads">
                  <span />
                  <span>{t('home.pipeline.bench.model')}</span>
                  <span>{t('home.pipeline.bench.result')}</span>
                  <span className="ft-num">{t('home.pipeline.bench.samples')}</span>
                  <span className="ft-num">{t('home.pipeline.bench.speed')}</span>
                  <span className="ft-num">{t('home.pipeline.compare.lastTest')}</span>
                </div>
                {set.results.map(r => <ResultRow key={r.key} r={r} series={seriesOf(r.key)} f={f} />)}
                {set.steps.map(step => (
                  <p key={step.model} className="ft-bench-step">
                    {t(Math.abs(step.points) < 0.05 ? 'home.pipeline.bench.stepSame' : step.points > 0 ? 'home.pipeline.bench.stepBetter' : 'home.pipeline.bench.stepWorse', {
                      model: step.model, newer: step.newer, older: step.older, points: f.number(Math.abs(step.points), 1),
                    })}
                  </p>
                ))}
              </div>
            ))}
            {task.pending.map(row => <PendingRow key={row.id} row={row} f={f} />)}
          </section>
        );
      })}
      {withoutResult.length > 0 && !showAll && (
        <button className="ft-gh-fold" aria-expanded="false" onClick={() => setShowAll(true)}>
          <ChevronRight className="w-3.5 h-3.5" aria-hidden="true" />
          {t('home.pipeline.compare.noResult')}<b>{withoutResult.length}</b>
        </button>
      )}
      {showAll && withoutResult.length > 0 && (
        <section className="ft-bench-task">
          <h2 className="ft-gh">{t('home.pipeline.compare.noResult')}<b>{withoutResult.length}</b></h2>
          <p className="ft-bench-names">{withoutResult.map(r => r.name).join(' · ')}</p>
        </section>
      )}
    </>
  );
}
