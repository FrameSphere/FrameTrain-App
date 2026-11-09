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

/**
 * Punkte, die enger als `px` beieinanderliegen (verkettet), als groesste Gruppe.
 * Diese Gruppe zeigt die Lupe vergroessert; einzeln waeren die Punkte nicht zu trennen.
 */
export function findTightCluster(points: { key: string; x: number; y: number }[], px = 12): string[] {
  const groups: string[][] = [];
  const seen = new Set<string>();
  for (const p of points) {
    if (seen.has(p.key)) continue;
    const group = [p.key];
    seen.add(p.key);
    for (let i = 0; i < group.length; i++) {
      const cur = points.find(q => q.key === group[i])!;
      for (const q of points) {
        if (!seen.has(q.key) && Math.hypot(q.x - cur.x, q.y - cur.y) < px) { seen.add(q.key); group.push(q.key); }
      }
    }
    if (group.length >= 2) groups.push(group);
  }
  return groups.sort((a, b) => b.length - a.length)[0] ?? [];
}

/**
 * Genauigkeit gegen Zeit pro Beispiel. Versionen eines Modells verbindet ein Pfeil
 * (aeltere zu neuerer). Liegen Punkte zu dicht, zeigt rechts eine Lupe sie
 * auseinandergezogen mit Werten. Nur drei Modelle farbig, weitere grau.
 */
function BenchScatter({ task, set, f, hover, onHover }: {
  task: BenchTask; set: BenchDataset; f: Format; hover: string | null; onHover: (key: string | null) => void;
}) {
  const { t } = useLanguage();
  const order = new Map(task.series.map((s, i) => [s.key, i]));
  const points = set.results
    .filter(r => r.metric.fraction !== null && r.msPerSample !== null)
    .sort((a, b) => (order.get(a.key) ?? 0) - (order.get(b.key) ?? 0));
  const W = 720, H = 210, L = 36, B = 34, T = 14, R = 240;
  const maxMs = Math.max(...points.map(p => p.msPerSample!)) * 1.15;
  const fracs = points.map(p => p.metric.fraction!);
  const lo = Math.max(0, Math.floor((Math.min(...fracs) - 0.05) * 20) / 20);
  const hi = Math.min(1, Math.ceil((Math.max(...fracs) + 0.02) * 20) / 20);
  const x = (ms: number) => L + (W - L - R) * (ms / maxMs);
  const y = (v: number) => T + (H - T - B) * (1 - (v - lo) / (hi - lo || 1));
  const ticks = [lo, (lo + hi) / 2, hi];
  const seriesOf = (key: string) => task.series.find(m => m.key === key);
  const fillOf = (key: string) => {
    const s = seriesOf(key);
    return s && s.color <= 3 ? seriesFill(s) : 'var(--ft-faint)';
  };
  const pos = new Map(points.map(p => [p.key, { x: x(p.msPerSample!), y: y(p.metric.fraction!) }]));
  const tight = findTightCluster(points.map(p => ({ key: p.key, ...pos.get(p.key)! })));
  const inLens = new Set(tight);
  const dim = (key: string) => (hover && hover !== key ? 0.25 : 1);
  const markerId = (key: string) => `ft-arrow-${task.key}-${key}`.replace(/[^a-zA-Z0-9_-]/g, '_');

  // Pfeile zwischen aufeinanderfolgenden Versionen desselben Modells
  const arrows = (keys: string[], at: (k: string) => { x: number; y: number }, gap: number) => {
    const out: { from: string; to: string; x1: number; y1: number; x2: number; y2: number }[] = [];
    for (let i = 1; i < keys.length; i++) {
      const a = points.find(p => p.key === keys[i - 1])!, b = points.find(p => p.key === keys[i])!;
      if (a.modelId !== b.modelId) continue;
      const pa = at(a.key), pb = at(b.key);
      const len = Math.hypot(pb.x - pa.x, pb.y - pa.y);
      if (len < gap * 2 + 4) continue;
      const ux = (pb.x - pa.x) / len, uy = (pb.y - pa.y) / len;
      out.push({ from: a.key, to: b.key, x1: pa.x + ux * gap, y1: pa.y + uy * gap, x2: pb.x - ux * (gap + 2), y2: pb.y - uy * (gap + 2) });
    }
    return out;
  };
  const mainArrows = arrows(points.map(p => p.key).filter(k => !inLens.has(k)), k => pos.get(k)!, 9);

  // Lupe
  const lens = tight.length >= 2 ? (() => {
    const bx = W - R + 18, bw = R - 26, by = T, bh = 128;
    const items = points.filter(p => inLens.has(p.key));
    // Die Werte unter den Punkten sind gut 90 px breit: genug Rand, damit nichts abgeschnitten wird.
    const pad = 54;
    const step = (bw - 2 * pad) / Math.max(1, items.length - 1);
    const at = new Map(items.map((p, i) => [p.key, { x: items.length === 1 ? bx + bw / 2 : bx + pad + i * step, y: by + 66 }]));
    const xs = items.map(p => pos.get(p.key)!.x), ys = items.map(p => pos.get(p.key)!.y);
    const sameModel = new Set(items.map(p => p.modelId)).size === 1;
    return {
      bx, bw, by, bh, items, at, sameModel,
      frame: { x: Math.min(...xs) - 11, y: Math.min(...ys) - 11, w: Math.max(...xs) - Math.min(...xs) + 22, h: Math.max(...ys) - Math.min(...ys) + 22 },
      arrows: arrows(items.map(p => p.key), k => at.get(k)!, 10),
    };
  })() : null;
  const labelOf = (key: string, short: boolean) => {
    const s = seriesOf(key);
    if (!s) return '';
    if (short && s.version) return s.version;
    return seriesName(s);
  };
  const usedArrowKeys = new Set([...mainArrows, ...(lens?.arrows ?? [])].map(a => a.to));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="ft-chart" role="img" aria-label={t('home.pipeline.bench.scatterTitle', { dataset: set.name })}>
      <defs>
        {[...usedArrowKeys].map(k => (
          <marker key={k} id={markerId(k)} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill={fillOf(k)} />
          </marker>
        ))}
      </defs>
      {ticks.map(v => (
        <g key={v}>
          <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} className="ft-chart-grid" />
          <text x={L - 8} y={y(v) + 4} textAnchor="end" className="ft-chart-tick">{Math.round(v * 100)}</text>
        </g>
      ))}
      <text x={L} y={H - 8} className="ft-chart-tick">{`0 ms  ← ${t('home.pipeline.bench.faster')}`}</text>
      <text x={W - R} y={H - 8} textAnchor="end" className="ft-chart-tick">{`${speedText(maxMs, f)} ${t('home.pipeline.bench.perSample')}`}</text>
      {mainArrows.map(a => (
        <line key={`${a.from}>${a.to}`} x1={a.x1} y1={a.y1} x2={a.x2} y2={a.y2} stroke={fillOf(a.to)} strokeWidth={1.6}
          markerEnd={`url(#${markerId(a.to)})`} opacity={Math.min(dim(a.from), dim(a.to)) === 1 ? 1 : 0.25} className="ft-fade-late" />
      ))}
      {lens && (
        <g className="ft-fade-late">
          <rect x={lens.frame.x} y={lens.frame.y} width={lens.frame.w} height={lens.frame.h} rx={6} className="ft-lens-frame" />
          <line x1={lens.frame.x + lens.frame.w} y1={lens.frame.y + lens.frame.h / 2} x2={lens.bx} y2={lens.by + lens.bh / 2} className="ft-lens-frame" />
          <rect x={lens.bx} y={lens.by} width={lens.bw} height={lens.bh} rx={10} className="ft-lens" />
          <text x={lens.bx + 12} y={lens.by + 20} className="ft-chart-tick ft-lens-title">
            {lens.sameModel ? lens.items[0].model : t('home.pipeline.bench.closeTogether')}
          </text>
          {lens.arrows.map(a => (
            <line key={`${a.from}>${a.to}`} x1={a.x1} y1={a.y1} x2={a.x2} y2={a.y2} stroke={fillOf(a.to)} strokeWidth={1.6}
              markerEnd={`url(#${markerId(a.to)})`} opacity={Math.min(dim(a.from), dim(a.to)) === 1 ? 1 : 0.25} />
          ))}
          {lens.items.map(p => {
            const at = lens.at.get(p.key)!;
            return (
              <g key={p.key} opacity={dim(p.key)} onMouseEnter={() => onHover(p.key)} onMouseLeave={() => onHover(null)}>
                <circle cx={at.x} cy={at.y} r={7} fill={fillOf(p.key)} className="ft-chart-dot" />
                <text x={at.x} y={at.y + 24} textAnchor="middle" className="ft-chart-axis ft-lens-label">{labelOf(p.key, lens.sameModel)}</text>
                <text x={at.x} y={at.y + 39} textAnchor="middle" className="ft-chart-tick">{`${valueText(p, f)} · ${speedText(p.msPerSample, f)}`}</text>
              </g>
            );
          })}
        </g>
      )}
      {points.map(p => {
        const at = pos.get(p.key)!;
        return (
          <circle key={p.key} cx={at.x} cy={at.y} r={hover === p.key ? 8 : 6} fill={fillOf(p.key)} opacity={dim(p.key)}
            className="ft-chart-dot ft-pop" onMouseEnter={() => onHover(p.key)} onMouseLeave={() => onHover(null)}>
            <title>{`${labelOf(p.key, false) || p.model}: ${valueText(p, f)}, ${speedText(p.msPerSample, f)}`}</title>
          </circle>
        );
      })}
      {placeScatterLabels(points.filter(p => !inLens.has(p.key)).map(p => {
        const s = seriesOf(p.key);
        return { ...pos.get(p.key)!, model: p.model, version: s?.multi ? s.version : null };
      }), W - R, lens ? [{ left: lens.frame.x, right: lens.frame.x + lens.frame.w, top: lens.frame.y, bottom: lens.frame.y + lens.frame.h }] : []).map(l => (
        <text key={`${l.x}:${l.y}:${l.text}`} x={l.x} y={l.y} textAnchor={l.anchor} className="ft-chart-axis ft-fade-late">{l.text}</text>
      ))}
    </svg>
  );
}

/**
 * Beschriftungen fuer das Streudiagramm. Fast deckungsgleiche Punkte bekommen
 * eine gemeinsame Beschriftung ("bench-electra-small · v1, v2"), nahe Beschriftungen
 * ruecken nach unten, und am rechten Rand steht der Text links vom Punkt.
 */
type Box = { left: number; right: number; top: number; bottom: number };

export function placeScatterLabels(
  points: { x: number; y: number; model: string; version: string | null }[],
  width: number,
  avoid: Box[] = [],
): { x: number; y: number; text: string; anchor: 'start' | 'end' | 'middle' }[] {
  const clusters: { x: number; y: number; items: typeof points }[] = [];
  for (const p of points) {
    const near = clusters.find(c => Math.abs(c.x - p.x) < 10 && Math.abs(c.y - p.y) < 10);
    if (near) near.items.push(p);
    else clusters.push({ x: p.x, y: p.y, items: [p] });
  }
  const textOf = (items: typeof points) => {
    const models = [...new Set(items.map(i => i.model))];
    if (models.length === 1) {
      const versions = items.map(i => i.version).filter((v): v is string => !!v);
      return versions.length ? `${models[0]} · ${versions.join(', ')}` : models[0];
    }
    return items.map(i => (i.version ? `${i.model} · ${i.version}` : i.model)).join(', ');
  };
  const charW = 6.6, lineH = 14;
  const placed: { x: number; y: number; text: string; anchor: 'start' | 'end' | 'middle'; left: number; right: number }[] = [];
  const hits = (left: number, right: number, base: number) =>
    avoid.some(b => left < b.right && b.left < right && base - 10 < b.bottom && b.top < base + 3);
  for (const c of [...clusters].sort((a, b) => a.y - b.y)) {
    const text = textOf(c.items);
    const w = text.length * charW;
    // Rechts vom Punkt, sonst links, sonst darunter: nie ueber einen Lupen-Rahmen.
    const options: { anchor: 'start' | 'end' | 'middle'; x: number; y: number; left: number }[] = [
      { anchor: 'start', x: c.x + 11, y: c.y + 4, left: c.x + 11 },
      { anchor: 'end', x: c.x - 11, y: c.y + 4, left: c.x - 11 - w },
      { anchor: 'middle', x: c.x, y: c.y + 22, left: c.x - w / 2 },
    ];
    const fits = (o: typeof options[number]) => o.left >= 0 && o.left + w <= width && !hits(o.left, o.left + w, o.y);
    const o = options.find(fits) ?? options.find(opt => !hits(opt.left, opt.left + w, opt.y)) ?? options[1];
    const left = o.left, right = left + w;
    let ly = o.y;
    while (placed.some(p => Math.abs(p.y - ly) < lineH && p.left < right && left < p.right)) ly += lineH;
    placed.push({ x: o.x, y: ly, text, anchor: o.anchor, left, right });
  }
  return placed.map(({ x, y, text, anchor }) => ({ x, y, text, anchor }));
}

function ResultRow({ r, series, f, hover, onHover }: {
  r: BenchResult; series: BenchSeries | undefined; f: Format; hover: string | null; onHover: (key: string | null) => void;
}) {
  const { t } = useLanguage();
  const flags: string[] = [];
  if (r.sameData) flags.push(t('home.pipeline.compare.sameData'));
  if (r.fewSamples && r.samples !== null) flags.push(t('home.pipeline.bench.fewSamples', { n: String(r.samples) }));
  return (
    <div
      className={`ft-bench ft-row ${hover === r.key ? 'is-hover' : hover ? 'is-dim' : ''}`}
      onMouseEnter={() => onHover(r.key)}
      onMouseLeave={() => onHover(null)}
    >
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
  // Zeile der Rangliste und Punkt im Streudiagramm heben sich gegenseitig hervor.
  const [hover, setHover] = useState<string | null>(null);
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
                    <BenchScatter task={task} set={scatterSet} f={f} hover={hover} onHover={setHover} />
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
                {set.results.map(r => <ResultRow key={r.key} r={r} series={seriesOf(r.key)} f={f} hover={set === scatterSet ? hover : null} onHover={set === scatterSet ? setHover : () => {}} />)}
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
