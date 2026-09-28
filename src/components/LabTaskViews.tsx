// Ergebnis- und Sample-Ansichten im Labor fuer Aufgaben, die mehr sind als
// "eine Klasse mit Konfidenz": markierte Entitaeten (NER), Aehnlichkeit
// (Embeddings), generierter Text (LLM, Seq2Seq, VLM, Spracherkennung) und
// Bild neben Referenz (Text-zu-Bild).

import { convertFileSrc } from '@tauri-apps/api/core';
import { useLanguage } from '../contexts/LanguageContext';
import { PAIR_SEPARATOR, wordErrorRate, type ExpectedEntity } from './labTaskSamples';

export interface EntitySpan { text?: string; label: string; start: number; end: number; score?: number }

const ENTITY_COLORS = ['#f59e0b', '#38bdf8', '#a78bfa', '#34d399', '#f472b6', '#fb7185', '#facc15', '#2dd4bf'];

/** Feste Farbe je Entitaets-Typ (PER bleibt ueberall dieselbe Farbe). */
export function entityColor(label: string): string {
  let h = 0;
  for (const ch of label) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return ENTITY_COLORS[h % ENTITY_COLORS.length];
}

/** Text mit markierten Entitaeten. Ueberlappende Spans werden uebersprungen. */
export function EntityText({ text, entities, dashed = false }: { text: string; entities: EntitySpan[]; dashed?: boolean }) {
  const sorted = [...entities]
    .filter(e => e.start >= 0 && e.end > e.start && e.end <= text.length)
    .sort((a, b) => a.start - b.start);
  const parts: React.ReactNode[] = [];
  let pos = 0;
  sorted.forEach((e, i) => {
    if (e.start < pos) return;
    if (e.start > pos) parts.push(<span key={`t${i}`}>{text.slice(pos, e.start)}</span>);
    const color = entityColor(e.label);
    parts.push(
      <mark
        key={`e${i}`}
        className="rounded px-0.5 text-inherit"
        style={{ background: `${color}33`, border: `1px ${dashed ? 'dashed' : 'solid'} ${color}` }}
        title={e.score != null ? `${e.label} ${(e.score * 100).toFixed(0)}%` : e.label}
      >
        {text.slice(e.start, e.end)}
        <sup className="ml-0.5 text-[9px] font-semibold" style={{ color }}>{e.label}</sup>
      </mark>,
    );
    pos = e.end;
  });
  if (pos < text.length) parts.push(<span key="rest">{text.slice(pos)}</span>);
  return <p className="text-gray-200 text-xs leading-7 whitespace-pre-wrap">{parts}</p>;
}

/** Vergleich Soll/Ist fuer NER: gleiche Spanne und gleicher Typ. */
export function compareEntities(expected: ExpectedEntity[], predicted: EntitySpan[]) {
  const key = (e: { start: number; end: number; label: string }) => `${e.start}:${e.end}:${e.label}`;
  const got = new Set(predicted.map(key));
  const want = new Set(expected.map(key));
  const hits = expected.filter(e => got.has(key(e))).length;
  return { hits, missing: expected.length - hits, extra: predicted.filter(e => !want.has(key(e))).length };
}

/** Position auf dem Aehnlichkeitsbalken: -1 → 0 %, 0 → 50 %, 1 → 100 %. */
export const similarityPercent = (v: number) => Math.max(0, Math.min(100, ((v + 1) / 2) * 100));

/** Zwei Saetze und ihre Kosinus-Aehnlichkeit als Balken (-1..1 → 0..100 %). */
export function SimilarityView({ similarity, expected }: { similarity: number; expected?: string }) {
  const { t } = useLanguage();
  const pct = similarityPercent(similarity);
  const exp = expected != null && expected !== '' && Number.isFinite(Number(expected)) ? Number(expected) : null;
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <span className="text-xs text-gray-400">{t('laboratoryPanel.taskViews.similarity')}</span>
        <span className="text-amber-300 font-mono text-lg font-semibold">{similarity.toFixed(3)}</span>
      </div>
      <div className="relative h-2.5 rounded-full bg-white/10">
        <div className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-sky-500 to-amber-400" style={{ width: `${pct}%` }} />
        {exp != null && (
          // Soll-Wert aus dem Dataset (0..1 bzw. 0..5 → normiert)
          <div
            className="absolute -top-1 -bottom-1 w-0.5 bg-white"
            // Gleiche Skala wie der Balken (-1..1); Soll 0..1 bzw. STS 0..5.
            style={{ left: `${similarityPercent(exp > 1 ? exp / 5 : exp)}%` }}
            title={t('laboratoryPanel.taskViews.expectedScore', { value: String(exp) })}
          />
        )}
      </div>
      <div className="flex justify-between text-[10px] text-gray-600">
        <span>-1</span><span>0</span><span>1</span>
      </div>
      {exp != null && (
        <p className="text-[11px] text-gray-400">{t('laboratoryPanel.taskViews.expectedScore', { value: String(exp) })}</p>
      )}
    </div>
  );
}

/** "Satz A ||| Satz B" als zwei Zeilen. */
export function PairView({ text }: { text: string }) {
  const [a, b] = text.split(PAIR_SEPARATOR.trim()).map(s => s.trim());
  if (b === undefined) return <p className="text-gray-200 text-xs whitespace-pre-wrap">{text}</p>;
  return (
    <div className="space-y-1.5 text-xs">
      <p className="text-gray-200"><span className="text-sky-300 font-mono mr-1.5">A</span>{a}</p>
      <p className="text-gray-200"><span className="text-amber-300 font-mono mr-1.5">B</span>{b}</p>
    </div>
  );
}

/** Generierter Text: mehrzeilig, nicht als fette Einzeile. */
export function GeneratedText({ text, inferenceMs }: { text: string; inferenceMs: number }) {
  return (
    <div className="px-4 py-3 rounded-xl bg-amber-500/10 border border-amber-500/20 space-y-1">
      <p className="text-amber-100 text-sm leading-relaxed whitespace-pre-wrap break-words max-h-64 overflow-y-auto">{text || '—'}</p>
      <p className="text-gray-600 text-[10px] text-right">{inferenceMs.toFixed(0)} ms</p>
    </div>
  );
}

/** Spracherkennung: Wortfehlerrate zum Soll-Transkript. */
export function TranscriptCheck({ expected, got }: { expected: string; got: string }) {
  const { t } = useLanguage();
  const wer = wordErrorRate(expected, got);
  const good = wer <= 0.1;
  return (
    <div className={`px-3 py-2 rounded-xl text-xs space-y-1 ${good ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-300' : 'bg-red-500/10 border border-red-500/20 text-red-300'}`}>
      <p className="font-medium">{t('laboratoryPanel.taskViews.wer', { value: (wer * 100).toFixed(1) })}</p>
      <p className="text-gray-400">{t('laboratoryPanel.taskViews.expected')}: <span className="text-gray-200">{expected}</span></p>
    </div>
  );
}

/** Text-zu-Bild: erzeugtes Bild neben dem Bild aus dem Dataset. */
export function GeneratedImage({ path, refImage, prompt }: { path: string; refImage?: string; prompt: string }) {
  const { t } = useLanguage();
  return (
    <div className={`grid gap-2 ${refImage ? 'grid-cols-2' : 'grid-cols-1'}`}>
      <figure className="space-y-1">
        <img src={convertFileSrc(path)} alt={prompt} className="w-full max-h-80 object-contain rounded-xl border border-white/10 bg-black/20" />
        <figcaption className="text-[10px] text-gray-500 text-center">{t('laboratoryPanel.taskViews.generated')}</figcaption>
      </figure>
      {refImage && (
        <figure className="space-y-1">
          <img src={convertFileSrc(refImage)} alt="" className="w-full max-h-80 object-contain rounded-xl border border-white/10 bg-black/20" />
          <figcaption className="text-[10px] text-gray-500 text-center">{t('laboratoryPanel.taskViews.reference')}</figcaption>
        </figure>
      )}
    </div>
  );
}

/** Vergleich fuer freien Text: Gross/klein, Leerraum und Satzzeichen am Ende egal. */
export function sameText(a: string, b: string): boolean {
  const norm = (s: string) => s.trim().toLowerCase().replace(/\s+/g, ' ').replace(/[.!?。]+$/, '');
  return norm(a) === norm(b);
}
