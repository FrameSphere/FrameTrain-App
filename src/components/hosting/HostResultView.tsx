// Ergebnis einer Hosting-Anfrage, passend zur Aufgabe des Modells. Nutzt die
// Ansichten des Labors weiter (Boxen, Entitaeten, Aehnlichkeit, Bilder).

import { useState } from 'react';
import { convertFileSrc } from '@tauri-apps/api/core';
import { Copy, Check } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { DetectionOverlay, type DetectionBox } from '../LaboratoryPanel';
import { EntityText, SimilarityView, type EntitySpan } from '../LabTaskViews';
import { MarkdownText } from '../ui/MarkdownText';
import { summarizeBoxes } from '../labGroundTruth';
import { formatMs, tokensPerSecond, topPredictions, type Attachment, type HostInfo, type InferResult } from './hostingModel';

export function CopyButton({ text, label }: { text: string; label: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      onClick={() => { void navigator.clipboard?.writeText(text).then(() => { setDone(true); setTimeout(() => setDone(false), 1200); }); }}
      className="inline-flex items-center gap-1 text-[11px] text-inherit opacity-90 hover:opacity-100 hover:text-white transition-colors"
    >
      {done ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
      {label}
    </button>
  );
}

function Scores({ r, compact }: { r: InferResult; compact: boolean }) {
  const top = topPredictions(r);
  if (!top.length) return null;
  return (
    <div className="space-y-1 mt-2">
      {top.slice(0, 5).map((p, i) => (
        <div key={`${p.label}-${i}`} className="flex items-center gap-2 text-xs">
          <span className={`w-28 truncate ${i === 0 ? 'text-white' : compact ? 'text-white/65' : 'text-gray-400'}`}>{p.label}</span>
          <div className="flex-1 h-1.5 rounded-full bg-white/10 overflow-hidden">
            <div className={`h-full rounded-full ${i === 0 ? 'bg-emerald-400' : 'bg-white/30'}`} style={{ width: `${Math.max(2, Math.min(100, p.score * 100))}%` }} />
          </div>
          <span className={`w-12 text-right tabular-nums ${compact ? 'text-white/65' : 'text-gray-500'}`}>{(p.score * 100).toFixed(1)} %</span>
        </div>
      ))}
    </div>
  );
}

function Detection({ r, file, classes, compact }: { r: InferResult; file?: Attachment; classes: string[]; compact: boolean }) {
  const { t } = useLanguage();
  const boxes = (r.boxes ?? []) as unknown as DetectionBox[];
  const w = r.image_width ?? 0;
  const h = r.image_height ?? 0;
  const shown = compact ? 200 : 300;
  return (
    // Schnell-Chat: Bild links, Ergebnis rechts daneben; Seite: untereinander.
    <div className={compact ? 'flex items-start gap-3' : 'space-y-2'}>
      {file && (
        // Feste Hoehe statt volle Breite: ein 512er-Bild fuellte sonst die ganze
        // Seite. Die Breite folgt dem Seitenverhaeltnis, Boxen bleiben deckungsgleich.
        <div className="relative rounded-xl overflow-hidden border border-white/10 bg-black/30 max-w-full flex-shrink-0"
          style={{ height: shown, aspectRatio: w && h ? `${w} / ${h}` : '4 / 3', maxWidth: compact ? '62%' : undefined }}>
          <img src={convertFileSrc(file.path)} alt={file.name} className="absolute inset-0 w-full h-full object-contain" />
          <DetectionOverlay boxes={boxes} classes={classes} width={w} height={h} displayHeight={shown} />
        </div>
      )}
      <div className="min-w-0">
        <p className="text-sm text-white">
          {boxes.length ? summarizeBoxes(boxes) : t('hosting.result.noObjects')}
        </p>
        {compact && boxes.slice(0, 6).map((b, i) => (
          <p key={i} className="text-[12px] text-white/60 tabular-nums truncate">{b.label} · {(b.confidence * 100).toFixed(0)} %</p>
        ))}
      </div>
    </div>
  );
}

export default function HostResultView({ result, host, file, inputText = '', compact = false }: { result: InferResult; host: HostInfo | null; file?: Attachment; inputText?: string; compact?: boolean }) {
  const { t } = useLanguage();
  const extra = (result.extra ?? {}) as Record<string, unknown>;
  const modality = host?.modality ?? '';
  const meta: string[] = [];
  const ms = formatMs(result.inference_ms);
  if (ms) meta.push(ms);
  const tps = tokensPerSecond(result);
  if (tps) meta.push(t('hosting.result.tokensPerSecond', { value: tps }));

  let body: JSX.Element;
  let copyText = result.predicted;

  // Auch ohne bekannte Aufgabe (alter Verlauf): Boxen samt Bildmassen sind YOLO.
  const isDetect = modality === 'detect' || (!!result.boxes?.length && !!result.image_width);
  if (isDetect && (host?.task ?? 'detect') !== 'classify') {
    body = <Detection r={result} file={file} classes={host?.classes ?? []} compact={compact} />;
    copyText = JSON.stringify(result.boxes ?? [], null, 2);
  } else if (modality === 'text_to_image' && typeof extra.image_path === 'string') {
    const p = extra.image_path;
    body = (
      <figure className="space-y-1">
        <img src={convertFileSrc(p)} alt={t('laboratoryPanel.taskViews.generated')} className={`max-w-full ${compact ? 'max-h-56' : 'max-h-80'} object-contain rounded-xl border border-white/10 bg-black/20`} />
        <figcaption className="text-[11px] text-gray-500 break-all">{p}</figcaption>
      </figure>
    );
    copyText = p;
  } else if (modality === 'token' && Array.isArray(extra.entities)) {
    body = <EntityText text={inputText} entities={extra.entities as EntitySpan[]} />;
  } else if (modality === 'embedding' && typeof extra.similarity === 'number') {
    body = <SimilarityView similarity={extra.similarity} />;
  } else if (modality === 'causal_lm') {
    body = <MarkdownText text={result.predicted} className="text-sm text-gray-100" />;
  } else if (['seq2seq', 'vlm', 'asr', 'embedding'].includes(modality)) {
    body = <p className="text-sm text-gray-100 whitespace-pre-wrap break-words leading-relaxed">{result.predicted}</p>;
  } else {
    body = (
      <div>
        <p className="text-sm text-white">
          <span className="font-medium">{result.predicted}</span>
          {typeof result.confidence === 'number' && (
            <span className={`ml-2 tabular-nums ${compact ? 'text-white/65' : 'text-gray-500'}`}>{(result.confidence * 100).toFixed(1)} %</span>
          )}
        </p>
        <Scores r={result} compact={compact} />
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {body}
      <div className={`flex items-center gap-3 text-[11px] ${compact ? 'text-white/55' : 'text-gray-500'}`}>
        {meta.length > 0 && <span className="tabular-nums">{meta.join(' · ')}</span>}
        <CopyButton text={copyText} label={t('hosting.result.copy')} />
      </div>
    </div>
  );
}
