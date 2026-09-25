// Rohdaten erzeugen lassen statt sie zu suchen.
//
// Die vierte Art, an Text zu kommen: nicht importieren, nicht holen, nicht
// selbst tippen, sondern schreiben lassen. Gefragt wird das Modell, das in den
// Einstellungen steht — ein trainiertes Modell braucht es dafuer nicht, und
// genau das ist der Punkt: am Anfang eines Projekts gibt es noch keins.
//
// Zwei Dinge sind hier nicht verhandelbar:
//   1. Nichts landet ungesehen im Projekt. Was erzeugt wurde, steht zuerst als
//      Vorschlag da und wird einzeln abgewaehlt oder uebernommen.
//   2. Jede uebernommene Zeile traegt mit, dass ein Modell sie geschrieben hat
//      (Herkunft samt Modellname). Ein Datensatz aus erzeugtem Text ist etwas
//      anderes als einer aus gesammeltem — wer das spaeter nicht mehr
//      auseinanderhalten kann, misst seine Genauigkeit gegen sich selbst.

import { useState } from 'react';
import { Loader2, Sparkles, Check, X } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useNotification } from '../../contexts/NotificationContext';
import { useAISettings } from '../../contexts/AISettingsContext';
import { callAI } from '../../ai/aiClient';
import { resolveModel, PROVIDER_META } from '../../ai/providerMeta';
import { classColor } from '../labGroundTruth';
import {
  parseGenerated, ohneDubletten, beispieleFuer, type GeneratedItem, type VorhandenerText,
} from './generatedTexts';
import type { StudioProject } from './studioTypes';
import ModalPortal from '../ui/ModalPortal';
import { useEscape } from './useEscape';

const MAX_PRO_KLASSE = 50;

export interface GenerateDialogProps {
  project: StudioProject;
  paare: boolean;
  /** Vorhandene Texte: gegen Dubletten und als Beispiel fuer den Stil. */
  vorhanden: VorhandenerText[];
  onCancel: () => void;
  onAdd: (items: GeneratedItem[], origin: string) => void;
}

export default function GenerateDialog({
  project, paare, vorhanden, onCancel, onAdd,
}: GenerateDialogProps) {
  const { t, language } = useLanguage();
  const { error } = useNotification();
  const { settings } = useAISettings();

  const classes = project.classes;
  const [gewaehlt, setGewaehlt] = useState<string[]>(classes);
  const [anzahl, setAnzahl] = useState(10);
  const [thema, setThema] = useState('');
  const [anlehnen, setAnlehnen] = useState(vorhanden.length > 0);
  const [busy, setBusy] = useState(false);
  const [lauf, setLauf] = useState<{ cur: number; total: number } | null>(null);
  const [vorschlaege, setVorschlaege] = useState<GeneratedItem[] | null>(null);
  const [aus, setAus] = useState<Set<number>>(new Set());

  const meta = PROVIDER_META[settings.provider];
  const bereit = settings.enabled && (!meta.needsKey || !!settings.apiKey);
  const modell = resolveModel(settings.provider, settings.selectedModel, settings.ollamaModel);
  const proLauf = Math.min(Math.max(1, anzahl), MAX_PRO_KLASSE);

  const toggleKlasse = (name: string) => {
    setGewaehlt(g => (g.includes(name) ? g.filter(x => x !== name) : [...g, name]));
  };

  // ── Ein Lauf: je Klasse eine Anfrage ─────────────────────────────────────
  //
  // Warum nicht alles in einer: ein Modell, das fuer sechs Klassen gleichzeitig
  // schreibt, verteilt ungleich und wiederholt sich. Je Klasse eine Anfrage
  // kostet mehr Aufrufe, liefert aber gleichmaessig viele Beispiele.
  const auftrag = (klasse: string | null): string => {
    const vorlage = anlehnen ? beispieleFuer(vorhanden, paare ? null : klasse) : [];
    const beispiele = vorlage.length > 0
      ? `\n\n${t('studio.generate.promptExamples')}\n${vorlage.map(b => `- ${b}`).join('\n')}`
      : '';
    const zusatz = thema.trim() ? `\n\n${t('studio.generate.promptTopic')} ${thema.trim()}` : '';
    if (paare) {
      return `${t('studio.generate.promptPairs', { count: proLauf })}${zusatz}${beispiele}`;
    }
    if (klasse) {
      return `${t('studio.generate.promptClass', { count: proLauf, klasse })}${zusatz}${beispiele}`;
    }
    return `${t('studio.generate.promptPlain', { count: proLauf })}${zusatz}${beispiele}`;
  };

  const run = async () => {
    const laeufe: (string | null)[] = paare || classes.length === 0 ? [null] : gewaehlt;
    if (laeufe.length === 0) return;
    setBusy(true);
    setVorschlaege(null);
    setAus(new Set());
    const gesammelt: GeneratedItem[] = [];
    try {
      for (let i = 0; i < laeufe.length; i++) {
        const klasse = laeufe[i];
        setLauf({ cur: i + 1, total: laeufe.length });
        const antwort = await callAI(settings, {
          system: t('studio.generate.system'),
          messages: [{ role: 'user', content: auftrag(klasse) }],
          maxTokens: 4000,
          // Beispiele sollen sich unterscheiden — bei 0 schreibt das Modell
          // zehnmal fast denselben Satz.
          temperature: 0.9,
          style: 'raw',
          responseLanguage: language,
        });
        for (const item of parseGenerated(antwort)) {
          gesammelt.push(klasse ? { ...item, label: klasse } : item);
        }
      }
      const frisch = ohneDubletten(gesammelt, vorhanden.map(v => v.text));
      if (frisch.length === 0) {
        error(t('studio.generate.emptyTitle'), t('studio.generate.emptyDetail'));
        return;
      }
      setVorschlaege(frisch);
    } catch (err: unknown) {
      error(t('studio.generate.errorTitle'), String(err));
    } finally {
      setBusy(false);
      setLauf(null);
    }
  };

  const aendern = (i: number, teil: Partial<GeneratedItem>) => {
    setVorschlaege(v => v ? v.map((item, k) => (k === i ? { ...item, ...teil } : item)) : v);
  };

  const uebernehmen = () => {
    if (!vorschlaege) return;
    // Was beim Korrigieren leer geworden ist, faellt weg.
    const behalten = vorschlaege
      .filter((item, i) => !aus.has(i) && item.text.trim())
      .map(item => ({ ...item, text: item.text.trim() }));
    if (behalten.length === 0) return;
    onAdd(behalten, t('studio.generate.origin', { model: modell }));
  };

  const behalten = vorschlaege ? vorschlaege.length - aus.size : 0;

  useEscape(onCancel, !busy);

  return (
    <ModalPortal><div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
      onClick={busy ? undefined : onCancel}>
      <div className="w-full max-w-xl rounded-2xl border border-white/10 bg-[#101218] p-6 space-y-4 max-h-[85vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}>
        <div>
          <h3 className="text-white font-semibold">{t('studio.generate.title')}</h3>
          <p className="text-gray-500 text-xs mt-1">{t('studio.generate.subtitle')}</p>
        </div>

        {!bereit && (
          <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/25 text-amber-200 text-xs">
            {t('studio.generate.noAI')}
          </div>
        )}

        {vorschlaege === null ? (
          <>
            {!paare && classes.length > 0 && (
              <div>
                <span className="text-gray-400 text-xs">{t('studio.generate.classesLabel')}</span>
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {classes.map(name => (
                    <button key={name} onClick={() => toggleKlasse(name)}
                      className={`px-2.5 py-1.5 rounded-lg border text-xs inline-flex items-center gap-1.5 transition-all ${gewaehlt.includes(name) ? 'bg-white/15 border-white/25 text-white' : 'bg-white/5 border-white/10 text-gray-500 hover:bg-white/10'}`}>
                      <span className="w-2 h-2 rounded-sm flex-shrink-0"
                        style={{ background: classColor(name, classes) }} />
                      {name}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <label className="block">
              <span className="text-gray-400 text-xs">
                {paare || classes.length === 0
                  ? t('studio.generate.countLabel')
                  : t('studio.generate.countPerClassLabel')}
              </span>
              <input type="number" min={1} max={MAX_PRO_KLASSE} value={anzahl}
                onChange={e => setAnzahl(Number(e.target.value) || 1)}
                className="mt-1 w-28 px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-white/25" />
            </label>

            <label className="block">
              <span className="text-gray-400 text-xs">{t('studio.generate.topicLabel')}</span>
              <textarea value={thema} onChange={e => setThema(e.target.value)} rows={3}
                placeholder={t('studio.generate.topicPlaceholder')}
                className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm placeholder-gray-600 focus:outline-none focus:border-white/25 resize-none" />
            </label>

            {vorhanden.length > 0 && (
              <label className="flex items-start gap-2 cursor-pointer">
                <input type="checkbox" checked={anlehnen}
                  onChange={e => setAnlehnen(e.target.checked)} className="mt-0.5" />
                <span className="text-gray-300 text-xs">{t('studio.generate.useExamples')}</span>
              </label>
            )}

            <p className="text-gray-500 text-xs">
              {t('studio.generate.plan', {
                count: (paare || classes.length === 0 ? 1 : gewaehlt.length) * proLauf,
                model: modell,
              })}
            </p>

            <div className="flex gap-2">
              <button onClick={onCancel} disabled={busy}
                className="flex-1 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white text-sm disabled:opacity-40">
                {t('common.cancel', 'Abbrechen')}
              </button>
              <button onClick={() => void run()}
                disabled={busy || !bereit || (!paare && classes.length > 0 && gewaehlt.length === 0)}
                className="flex-1 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm inline-flex items-center justify-center gap-2 disabled:opacity-40">
                {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
                {busy && lauf
                  ? t('studio.generate.running', { cur: lauf.cur, total: lauf.total })
                  : t('studio.generate.runButton')}
              </button>
            </div>
          </>
        ) : (
          <>
            <p className="text-gray-400 text-xs">
              {t('studio.generate.reviewHint', { count: vorschlaege.length })}
            </p>

            <div className="space-y-1.5 max-h-72 overflow-y-auto pr-1">
              {vorschlaege.map((item, i) => (
                <div key={i}
                  className={`flex items-start gap-2 p-2 rounded-lg border text-xs transition-all ${aus.has(i) ? 'bg-white/[0.02] border-white/5 text-gray-600 line-through' : 'bg-white/5 border-white/10 text-gray-200'}`}>
                  <button
                    aria-label={aus.has(i) ? t('studio.generate.keep') : t('studio.generate.drop')}
                    onClick={() => setAus(s => {
                      const n = new Set(s);
                      if (n.has(i)) n.delete(i); else n.add(i);
                      return n;
                    })}
                    className="mt-0.5 p-1 rounded hover:bg-white/10 text-gray-400 flex-shrink-0">
                    {aus.has(i) ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
                  </button>
                  {/* Direkt korrigierbar: ein fast richtiges Beispiel soll nicht
                      verworfen werden muessen, nur weil ein Wort nicht passt. */}
                  <div className="min-w-0 flex-1 space-y-1">
                    <textarea value={item.text} disabled={aus.has(i)}
                      aria-label={t('studio.generate.editItem', { n: i + 1 })}
                      rows={Math.min(5, Math.max(1, Math.ceil(item.text.length / 70)))}
                      onChange={e => aendern(i, { text: e.target.value })}
                      className={`w-full bg-transparent resize-none rounded px-1 -mx-1 focus:outline-none focus:bg-black/30 ${aus.has(i) ? 'line-through' : ''}`} />
                    {item.target !== undefined && item.target !== null && (
                      <div className="flex items-start gap-1 text-gray-400">
                        <span className="mt-0.5">→</span>
                        <textarea value={item.target} disabled={aus.has(i)}
                          aria-label={t('studio.generate.editTarget', { n: i + 1 })}
                          rows={Math.min(5, Math.max(1, Math.ceil(item.target.length / 70)))}
                          onChange={e => aendern(i, { target: e.target.value })}
                          className="w-full bg-transparent resize-none rounded px-1 focus:outline-none focus:bg-black/30" />
                      </div>
                    )}
                  </div>
                  {item.label && (
                    <span className="px-1.5 py-0.5 rounded bg-white/10 text-gray-300 flex-shrink-0 inline-flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-sm"
                        style={{ background: classColor(item.label, classes) }} />
                      {item.label}
                    </span>
                  )}
                </div>
              ))}
            </div>

            <div className="flex gap-2">
              <button onClick={() => setVorschlaege(null)}
                className="flex-1 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white text-sm">
                {t('studio.generate.again')}
              </button>
              <button onClick={uebernehmen} disabled={behalten === 0}
                className="flex-1 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm inline-flex items-center justify-center gap-2 disabled:opacity-40">
                <Check className="w-4 h-4" /> {t('studio.generate.addButton', { count: behalten })}
              </button>
            </div>
          </>
        )}
      </div>
    </div></ModalPortal>
  );
}
