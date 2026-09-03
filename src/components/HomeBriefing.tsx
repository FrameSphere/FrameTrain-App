// KI-Briefing auf der Startseite.
//
// Beantwortet in ein paar Saetzen, was seit dem letzten Blick passiert ist und
// was als naechstes sinnvoll waere — auf Basis derselben Fakten, die die Seite
// ohnehin schon geladen hat.
//
// Bewusst NICHT automatisch: jeder Aufruf kostet Tokens. Der Text wird zusammen
// mit einem Hash der Fakten gespeichert; solange sich am Stand nichts aendert,
// wird beim naechsten Besuch der gespeicherte Text gezeigt statt neu bezahlt.

import { useState, useEffect } from 'react';
import { Sparkles, Loader2, AlertTriangle, RefreshCw, ArrowRight, Settings as SettingsIcon } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useLanguage } from '../contexts/LanguageContext';
import { useAISettings } from '../contexts/AISettingsContext';
import { callAI } from '../ai/aiClient';
import { PROVIDER_META, resolveModel } from '../ai/providerMeta';
import { navigateTo } from '../ui/navigationEvents';
import { MarkdownText } from './ui/MarkdownText';
import { splitBriefing } from './homeInsights';
import { dateLocale } from '../utils/dateLocale';

interface HomeBriefingProps {
  /** Verdichteter Projektstand (siehe buildBriefingFacts). */
  facts: string;
  /** Hash ueber genau diese Fakten — Cache-Schluessel. */
  factsKey: string;
  userId: string;
}

interface CachedBriefing {
  hash: string;
  text: string;
  at: string;
  model: string;
}

const CACHE_PREFIX = 'ft_home_briefing_';

const SYSTEM_PROMPT = {
  de: `Du bist der Trainings-Assistent von FrameTrain, einer Desktop-App fuer lokales ML-Training.
Der Nutzer sieht gerade die Startseite und will in wenigen Sekunden wissen, wo sein Projekt steht.

Antworte in Markdown, genau in dieser Struktur:
1. EIN Einleitungsabsatz mit 1-2 Saetzen zur Gesamtlage.
2. Danach 2 bis 4 Stichpunkte, jeder mit "- " am Zeilenanfang. Ein Stichpunkt = eine Beobachtung.
3. Zuletzt eine eigene Zeile, die genau so beginnt: "**Naechster Schritt:** " — dahinter EIN konkreter Schritt, den der Nutzer in dieser App tun kann.

Formatregeln:
- Setze Modellnamen, Zahlen und Kennwerte (Loss, Accuracy) in **Fettschrift**.
- Keine Ueberschriften, keine Code-Bloecke, keine Tabellen, keine verschachtelten Listen.
- Hoechstens 150 Woerter insgesamt.

Inhaltsregeln:
- Beginne mit dem Wichtigsten, das seit den letzten Laeufen passiert ist.
- Nenne konkrete Zahlen aus den Fakten, wenn sie etwas aussagen.
- Erfinde nichts. Stehen kaum Daten zur Verfuegung, sage das offen und schlage den ersten Schritt vor.`,
  en: `You are the training assistant of FrameTrain, a desktop app for local ML training.
The user is looking at the home screen and wants to know within seconds where their project stands.

Answer in Markdown, in exactly this structure:
1. ONE opening paragraph of 1-2 sentences on the overall situation.
2. Then 2 to 4 bullet points, each starting with "- ". One bullet = one observation.
3. Finally a line of its own starting exactly with: "**Next step:** " — followed by ONE concrete step the user can take inside this app.

Formatting rules:
- Put model names, numbers and metrics (loss, accuracy) in **bold**.
- No headings, no code blocks, no tables, no nested lists.
- 150 words maximum.

Content rules:
- Lead with the most important thing that happened in the recent runs.
- Quote concrete numbers from the facts where they say something.
- Invent nothing. If there is barely any data, say so plainly and suggest the first step.`,
};

export default function HomeBriefing({ facts, factsKey, userId }: HomeBriefingProps) {
  const { currentTheme } = useTheme();
  const { t, language } = useLanguage();
  const { settings } = useAISettings();

  const [cached, setCached] = useState<CachedBriefing | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const cacheKey = CACHE_PREFIX + userId;

  useEffect(() => {
    try {
      const raw = localStorage.getItem(cacheKey);
      setCached(raw ? (JSON.parse(raw) as CachedBriefing) : null);
    } catch {
      setCached(null);
    }
  }, [cacheKey]);

  const meta = PROVIDER_META[settings.provider];
  const aiReady = settings.enabled && (!meta.needsKey || !!settings.apiKey);
  // Der gespeicherte Text passt nur, solange er zu genau diesem Stand gehoert.
  const isStale = !!cached && cached.hash !== factsKey;

  const generate = async () => {
    setLoading(true);
    setError(null);
    try {
      const text = await callAI(settings, {
        system: SYSTEM_PROMPT[language === 'en' ? 'en' : 'de'],
        messages: [{ role: 'user', content: `${t('home.briefing.userPrompt')}\n\n${facts}` }],
        maxTokens: 700,
        temperature: 0.4,
        responseLanguage: language,
      });
      const trimmed = text.trim();
      if (!trimmed) throw new Error(t('home.briefing.emptyAnswer'));
      const entry: CachedBriefing = {
        hash: factsKey,
        text: trimmed,
        at: new Date().toISOString(),
        model: resolveModel(settings.provider, settings.selectedModel, settings.ollamaModel),
      };
      try { localStorage.setItem(cacheKey, JSON.stringify(entry)); } catch { /* Quota — der Text steht trotzdem */ }
      setCached(entry);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const split = splitBriefing(cached?.text ?? '');

  const header = (
    <div className="flex items-center gap-2">
      <Sparkles className="w-4 h-4 text-gray-300" />
      <h3 className="text-sm font-semibold text-white">{t('home.briefing.title')}</h3>
    </div>
  );

  // KI aus oder ohne Key — hier hilft nur der Weg in die Einstellungen.
  if (!aiReady) {
    return (
      <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
        {header}
        <p className="text-sm text-gray-400 mt-3">{t('home.briefing.disabled')}</p>
        <button
          onClick={() => navigateTo('settings')}
          className="mt-4 inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-sm text-gray-300 hover:bg-white/10 hover:text-white transition-all"
        >
          <SettingsIcon className="w-4 h-4" />
          {t('home.briefing.toSettings')}
        </button>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
      <div className="flex items-start justify-between gap-4">
        {header}
        {cached && !loading && (
          <button
            onClick={() => void generate()}
            title={t('home.briefing.regenerate')}
            className="p-1.5 rounded-lg text-gray-500 hover:text-white hover:bg-white/10 transition-all flex-shrink-0"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-sm text-gray-400 mt-3">
          <Loader2 className="w-4 h-4 animate-spin" />
          {t('home.briefing.loading', { provider: meta.label })}
        </div>
      )}

      {!loading && error && (
        <div className="flex items-start gap-2 mt-3 text-sm text-red-300">
          <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <span className="min-w-0 break-words">{error}</span>
        </div>
      )}

      {!loading && cached && (
        <>
          <div className="mt-3 text-gray-300">
            <MarkdownText text={split.body} className="space-y-1.5" />
          </div>
          {split.nextStep && (
            <div className="mt-3 flex items-start gap-2.5 rounded-xl border border-white/10 bg-white/[0.07] px-3 py-2.5 text-gray-200">
              <ArrowRight className="w-4 h-4 mt-0.5 flex-shrink-0" style={{ color: currentTheme.colors.accent }} />
              <MarkdownText text={split.nextStep} />
            </div>
          )}
          <p className="text-xs text-gray-600 mt-3">
            {t('home.briefing.meta', {
              model: cached.model,
              when: new Date(cached.at).toLocaleString(dateLocale(language), {
                day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit',
              }),
            })}
            {isStale && <span className="text-amber-400/80"> · {t('home.briefing.stale')}</span>}
          </p>
        </>
      )}

      {!loading && (!cached || isStale) && (
        <button
          onClick={() => void generate()}
          className={`mt-4 inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-xl text-white text-sm font-medium hover:opacity-90 transition-all`}
        >
          <Sparkles className="w-4 h-4" />
          {cached ? t('home.briefing.refresh') : t('home.briefing.create')}
        </button>
      )}

      {!loading && !cached && (
        <p className="text-xs text-gray-600 mt-2">{t('home.briefing.costHint')}</p>
      )}
    </div>
  );
}
