import type { Language } from '../contexts/LanguageContext';
import { dateLocale } from './dateLocale';
// KI-Vorlagen speichern Name und Beschreibung als Text — in der Sprache, die
// beim Anlegen eingestellt war ("KI-Empfehlung · distilbert v6"). Angezeigt
// wird der feste Vorsatz in der aktuellen Sprache.

type Translate = (key: string, paramsOrFallback?: string | Record<string, string | number>) => string;

const NAME_PREFIXES = ['KI-Empfehlung', 'AI Recommendation'];
const DESC_PREFIXES = ['KI-empfohlene Parameter', 'AI-recommended parameters'];

function swapPrefix(text: string, prefixes: string[], current: string): string {
  const hit = prefixes.find(p => text.startsWith(p));
  return hit ? current + text.slice(hit.length) : text;
}

export function templateName(tpl: { name: string; source?: string }, t: Translate): string {
  return tpl.source === 'ai' ? swapPrefix(tpl.name, NAME_PREFIXES, t('analysisPanel.templates.sourceAI')) : tpl.name;
}

export function templateDescription(tpl: { description?: string | null; source?: string }, t: Translate, language?: Language): string {
  const d = tpl.description ?? '';
  if (tpl.source !== 'ai') return d;
  const swapped = swapPrefix(d, DESC_PREFIXES, t('analysisPanel.templates.aiDescriptionPrefix'));
  return language ? relocalizeDate(swapped, language) : swapped;
}

/** Das Datum steht als Text in der Beschreibung — im Format der Sprache von
 *  damals ("07.09.2026, 23:33" auch in der englischen Oberflaeche). */
export function relocalizeDate(text: string, language: Language): string {
  return text.replace(/\b(\d{2})[./](\d{2})[./](\d{4}), (\d{2}):(\d{2})\b/, (all, dd, mm, yyyy, hh, mi) => {
    const date = new Date(Number(yyyy), Number(mm) - 1, Number(dd), Number(hh), Number(mi));
    if (Number.isNaN(date.getTime())) return all;
    return date.toLocaleDateString(dateLocale(language), { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
  });
}
