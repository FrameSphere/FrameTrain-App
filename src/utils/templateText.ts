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

export function templateDescription(tpl: { description?: string | null; source?: string }, t: Translate): string {
  const d = tpl.description ?? '';
  return tpl.source === 'ai' ? swapPrefix(d, DESC_PREFIXES, t('analysisPanel.templates.aiDescriptionPrefix')) : d;
}
