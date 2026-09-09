export type AutoMode = 'auto' | 'chat' | 'edit';

export type AutoAction = {
  mode: 'chat' | 'edit' | 'rewrite';
  rationale: string;
  title?: string;
  // When mode === edit: assistant should also include ##EDIT_START## blocks.
  // When mode === rewrite: assistant should also include a full ```python``` block.
};

const ACTION_FENCE = 'ft_action';

export function buildAutoSystemPrompt(base: string) {
  return `${base}

AUTO-MODUS (Antwortformat):
Du musst IMMER mit folgendem Format starten — NICHT mit \`\`\`json\`\`\`:

\`\`\`ft_action
{"mode":"chat|edit|rewrite","rationale":"Deine kurze Begründung","title":"Optional: Kurztitel"}
\`\`\`

WICHTIG: Verwende IMMER \`\`\`ft_action\`\`\` nicht \`\`\`json\`\`\`!
Die "rationale" ist nur eine interne Kurznotiz (max. 10 Woerter) und wird dem
User NICHT angezeigt. Die eigentliche Erklaerung steht IMMER im Text darunter —
niemals ausschliesslich in der rationale.

Dann antworte normal:
- Bei mode="chat": Beantwortung + optional 1-2 Rückfragen
- Bei mode="edit": Normale Erklärung + ##EDIT_START## Blöcke mit find/replace
- Bei mode="rewrite": Normale Erklärung + kompletter \`\`\`python\`\`\` Block

EDIT-Format (WICHTIG: Kein Code-Fence drum herum, kein \`\`\`python in FIND/REPLACE \u2013 nur roher Code):
##EDIT_START##
FIND:
...alter code exakt wie im Skript...
REPLACE:
...neuer code...
##EDIT_END##`;
}


/**
 * Erstes vollstaendiges JSON-Objekt ab `from` — mit Klammerzaehlung statt
 * Regex, damit verschachtelte Objekte und Klammern in Strings nicht stoeren.
 */
function firstJsonObject(input: string, from: number): { json: string; end: number } | null {
  const start = input.indexOf('{', from);
  if (start < 0) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < input.length; i++) {
    const c = input[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (c === '\\') escaped = true;
      else if (c === '"') inString = false;
      continue;
    }
    if (c === '"') inString = true;
    else if (c === '{') depth++;
    else if (c === '}') {
      depth--;
      if (depth === 0) return { json: input.slice(start, i + 1), end: i + 1 };
    }
  }
  return null;
}

/**
 * Liest den Steuerblock am Anfang der Antwort und schneidet ihn aus dem Text.
 *
 * Der Block kommt nicht immer sauber eingerahmt zurueck: manche Modelle
 * schliessen den ```ft_action-Block nicht, andere setzen ihn in einen
 * ```json-Block. Blieb er unerkannt, stand das rohe Steuer-JSON
 * ("{"mode":"edit","rationale":...}") als Code-Block sichtbar im Chat.
 * Deshalb wird das JSON per Klammerzaehlung gelesen und der Bereich vom
 * Fence bis zur schliessenden Klammer (samt optionalem End-Fence) entfernt.
 */
export function parseAutoAction(text: string): { action: AutoAction | null; cleaned: string } {
  // 1. Regulaer: der Block steckt in einem ```ft_action- oder ```json-Fence.
  const fence = new RegExp('```[ \\t]*(?:' + ACTION_FENCE + '|json)\\b', 'i');
  let hit: { index: number; length: number } | null = null;
  const fenced = fence.exec(text);
  if (fenced) {
    hit = { index: fenced.index, length: fenced[0].length };
  } else {
    // 2. Ohne Fence: groq/compound-mini schreibt schlicht die Zeile
    //    "ft_action" und darunter das JSON. Ohne diesen Fall stand der
    //    Steuerblock als Text ueber jeder Antwort im Chat.
    //    Bewusst eng: nur am ANFANG der Antwort und nur mit gueltigem mode.
    const bare = new RegExp('^\\s*(?:' + ACTION_FENCE + '\\s*)?(?=\\{)', 'i').exec(text);
    if (bare) hit = { index: bare.index, length: bare[0].length };
  }
  if (!hit) return { action: null, cleaned: text };

  const found = firstJsonObject(text, hit.index + hit.length);
  if (!found) return { action: null, cleaned: text };

  let action: AutoAction;
  try {
    action = JSON.parse(found.json) as AutoAction;
  } catch {
    return { action: null, cleaned: text };
  }
  if (!action || (action.mode !== 'chat' && action.mode !== 'edit' && action.mode !== 'rewrite')) {
    return { action: null, cleaned: text };
  }
  if (typeof action.rationale !== 'string') action.rationale = '';

  // Ein direkt folgender End-Fence gehoert mit weg.
  const after = text.slice(found.end);
  const closing = after.match(/^\s*```/);
  const cutEnd = found.end + (closing ? closing[0].length : 0);
  const cleaned = (text.slice(0, hit.index) + text.slice(cutEnd)).trim();
  return { action, cleaned };
}

