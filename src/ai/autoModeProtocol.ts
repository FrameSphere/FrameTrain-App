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


export function parseAutoAction(text: string): { action: AutoAction | null; cleaned: string } {
  // Try matching the preferred ft_action fence first
  let match = text.match(new RegExp("```" + ACTION_FENCE + "\\s*([\\s\\S]*?)\\s*```", 'm'));
  
  // Fallback: Try matching ```json fence with valid AutoAction JSON
  if (!match) {
    match = text.match(/```json\s*([\s\S]*?)\s*```/m);
    if (match) {
      try {
        const parsed = JSON.parse(match[1].trim());
        // Only accept as AutoAction if it has the expected shape
        if (parsed?.mode && typeof parsed.rationale === 'string') {
          // This looks like an AutoAction, so we'll proceed
        } else {
          match = null;
        }
      } catch {
        match = null;
      }
    }
  }
  
  if (!match) return { action: null, cleaned: text };
  
  const rawJson = match[1].trim();
  try {
    const action = JSON.parse(rawJson) as AutoAction;
    const cleaned = (text.slice(0, match.index) + text.slice((match.index ?? 0) + match[0].length)).trim();
    if (!action || (action.mode !== 'chat' && action.mode !== 'edit' && action.mode !== 'rewrite')) {
      return { action: null, cleaned: text };
    }
    if (typeof action.rationale !== 'string') action.rationale = '';
    return { action, cleaned };
  } catch {
    return { action: null, cleaned: text };
  }
}

