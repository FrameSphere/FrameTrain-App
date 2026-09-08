/**
 * Findet das letzte JSON-Objekt in einer KI-Antwort.
 *
 * Modelle liefern ihre Parameter mal in einem ```json-Block, mal roh am Ende
 * des Textes — und wenn max_tokens erreicht ist, mitten im Objekt abgeschnitten:
 *
 *   {"epochs":100,"optimizer":"sgd","logging_steps":20,
 *
 * Ohne schliessende Klammer scheiterte jeder Parser, und ausgerechnet die als
 * String notierten Empfehlungen ("optimizer": "sgd") gingen verloren. Deshalb
 * wird ein angeschnittenes Objekt bis zum letzten vollstaendigen Paar
 * zurueckgeschnitten und selbst geschlossen.
 *
 * Bei mehreren Treffern gewinnt der letzte: Berichte nennen erst den
 * Ist-Zustand und danach die Empfehlung.
 */
export function findLastJsonObject(text: string): Record<string, unknown> | null {
  const parse = (candidate: string): Record<string, unknown> | null => {
    try {
      const parsed = JSON.parse(candidate.trim());
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch { /* naechste Strategie */ }
    return null;
  };

  const lastMatch = (re: RegExp, group: number): Record<string, unknown> | null => {
    let hit: Record<string, unknown> | null = null;
    for (const m of text.matchAll(re)) hit = parse(m[group]) ?? hit;
    return hit;
  };

  // 1. Geschlossener Code-Block.
  const fenced = lastMatch(/```json\s*([\s\S]*?)```/gi, 1) ?? lastMatch(/```\s*(\{[\s\S]*?\})\s*```/g, 1);
  if (fenced) return fenced;

  // 2. Freistehendes JSON-Objekt irgendwo im Text.
  const loose = lastMatch(/\{[^{}]*\}/g, 0);
  if (loose) return loose;

  // 3. Am Token-Limit abgeschnittenes Objekt.
  return repairTruncated(text);
}

/** Schliesst ein angeschnittenes Objekt nach dem letzten vollstaendigen Paar. */
function repairTruncated(text: string): Record<string, unknown> | null {
  const open = text.lastIndexOf('{');
  if (open < 0 || text.indexOf('}', open) >= 0) return null;
  const partial = text.slice(open);
  const lastComma = partial.lastIndexOf(',');
  if (lastComma <= 0) return null;
  try {
    const parsed = JSON.parse(`${partial.slice(0, lastComma)}}`);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch { /* nicht reparierbar */ }
  return null;
}
