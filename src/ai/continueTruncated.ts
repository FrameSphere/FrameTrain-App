// Abgeschnittene Code-Antworten fortsetzen (Code-Assistent in Dev Train / Dev Test).
//
// Ein komplettes Trainingsskript (z. B. Chronos-2 mit Fenstern, Quantil-Loss
// und Validierung) passt oft nicht in ein Antwort-Budget. Die Antwort endete
// mitten im Code, wurde zu Recht NICHT in die Datei geschrieben — und der User
// sah nur den Hinweis aufs Token-Limit. Jetzt fragt der Assistent nach der
// Fortsetzung und setzt die Teile zusammen, bis der Codeblock geschlossen ist.

import { parseAutoAction } from './autoModeProtocol';

export const MAX_CONTINUATIONS = 3;

export const CONTINUE_PROMPT =
  'Deine letzte Antwort wurde am Token-Limit abgeschnitten. Setze sie GENAU nach dem letzten Zeichen fort — '
  + 'ohne ft_action-Block, ohne Einleitung, ohne Wiederholung von bereits geschriebenem Code, '
  + 'ohne neuen ```python-Fence. Schliesse den Codeblock mit ``` ab, sobald das Skript fertig ist.';

/** Ist in `text` ein ```python-Block geoeffnet, aber nicht geschlossen? */
export function hasOpenCodeBlock(text: string): boolean {
  const open = text.lastIndexOf('```python');
  if (open < 0) return false;
  return !text.slice(open + '```python'.length).includes('```');
}

/**
 * Haengt eine Fortsetzung an. Entfernt, was Modelle trotz Anweisung oft
 * mitschicken: einen neuen Steuerblock, einen neuen ```python-Fence und die
 * wiederholte letzte (halbe) Zeile.
 */
export function joinContinuation(soFar: string, next: string): string {
  let rest = next;
  const { action, cleaned } = parseAutoAction(rest);
  if (action) rest = cleaned;
  if (hasOpenCodeBlock(soFar)) {
    rest = rest.replace(/^\s*```(?:python|py)[^\n]*\n/, '');
  }
  // Ueberlappung: laengstes Ende von soFar, mit dem rest beginnt (max. 400 Zeichen).
  const tail = soFar.slice(-400);
  for (let n = Math.min(tail.length, rest.length); n >= 8; n--) {
    if (tail.endsWith(rest.slice(0, n))) {
      rest = rest.slice(n);
      break;
    }
  }
  // Wiederholte angefangene Zeile: Modelle beginnen gern die letzte Zeile neu.
  // Exakter Vergleich inkl. Einrueckung — sonst wuerde eine echte Fortsetzung,
  // die zufaellig mit denselben Buchstaben beginnt, die halbe Zeile ersetzen.
  const lastLine = soFar.slice(soFar.lastIndexOf('\n') + 1);
  if (lastLine.trim().length >= 2 && rest.split('\n', 1)[0].startsWith(lastLine)) {
    return soFar.slice(0, soFar.length - lastLine.length) + rest;
  }
  return soFar + rest;
}

/** Soll eine abgeschnittene Antwort fortgesetzt werden? */
export function shouldContinue(text: string, mode?: string | null): boolean {
  if (hasOpenCodeBlock(text)) return true;
  // Rewrite angekuendigt, aber der Code hat noch gar nicht begonnen.
  return mode === 'rewrite' && !text.includes('```python');
}

/**
 * Setzt eine abgeschnittene Antwort bis zu MAX_CONTINUATIONS-mal fort.
 * `call(soFar)` holt die naechste Fortsetzung und meldet, ob auch sie
 * abgeschnitten wurde. Rueckgabe: zusammengesetzter Text und ob er am Ende
 * immer noch unvollstaendig ist.
 */
export async function continueTruncated(
  first: string,
  mode: string | null | undefined,
  call: (soFar: string) => Promise<{ text: string; cut: boolean }>,
  onRound?: (round: number) => void,
): Promise<{ text: string; cut: boolean }> {
  let text = first;
  for (let round = 1; round <= MAX_CONTINUATIONS; round++) {
    if (!shouldContinue(text, mode)) return { text, cut: false };
    onRound?.(round);
    const next = await call(text);
    text = joinContinuation(text, next.text);
    if (!next.cut) {
      // Regulaer beendet: ein vergessener schliessender Fence wird ergaenzt,
      // statt erneut nachzufragen.
      if (hasOpenCodeBlock(text)) text = `${text.trimEnd()}\n\`\`\``;
      return { text, cut: false };
    }
  }
  return { text, cut: hasOpenCodeBlock(text) || shouldContinue(text, mode) };
}
