// Aktives Lernen ueber das Labor: was dort falsch oder unsicher war, geht als
// Arbeit in ein Werkstatt-Projekt.
//
// Genau dort traegt ein Mensch am meisten bei. Ein Modell, das bei 97 % der
// Samples richtig liegt, lernt aus weiteren sicheren Treffern kaum etwas —
// aus den 3 %, bei denen es daneben lag oder zoegerte, sehr viel.
//
// Was als bestaetigt ankommt und was als Vorschlag:
//   * Korrektur im Labor          -> bestaetigt (ein Mensch hat sie gesetzt)
//   * als "richtig" bewertet      -> bestaetigt mit der Vorhersage
//   * erwartetes Label im Dataset -> bestaetigt
//   * sonst                       -> Vorschlag des Modells mit seiner Sicherheit

import type { Correction } from '../labCorrection';
import type { TruthBox } from '../labGroundTruth';

export interface LabErgebnis {
  inputText: string;
  filePath?: string;
  expectedLabel?: string;
  correction?: Correction;
  predicted: string;
  confidence?: number;
  userRating: 'correct' | 'wrong' | 'skipped';
}

export interface LabItem {
  filePath:   string | null;
  text:       string | null;
  predicted:  string | null;
  confidence: number | null;
  label:      string | null;
  target:     string | null;
  boxes:      TruthBox[] | null;
}

export type LabAuswahl = 'wrong' | 'uncertain' | 'all';

/** Unter dieser Sicherheit gilt eine Vorhersage als unsicher. */
export const UNSICHER_UNTER = 0.6;

const BILD = /\.(jpe?g|png|webp|gif|bmp|tiff?)$/i;
const AUDIO = /\.(wav|mp3|flac|ogg|m4a|aac|aiff?)$/i;
const VIDEO = /\.(mp4|mov|m4v|webm|mkv|avi)$/i;

/** Welche Art Werkstatt-Projekt zu den Ergebnissen passt. */
export function labModalitaet(ergebnisse: LabErgebnis[]): 'text' | 'image' | 'audio' | 'video' {
  const pfad = ergebnisse.find(r => r.filePath)?.filePath;
  if (!pfad) return 'text';
  if (VIDEO.test(pfad)) return 'video';
  if (AUDIO.test(pfad)) return 'audio';
  if (BILD.test(pfad)) return 'image';
  return 'image';
}

export function labAuswahl(ergebnisse: LabErgebnis[], wie: LabAuswahl, schwelle = UNSICHER_UNTER): LabErgebnis[] {
  if (wie === 'all') return ergebnisse;
  if (wie === 'wrong') return ergebnisse.filter(r => r.userRating === 'wrong' || !!r.correction);
  return ergebnisse.filter(r => r.confidence != null && r.confidence < schwelle);
}

export function labItems(ergebnisse: LabErgebnis[]): LabItem[] {
  return ergebnisse.map(r => {
    const c = r.correction;
    const richtig = r.userRating === 'correct' ? r.predicted : null;
    return {
      filePath: r.filePath ?? null,
      text: r.filePath ? null : r.inputText,
      predicted: r.predicted || null,
      confidence: r.confidence ?? null,
      label: c?.kind === 'label' ? c.label : (richtig ?? r.expectedLabel ?? null),
      target: c?.kind === 'text' ? c.text : null,
      boxes: c?.kind === 'boxes' ? c.boxes : null,
    };
  });
}
