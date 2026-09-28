// Meldungen zu Datasets als Schluessel + Parameter statt fertiger Saetze.
//
// Backend (DatasetHint) und Kompatibilitaets-Plugins liefern einen Code; die
// Oberflaeche uebersetzt ihn in der gewaehlten Sprache. Der deutsche Text
// bleibt jeweils als Fallback daneben stehen (alte Metadaten, Logs).

import { translate } from '../contexts/LanguageContext';

export type MsgParam = string | number | Msg | Msg[];

export interface Msg {
  key:     string;
  params?: Record<string, MsgParam>;
}

/** Spiegelt dataset_manager.rs::DatasetHint */
export interface DatasetHint {
  code:    string;
  params?: Record<string, string | number> | null;
  text?:   string;
}

type TFn = (key: string, paramsOrFallback?: string | Record<string, string | number>) => string;

/** Loest eine Meldung auf; Parameter duerfen selbst Meldungen sein
 *  (z. B. ein Typname, der ebenfalls uebersetzt wird). */
export function msgText(t: TFn, msg: Msg, fallback?: string): string {
  const params: Record<string, string | number> = {};
  for (const [k, v] of Object.entries(msg.params ?? {})) {
    params[k] = Array.isArray(v) ? v.map(m => msgText(t, m)).join(', ')
      : typeof v === 'object' ? msgText(t, v) : v;
  }
  const out = t(msg.key, params);
  return out === msg.key ? (fallback ?? msg.key) : out;
}

/** Deutscher Text einer Meldung — fuer die Fallback-Felder. */
export const msgDe = (msg: Msg): string =>
  msgText((k, p) => translate('de', k, p), msg);

// Metadaten aus Versionen vor den Hinweis-Codes kennen nur den deutschen
// Text. Die bekannten Saetze werden auf ihren Code zurueckgefuehrt, damit
// auch diese Karten in der gewaehlten Sprache erscheinen.
const LEGACY: [RegExp, (m: RegExpMatchArray) => DatasetHint][] = [
  [/^(\d+) Bild\(er\) ohne Label\.$/, m => ({ code: 'orphan_images', params: { count: Number(m[1]) } })],
  [/^Kein Validierungs-Split gefunden/, () => ({ code: 'no_val_split' })],
  [/^(\d+) Audio-Datei\(en\) ohne Transkript\.$/, m => ({ code: 'orphan_audio', params: { count: Number(m[1]) } })],
  [/^Dataset enth(ae|ä)lt gemischte Dateitypen\.$/, () => ({ code: 'mixed_file_types' })],
  [/^images\/ und labels\/ gefunden, aber die Struktur passt nicht/, () => ({ code: 'images_labels_mismatch' })],
  [/^Dataset-Typ konnte nicht erkannt werden\.$/, () => ({ code: 'type_unknown' })],
  [/^Dataset hat (\d+) Konfigurationen \((.*)\)\. Importiert wurde '(.*)'\.$/,
    m => ({ code: 'hf_configs', params: { count: Number(m[1]), configs: m[2], chosen: m[3] } })],
  [/^Splits ohne verwertbare Labels (ue|ü)bersprungen: (.*)\.$/, m => ({ code: 'hf_splits_skipped', params: { splits: m[2] } })],
  [/^Keine Standard-Splits erkannt/, () => ({ code: 'hf_no_standard_splits' })],
];

function ausAltemText(text: string): DatasetHint {
  for (const [re, bau] of LEGACY) {
    const m = text.match(re);
    if (m) return { ...bau(m), text };
  }
  return { code: '', text };
}

/** Die Hinweise eines Datasets in der Sprache der Oberflaeche. */
export function datasetHintTexts(
  t: TFn, hints: DatasetHint[] | null | undefined, warnings: string[] | null | undefined,
): string[] {
  const liste = hints?.length ? hints : (warnings ?? []).map(ausAltemText);
  return liste.map(h => h.code
    ? msgText(t, { key: `datasetHints.${h.code}`, params: h.params ?? undefined }, h.text)
    : (h.text ?? ''));
}
