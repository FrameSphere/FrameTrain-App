// Fehler aus der Python-Engine kommen als {error: Titel, details: Text}.
// Der Titel ist deutsch (die Engine kennt die Sprache der Oberflaeche nicht),
// und die Details fielen bisher ganz weg — im Test stand dann nur
// "Fehlendes Python-Paket", ohne zu sagen welches.

import { brokenPackageFromError, type BrokenPackage } from '../components/usePackageInstall';

type Translate = (key: string, paramsOrFallback?: string | Record<string, string | number>) => string;

export interface EngineErrorData { error?: string; details?: string }

const TITLE_KEYS: Record<string, string> = {
  'Fehlendes Python-Paket': 'missingPackage',
  'Datei nicht gefunden': 'fileNotFound',
  'Datei/Ordner nicht gefunden': 'fileNotFound',
  'Dataset nicht gefunden': 'datasetNotFound',
  'Config nicht gefunden': 'configNotFound',
  'Konfigurationsfehler': 'configError',
  'Modell nicht initialisiert': 'modelNotInitialized',
  'Model nicht verfügbar': 'modelUnavailable',
  'Zu wenig Klassen': 'tooFewClasses',
  'Ultralytics nicht installiert': 'ultralyticsMissing',
  'Pipeline nicht unterstuetzt': 'pipelineUnsupported',
};

/** Erste aussagekraeftige Zeile der Details (ohne Traceback). */
function firstDetailLine(details: string): string {
  const line = details.split('\n').map(l => l.trim())
    .find(l => l && !l.startsWith('Traceback') && !l.startsWith('File "'));
  return (line ?? '').slice(0, 300);
}

export function engineErrorText(data: EngineErrorData | undefined, t: Translate, fallback: string): string {
  if (!data?.error) return fallback;
  const key = TITLE_KEYS[data.error];
  const title = key ? t(`engineErrors.${key}`, data.error) : data.error;
  const detail = data.details ? firstDetailLine(data.details) : '';
  return detail && detail !== data.error ? `${title}: ${detail}` : title;
}

export function engineErrorBroken(data: EngineErrorData | undefined): BrokenPackage | null {
  return brokenPackageFromError(data?.details) ?? brokenPackageFromError(data?.error);
}
