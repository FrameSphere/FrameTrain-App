import type { Language } from '../contexts/LanguageContext';

/**
 * BCP-47-Tag für Datums- und Zeitformatierung.
 *
 * Vorher stand in den Komponenten überall fest 'de-DE'. Dadurch blieben
 * Datumsangaben auch im englischen UI im deutschen Format (07.08.26).
 * 'en-GB' behält die Tag-vor-Monat-Reihenfolge und das 24-Stunden-Format
 * bei — das passt zum restlichen UI besser als 'en-US'.
 */
export function dateLocale(language: Language): string {
  return language === 'en' ? 'en-GB' : 'de-DE';
}
