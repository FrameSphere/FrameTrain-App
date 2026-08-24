import { useEffect } from 'react';

/**
 * Schliesst einen Dialog per Escape-Taste. Bisher reagierte nur ein Teil der
 * Modals auf Escape — das war inkonsistent. Der Hook registriert den Listener
 * nur, solange `enabled` (der Dialog offen) ist, damit geschlossene Dialoge
 * kein Escape abfangen.
 *
 * @param onEscape  Wird bei Escape aufgerufen (typischerweise onClose).
 * @param enabled   Nur lauschen, wenn true (Dialog offen). Default: true.
 */
export function useEscapeKey(onEscape: () => void, enabled: boolean = true): void {
  useEffect(() => {
    if (!enabled) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onEscape();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onEscape, enabled]);
}
