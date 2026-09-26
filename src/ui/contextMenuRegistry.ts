// Registry für das app-weite Rechtsklick-Menü.
//
// Jede Seite registriert ihre Aktionen über useContextMenuActions() — solange
// die Seite gemountet ist, erscheinen ihre Aktionen im Menü. Das Menü ruft
// die Provider erst beim Öffnen auf, Labels/Disabled-Zustände sind damit
// immer aktuell (Provider laufen mit frischem Component-State).

import { useEffect, useRef } from 'react';
import type { LucideIcon } from 'lucide-react';

export interface ContextMenuAction {
  id: string;
  /** Bereits übersetztes Label (Provider hat useLanguage) */
  label: string;
  /** Gruppen-Überschrift (bereits übersetzt); Aktionen gleicher Gruppe stehen zusammen */
  group?: string;
  icon?: LucideIcon;
  disabled?: boolean;
  /** Rot hervorgehoben — fuer Loeschen und Entfernen. */
  danger?: boolean;
  /** Tastenkuerzel, rechts in der Zeile angezeigt (z. B. "⌘D"). Nur Anzeige —
   *  ausgefuehrt wird es von der Seite selbst. So lernt man die Kuerzel beim
   *  Klicken, statt sie in einer Hilfe nachlesen zu muessen. */
  shortcut?: string;
  /** Untermenue, etwa die Klassen eines Projekts. */
  submenu?: ContextMenuAction[];
  onSelect: () => void;
}

/** Worauf rechts geklickt wurde — damit eine Seite passende Aktionen zeigt
 *  (die Box unter dem Zeiger, die Karte eines Datensatzes). */
export interface ContextMenuContext {
  target: HTMLElement | null;
}

type Provider = (ctx: ContextMenuContext) => ContextMenuAction[];

const providers = new Set<Provider>();

/** Sammelt alle Aktionen der aktuell gemounteten Seiten ein. */
export function collectContextMenuActions(ctx: ContextMenuContext = { target: null }): ContextMenuAction[] {
  const out: ContextMenuAction[] = [];
  providers.forEach((p) => {
    try { out.push(...p(ctx)); } catch { /* defekter Provider blockiert das Menü nicht */ }
  });
  return out;
}

/**
 * Registriert Seiten-Aktionen fürs Rechtsklick-Menü (solange gemountet).
 * Die factory wird bei jedem Menü-Öffnen frisch aufgerufen — einfach den
 * aktuellen State/Handler der Komponente verwenden.
 */
export function useContextMenuActions(factory: Provider): void {
  const ref = useRef(factory);
  ref.current = factory;
  useEffect(() => {
    const provider: Provider = (ctx) => ref.current(ctx);
    providers.add(provider);
    return () => { providers.delete(provider); };
  }, []);
}
