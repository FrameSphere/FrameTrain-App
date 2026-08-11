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
  onSelect: () => void;
}

type Provider = () => ContextMenuAction[];

const providers = new Set<Provider>();

/** Sammelt alle Aktionen der aktuell gemounteten Seiten ein. */
export function collectContextMenuActions(): ContextMenuAction[] {
  const out: ContextMenuAction[] = [];
  providers.forEach((p) => {
    try { out.push(...p()); } catch { /* defekter Provider blockiert das Menü nicht */ }
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
    const provider: Provider = () => ref.current();
    providers.add(provider);
    return () => { providers.delete(provider); };
  }, []);
}
