// Globale Navigation per Event — erlaubt es Menü/Aktionen von überall die
// Hauptansicht zu wechseln, ohne Props durch den Baum zu reichen.

export type AppView =
  | 'home'
  | 'models' | 'training' | 'dataset' | 'analysis'
  | 'tests' | 'versions' | 'settings' | 'laboratory' | 'synapse' | 'studio' | 'hosting';

const EVENT_NAME = 'ft_navigate';

export function navigateTo(view: AppView) {
  try {
    window.dispatchEvent(new CustomEvent<AppView>(EVENT_NAME, { detail: view }));
  } catch { /* ignore */ }
}

export function onNavigate(handler: (view: AppView) => void) {
  const listener = (e: Event) => handler((e as CustomEvent<AppView>).detail);
  window.addEventListener(EVENT_NAME, listener as EventListener);
  return () => window.removeEventListener(EVENT_NAME, listener as EventListener);
}

// Welcher Einstellungs-Tab beim naechsten Oeffnen der Einstellungen aktiv sein
// soll (z. B. "Schnell-Zugriff" aus dem Hosting). Wird beim Lesen geleert.
let requestedSettingsTab: string | null = null;

export function requestSettingsTab(tab: string) {
  requestedSettingsTab = tab;
}

export function takeRequestedSettingsTab(): string | null {
  const t = requestedSettingsTab;
  requestedSettingsTab = null;
  return t;
}
