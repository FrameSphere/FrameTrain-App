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

// Welches Modell die Zielseite vorauswaehlen soll (Aktionen der Startseite).
// Gilt nur fuer genau diese Seite und wird beim Lesen geleert.
export interface ModelFocus { modelId: string; versionId?: string | null; datasetId?: string | null }

let requestedFocus: { view: AppView; focus: ModelFocus } | null = null;

export function navigateToModel(view: AppView, focus: ModelFocus) {
  requestedFocus = { view, focus };
  navigateTo(view);
}

export function takeModelFocus(view: AppView): ModelFocus | null {
  if (requestedFocus?.view !== view) return null;
  const { focus } = requestedFocus;
  // Erst nach dem Render leeren: StrictMode ruft Effekte im Dev-Modus doppelt auf.
  setTimeout(() => { requestedFocus = null; }, 0);
  return focus;
}
