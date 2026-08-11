// Globale Navigation per Event — erlaubt es Menü/Aktionen von überall die
// Hauptansicht zu wechseln, ohne Props durch den Baum zu reichen.

export type AppView =
  | 'models' | 'training' | 'dataset' | 'analysis'
  | 'tests' | 'versions' | 'settings' | 'laboratory' | 'synapse';

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
