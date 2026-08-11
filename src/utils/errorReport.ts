// Zentraler Helper für Fehler-Reports an das FrameTrain-Team (WebControl HQ).
// Genutzt von TrainingDashboard/DevTrainPanel (explizite Reports) UND vom
// globalen Handler (installGlobalErrorReporting), der jeden uncaught Fehler
// in der App automatisch an den Manager meldet → speist die Auto-Fix-Pipeline.
//
// API-Kontrakt (POST /api/app-errors): `error_type` und `message` sind Pflicht.

export const APP_ERROR_ENDPOINT =
  'https://webcontrol-hq-api.karol-paschek.workers.dev/api/app-errors';

export interface AppErrorReport {
  /** z. B. "training:memory", "devtrain:code", "synapse:shape", "runtime:uncaught" */
  error_type: string;
  title: string;
  message: string;
  details?: string;
  logs?: string;
  script_full?: string;
  error_analysis?: string;
  error_category?: string;
  config?: Record<string, unknown>;
  /** App-Screen/Seite, auf der der Fehler auftrat (z. B. "training", "synapse"). */
  screen?: string;
}

// ── Laufzeit-Kontext, den die globalen Handler mitschicken ──────────
let currentScreen = 'unknown';
let appVersion = 'desktop-app';

/** Von der App aufgerufen, wenn der User die Seite wechselt (siehe PageContext). */
export function setCurrentScreen(screen: string | null | undefined): void {
  if (screen) currentScreen = String(screen);
}

/** Sendet einen Fehler-Report. Liefert true bei HTTP 2xx, wirft bei Netzwerkfehler. */
export async function sendAppErrorReport(report: AppErrorReport): Promise<boolean> {
  const response = await fetch(APP_ERROR_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      site_id: 'frametrain',
      platform: navigator.platform || 'unknown',
      app_version: appVersion,
      screen: report.screen || currentScreen,
      timestamp: new Date().toISOString(),
      ...report,
    }),
  });
  return response.ok;
}

// ── Globales, automatisches Error-Reporting ─────────────────────────
// Fängt window.onerror + unhandledrejection ab und meldet sie einmalig
// pro Signatur (Client-seitige Drossel; der Server dedupliziert zusätzlich).
const seen = new Map<string, number>();
const THROTTLE_MS = 60_000;

function shouldSend(signature: string): boolean {
  const now = Date.now();
  const last = seen.get(signature) || 0;
  if (now - last < THROTTLE_MS) return false;
  seen.set(signature, now);
  // Map klein halten
  if (seen.size > 200) seen.clear();
  return true;
}

function reportRuntime(error_type: string, message: string, stack?: string): void {
  const msg = String(message || 'Unknown error').slice(0, 1000);
  if (!msg || msg === 'Script error.') return; // CSP/cross-origin Rauschen ignorieren
  if (!shouldSend(error_type + '|' + msg.slice(0, 120))) return;
  // Fire-and-forget; Netzwerkfehler dürfen die App nie beeinflussen
  sendAppErrorReport({
    error_type,
    title: msg.slice(0, 120),
    message: msg,
    details: stack ? String(stack).slice(0, 4000) : undefined,
    screen: currentScreen,
  }).catch(() => { /* offline o.ä. – bewusst geschluckt */ });
}

let installed = false;

/**
 * Einmalig beim App-Start aufrufen (in main.tsx). Registriert globale
 * Fehler-Handler und lädt die echte App-Version (Tauri) nach.
 */
export function installGlobalErrorReporting(): void {
  if (installed) return;
  installed = true;

  // Echte Version aus Tauri nachladen (best effort)
  import('@tauri-apps/api/app')
    .then(m => m.getVersion())
    .then(v => { if (v) appVersion = v; })
    .catch(() => { /* nicht im Tauri-Kontext */ });

  window.addEventListener('error', (e: ErrorEvent) => {
    const err = e.error;
    reportRuntime('runtime:uncaught', e.message || (err && err.message) || 'Uncaught error', err && err.stack);
  });

  window.addEventListener('unhandledrejection', (e: PromiseRejectionEvent) => {
    const reason: any = e.reason;
    const message = reason?.message || (typeof reason === 'string' ? reason : JSON.stringify(reason)?.slice(0, 500)) || 'Unhandled promise rejection';
    reportRuntime('runtime:promise', message, reason?.stack);
  });
}
