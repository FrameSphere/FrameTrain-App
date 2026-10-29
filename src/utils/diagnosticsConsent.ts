// Abgleich des Schalters "Fehler automatisch melden" mit dem Konto.
//
// Quelle der Wahrheit ist das Konto (users.diagnostics_consent, gesetzt bei der
// Registrierung oder hier in der App). Lokal liegt nur eine Kopie, damit der
// globale Fehler-Handler ohne Netz entscheiden kann. Klappt das Zurückschreiben
// nicht (offline), merkt sich die App die Wahl als "pending" und holt sie beim
// nächsten Login nach — sonst würde ein offline erklärter Widerruf vom alten
// Server-Wert überschrieben.

import { invoke } from '@tauri-apps/api/core';
import { setAutoErrorReportingEnabled } from './errorReport';

export const PENDING_CONSENT_KEY = 'ft_auto_error_reporting_pending';

function readPending(): boolean | null {
  try {
    const v = localStorage.getItem(PENDING_CONSENT_KEY);
    return v === 'on' ? true : v === 'off' ? false : null;
  } catch { return null; }
}

function writePending(on: boolean | null): void {
  try {
    if (on === null) localStorage.removeItem(PENDING_CONSENT_KEY);
    else localStorage.setItem(PENDING_CONSENT_KEY, on ? 'on' : 'off');
  } catch { /* ohne Speicher kein Nachholen */ }
}

async function pushToAccount(apiKey: string, on: boolean): Promise<boolean> {
  try {
    await invoke('set_diagnostics_consent', { apiKey, consent: on });
    writePending(null);
    return true;
  } catch {
    writePending(on);
    return false;
  }
}

/** Schalter in der App: wirkt sofort lokal, wird ins Konto geschrieben. */
export async function changeDiagnosticsConsent(apiKey: string, on: boolean): Promise<boolean> {
  setAutoErrorReportingEnabled(on);
  return pushToAccount(apiKey, on);
}

/**
 * Nach dem Login: Wert aus dem Konto übernehmen. Eine offline getroffene Wahl
 * hat Vorrang und wird nachgereicht. `undefined` = Server kennt das Feld noch
 * nicht, dann bleibt die lokale Einstellung.
 */
export async function applyAccountDiagnosticsConsent(apiKey: string, fromAccount: boolean | null | undefined): Promise<void> {
  const pending = readPending();
  if (pending !== null) {
    setAutoErrorReportingEnabled(pending);
    await pushToAccount(apiKey, pending);
    return;
  }
  if (typeof fromAccount === 'boolean') setAutoErrorReportingEnabled(fromAccount);
}
