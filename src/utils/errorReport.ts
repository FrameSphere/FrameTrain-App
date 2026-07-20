// Zentraler Helper für Fehler-Reports an das FrameTrain-Team (WebControl HQ).
// Wird von TrainingDashboard (Standard-/Dev-Training) und DevTrainPanel genutzt.
//
// API-Kontrakt (POST /api/app-errors): `error_type` und `message` sind Pflicht.

export const APP_ERROR_ENDPOINT =
  'https://webcontrol-hq-api.karol-paschek.workers.dev/api/app-errors';

export interface AppErrorReport {
  /** z. B. "training:memory", "devtrain:code", "synapse:shape" */
  error_type: string;
  title: string;
  message: string;
  details?: string;
  logs?: string;
  script_full?: string;
  error_analysis?: string;
  error_category?: string;
  config?: Record<string, unknown>;
}

/** Sendet einen Fehler-Report. Liefert true bei HTTP 2xx, wirft bei Netzwerkfehler. */
export async function sendAppErrorReport(report: AppErrorReport): Promise<boolean> {
  const response = await fetch(APP_ERROR_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      site_id: 'frametrain',
      platform: navigator.platform || 'unknown',
      app_version: 'desktop-app',
      timestamp: new Date().toISOString(),
      ...report,
    }),
  });
  return response.ok;
}
