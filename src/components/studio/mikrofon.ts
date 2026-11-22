// Warum das Mikrofon nicht geht. "Systemeinstellungen oeffnen" hilft nur, wenn
// die Freigabe fehlt; ohne angeschlossenes Mikrofon sucht man dort vergeblich.

export function mikrofonFehler(err: unknown): { title: string; detail: string; text: string } {
  const name = (err as { name?: string } | null)?.name ?? '';
  const text = err instanceof Error && err.message ? err.message : String(err);
  const art = /NotFound|DevicesNotFound|Overconstrained/i.test(name) ? 'None'
    : /NotReadable|TrackStart|Abort/i.test(name) ? 'Busy'
      : /NotSupported|TypeError|ReferenceError/i.test(name) ? 'Unsupported'
        : '';
  return art
    ? { title: `studio.audio.mic${art}Title`, detail: `studio.audio.mic${art}Detail`, text }
    : { title: 'studio.audio.micErrorTitle', detail: 'studio.audio.micErrorDetail', text };
}
