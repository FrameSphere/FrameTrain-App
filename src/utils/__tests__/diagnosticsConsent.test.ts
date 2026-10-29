import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('@tauri-apps/api/core', () => ({ invoke: vi.fn() }));
import { invoke } from '@tauri-apps/api/core';
import { isAutoErrorReportingEnabled } from '../errorReport';
import {
  PENDING_CONSENT_KEY,
  applyAccountDiagnosticsConsent,
  changeDiagnosticsConsent,
} from '../diagnosticsConsent';

const invokeMock = vi.mocked(invoke);

describe('Fehlerbericht-Einwilligung aus dem Konto', () => {
  beforeEach(() => {
    localStorage.clear();
    invokeMock.mockReset();
    invokeMock.mockResolvedValue(undefined);
  });

  it('uebernimmt die Wahl aus der Registrierung', async () => {
    await applyAccountDiagnosticsConsent('ft_key', true);
    expect(isAutoErrorReportingEnabled()).toBe(true);
    await applyAccountDiagnosticsConsent('ft_key', false);
    expect(isAutoErrorReportingEnabled()).toBe(false);
  });

  it('laesst die lokale Einstellung bei aelteren Servern ohne Feld stehen', async () => {
    await changeDiagnosticsConsent('ft_key', true);
    await applyAccountDiagnosticsConsent('ft_key', null);
    expect(isAutoErrorReportingEnabled()).toBe(true);
  });

  it('schreibt den Schalter ins Konto', async () => {
    expect(await changeDiagnosticsConsent('ft_key', true)).toBe(true);
    expect(invokeMock).toHaveBeenCalledWith('set_diagnostics_consent', { apiKey: 'ft_key', consent: true });
    expect(localStorage.getItem(PENDING_CONSENT_KEY)).toBeNull();
  });

  it('ein offline erklaerter Widerruf schlaegt den alten Kontowert und wird nachgereicht', async () => {
    invokeMock.mockRejectedValueOnce(new Error('offline'));
    expect(await changeDiagnosticsConsent('ft_key', false)).toBe(false);
    expect(isAutoErrorReportingEnabled()).toBe(false);

    // Naechster Login: Konto meldet noch "an"
    await applyAccountDiagnosticsConsent('ft_key', true);
    expect(isAutoErrorReportingEnabled()).toBe(false);
    expect(invokeMock).toHaveBeenLastCalledWith('set_diagnostics_consent', { apiKey: 'ft_key', consent: false });
    expect(localStorage.getItem(PENDING_CONSENT_KEY)).toBeNull();
  });
});
