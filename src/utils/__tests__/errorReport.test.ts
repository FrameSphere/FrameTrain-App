import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  APP_ERROR_ENDPOINT,
  buildAppErrorPayload,
  isAutoErrorReportingEnabled,
  sendAutoErrorReport,
  setAutoErrorReportingEnabled,
} from '../errorReport';

describe('Fehlerberichte: Opt-in', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it('ist standardmaessig aus', () => {
    expect(isAutoErrorReportingEnabled()).toBe(false);
  });

  it('sendet ohne Einwilligung nichts', () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('{}'));
    sendAutoErrorReport({ error_type: 'runtime:uncaught', title: 'x', message: 'x' });
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('sendet nach Einwilligung und stoppt nach Widerruf', () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('{}'));
    setAutoErrorReportingEnabled(true);
    sendAutoErrorReport({ error_type: 'runtime:uncaught', title: 'x', message: 'x' });
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(fetchSpy.mock.calls[0][0]).toBe(APP_ERROR_ENDPOINT);

    setAutoErrorReportingEnabled(false);
    sendAutoErrorReport({ error_type: 'runtime:uncaught', title: 'x', message: 'x' });
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it('Vorschau zeigt denselben Inhalt, der gesendet wird', () => {
    const payload = buildAppErrorPayload({ error_type: 'devtrain:code', title: 't', message: 'm', script_full: 'print(1)' });
    expect(payload).toMatchObject({ site_id: 'frametrain', error_type: 'devtrain:code', script_full: 'print(1)' });
  });
});
