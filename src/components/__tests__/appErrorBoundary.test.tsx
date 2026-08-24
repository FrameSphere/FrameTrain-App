// Regression aus dem Library-Test (23.08.2026): Ein Klick auf "Details" machte
// die gesamte App schwarz — die Detailansicht las script.script, das die API in
// der Listenantwort nicht mitliefert. Ohne Sicherheitsnetz reisst ein einzelner
// Renderfehler alles mit.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { AppErrorBoundary } from '../AppErrorBoundary';

vi.mock('../../utils/errorReport', () => ({
  sendAppErrorReport: vi.fn().mockResolvedValue(true),
}));
import { sendAppErrorReport } from '../../utils/errorReport';

function Boom(): React.ReactElement {
  // Genau der Fehler aus der Library: Zugriff auf ein fehlendes Feld.
  const script = {} as { script?: string };
  return <div>{(script.script as string).split('\n').length}</div>;
}

describe('AppErrorBoundary', () => {
  beforeEach(() => {
    // React protokolliert abgefangene Fehler auf der Konsole — hier erwartet.
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.mocked(sendAppErrorReport).mockClear();
  });
  afterEach(() => vi.restoreAllMocks());

  it('faengt einen Renderfehler ab statt die App zu leeren', () => {
    render(<AppErrorBoundary><Boom /></AppErrorBoundary>);
    expect(screen.getByText(/abgestuerzt/i)).toBeTruthy();
    expect(screen.getByText(/nicht betroffen/i)).toBeTruthy();
  });

  it('meldet den Fehler an die Auto-Fix-Pipeline', () => {
    render(<AppErrorBoundary><Boom /></AppErrorBoundary>);
    expect(sendAppErrorReport).toHaveBeenCalled();
    const arg = vi.mocked(sendAppErrorReport).mock.calls[0][0];
    expect(arg.error_type).toBe('runtime:render');
    expect(arg.details).toContain('Component stack');
  });

  it('laesst fehlerfreie Inhalte unangetastet durch', () => {
    render(<AppErrorBoundary><span>alles gut</span></AppErrorBoundary>);
    expect(screen.getByText('alles gut')).toBeTruthy();
    expect(sendAppErrorReport).not.toHaveBeenCalled();
  });

  it('kehrt ueber "Zurueck zur App" wieder zum Inhalt zurueck', () => {
    let explode = true;
    function Maybe(): React.ReactElement {
      if (explode) throw new Error('kaputt');
      return <span>wieder da</span>;
    }
    render(<AppErrorBoundary><Maybe /></AppErrorBoundary>);
    explode = false;
    fireEvent.click(screen.getByText(/Zurueck zur App/i));
    expect(screen.getByText('wieder da')).toBeTruthy();
  });
});
