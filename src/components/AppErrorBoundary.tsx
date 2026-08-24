// Sicherheitsnetz um die gesamte Oberflaeche.
//
// Anlass: Ein Klick auf "Details" in der Open Script Library liess die App
// komplett schwarz werden — die Detailansicht las script.script, das die API in
// der Listenantwort gar nicht liefert. Ohne Boundary reisst ein einzelner
// Renderfehler die ganze Anwendung mit; der Nutzer sieht eine tote Flaeche und
// verliert seinen Arbeitsstand.

import React from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';
import { sendAppErrorReport } from '../utils/errorReport';

interface Props { children: React.ReactNode }
interface State { error: Error | null }

export class AppErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    // Speist dieselbe Auto-Fix-Pipeline wie der globale Handler.
    void sendAppErrorReport({
      error_type: 'runtime:render',
      title: error.message.slice(0, 200),
      message: error.message,
      details: `${error.stack ?? ''}\n--- Component stack ---${info.componentStack ?? ''}`,
    });
  }

  private reset = (): void => this.setState({ error: null });

  render(): React.ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950 p-6">
        <div className="max-w-lg w-full rounded-2xl border border-red-500/20 bg-red-500/5 p-6 space-y-4">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0" />
            <h1 className="text-white font-semibold">Diese Ansicht ist abgestuerzt</h1>
          </div>
          <p className="text-gray-400 text-sm">
            Der Fehler wurde automatisch gemeldet. Deine Modelle, Datensaetze und
            Trainingslaeufe sind davon nicht betroffen.
          </p>
          <pre className="text-[11px] font-mono text-red-300/80 bg-black/30 rounded-lg p-3 overflow-x-auto max-h-32">
            {error.message}
          </pre>
          <div className="flex gap-2">
            <button
              onClick={this.reset}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/10 hover:bg-white/15 text-white text-sm transition-all"
            >
              <RotateCcw className="w-4 h-4" /> Zurueck zur App
            </button>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 rounded-xl bg-white/5 hover:bg-white/10 text-gray-300 text-sm transition-all"
            >
              Neu laden
            </button>
          </div>
        </div>
      </div>
    );
  }
}
