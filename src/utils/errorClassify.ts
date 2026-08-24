// Gemeinsame Fehler-Klassifikation fuer Trainings-/Dev-Train-Fehler.
//
// Vorher hatten TrainingDashboard und DevTrainPanel je eine eigene
// analyzeError-Funktion mit unterschiedlicher Reihenfolge — derselbe Fehler
// wurde einmal als "Code-Fehler", einmal als "Numerischer Fehler (NaN/Inf)"
// eingestuft. Die Kategorie kommt jetzt aus einer einzigen Quelle; nur die
// Titel/Hinweis-Texte bleiben je Ansicht (unterschiedliche Locale-Namespaces).

export type ErrorCategory =
  | 'memory' | 'dataset' | 'labels' | 'architecture' | 'packages'
  | 'network' | 'cuda' | 'config' | 'code' | 'unknown';

export function classifyError(errorMsg: string): ErrorCategory {
  const e = (errorMsg ?? '').toLowerCase();
  if (e.includes('cuda out of memory') || e.includes('out of memory') || e.includes('oom'))
    return 'memory';

  // Label-/Klassenprobleme VOR der Geraete-Pruefung: PyTorch meldet den
  // Regressions-Fallback bei num_labels=1 als "mse_loss_out_mps: only defined
  // for floating types". Wer nur auf "mps" prueft, verkauft dem Nutzer einen
  // Datenfehler als Hardwareproblem.
  if (e.includes('mse_loss') || e.includes('only defined for floating types')
      || e.includes('label-spalte') || e.includes('label column')
      || e.includes('num_labels') || e.includes('nur einen einzigen wert'))
    return 'labels';

  if (e.includes('wird noch nicht unterstützt') || e.includes('not yet supported')
      || e.includes('modell-architektur') || e.includes('model architecture')
      || e.includes('unsupported architecture'))
    return 'architecture';

  // Geraete-Fehler nur bei echten Geraete-Meldungen, nicht bei jedem Vorkommen
  // von "mps"/"device" irgendwo in einem Traceback.
  if (e.includes('cuda error') || e.includes('cuda unavailable') || e.includes('no cuda')
      || e.includes('cuda is not available') || e.includes('mps not available')
      || e.includes('mps backend') || e.includes('device-side assert')
      || e.includes('no gpu') || e.includes('device not found')
      || /device .*(unavailable|not available|mismatch)/.test(e))
    return 'cuda';

  // Netzwerk- und Zertifikatsfehler VOR der Paket-Pruefung: torchvision laedt
  // vortrainierte Gewichte nach, und ein SSL-Fehler dabei enthaelt das Wort
  // "torchvision".
  if (e.includes('certificate_verify_failed') || e.includes('certificate verify failed')
      || e.includes('unable to get local issuer certificate') || e.includes('ssl:')
      || e.includes('sslcertverificationerror') || e.includes('urlopen error')
      || e.includes('max retries exceeded') || e.includes('connection refused')
      || e.includes('connectionerror') || e.includes('name or service not known')
      || e.includes('temporary failure in name resolution') || e.includes('proxyerror'))
    return 'network';

  if (e.includes('modulenotfounderror') || e.includes('importerror') || e.includes('no module')
      || e.includes('torchvision') || e.includes('versionskonflikt') || e.includes('version conflict'))
    return 'packages';

  // Python-Fehlertypen VOR der Dataset-Pruefung. Ein Traceback aus einem
  // Dev-Train-Script nennt fast immer DATASET_PATH — vorher wurde deshalb jeder
  // NameError als "Dataset / Pfad Fehler" ausgegeben.
  if (e.includes('syntaxerror') || e.includes('indentationerror') || e.includes('nameerror')
      || e.includes('typeerror') || e.includes('attributeerror') || e.includes('keyerror')
      || e.includes('indexerror') || e.includes('unboundlocalerror')
      || e.includes('zerodivisionerror') || e.includes('recursionerror'))
    return 'code';

  // Nur echte Datei-/Dataset-Meldungen. Ein blosses "path" irgendwo im
  // Traceback reicht nicht.
  if (e.includes('filenotfounderror') || e.includes('file not found')
      || e.includes('no such file') || e.includes('dataset')
      || e.includes('existiert nicht') || e.includes('keine daten-dateien')
      || e.includes('permission denied') || e.includes('isadirectoryerror'))
    return 'dataset';

  // \b-Grenzen: sonst matchen deutsche Woerter wie "Einfach" oder "Finanzen".
  if (/\bnan\b|\binf\b/.test(e) || e.includes('gradient') || e.includes('loss'))
    return 'config';

  return 'unknown';
}
