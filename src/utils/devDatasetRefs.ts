// Pfad-Referenzen fuer Dev Train und Dev Test.
//
// FrameTrain setzt diese Werte beim Start als Umgebungsvariablen:
//   DATASET_PATH / DATASET_YAML         → das ausgewaehlte Dataset
//   DATASET_PATH_<n> / DATASET_YAML_<n> → alle Datasets des Modells in fester
//                                          Reihenfolge (Importdatum)
//
// Vorher hing die Nummer an der Position in einer Liste, in der das gewaehlte
// Dataset nach vorne sortiert wurde: Wer ein anderes Dataset waehlte, bekam
// unter DATASET_PATH_2 ploetzlich andere Daten. Die Nummern sind jetzt
// unabhaengig von der Auswahl. DATASET_YAML gibt es nur, wenn das Dataset eine
// dataset.yaml/data.yaml hat (YOLO) — das Backend prueft und repariert sie.

export interface DevRefDataset {
  id: string;
  name: string;
  storage_path?: string;
  dataset_yaml_path?: string | null;
}

export interface DevRef {
  key: string;
  value: string;
  name: string;
  kind: 'path' | 'yaml';
}

/** Das gewaehlte Dataset, sonst das erste. */
export function pickSelectedDataset<T extends DevRefDataset>(datasets: T[], selectedId?: string | null): T | undefined {
  return datasets.find(d => d.id === selectedId) ?? datasets[0];
}

/** Liste mit dem gewaehlten Dataset an Position 0 (fuer Templates und Job-Metadaten). */
export function selectedFirst<T extends DevRefDataset>(datasets: T[], selectedId?: string | null): T[] {
  const sel = pickSelectedDataset(datasets, selectedId);
  if (!sel) return datasets;
  return [sel, ...datasets.filter(d => d !== sel)];
}

function refsFor(d: DevRefDataset, suffix: string): DevRef[] {
  const refs: DevRef[] = [{ key: `DATASET_PATH${suffix}`, value: d.storage_path || '', name: d.name, kind: 'path' }];
  if (d.dataset_yaml_path) {
    refs.push({ key: `DATASET_YAML${suffix}`, value: d.dataset_yaml_path, name: d.name, kind: 'yaml' });
  }
  return refs;
}

export function buildDatasetRefs(datasets: DevRefDataset[], selectedId?: string | null): DevRef[] {
  const sel = pickSelectedDataset(datasets, selectedId);
  if (!sel) return [];
  const refs = refsFor(sel, '');
  // Nummerierte Referenzen nur, wenn es mehr als ein Dataset gibt — sonst
  // stuende dasselbe Dataset doppelt da.
  if (datasets.length > 1) {
    datasets.forEach((d, i) => refs.push(...refsFor(d, `_${i + 1}`)));
  }
  return refs;
}

/** Umgebungsvariablen fuer start_dev_training / start_dev_test. */
export function refsToEnv(modelPath: string, refs: DevRef[]): Record<string, string> {
  return {
    MODEL_PATH: modelPath,
    ...Object.fromEntries(refs.filter(r => r.value).map(r => [r.key, r.value])),
  };
}

/** Zeilen fuer den System-Prompt der Code-KI. Ohne Pfad steht dort "(kein Pfad)", nie der Name. */
export function refsForPrompt(refs: DevRef[]): string {
  return refs.map(r => `- ${r.key} = "${r.value || '(kein Pfad)'}" (${r.name})`).join('\n');
}
