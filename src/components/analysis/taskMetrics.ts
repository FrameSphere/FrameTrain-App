// Aufgabenspezifische Kennzahlen fuer die Analyse-Seite.
//
// Jedes Plugin meldet eigene Zahlen (LLM: Perplexitaet, Exact Match;
// Spracherkennung: WER; Embeddings: Recall@k; YOLO-Segmentierung: Masken-mAP).
// Die Seite kannte nur Accuracy/F1 und mAP — alles andere war unsichtbar.
// Hier werden die Werte sortiert, als Prozent oder Zahl formatiert und mit
// ihrem Vorher-Wert (baseline_x bzw. x_before) zusammengelegt.

export interface TaskMetricTile {
  key: string;
  value: number;
  before?: number;
  /** true = Anteil 0..1, als Prozent zeigen */
  rate: boolean;
  /** true = kleiner ist besser (WER, Perplexitaet, Loss) */
  lowerIsBetter: boolean;
}

const RATE_PATTERNS = [
  /^exact_match$/, /^rouge/i, /^bleu$/, /^preference_accuracy$/, /^wer$/, /^cer$/,
  /recall@?\d*/i, /^mrr/i, /ndcg/i, /accuracy/i, /map\d*/i, /^mask_/, /^pose_/, /^box_/,
  /^entity_/, /^f1_/, /^precision_/, /^recall_/, /spearman/i, /pearson/i, /^r2$/,
];
const LOWER_IS_BETTER = [/^wer$/, /^cer$/, /perplexity/, /loss/, /^mse$/, /^rmse$/, /^mae$/];
/** Reine Zaehl- oder Protokollwerte ohne Aussage ueber die Qualitaet. */
const HIDDEN = [/_seconds$/, /^tokens_per_second$/, /^peak_memory/, /^quant_bits$/, /^lora_r$/];

function beforeKey(key: string, all: Record<string, number>): number | undefined {
  if (typeof all[`baseline_${key}`] === 'number') return all[`baseline_${key}`];
  if (typeof all[`${key}_before`] === 'number') return all[`${key}_before`];
  return undefined;
}

export function buildTaskMetricTiles(metrics: Record<string, unknown> | null | undefined): TaskMetricTile[] {
  if (!metrics) return [];
  const nums: Record<string, number> = {};
  for (const [k, v] of Object.entries(metrics)) {
    if (typeof v === 'number' && Number.isFinite(v)) nums[k] = v;
  }
  const tiles: TaskMetricTile[] = [];
  for (const [key, value] of Object.entries(nums)) {
    if (key.startsWith('baseline_') || key.endsWith('_before')) continue;
    if (HIDDEN.some(r => r.test(key))) continue;
    // Pearson/Spearman/R2 duerfen negativ sein, bleiben aber "Anteile" bis 1.
    const rate = RATE_PATTERNS.some(r => r.test(key)) && value <= 1 && value >= -1;
    tiles.push({ key, value, before: beforeKey(key, nums), rate, lowerIsBetter: LOWER_IS_BETTER.some(r => r.test(key)) });
  }
  // Erst die mit Vorher-Wert (zeigen, was das Training gebracht hat), dann alphabetisch.
  return tiles.sort((a, b) => Number(b.before !== undefined) - Number(a.before !== undefined) || a.key.localeCompare(b.key));
}

export function formatMetric(value: number, rate: boolean): string {
  if (rate) return `${(value * 100).toFixed(1)}%`;
  if (Number.isInteger(value)) return value.toLocaleString();
  return Math.abs(value) >= 100 ? value.toFixed(1) : value.toFixed(4);
}

/** Hat sich der Wert verbessert? undefined = kein Vorher-Wert. */
export function improved(tile: TaskMetricTile): boolean | undefined {
  if (tile.before === undefined || tile.before === tile.value) return undefined;
  return tile.lowerIsBetter ? tile.value < tile.before : tile.value > tile.before;
}
