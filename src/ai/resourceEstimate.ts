// ============================================================================
// Ressourcen-Schätzung — EINE Rechnung für Oberflaeche und KI
// ----------------------------------------------------------------------------
// Vorher gab es zwei: der RAM-Rechner im Training-Panel und der Seiten-Kontext
// des Coaches rechneten unabhaengig voneinander — mit unterschiedlichen
// Faktoren fuer Optimizer und Aktivierungen, und der Kontext liess den
// Framework-Overhead ganz weg. Auf demselben Bildschirm stand deshalb einmal
// "~12,5 GB" (Panel) und einmal "~17,7 GB" (Coach). Genau eine Rechnung,
// beide lesen sie.
// ============================================================================

/** Nur die Felder, die den Speicherbedarf bestimmen. */
export interface RamRelevantConfig {
  batch_size: number;
  max_seq_length: number;
  gradient_checkpointing: boolean;
  fp16: boolean;
  bf16: boolean;
  use_lora: boolean;
  load_in_4bit: boolean;
  load_in_8bit: boolean;
  optimizer: string;
}

export interface RamEstimate {
  /** Modell-Gewichte im Speicher */
  weights: number;
  /** Gradienten */
  gradients: number;
  /** Optimizer-Zustand (Adam: Momente, ggf. Master-Copy) */
  optimizer: number;
  /** Aktivierungen (haengt an Batch-Groesse und Sequenzlaenge) */
  activations: number;
  /** CUDA-Runtime, PyTorch-Caches, Tokenizer */
  overhead: number;
  /** Summe der Posten */
  total: number;
  /** true, wenn nur Adapter trainiert werden (LoRA oder quantisiert) */
  adapterOnly: boolean;
  /** true, wenn in halber Praezision gerechnet wird */
  mixedPrecision: boolean;
}

/** Framework-Overhead: CUDA-Runtime, PyTorch-Caches, Tokenizer. */
const FRAMEWORK_OVERHEAD_GB = 1.2;

export interface RamEstimateOptions {
  /**
   * Bildkantenlaenge in Pixeln (z. B. YOLO imgsz). Gesetzt fuer Bildmodelle:
   * dort bestimmt die Aufloesung die Aktivierungen, nicht max_seq_length —
   * das Feld existiert bei YOLO nicht einmal im Formular.
   */
  imageSize?: number;
}

/** Referenz-Aufloesung, auf die sich der Aktivierungs-Faktor bezieht. */
const REFERENCE_IMAGE_SIZE = 640;

/**
 * Grobe Peak-Schaetzung des Trainings-Speicherbedarfs in GB.
 *
 * Bewusst konservativ und einfach nachvollziehbar — es geht darum, ob ein Lauf
 * ueberhaupt auf die Maschine passt, nicht um zwei Nachkommastellen.
 */
export function estimateTrainingRam(
  config: RamRelevantConfig,
  modelSizeGb: number,
  options: RamEstimateOptions = {},
): RamEstimate {
  const size = Number.isFinite(modelSizeGb) && modelSizeGb > 0 ? modelSizeGb : 0;
  const mixedPrecision = !!(config.fp16 || config.bf16);
  const is4bit = !!config.load_in_4bit;
  const is8bit = !!config.load_in_8bit;
  const quantized = is4bit || is8bit;
  const adapterOnly = !!config.use_lora || quantized;

  // 1. Gewichte: FP32-Cast = 2x, FP16/BF16 = 1x, 8-bit = 0.5x,
  //    4-bit = 0.25x (+ ~5% fuer die LoRA-Adapter obendrauf)
  let weights: number;
  if (is4bit) weights = size * 0.25 + (config.use_lora ? size * 0.05 : 0);
  else if (is8bit) weights = size * 0.5;
  else if (mixedPrecision) weights = size;
  else weights = size * 2;

  // 2. Gradienten: nur fuer trainierte Parameter
  let gradients: number;
  if (adapterOnly) gradients = size * 0.05;
  else if (mixedPrecision) gradients = size;
  else gradients = size * 2;

  // 3. Optimizer-Zustand: AdamW haelt zwei Momente; bei Mixed Precision kommt
  //    die FP32-Master-Copy dazu (=> 4x statt 2x). Adafactor braucht deutlich
  //    weniger, weil es die zweite Ordnung faktorisiert.
  const trainedGb = size * (adapterOnly ? 0.05 : 1.0);
  const optimizer = /adafactor/i.test(config.optimizer || '')
    ? trainedGb * (mixedPrecision ? 1.0 : 0.5)
    : trainedGb * (mixedPrecision ? 4 : 2);

  // 4. Aktivierungen: ~0.30 GB je Sample bei 128 Tokens bzw. 640 px (FP32),
  //    halb so viel in Mixed Precision. Gradient Checkpointing spart rund 70%.
  //    Text skaliert linear mit der Sequenzlaenge, Bilder quadratisch mit der
  //    Kantenlaenge (die Pixelzahl waechst mit der Flaeche).
  const imageSize = options.imageSize;
  const seqFactor = imageSize && Number.isFinite(imageSize) && imageSize > 0
    ? (imageSize / REFERENCE_IMAGE_SIZE) ** 2
    : Math.max(1, config.max_seq_length || 128) / 128;
  const bytesPerSample = mixedPrecision ? 0.15 : 0.30;
  const batch = Math.max(1, config.batch_size || 1);
  const activations = batch * seqFactor * bytesPerSample * (config.gradient_checkpointing ? 0.3 : 1.0);

  const overhead = FRAMEWORK_OVERHEAD_GB;
  const total = weights + gradients + optimizer + activations + overhead;

  return { weights, gradients, optimizer, activations, overhead, total, adapterOnly, mixedPrecision };
}

/** Wie der geschaetzte Bedarf zum vorhandenen Arbeitsspeicher steht. */
export type RamVerdict = 'fits' | 'tight' | 'exceeds' | 'unknown';

/**
 * Vergleicht die Schaetzung mit dem tatsaechlichen System-RAM.
 *
 * Ohne diesen Vergleich nannte der Coach eine Zahl, die dem Nutzer nichts
 * sagte: "~17 GB" ist auf einem 64-GB-Rechner unkritisch und auf einem
 * 16-GB-Rechner ein sicherer Abbruch.
 */
export function ramVerdict(totalGb: number, systemRamGb: number | null): RamVerdict {
  if (!systemRamGb || !Number.isFinite(systemRamGb) || systemRamGb <= 0) return 'unknown';
  // Das Betriebssystem und die App selbst brauchen auch etwas.
  const usable = systemRamGb * 0.8;
  if (totalGb > usable) return 'exceeds';
  if (totalGb > usable * 0.75) return 'tight';
  return 'fits';
}

/**
 * Die Schaetzung als Kontext-Zeilen fuer die KI (Coach + Metrik-Assistent).
 * Bewusst kompakt: fuenf Zeilen, damit der Block bei kleinem Token-Budget
 * nicht die eigentliche Frage verdraengt.
 */
export function ramEstimateLines(
  est: RamEstimate,
  modelSizeGb: number,
  systemRamGb: number | null,
): string[] {
  const lines = [
    `Modellgröße: ~${modelSizeGb.toFixed(2)} GB`,
    `Weights ~${est.weights.toFixed(1)} GB, Gradients ~${est.gradients.toFixed(1)} GB, ` +
      `Optimizer ~${est.optimizer.toFixed(1)} GB, Activations ~${est.activations.toFixed(1)} GB, ` +
      `Overhead ~${est.overhead.toFixed(1)} GB`,
    `Geschätzter Peak-RAM/VRAM: ~${est.total.toFixed(1)} GB`,
  ];
  const verdict = ramVerdict(est.total, systemRamGb);
  if (verdict !== 'unknown' && systemRamGb) {
    lines.push(`Verfügbarer System-RAM: ${systemRamGb.toFixed(1)} GB`);
    lines.push(
      verdict === 'exceeds'
        ? 'Bewertung: PASST NICHT — der Lauf würde den Arbeitsspeicher sprengen (OOM). Sag das dem User deutlich und nenne konkrete Sparmaßnahmen.'
        : verdict === 'tight'
          ? 'Bewertung: KNAPP — läuft vermutlich, aber ohne Reserve. Sparmaßnahmen erwähnen.'
          : 'Bewertung: PASST — genug Reserve vorhanden.',
    );
  } else {
    lines.push('Verfügbarer System-RAM: unbekannt — nenne keine Aussage darüber, ob es auf diese Maschine passt.');
  }
  return lines;
}
