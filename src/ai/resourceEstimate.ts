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

/** Architektur aus der config.json (get_model_ram_info). */
export interface ModelArch {
  hiddenSize: number;
  numLayers: number;
  numHeads: number;
  vocabSize: number;
  /** Parameter in Milliarden (aus der Architektur geschaetzt). */
  paramBillion?: number;
}

export interface RamEstimateOptions {
  /**
   * Bildkantenlaenge in Pixeln (z. B. YOLO imgsz). Gesetzt fuer Bildmodelle:
   * dort bestimmt die Aufloesung die Aktivierungen, nicht max_seq_length —
   * das Feld existiert bei YOLO nicht einmal im Formular.
   */
  imageSize?: number;
  /**
   * Bekannte Architektur: dann rechnen die Aktivierungen mit Layern, Breite,
   * Heads und Sequenzlaenge (quadratisch, Attention) statt mit einer
   * Pauschale je Sample.
   */
  arch?: ModelArch | null;
  /** Sprachmodell-Kopf (LLM, Seq2Seq): Logits ueber das ganze Vokabular. */
  lmHead?: boolean;
}

/*
 * Kalibriert am 28.09.2026 mit echten Trainingsschritten auf Apple Silicon
 * (PyTorch/MPS, fp32, AdamW, Spitzenwert des Prozesses):
 *   SmolLM2-135M + LoRA  b4×128: 2,7 GB  b4×512: 8,2  b1×1024: 5,9  b2×1024: 9,3  b4×1024: 17,4
 *   SmolLM2-135M voll    b8×512: 20,1 GB
 *   DistilBERT voll      b16×128: 3,4 GB  b8×512: 3,4  b16×512: 6,8
 * Die alte Pauschale (0,3 GB je Sample bei 128 Tokens, linear) sagte fuer
 * b4×1024 11 GB — gemessen 17: die Attention waechst quadratisch mit der
 * Laenge, und ein 135M-Modell mit 30 Layern braucht mehr als ein BERT mit 6.
 */
/** Byte je Token, Layer und Hidden-Einheit (fp32). LoRA spart die Eingaben eingefrorener Schichten. */
const ACT_BYTES_LINEAR_FULL = 128;
const ACT_BYTES_LINEAR_ADAPTER = 86;
/** Byte je Token², Layer und Head (fp32): Attention-Scores samt Softmax fuer die Rueckrechnung. */
const ACT_BYTES_ATTENTION = 6.4;
/** Byte je Token und Vokabel-Eintrag: Logits, fp32-Kopie fuer den Loss, Gradient. */
const ACT_BYTES_LOGITS = 12;
/** Gradient Checkpointing sparte in der Messung (LoRA, 1024 Tokens) nur ~7 % — die Attention bleibt. */
const GRAD_CKPT_FACTOR_ARCH = 0.85;

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
  const arch = options.arch && options.arch.hiddenSize > 0 && options.arch.numLayers > 0 ? options.arch : null;
  // Mit Architektur: Parameterzahl in "GB bei 16 Bit" — die Ordnergroesse
  // taeuscht, wenn die Datei fp32 speichert (DistilBERT: doppelt so gross).
  const fromArch = arch?.paramBillion && arch.paramBillion > 0 ? arch.paramBillion * 2 : 0;
  const size = fromArch || (Number.isFinite(modelSizeGb) && modelSizeGb > 0 ? modelSizeGb : 0);
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
  const batch = Math.max(1, config.batch_size || 1);
  let activations: number;
  if (arch && !imageSize) {
    const seq = Math.max(1, config.max_seq_length || 128);
    const tokens = batch * seq;
    const precision = mixedPrecision ? 0.5 : 1;
    const linear = tokens * arch.numLayers * arch.hiddenSize
      * (adapterOnly ? ACT_BYTES_LINEAR_ADAPTER : ACT_BYTES_LINEAR_FULL) * precision;
    const attention = batch * seq * seq * arch.numLayers * Math.max(1, arch.numHeads) * ACT_BYTES_ATTENTION * precision;
    // Die Logits rechnet der Loss auch bei Mixed Precision in fp32.
    const logits = options.lmHead ? tokens * Math.max(0, arch.vocabSize) * ACT_BYTES_LOGITS : 0;
    activations = (linear + attention + logits) / 1e9 * (config.gradient_checkpointing ? GRAD_CKPT_FACTOR_ARCH : 1);
  } else {
    const seqFactor = imageSize && Number.isFinite(imageSize) && imageSize > 0
      ? (imageSize / REFERENCE_IMAGE_SIZE) ** 2
      : Math.max(1, config.max_seq_length || 128) / 128;
    const bytesPerSample = mixedPrecision ? 0.15 : 0.30;
    activations = batch * seqFactor * bytesPerSample * (config.gradient_checkpointing ? 0.3 : 1.0);
  }

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
