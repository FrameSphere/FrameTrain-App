// Modell- und Dataset-Fakten fuer die Code-KI in Dev Train und Dev Test.
//
// Vorher kannte die KI nur die Pfade. Fuer Chronos-2 mit einer Zeitreihen-
// Tabelle schrieb sie deshalb: "Since the dataset type isn't known yet, the
// script auto-detects …" und lieferte ein Skript mit YOLO-, Klassifikations-
// und LLM-Zweig. Das Backend (describe_dev_context → ft_data/dev_context.py)
// liest config.json, README, Splits, Spalten und installierte Pakete.

import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';

// Kurz zwischenspeichern: Die Beschreibung enthaelt die installierten Pakete.
// Ohne Ablauf blieb "chronos-forecasting NICHT installiert" bis zum
// App-Neustart stehen, obwohl es laengst installiert war.
const CACHE_TTL_MS = 60_000;
const cache = new Map<string, { text: string; at: number }>();
const fresh = (key: string) => {
  const hit = cache.get(key);
  return hit && Date.now() - hit.at < CACHE_TTL_MS ? hit.text : null;
};

/** Fakten zu Modell und Dataset; null solange sie laden oder nicht lesbar sind. */
export function useDevContext(modelPath: string, datasetPath?: string): string | null {
  const key = `${modelPath}\u0000${datasetPath ?? ''}`;
  const [text, setText] = useState<string | null>(() => fresh(key));
  useEffect(() => {
    if (!modelPath) { setText(null); return; }
    const hit = fresh(key);
    if (hit) { setText(hit); return; }
    let alive = true;
    setText(null);
    invoke<string>('describe_dev_context', { modelPath, datasetPath: datasetPath || null })
      .then(t => { cache.set(key, { text: t, at: Date.now() }); if (alive) setText(t); })
      .catch(() => { if (alive) setText(null); });
    return () => { alive = false; };
  }, [key, modelPath, datasetPath]);
  return text;
}

/** Abschnitt fuer den System-Prompt. Ohne Fakten bleibt der bisherige Hinweis stehen. */
export function devContextPrompt(facts: string | null, mode: 'train' | 'test'): string {
  const factsBlock = facts
    ? `MODELL, DATASET UND PYTHON (aus den Dateien gelesen — verbindlich):
${facts}

Schreibe das Skript fuer GENAU dieses Modell und dieses Dataset. Keine
Fallunterscheidung fuer andere Modelltypen (YOLO / Text / LLM), wenn oben klar
ist, was vorliegt. Nutze die echten Spaltennamen. Fehlt ein Paket, das das
Skript braucht, sage es zu Beginn deiner Antwort und nenne den
Installationsbefehl von oben.`
    : 'INSTALLIERTE PAKETE: torch, transformers, datasets, scikit-learn, numpy, accelerate, peft, ultralytics (YOLO)';

  if (mode === 'test') {
    return `${factsBlock}

ERGEBNIS: Schreibe die Auswertung nach OUTPUT_PATH/results.json und drucke die
wichtigsten Kennzahlen auf stdout.`;
  }
  return `${factsBlock}

FORTSCHRITT AN FRAMETRAIN MELDEN (sonst bleiben Balken und Loss-Kurve leer):
eine JSON-Zeile pro Event auf stdout, mit flush=True:
    {"type": "progress", "data": {"epoch": 1, "total_epochs": 3, "step": 10, "total_steps": 100, "train_loss": 0.42, "val_loss": 0.40}}
    {"type": "complete", "data": {"model_path": OUTPUT_PATH, "final_metrics": {"val_loss": 0.38, "total_epochs": 3}}}
SPEICHERN: Das fertige Modell gehoert DIREKT nach OUTPUT_PATH (nicht in einen
Unterordner wie OUTPUT_PATH/final) — daraus legt FrameTrain nach Exit-Code 0
die neue Modellversion an, die Tests und Labor laden.`;
}

/** Python-Codeblöcke aus einer Erklaerung entfernen (der Code steht separat darunter). */
export function stripPythonBlocks(text: string): string {
  return text.replace(/```(?:python|py)\b[\s\S]*?```/g, '').replace(/\n{3,}/g, '\n\n').trim();
}
