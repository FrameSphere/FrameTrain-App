# Test-Kennzahlen (Session test-kennzahlen)

Branch `claude/engine-test-kennzahlen-c3d40f`, Stand 14.11.2026. Nicht gepusht, nicht gemergt, keine Versionsnummer geändert.

## Was gebaut wurde

### 1. Klassifikation (Bild HF, Bild torchvision, Audio, Video, Text)

- `test_engine/plugins/_shared_classify.py`: neue Funktion `classification_metrics(expected, predicted)`, rein in Python, Ergebnis wie sklearn (`labels=` gesehene Klassen, `zero_division=0`):
  `accuracy`, `macro_f1`, `macro_precision`, `macro_recall`, `weighted_f1`, `per_class_f1` (altes Feld des Textplugins, bleibt), `per_class` {precision, recall, f1, support}, `confusion_matrix` {labels, matrix, total_classes, truncated}, `top_confusions`.
  - Vorhersage `ERROR` ist keine eigene Klasse, zählt aber als verpasster Treffer.
  - Matrix in `results.json` höchstens 20 Klassen (bei mehr: die am häufigsten verwechselten), damit die Test-Historie bei 1000 Klassen nicht eine Million Zellen speichert.
- `run_dataset_classification` schreibt `results.json` jetzt als `{predictions, metrics}` (Rust liest beide Formen, siehe `parse_results_json`) und meldet die Kennzahlen auch in der Abschlussmeldung. Dazu `hard_examples.json` wie beim Textplugin.
- Text (`seq_classification`): `macro_f1_scores` nutzt die gemeinsame Rechnung. Unterschied zu vorher: Macro-F1 mittelt jetzt wie sklearn auch über Klassen, die nur vorhergesagt wurden (vorher nur über erwartete).
- Neu: Test-Plugin `test_engine/plugins/image_classification/` für das torchvision-Bildklassifikations-Training (das Task-Briefing nennt es timm; trainiert wird über torchvision). Vorher gab es keins, Tests dieser Versionen endeten mit "Kein Test-Plugin". Lädt `model.pt` (arch, classes, image_size). Der Eintrag `image_classification` in `TRAIN_ONLY` von `train_engine/test_plugin_consistency.py` ist entfernt (einzige Änderung in train_engine, nur Testdatei).

### 2. Vision-Language

- `test_engine/plugins/_shared_text_metrics.py`: `tolerant_match` (normalisiert gleich oder Antwort enthält die Referenz als ganze Wortfolge, führende Artikel egal), `character_error_rate` (Levenshtein, über das Dataset summiert wie jiwer), `looks_like_ocr` (Mehrheit der Prompts fragt nach Text im Bild).
- `vision_language`: Kennzahlen `exact_match`, `rougeL` (unverändert berechnet), neu `tolerant_match`, bei Texterkennung `cer`. `plugin_config.ocr` true/false erzwingt bzw. unterdrückt CER. Je Zeile `tolerant_correct`. `accuracy` bleibt Exact Match (keine Umdeutung bestehender Felder).

### 3. Text-zu-Bild

- `test_engine/plugins/_shared_clip.py`: CLIP-Score = max(100 · cos, 0), Skala wie torchmetrics. Modell nur lokal (`local_files_only`), Standard `openai/clip-vit-base-patch32`, sonst `plugin_config.clip_model` (HF-Name im Cache oder Ordner).
- `text_to_image_lora`: `clip_score` je Bild und im Mittel; mit `plugin_config.compare_base: true` dasselbe Bild mit gleichem Seed und abgeschalteter LoRA (`clip_score_base`, `clip_score_delta`, Bilder unter `images/base/`). Bei gemergter Pipeline kein Vergleich.
- Fehlt CLIP: Test läuft weiter, Warnung im Status, in `metrics` stehen `metrics_note` (Text für MCP/Logs) und `metrics_note_codes` (`clip_unavailable`, `base_merged`, `base_unsupported`), die die Ansicht auf EN/DE übersetzt.

### 4. Ansicht und Auswertung

- `src/plugins/ConfusionMatrixView.tsx` (neu): Matrix (Zeile erwartet, Spalte vorhergesagt; Diagonale grün, Verwechslungen rot) auf höchstens 10 Klassen begrenzt, dazu die Tabelle je Klasse (schlechteste zuerst).
- `GenericTestPanel.tsx`: behält die Rohkennzahlen, zeigt Matrix und Hinweise; neue Prozent-Kennzahlen (Macro-*, Exact/Tolerant Match), CLIP-Score als Zahl; `accuracy` nicht mehr doppelt in der Liste. Damit für HF-Bild, Audio, Video, VLM, Text-zu-Bild ohne Änderung an deren `TestPlugin.tsx`.
- `image-classification/TestPlugin.tsx` (eigene Ansicht): Macro-F1/-Precision/-Recall und Matrix nach dem Dataset-Test.
- Text (`xlm-roberta`/`hf-encoder`-Ansichten) zeigt die Matrix **nicht** — dort eigene Ansichten, nicht angefasst. Die Kennzahlen kommen trotzdem in Historie/MCP an.
- Übersetzungen EN+DE additiv: `testPlugins.generic.metricLabels.*`, `testPlugins.confusion.*`, `testPlugins.metricNotes.*`.
- Weg der Kennzahlen geprüft: `results.json` → `metrics` → `TestResults.metrics` (Rust, DB und `test_jobs.json`) → Test-Historie, Benchmark (`homePipeline.testMetric`) und MCP `tests_results`/`tests_history` (geben `metrics` unverändert weiter). Keine Felder umbenannt.
- Eine Änderung außerhalb des Auftragsrahmens, nötig für Punkt 4: `homePipeline.ts` nimmt `clip_score` vor einer beliebigen Zahl (vorher hätte der Benchmark für Text-zu-Bild zufällig Bildanzahl oder Schrittzahl gezeigt, weil Rust `metrics` als HashMap ohne Reihenfolge speichert); `BenchmarkView.tsx` kennt den Namen "CLIP score". Klassifikation rankt weiter über Accuracy, VLM über Exact Match (= accuracy).

## Tests

Python (aus `src-tauri/python/test_engine/`: `python3.11 -m unittest plugins.test_shared_classify plugins.test_shared_text_metrics plugins.test_shared_clip`):

```
test_accuracy_und_macro ... ok
test_gleich_wie_sklearn ... ok
test_je_klasse ... ok
test_konfusionsmatrix ... ok
test_error_ist_keine_klasse_aber_ein_fehler ... ok
test_grosse_matrix_wird_auf_verwechselte_klassen_begrenzt ... ok
test_ohne_labels_leer ... ok
test_vorhergesagte_fremdklasse_zaehlt_mit ... ok
test_cer_ueber_dataset_summiert ... ok
test_cer_wie_jiwer ... ok
test_levenshtein ... ok
test_cer_nur_bei_texterkennung ... ok
test_kennzahlen ... ok
test_normalisierung ... ok
test_treffer ... ok
test_bekannte_vektoren ... ok
test_mittel_ohne_none ... ok
test_wie_torchmetrics_formel ... ok
test_fehlendes_modell_liefert_grund_statt_fehler ... ok
test_winziges_clip_rechnet ... ok
Ran 20 tests — OK
```

`train_engine/test_plugin_consistency.py`: 4 Tests OK (neues Test-Plugin hat Train-Gegenstück, Architekturliste gleich).

vitest (`npm test`): **112 Dateien, 1208 Tests grün** (vorher 1197). Neu `src/plugins/__tests__/testMetrics.test.tsx` (11 Tests): Matrix-Begrenzung, Formatierung, Matrix und Werte je Klasse in HF-Bild- und torchvision-Ansicht, CLIP-Score und übersetzter Hinweis in Text-zu-Bild, Benchmark wählt `clip_score`, MCP `tests_results` gibt Matrix und `per_class` weiter.

## Echte Mini-Läufe durch `test_engine.py`

Winzige Modelle mit Zufallsgewichten (ViT 1 Schicht, mobilenet_v3_small, Wav2Vec2 1 Schicht, tiny VideoMAE, BERT 1 Schicht), 6–12 Beispiele, 3 Klassen. Die niedrigen Werte sind erwartet; belegt wird, dass die Kennzahlen entstehen und in `results.json` stehen.

```
== hf_image_classification | Beispiele 12 | accuracy 0.6667
  complete.metrics: {"accuracy": 0.6667, "macro_f1": 0.5556, "macro_precision": 0.5, "macro_recall": 0.6667, "weighted_f1": 0.5556}
  results.json confusion_matrix: ['blau', 'gruen', 'rot'] [[0, 4, 0], [0, 4, 0], [0, 0, 4]] truncated False
  top_confusions: [{"expected": "blau", "predicted": "gruen", "count": 4}]
== image_classification | Beispiele 12 | accuracy 0.3333
  complete.metrics: {"accuracy": 0.3333, "macro_f1": 0.1667, "macro_precision": 0.1111, "macro_recall": 0.3333, "weighted_f1": 0.1667}
  results.json confusion_matrix: ['blau', 'gruen', 'rot'] [[0, 4, 0], [0, 4, 0], [0, 4, 0]] truncated False
== audio_classification | Beispiele 9 | accuracy 0.3333
  complete.metrics: {"accuracy": 0.3333, "macro_f1": 0.1667, ...}
  results.json confusion_matrix: ['blau', 'gruen', 'rot'] [[3, 0, 0], [3, 0, 0], [3, 0, 0]]
== video_classification | Beispiele 6 | accuracy 0.3333
  complete.metrics: {"accuracy": 0.3333, "macro_f1": 0.1667, ...}
  results.json confusion_matrix: ['blau', 'gruen', 'rot'] [[0, 0, 2], [0, 0, 2], [0, 0, 2]]
== seq_classification | Beispiele 12 | accuracy 0.3333
  complete.metrics: {"macro_f1": 0.1667, "macro_precision": 0.1111, "macro_recall": 0.3333, "weighted_f1": 0.1667}
  results.json confusion_matrix: ['blau', 'gruen', 'rot'] [[0, 0, 4], [0, 0, 4], [0, 0, 4]]
```

Vision-Language mit SmolVLM-256M-Instruct (aus dem HF-Cache), 3 Bilder:

```
== vision_language (Frage nach der Farbe) | accuracy 0.0
  complete.metrics: {"exact_match": 0.0, "rougeL": 0.4, "tolerant_match": 1.0}
  erste Zeile: expected "blue", predicted "The square is blue.", tolerant_correct true
== vision_language (Schild ablesen, "Read the text in the image.") | accuracy 1.0
  complete.metrics: {"exact_match": 1.0, "rougeL": 1.0, "tolerant_match": 1.0, "cer": 0.0}
  EXIT 12 -> 'EXIT 12', GO 30 -> 'GO 30', STOP -> 'STOP'
```

Text-zu-Bild mit `hf-internal-testing/tiny-stable-diffusion-pipe` und einer LoRA mit Zufallsgewichten, 2 Captions, 2 Schritte:

```
## LoRA + tiny CLIP + compare_base
  complete.metrics: {"clip_score": 0.0, "clip_model": "hf-internal-testing/tiny-random-CLIPModel", "clip_score_base": 0.0, "clip_score_delta": 0.0}
## gemergte Pipeline + compare_base
  WARNUNG: Vergleich mit dem Basismodell nicht moeglich: die LoRA ist in die Pipeline eingerechnet (merge_lora).
  complete.metrics: {"clip_score": 0.0, ..., "metrics_note_codes": ["base_merged"]}
## CLIP fehlt (Standardmodell nicht im Cache)
  WARNUNG: CLIP-Score nicht berechnet: CLIP-Modell 'openai/clip-vit-base-patch32' ist lokal nicht vorhanden (OSError). ...
  complete.metrics: {"metrics_note": "...", "metrics_note_codes": ["clip_unavailable"], "clip_model": "openai/clip-vit-base-patch32"}
```

Nachgeprüft: CLIP-Score 0.0 kommt vom Zufalls-CLIP (roher Kosinus -0,054, auf 0 begrenzt; Prompt "hello world" ergibt 6,25). LoRA- und Basisbild unterscheiden sich (mittlere Pixelabweichung 26 bzw. 22 von 255), das Abschalten der LoRA wirkt.

## Nicht belegt

- **Ein aussagekräftiger CLIP-Score** mit `openai/clip-vit-base-patch32`: das Modell liegt nicht lokal, und während der Läufe war kein Netz da. Belegt sind Formel (gegen torchmetrics-Formel), Ladeweg, Ablauf und der Fall „fehlt“.
- **Die Ansicht in der echten App** (Tauri): nur vitest/jsdom, kein Screenshot aus einer gebauten Version.
- **Rust-Seite**: nicht geändert und nicht neu getestet. Die Annahme, dass verschachtelte `metrics` (Matrix, `per_class`) in DB und `test_jobs.json` ankommen, folgt aus `TestResults.metrics: HashMap<String, serde_json::Value>` und `parse_results_json`; ein Lauf über die App fehlt.
- **Text-Klassifikation in der Ansicht**: `xlm-roberta`/`hf-encoder` haben eigene Test-Ansichten ohne Matrix.
- Große Klassenzahlen nur im Unit-Test (30 Klassen), nicht in einem Engine-Lauf.

## Nebenbefund (nicht behoben, außerhalb des Auftrags)

`ft_data/image_text.resolve_image_text` lehnt ein Bild-Text-Dataset ab, das nur `test/` enthält ("gibt es Daten (test), aber keinen train-Split"). Für Klassenordner (ft_data.media) sind reine Test-Datasets erlaubt, für VLM/Text-zu-Bild nicht. Für die Mini-Läufe lagen die Dateien deshalb ohne Split-Ordner.
