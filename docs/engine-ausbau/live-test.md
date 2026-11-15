# Engine-Ausbau: Zusammenführung und Live-Test (15.11.2026)

Branch `feat/engine-ausbau` = main (1.8.3) + sechs `claude/engine-*`-Branches + Fixes aus dem Live-Test.

## Automatische Tests auf dem zusammengeführten Stand
- `tsc --noEmit`: sauber
- vitest: 1337 bestanden (117 Dateien)
- `cargo test`: 292 bestanden, 1 ignoriert (+ 3 neue Fälle in `leitet_aufgabe_aus_dem_versionsordner_ab`)
- Python: alle `test_*.py` unter `src-tauri/python` grün (jeweils mit dem im Dateikopf genannten Aufruf)

## Live in der Preview-App (Build aus diesem Branch, über die lokale KI-Verbindung)
Echte Modelle aus Hugging Face, selbst erzeugte Mini-Datensätze, Training und Test durch die App.

| Plugin | Modell | Training | Test | Einzeltest |
|---|---|---|---|---|
| Segmentierung | nvidia/mit-b0 | ok, mIoU 0,91 (Val) | mIoU 0,908, IoU je Klasse | Maske, Überlagerung, Legende |
| HF-Erkennung | hustvl/yolos-tiny | ok, mAP50 0,32 nach 3 Epochen | mAP50 0,46, mAP50-95 0,28 | läuft, findet nach 3 Epochen noch nichts |
| Frage-Antwort | mrm8488/bert-tiny-finetuned-squadv2 | ok, EM 28,6 % -> 71,4 % | EM 62,5 %, F1 65,3 % | Antwortstelle markiert |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | ok | MRR@10 1,0, nDCG@10 1,0 | Relevanzwert |
| TTS | microsoft/speecht5_tts | ok, Hörprobe gespeichert | WER 1,6 %, CER 1,6 % (whisper-tiny) | WAV erzeugt |

## Befunde aus dem Live-Test
Behoben (Commit 83fff21), danach live gegengeprüft:
1. `training_start` über die KI-Verbindung nahm ohne `task_type` immer `seq_classification`; SegFormer scheiterte mit „Architektur nicht unterstützt“. Jetzt kommt die Aufgabe samt Standard-`plugin_config` aus dem erkannten Plugin.
2. Tests ohne `task_type` liefen bei Segmentierung, Frage-Antwort und Reranker als `seq_classification` (Reranker: Accuracy 0 statt MRR). `infer_task_type` kennt die drei jetzt.

## Runde 2 (15.11. nachmittags): Befunde behoben, Rest live geprüft

Stand der Tests danach: `tsc` sauber, vitest 1338, `cargo test` 293, Python-Tests der betroffenen Plugins grün.

Behoben und live gegengeprüft:
3. `datasets_check` erkennt Bild+Maske als `image_mask` (40 Bilder, 40 Masken, Klassen aus `classes.txt`, keine Warnung).
4. Neues Werkzeug `models_create_blank` (Zeitreihen, Tabellen) für die KI-Verbindung.
5. RT-DETR (Ultralytics) brach auf Apple Silicon ab (`grid_sampler_2d_backward` fehlt auf MPS). Training läuft dort jetzt auf der CPU, mit Hinweis.
6. Kennzahlen, die ein Plugin nur in der Abschlussmeldung sendet, fehlten in den gespeicherten Ergebnissen (LLM: Perplexität, ROUGE-L). Rust ergänzt sie jetzt.
7. Einzeltest eines auf Rohtext trainierten LLM lief durch die Chat-Vorlage und wiederholte „### Assistant:“. Er setzt den Text jetzt fort.
8. HF-Erkennung nennt Schwelle und höchsten Wert, wenn nichts über der Schwelle liegt (Precision/Recall 0 war kein Fehler).

Zusätzlich live in der Preview:

| Was | Ergebnis |
|---|---|
| Zeitreihen (leeres Modell, PatchTST) | Training ok; Test MAE 0,447 gegen naiv 5,494, MASE 0,42; Einzeltest liefert 16 Werte |
| Tabellen (leeres Modell) | Training ok; Accuracy 0,717 gegen Basis 0,567, Macro-F1 0,716; Einzeltest „ja“ 0,80 |
| RT-DETR-l (Ultralytics) | Training ok (CPU); Test mAP50 0,82, mAP50-95 0,82; Einzelbild findet Boxen |
| LLM volles Fine-Tuning auf Rohtext (SmolLM2-135M, MLX) | Perplexität 11,78 -> 1,25; Test speichert Perplexität 1,123 |
| TTS, längerer Lauf | Loss 0,57 / Val 0,60 sichtbar (der Wert 0 im Kurzlauf kam von nur 6 Schritten) |
| TTS im Labor | Server lädt, `lab_infer` erzeugt WAV (1,6 s) |
| Neue Klassifikations-Kennzahlen (bench-electra-small) | Macro-F1 0,90, Konfusionsmatrix [[25,0],[6,29]], Werte je Klasse |
| CLIP-Score mit echtem `openai/clip-vit-base-patch32` (Engine-Ebene) | passender Prompt 28,1, falscher 13,7 |

In der Oberfläche gesehen (Bildschirmsteuerung): Startseite listet alle neuen Modelle nach dem Neuladen, Modelle-Seite zeigt für jedes das richtige Plugin, Import-Reiter „Neu“ mit Zeitreihen/Tabellen, Segmentierungs-Einzeltest mit Überlagerung, Deckkraft-Regler und Legende, TTS-Einzeltest mit Audio-Player (Dauer wird geladen), Zeitreihen-Test mit Kennzahlen und Kurve (Ist, Modell, naiv).

## Runde 3 (15.11. abends): Labor und Hosting für die neuen Aufgaben

Der Modell-Server (`test_engine/model_server.py`) bedient Segmentierung, HF-Erkennung, Frage-Antwort, Reranker, Zeitreihen und Tabellen jetzt über die Test-Plugins der Aufgaben: das Plugin wird einmal geladen, jede Anfrage ist ein Einzeltest. Damit rechnen Test, Labor und Hosting mit demselben Code. Rust schätzt die Aufgabe vor dem Laden (`guess_task`) und reicht bei Frage-Antwort und Reranker die Frage aus dem zweiten Feld weiter. Ansichten: Überlagerung mit Legende (Segmentierung), freier Text (Frage-Antwort, Zeitreihe), Rangliste (Reranker), Klasse mit Konfidenz (Tabelle), Boxen wie bei YOLO (HF-Erkennung). Texte EN/DE.

Tests: 9 neue Python-Tests (`test_model_server_plugins.py`), neue Fälle in `hosting_manager.rs` und `model_host.rs`, 5 neue vitest-Fälle. Stand: vitest 1343, `cargo test` 293.

Live in der Preview, jeweils Labor (`lab_server_start` + `lab_infer`) und Hosting (`hosting_load` + `hosting_infer`):

| Aufgabe | Erkannt als | Labor | Hosting |
|---|---|---|---|
| Segmentierung | segmentation / Bild | Klasse, Legende, Überlagerung | ok |
| HF-Erkennung | hf_detect / Bild | läuft; yolos-tiny findet nichts ab 0,25 (höchster Wert 0,17) | ok |
| Frage-Antwort | qa / Text | Antwort mit markierter Stelle, auch mit Frage im zweiten Feld | ok |
| Reranker | reranker / Text | Platz 1 mit Wert, Rangliste | ok |
| Zeitreihe | time_series / Text | 16 Vorhersagewerte | ok |
| Tabelle | tabular / Text | „ja“ 0,80 | ok |

Nicht im Fenster gesehen: die neuen Ansichten in Labor und Hosting (nur per vitest und über die Antworten der KI-Verbindung geprüft).

Weiter offen:
- Startseite lädt nach Läufen über die KI-Verbindung nicht von selbst neu (erst der Neuladen-Knopf zeigt neue Modelle und Tests); TTS-Modell steht dort als „Pretrained“.
- Nicht im Fenster gesehen: Boxen-Ansicht der HF-Erkennung (yolos-tiny lag nach 3 Epochen unter der Schwelle), markierte Antwort bei Frage-Antwort, Konfusionsmatrix-Ansicht (Textklassifikatoren haben eigene Ansichten ohne Matrix), Audio wurde geladen, aber nicht abgehört.
- CLIP-Score und VLM-Kennzahlen nicht über einen Test in der App, nur auf Engine-Ebene bzw. per Unit-Test.
- Labor: Datasets der neuen Aufgaben werden noch nicht als Stichproben aufbereitet (kein Soll-Vergleich für Maske, Antwort, Vorhersage); Eingabe von Hand oder per Datei geht.
- Reranker-Werte sind Logits, keine 0..1-Werte; die Balken in der Rangliste sind dafür nicht skaliert.
- `models_create_blank` ändert die Werkzeugliste der KI-Verbindung: vor einem Release mit den laufenden Prüfungen bei Anthropic/OpenAI abstimmen.
- Windows, Linux, CUDA: weiterhin ungetestet.
