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

Offen:
- `datasets_check` kennt Bild+Maske nicht: meldet „Images without labels … not directly for supervised training“, obwohl der Import `image_mask` erkennt.
- Zeitreihen und Tabellen sind live NICHT geprüft: das leere Modell entsteht nur über den Reiter „Neu“, ein Werkzeug für die KI-Verbindung fehlt.
- Oberfläche nur angesehen, nicht durchgeklickt: Startseite listet die neuen Modelle, die Tests-Seite erkennt SegFormer als „Semantic Segmentation“. Klicks in der Preview wurden von der Bildschirmsteuerung abgewiesen; Masken-Überlagerung, Boxen, Audio-Player, Konfusionsmatrix und Kurve sind im Webview nicht gesehen. CSP erlaubt `asset:` für Bilder und Audio.
- HF-Erkennung: Precision/Recall 0 bei mAP50 0,46 (Schwelle bei schwach trainiertem Modell); prüfen, ob die Schwelle für die beiden Werte sinnvoll ist.
- TTS: Trainings-Fortschritt meldet `train_loss` 0; Startseite zeigt das Modell als „Pretrained“.
- Startseite zeigt beim Reranker weiter „0.0 % in the last test“ aus dem ersten, falsch zugeordneten Lauf.
- Nicht live: RT-DETR über Ultralytics, volles Fine-Tuning/Rohtext bei LLMs, neue Kennzahlen-Ansicht (Konfusionsmatrix, CLIP-Score), Labor/Hosting für die neuen Aufgaben, Windows/CUDA.
