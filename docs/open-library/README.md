# Skripte für die Open Library

Sieben Skripte für Aufgaben, die kein Plugin abdeckt. `manifest.json` enthält je Skript Name, Beschreibung und die Felder des Upload-Formulars (Werte aus `MODEL_TYPES`, `TASK_TYPES`, `FRAMEWORKS` in `src/components/OpenLibraryModal.tsx`).

Jedes Skript lief am 15.11.2026 in der Preview über Dev Train bzw. Dev Test (KI-Verbindung, `devtrain_start` / `devtrain_test_start`):

| Skript | Modell / Daten im Lauf | Ergebnis |
|---|---|---|
| `train/mlm_domain_adaptation.py` | bert-tiny, 400 Texte rotten-tomatoes | Perplexität 132,1 -> 114,9 nach 1 Epoche |
| `train/clip_contrastive_finetune.py` | openai/clip-vit-base-patch32, 48 Bild-Text-Paare | Val-Loss 0,58 -> 0,40, Recall@1 1,0 |
| `train/whisper_lora.py` | openai/whisper-tiny, 12 Aufnahmen | läuft, 0,29 Mio. trainierbare Parameter, WER 0,0 vorher und nachher (Daten zu leicht) |
| `train/tabular_anomaly_autoencoder.py` | leeres Tabellenmodell, 400 Zeilen | Val-Loss 0,17, 10 Zeilen über der Schwelle |
| `test/classifier_calibration.py` | electra-small (trainiert), 500 Zeilen | ECE 0,018, Accuracy 0,824 |
| `test/text_classifier_robustness.py` | electra-small (trainiert), 300 Zeilen | 0,820 -> 0,813 bei Tippfehlern, 4,7 % gekippt |
| `test/chronos_zero_shot_forecast.py` | chronos-bolt-tiny, Temperatur-Zeitreihe | MAE 0,375 gegen naiv 4,054, MASE 0,09 |

Nicht geprüft: CUDA, Windows, große Datenmengen. Whisper-LoRA zeigt auf den Testdaten keine Verbesserung, weil whisper-tiny sie schon fehlerfrei erkennt.
