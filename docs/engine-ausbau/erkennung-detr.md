# Erkennung über YOLO hinaus (Session erkennung-detr) — Zwischenstand

## Gebaut
- YOLO-Plugin (train + test): Ultralytics-Familien über `ft_data/ultralytics_family.py`; Laden mit der richtigen Klasse (RTDETR, YOLOWorld, YOLOE, YOLO), Familie in model.json.
  - Trainieren und testen (Ultralytics 8.4.80, gemessen): YOLOv10, YOLO12, YOLO26, RT-DETR (rtdetr-l/-x).
  - Klare Fehlermeldung statt Absturz: YOLO-World ohne Paket `clip`, YOLOE, `-sem`-Gewichte.
  - RT-DETR bekommt `optimizer=auto` statt SGD 0.01. Fix: bei Adam-Optimierern `warmup_bias_lr=0` (vorher RT-DETR mit AdamW mAP50 0.00, jetzt 0.50).
- Neues Plugin `hf_object_detection` (train + test): AutoModelForObjectDetection (DETR, Conditional/Deformable DETR, RT-DETR/v2, YOLOS, DAB-DETR, D-FINE). Daten YOLO oder COCO-JSON (`ft_data/od_formats.py`), mAP50/mAP50-95 nach COCO-Definition (`ft_data/od_metrics.py`), Ergebnisaufbau wie beim YOLO-Test.

## Tests (alle grün)
- `train_engine/plugins/yolo/test_families.py` — 14 Tests
- `train_engine/plugins/hf_object_detection/test_formats_metrics.py` — 18 Tests
- `train_engine/plugins/hf_object_detection/test_plugin_offline.py` — 4 Tests (Zufalls-YOLOS, Train -> Test)
- Regression: yolo/test_weights (21), yolo/test_tasks (28), test_plugin_consistency (4), test_yolo_inference_server + ft_data.test_detection (15)

## Echte Mini-Läufe (coco128-Ausschnitt: 8 train / 4 val / 4 test, CPU, imgsz 160–256)
- RT-DETR (rtdetr-l.pt) Train-Engine 1 Epoche: Val mAP50 0.499; Test-Engine: mAP50 0.722, mAP50-95 0.388.
- YOLOv10n Train-Engine 1 Epoche: mAP50 0.532. YOLO-World: Abbruch mit Meldung „braucht das Paket 'clip'“.
- hustvl/yolos-tiny Train-Engine 2 Epochen: läuft, Loss 7.18 -> 6.13, mAP 0 (Kopf neu angelegt, nur 4 Schritte).
- yolos-tiny vortrainiert, Test-Engine: mAP50 0.842, mAP50-95 0.668; Einzelbild liefert Boxen.

## Offen (nicht gebaut / nicht belegt)
- Frontend `src/plugins/hf-object-detection/` + Eintrag in registry.ts; `detectYOLO` greift sonst `hustvl/yolos-tiny` (enthält „/yolo“) — muss ausgeschlossen werden; `rtdetr` in YOLO_MODEL_TYPES/manifest ergänzen; KNOWN_UNSUPPORTED.detr entfernen.
- Rust: `test_manager.rs infer_task_type` (`ForObjectDetection` -> hf_object_detection, `rtdetr*.pt` -> detect), `plugin_commands.rs packages_for_task`.
- vitest (`npm test`) nicht gelaufen.
- Training auf MPS nicht gemessen (nur CPU, wegen paralleler Trainings).
