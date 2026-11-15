# Erkennung über YOLO hinaus (Session erkennung-detr)

Branch `claude/engine-erkennung-detr-b93934`. Nicht gepusht, nicht gemergt, keine Versionsnummer geändert.

## Gebaut

### 1. YOLO-Plugin: weitere Ultralytics-Familien
- `ft_data/ultralytics_family.py`: erkennt die Familie (Name, sonst `family` in model.json) und lädt mit der passenden Klasse: `RTDETR`, `YOLOWorld`, `YOLOE`, sonst `YOLO`. Train- und Test-Plugin nutzen den Loader; model.json speichert `family`.
- Stand Ultralytics 8.4.80, je 1 Epoche gemessen:
  | Familie | Training | Test |
  |---|---|---|
  | RT-DETR (`rtdetr-l.pt`, `rtdetr-x.pt`) | ja | ja |
  | YOLOv10, YOLO12, YOLO26 | ja | ja |
  | YOLO-World | nur mit Paket `clip`, sonst klare Meldung mit pip-Befehl | ja (Vorab-Probe: mAP50 0.72) |
  | YOLOE | klare Meldung „nicht eingebunden“ | nicht geprüft |
  | `-sem` (semantische Segmentierung) | klare Meldung | – |
- RT-DETR bekommt `optimizer=auto` statt der YOLO-Standardwerte SGD/0.01, solange der Nutzer sie nicht geändert hat.
- Fehlerbehebung: Ultralytics wärmt Bias-Parameter mit `warmup_bias_lr=0.1` auf und setzt das nur bei `optimizer=auto` für Adam auf 0. Bei ausdrücklich gewähltem AdamW (auch bei normalen YOLO-Läufen) wurde das Modell dadurch zerstört (RT-DETR: mAP50 0.00). Das Plugin übergibt bei Adam-Optimierern jetzt 0.0 (überschreibbar über `plugin_config.warmup_bias_lr`).

### 2. Neues Plugin `hf_object_detection`
- Train: `src-tauri/python/train_engine/plugins/hf_object_detection/` — AutoModelForObjectDetection (DETR, Conditional/Deformable DETR, RT-DETR/RT-DETRv2 im HF-Format, YOLOS, DAB-DETR, D-FINE, Table Transformer). Eigene Schleife mit AdamW (Backbone 1/10 der Lernrate), Gradient-Clipping, Spiegeln als Augmentierung, mAP je Epoche auf val, bestes Epochen-Ergebnis wird gespeichert. Schreibt `model.json` mit `task: hf_object_detection`.
- Test: `src-tauri/python/test_engine/plugins/hf_object_detection/` — Ergebnisaufbau wie YOLO-Test: Kennzahlen `mAP50` (zuerst), `mAP50-95`, `precision`, `recall`, `images_exact`; je Bild `detections` mit `label, confidence, x1..y2`. Einzelbild liefert zusätzlich `image_path`, `image_width`, `image_height`.
- Gemeinsam (`ft_data/`): `od_formats.py` (YOLO-Labels und COCO-JSON → Pixelboxen → COCO-Annotation für den HF-Prozessor; Klassen werden beim Testen über den Namen auf die Modellklassen abgebildet), `od_metrics.py` (COCO-mAP, 101-Punkt-Interpolation, ohne pycocotools), `hf_detection.py` (Modellprüfung, Bildgröße, Gerät, Vorhersage).
- Frontend: `src/plugins/hf-object-detection/` (detect.ts, index.ts, TestPlugin.tsx), registriert direkt nach YOLO. Test-UI über `GenericTestPanel` mit neuem `resultKind="boxes"` (Boxen über dem Eingabebild).
- Erkennung: `hustvl/yolos-tiny` landete vorher beim YOLO-Plugin (Name enthält „/yolo“) — jetzt ausgeschlossen. `rtdetr-l.pt` geht an YOLO, `PekingU/rtdetr_r18vd` (HF-Format) an das neue Plugin. Der Hinweis „DETR wird nicht unterstützt“ ist entfernt; der HF-Download-Check lässt DETR/RT-DETR/YOLOS durch, unbekannte Detektoren (OWLv2) werden weiter mit Begründung abgelehnt.
- Texte EN+DE: Plugin-Name/-Beschreibung, `testPlugins.generic.detectedObjects`, Dataset-Hinweis `missing.hfDetect`; YOLO-Texte nennen v10/12/26 und RT-DETR.
- Rust: `test_manager.rs infer_task_type` erkennt `*ForObjectDetection` → `hf_object_detection` und `rtdetr*.pt` → `detect`; `plugin_commands.rs packages_for_task` kennt `hf_object_detection` (transformers-Stack + pillow, timm, scipy).
- Gemeinsame Dateien nur additiv angefasst: `registry.ts` (Import, Eintrag, eine Zeile entfernt), `GenericTestPanel.tsx`, `genericDatasetCompat.ts`, `TrainingPanel.tsx` (Task-Liste), `ModelManager.tsx` (Hinweistext), Locales.

## Tests

Python (unittest, aus `src-tauri/python/train_engine/`):
```
plugins/yolo/test_families.py                       Ran 14 tests  OK
plugins/hf_object_detection/test_formats_metrics.py Ran 18 tests  OK
plugins/hf_object_detection/test_plugin_offline.py  Ran 4 tests   OK   (Zufalls-YOLOS: Train -> Export -> Test-Plugin Dataset + Einzelbild)
Regression: plugins/yolo/test_weights.py 21 OK, plugins/yolo/test_tasks.py 28 OK, test_plugin_consistency.py 4 OK,
            python -m unittest train_engine.plugins.yolo.test_yolo_inference_server ft_data.test_detection: 15 OK
```

Frontend (`npm test`):
```
Test Files  112 passed (112)
     Tests  1229 passed (1229)
```
Neu: `src/plugins/__tests__/hfObjectDetection.test.ts` (31 Tests); angepasst: `modalities.test.ts` (DETR jetzt unterstützt), `pluginManifests.test.ts` (Abgleich HF_DETECTION_MODEL_TYPES ↔ Manifest), `hfModelSupport.test.ts`.

Rust (`cargo test -- test_manager:: plugin_commands::`):
```
test test_manager::task_type_tests::detection_heisst_detect ... ok
test test_manager::task_type_tests::leitet_aufgabe_aus_dem_versionsordner_ab ... ok   (neu: rtdetr-l.pt, RTDetrForObjectDetection, model.json)
test plugin_commands::dependency_version_tests::aufgaben_haben_ihre_gruppe ... ok     (neu: hf_object_detection enthaelt timm)
test result: ok. 19 passed; 0 failed; 1 ignored
```

## Echte Mini-Läufe

coco128-Ausschnitt: 8 train / 4 val / 4 test, CPU (parallel liefen MPS-Trainings), kleine Bildgrößen.

| Lauf | Engine | Ergebnis |
|---|---|---|
| RT-DETR `rtdetr-l.pt`, 1 Epoche, imgsz 160 | Train | Val mAP50 0.499, mAP50-95 0.156; Familie `rtdetr` in model.json |
| dasselbe Modell | Test (`detect`) | mAP50 0.722, mAP50-95 0.388, Precision 0.93, Recall 0.70 |
| YOLOv10n, 1 Epoche | Train | mAP50 0.532 |
| YOLO-World `yolov8s-world.pt` | Train | Abbruch mit „YOLO-World braucht zum Training das Paket 'clip'“ |
| `hustvl/yolos-tiny`, 2 Epochen, imgsz 256 | Train | läuft durch, 13 s; Loss 7.18 → 6.13; mAP 0 (Klassenkopf neu, nur 4 Schritte) |
| `hustvl/yolos-tiny` vortrainiert | Test (`hf_object_detection`) | mAP50 0.842, mAP50-95 0.668; Einzelbild: Fahrrad, Hund, Personen mit Boxen |

## Nicht belegt / offen
- Training auf MPS nicht gemessen (nur CPU). Fällt eine Operation auf MPS aus (`NotImplementedError`), wechselt das HF-Plugin auf CPU — dieser Pfad ist nicht durch einen echten Lauf belegt.
- HF-Fine-Tuning mit mehr Schritten (sinkt die mAP-Lücke nach Kopf-Neuanlage?) nicht gemessen; der vortrainierte Kopf wird bei passenden Klassennamen nicht übernommen.
- DETR mit ResNet-Backbone (`facebook/detr-resnet-50`, braucht timm) und RT-DETR im HF-Format nicht als echter Lauf geprüft — nur YOLOS.
- YOLO-World-Training mit installiertem `clip` und YOLOE nicht geprüft.
- Box-Anzeige im Einzeltest nur per Unit-Test der Datenform geprüft, nicht in der laufenden App.
- Labor/Hosting kennen `hf_object_detection` noch nicht (dort läuft nur Ultralytics-Inferenz).
