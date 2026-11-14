# Engine-Ausbau: Semantische Segmentierung (HuggingFace)

Branch `claude/engine-segmentierung-57b9a4`. Neue Aufgabe `semantic_segmentation`
neben dem YOLO-Plugin (YOLO unverändert).

## Was gebaut wurde

| Teil | Datei(en) |
|---|---|
| Gemeinsame Datenlogik (beide Engines) | `src-tauri/python/ft_data/segmentation.py` |
| Train-Plugin | `src-tauri/python/train_engine/plugins/semantic_segmentation/` |
| Test-Plugin | `src-tauri/python/test_engine/plugins/semantic_segmentation/` |
| Frontend-Plugin | `src/plugins/semantic-segmentation/` (detect, index, TestPlugin) |
| Registrierung | `src/plugins/registry.ts` (+4 Zeilen, vor `hfImageClassificationPlugin`) |
| Dataset-Typ `image_mask` | `datasetCompatHelpers.ts`, `genericDatasetCompat.ts`, `src-tauri/src/dataset_manager.rs` |
| Übersetzungen EN+DE | `src/locales/{en,de}.json` (nur neue Schlüssel) |

**Modelle:** alles, was `AutoModelForSemanticSegmentation` lädt: SegFormer (auch der
reine Encoder `nvidia/mit-b0…b5`), UperNet, DPT (nur mit Segmentierungs-Hinweis, nicht
die Tiefenschätzer), BEiT / Data2Vec-Vision / MobileViT / MobileViTV2 / MobileNetV2-DeepLab
(nur mit Hinweis wie `ade`, `deeplab`, `…ForSemanticSegmentation`, sonst bleiben sie
Bildklassifikation). **Mask2Former/MaskFormer nicht**: die trainieren mit Masken- und
Klassenlisten je Bild (Universal Segmentation) und laufen nicht über
`AutoModelForSemanticSegmentation`.

**Daten:**
- `images/` + `masks/` (ungeteilt, Val wird automatisch abgetrennt), `train|val|test/images|masks`
  (so teilt die App auf), `images/<split>` + `annotations/<split>` (ADE20K-Stil).
  Maskenordner auch `labels`, `annotation(s)`, `segmentation(class)`, `gt`; Bildordner auch `jpegimages`.
- Paarung über den Dateinamen, Endungen `_mask`, `_seg`, `_label`, `_gt` am Maskennamen werden ignoriert.
- Masken mit Klassen-IDs (Graustufen, 16 bit, Paletten-PNG wie Pascal VOC; 255 = ignorieren)
  oder Farbmasken (RGB). Farbtabelle aus `classes.json` / `labelmap.txt` (CVAT/VOC) /
  `classes.txt` mit RGB, sonst aus den Masken abgeleitet (Schwarz zuerst).
- Klassenliste aus `classes.json`, `id2label.json`, `label_map.json`, `labelmap.txt`,
  `classes.txt`, `class_names.txt`, `labels.txt`, `obj.names`; sonst `class_0 …` aus den Masken.

**Training:** Loss, mIoU, Pixel-Accuracy, mittlere Klassen-Accuracy je Auswertung
(erscheinen als `eval_mean_iou` usw. im Fortschritt). Das Modell mit dem besten Val-mIoU
wird gemerkt und gespeichert, nicht das letzte. `label_mapping.json` im Modellordner
enthält Klassen, Farben, Maskenformat und Eingabegröße; die Test-Engine liest Masken damit
genau wie das Training. `plugin_config`: `image_size` (Standard 512), `augment` (Spiegeln),
`val_fraction`.

**Test:** Datensatz-Lauf auf test → val → train: mIoU, IoU je Klasse, Pixel-Accuracy,
bis zu 12 Überlagerungen, `hard_examples.json` (Bilder mit dem schlechtesten mIoU).
Einzelbild: Klassen-ID-Maske, Farbmaske und Überlagerung als PNG; Legende mit Farbe
und Flächenanteil. Die Oberfläche legt die Farbmaske mit Deckkraft-Regler über das Bild.

**Rust:** `DatasetType::ImageMask` (`image_mask`). Erkennung vor YOLO und Ordner-Klassen
— vorher galt `images/` + `masks/` als Ordner-Klassifikation mit den Klassen „images“
und „masks“. `labels/` zählt als Maskenordner nur, wenn dort ausschließlich Bilder liegen
(YOLO mit `.txt` bleibt YOLO). Aufteilen nutzt das vorhandene `split_paired_dirs`.

## Befund: SegFormer-Training stürzt auf Apple MPS ab (behoben)

Mit torch 2.12.1 bricht der Rückweg jedes SegFormer bei 128–512 px auf MPS ab:
`view size is not compatible with input tensor's size and stride`. Ursache: der
Decoder verkettet die Stufen im `channels_last`-Layout, der Rückweg durch
`linear_fuse`/BatchNorm verträgt das nicht. Fix im Train-Plugin
(`apply_mps_fixes`): `linear_fuse` bekommt eine zusammenhängende Eingabe.

| Batch × px (mit-b0-Konfiguration, MPS) | ohne Fix | mit Fix |
|---|---|---|
| 2 × 512 | Fehler | ok |
| 4 × 256 | Fehler | ok |
| 8 × 128 | Fehler | ok |
| 4 × 64 | ok | ok |
| 4 × 32 | Fehler | Metal-Assert (Prozess stirbt) |

Bei 32 px ist die letzte Stufe 1×1 Pixel, dort scheitert ein zweiter MPS-Fehler in der
Faltung. Deshalb hebt das Plugin die Eingabegröße auf mindestens 64 px an.

## Tests (alle grün)

```
$ cd src-tauri/python && python3.11 -m unittest ft_data.test_segmentation
.................
Ran 17 tests in 0.114s
OK
```
Layouts (ungeteilt, App-Split, ADE-Stil, fehlende Paare), ID-Masken, Paletten-PNG mit 255,
Farbmasken mit `classes.json`, Farbmasken ohne Tabelle, unbekannte Farben,
`labelmap.txt`/`classes.txt`-Formate, Klassenliste begrenzt IDs, Farbtabelle des Modells
in der Test-Engine, mIoU mit bekannten Werten und Ignore-Pixeln, Vorverarbeitung
(nearest, Spiegeln), Legende und Überlagerung.

```
$ cd src-tauri/python/train_engine && python3.11 -m unittest -v plugins.semantic_segmentation.test_plugin
test_farbmasken_mit_farbtabelle ... [Farbmasken] Train mIoU 0.946 | Test mIoU 0.915 | Pixel-Acc 0.974
ok
test_klassen_id_masken ... [ID-Masken] Train mIoU 0.404 | Test mIoU 0.404 | Pixel-Acc 0.809
ok
Ran 2 tests in 70.243s
OK
```
Mini-Trainingslauf Ende zu Ende: winziger SegFormer (≈9.000 Parameter, lokal erzeugt)
durch `train_engine.py` (MPS), danach `test_engine.py` im Datensatz- und Einzelbild-Modus.
Der ID-Masken-Fall lernt in 3 Epochen nur den Hintergrund sicher (mIoU 0,40); geprüft
wird dort der Weg, nicht die Güte.

```
$ python3.11 test_plugin_consistency.py      -> Ran 4 tests, OK
$ npm test                                   -> Test Files 112 passed, Tests 1207 passed
$ npx tsc --noEmit                           -> keine Fehler
$ cargo test --bins dataset_manager::        -> 49 passed; 0 failed (2 neu: image_mask, labels mit PNG vs. YOLO)
```
vitest `src/plugins/__tests__/semanticSegmentation.test.tsx` (10 Tests): Architekturliste
== Train-Manifest, Erkennung (SegFormer, mit-b0, UperNet, BEiT/MobileViT nur mit Hinweis,
DPT-Tiefe und Mask2Former nicht, lokaler Import), Dataset-Kompatibilität `image_mask`,
Übersetzungen EN/DE vollständig, Engine-Antworten, Test-Oberfläche (Maske über dem Bild,
Deckkraft, Legende; Datensatz mit mIoU, IoU je Klasse, abwesende Klasse).

## Echter Lauf mit `nvidia/mit-b0`

Selbst erzeugter Datensatz: 16 Bilder 128 px mit Quadrat und Kreis (`images/` + `masks/`,
ID-Masken, `classes.txt`), getrennter Test-Datensatz mit `test/` (6 Bilder).
Train-Engine: 6 Epochen, Batch 4, lr 6e-4, `image_size` 128, Gerät MPS.

```
Kein val-Split - 3 von 16 Paaren werden zur Validierung abgetrennt.
Klassen-ID-Masken | Klassen (3): hintergrund, quadrat, kreis
MPS-Anpassung fuer den SegFormer-Decoder aktiv.
Modell geladen (SegformerForSemanticSegmentation) | Parameter: 3.7M | Klassen: 3
ep 2  val 0.9422  mIoU 0.6453  Pixel-Acc 0.8938
ep 3  val 0.9121  mIoU 0.8480  Pixel-Acc 0.9727
ep 4  val 0.7477  mIoU 0.8889  Pixel-Acc 0.9754
ep 5  val 0.6874  mIoU 0.9120  Pixel-Acc 0.9796
ep 6  val 0.5199  mIoU 0.9442  Pixel-Acc 0.9892
COMPLETE mean_iou 0.944, pixel_accuracy 0.989, best_epoch 6, total_steps 24, device mps
         per_class_iou: hintergrund 0.989, quadrat 0.940, kreis 0.903
```
Test-Engine auf dem Test-Split (6 Bilder): **mIoU 0,946**, Pixel-Accuracy 0,988,
IoU hintergrund 0,987 / quadrat 0,938 / kreis 0,912, 6 Überlagerungen gespeichert.
Einzelbild: Legende hintergrund 85,5 % / quadrat 10,0 % / kreis 4,5 %, Überlagerung
geprüft (Quadrat und Kreis an der richtigen Stelle eingefärbt).

## Nicht belegt / offen

- **Nicht in der laufenden App geprüft** (kein Release, kein `tauri:dev`): Import eines
  `images/` + `masks/`-Ordners, Anzeige „Bild + Maske“, Aufteilen per App-Split, Test-Seite
  mit echten `asset:`-Bildern.
- UperNet, DPT, BEiT, MobileViT: nur über die Erkennung und denselben Codepfad
  (`AutoModelForSemanticSegmentation`) abgedeckt, nicht mit echten Gewichten trainiert.
  Der MPS-Fix greift nur bei Modellen mit `decode_head.linear_fuse` (SegFormer); ob die
  anderen Köpfe auf MPS denselben Fehler haben, ist offen.
- Rust-Paarungsprüfung (`check_basename_pairing`) kennt die Endungen `_mask`/`_seg` nicht:
  Masken wie `a_mask.png` erscheinen dort als „Bild ohne Label“, obwohl Python sie paart.
  Der App-Split (`split_paired_dirs`) würde solche Paare trennen.
- Keine Verbindung zu Labor/Hosting (`lab_infer`, `hosting_infer`) und keine eigene
  Anzeige auf der Analyse-Seite für `per_class_iou` (die Analyse zeigt mIoU über die
  normalen Kennzahlen-Felder, `accuracy` = Pixel-Accuracy).
- Klasse 0 bekommt in der VOC-Standardpalette Schwarz; die Überlagerung dunkelt den
  Hintergrund dadurch ab. Eigene Farben über die Farbtabelle möglich.
- Die Mindestgröße 64 px gilt für alle Architekturen, gemessen nur an SegFormer.
