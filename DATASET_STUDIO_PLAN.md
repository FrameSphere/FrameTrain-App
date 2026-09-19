# FrameTrain Dataset Studio – Systemplanung

Stand: 2026-09-20 — Beta: S1, S2, Qualitaetspruefung und Video-Erfassung gebaut
Ergaenzt `DATASET_ROADMAP.md`: dort geht es um *vorhandene* Datensaetze (erkennen,
splitten, importieren), hier um das *Erzeugen* neuer Datensaetze.

---

## 1. Ziel in einem Satz

Ein Werkzeug, mit dem aus Rohmaterial (Dateien, Aufnahmen, Web, eigenen Texten)
ein trainierbarer Datensatz wird – manuell, regelbasiert, modellgestuetzt oder
im aktiven Lernkreis, je nachdem was die Datenlage hergibt.

Einstieg: Button **"Datensatz bauen"** im Kopf des Dataset-Bereichs, links neben
"Dataset hinzufuegen". Neue View `studio` in `Dashboard.tsx` (`type View`); die
Seitenleiste bekommt keinen eigenen Eintrag (sie hat schon neun) und behaelt
waehrenddessen "Datasets" markiert, weil das Studio dessen Unterseite ist.

---

## 2. Was es schon gibt (und was der Studio davon erbt)

| Vorhanden | Datei | Nutzung im Studio |
|---|---|---|
| Typ-Erkennung fuer 10 Dataset-Typen | `dataset_manager.rs::detect_dataset_type` | Export prueft sich selbst gegen die Erkennung |
| `dataset.yaml`-Generator | `dataset_manager.rs::generate_dataset_yaml` | Export-Schritt fuer YOLO/Pascal |
| Typ-aware Split | `dataset_manager.rs::split_dataset` | Split bleibt beim Export, nicht im Studio |
| Box-Overlay in Bildkoordinaten (SVG) | `LaboratoryPanel.tsx::DetectionOverlay` | Basis der Bild-Werkbank (aus Ansicht wird Editor) |
| YOLO-Label-Parser, Klassenfarben, `data.yaml`-Klassen | `labGroundTruth.ts` | Lesen *und* Schreiben von Labeldateien |
| Persistenter Inferenz-Server (stdin/stdout JSON) | `laboratory_manager.rs`, `test_engine/model_server.py` | Vorschlags-Engine, gleiches Protokoll |
| Plugin-Installer mit pip + Fortschritts-Events | `plugin_commands.rs` | Optionale Assist-Modelle (Whisper, SAM) nachinstallieren |
| AI-Schicht mit eigenen Keys | `src/ai/aiClient.ts`, `ai_proxy.rs` | LLM-Labeling fuer Text, Klassenvorschlaege, Synthese |
| Modell-Plugin-Erkennung | `registry.ts::detectPluginForModel` | Welches Modell darf welche Vorschlaege machen |
| HTTP-Stack | `reqwest` in `Cargo.toml` | Web-Erfassung ohne neue Abhaengigkeit |
| Leerer Haken | `laboratory_manager.rs::lab_export_as_dataset` ("Noch nicht implementiert") | Lab-Session -> Studio-Projekt |

Neu zu bauen ist im Kern: die Projekt-Ablage, drei Editoren, die Vorschlags-Pipeline,
die Qualitaetspruefung und die Export-Writer.

---

## 3. Zentrale Entscheidung: ein neutrales Zwischenformat

Der Studio arbeitet **nie** direkt in YOLO- oder COCO-Struktur. Sonst muesste jeder
Editor jedes Zielformat kennen, und ein Wechsel des Zielformats waere ein Neuanfang.

Stattdessen ein Projektordner unter `<app_data>/datasets/<user_id>/_studio/<project_id>/`:

```
project.json        Name, Modalitaet, Aufgabe, Klassen-Schema, Zielformat, Quellen
media/ab/cd/<sha256>.<ext>   inhaltsadressiert -> exakte Duplikate kosten nichts
samples.jsonl       eine Zeile pro Sample (append-only, wird kompaktiert)
events.jsonl        jede Annotationsaktion (Undo/Redo, absturzsicher, Audit)
suggestions.jsonl   Maschinenvorschlaege, getrennt von bestaetigten Labels
thumbs/             Vorschaubilder, Wellenform-Peaks
exports/            erzeugte Datensaetze + Report
```

Eine Sample-Zeile:

```json
{"id":"s_00412","media":"ab/cd/9f3c….jpg","mime":"image/jpeg","status":"confirmed",
 "ann":{"boxes":[{"cls":0,"x":0.51,"y":0.40,"w":0.22,"h":0.31}]},
 "src":{"kind":"web","url":"https://…","license":"CC-BY-4.0","fetched":"2026-09-16"},
 "meta":{"w":1920,"h":1080,"group":"video_07","near_dup_of":null},
 "hist":[{"t":"suggest","by":"yolo:best.pt","conf":0.82},{"t":"confirm","by":"user"}]}
```

Drei Punkte daran sind bewusst so:

1. **`status` statt "gelabelt ja/nein":** `new -> suggested -> confirmed -> rejected/skipped`.
   Ein Maschinenvorschlag wird nie stillschweigend zur Wahrheit. Der Export nimmt
   per Default nur `confirmed`; "Vorschlaege ab Konfidenz X mitnehmen" ist ein
   bewusster Schalter und steht im Export-Report.
2. **`src` ist Pflichtfeld.** Damit faellt beim Export automatisch eine
   `PROVENANCE.csv` + `DATA_CARD.md` ab. Wer aus dem Netz sammelt, braucht das –
   sonst ist der Datensatz spaeter nicht weitergebbar.
3. **`meta.group`** (Video, Sprecher, Aufnahmesession, Quelldomain). Der Split beim
   Export ist gruppenbewusst: Frames aus demselben Video landen nie gleichzeitig in
   Train und Val. Das ist die haeufigste stille Ursache fuer zu gute Val-Werte.

Kein SQLite. Phase 10 der bestehenden Roadmap empfiehlt ohnehin, JSON zur einzigen
Quelle zu machen – ein neues Subsystem sollte den Fehler nicht wiederholen.

---

## 4. Schichten

**Frontend** (`src/components/studio/`)
- `StudioPanel.tsx` – Projektliste, Anlegen, Zielformat waehlen
- `workbench/TextWorkbench.tsx` | `ImageWorkbench.tsx` | `AudioWorkbench.tsx`
- geteilt: `SampleQueue.tsx` (virtualisiert), `LabelPalette.tsx`, `SuggestionBar.tsx`,
  `QualityPanel.tsx`, `ExportDialog.tsx`, `studioShortcuts.ts`
- reine Logik testbar ausgelagert (wie `labGroundTruth.ts`): `studioSample.ts`,
  `studioExport.ts`, `studioDedup.ts`

**Rust**
- `studio_manager.rs` – Projekte, samples.jsonl lesen/anhaengen/kompaktieren, Undo-Log
- `studio_sources.rs` – Ordner-Import, Web-Fetch, Hashing, Thumbnails anstossen
- `studio_export.rs` – Writer je Zielformat + Registrierung als normales Dataset

**Python** (`src-tauri/python/label_engine/`, neben `train_engine`/`test_engine`,
gleiches stdin/stdout-Zeilenprotokoll wie `model_server.py`)
- `predict_server.py` – Batch-Vorschlaege mit einem FrameTrain-Modell
- `assist/` – optionale Modelle (Whisper, SAM, CLIP/Grounding-DINO), per Plugin-Installer
- `media_ops.py` – Audio nach 16 kHz mono WAV, Bilder normalisieren, pHash, Embeddings
- `augment.py` – Augmentierung beim Export (opt-in, nur Train-Split)

**AI-Schicht** – unveraendert genutzt: `callAI` fuer Textlabels, Klassenvorschlaege,
synthetische Beispiele. Keine neue Infrastruktur, keine neuen Keys.

---

## 5. Die vier Erfassungswege

1. **Import** – Ordner/Dateien, inklusive "Ordnername = Klasse" als Regel
2. **Aufnahme** – Mikrofon, Kamera, Bildschirmausschnitt
3. **Erstellen** – Texteditor mit Schema (Klassifikation, Spans, Paare), LLM-Synthese
4. **Web** – HuggingFace (vorhanden), URL-/Sitemap-Liste, Such-APIs

---

## 6. Die fuenf Automatisierungsgrade

| Stufe | Was passiert | Kosten | Realistischer Anteil |
|---|---|---|---|
| 0 Manuell | Tastatur-first: 1–9 = Klasse, Enter = bestaetigen + weiter | – | immer noetig |
| 1 Regeln | Ordner-/Dateiname, Regex, Metadaten -> Label | 0 | bei sortiertem Material 50–80 % |
| 2 Modellvorschlag | vorhandenes FrameTrain-Modell schlaegt vor, Mensch bestaetigt | lokal | ab erstem Training |
| 3 Assist-Modelle | Whisper -> Transkript, SAM -> Maske aus Klick, CLIP/DINO -> Zero-Shot | Download | modalitaetsabhaengig |
| 4 Aktives Lernen | labeln -> kurz trainieren -> Rest vorhersagen -> nach Unsicherheit sortieren | Rechenzeit | ab ~200 bestaetigten Samples |

Stufe 1 ist der unterschaetzte Hebel: deterministisch, sofort, ohne Modell. Stufe 4
ist der, bei dem Werkzeuge gern mehr versprechen als sie halten – sie braucht einen
schnellen Trainingsmodus (wenige Epochen, kleiner Kopf), sonst wartet man bei jeder
Runde und benutzt es nicht.

Zusatznutzen aus Stufe 2/4: der Filter **"Modell widerspricht dem Label"** findet
Fehler in bereits gelabelten Daten – das ist oft wertvoller als neue Labels.

---

## 7. Qualitaet (vor dem Export, nicht danach)

- Duplikate: exakt ueber sha256 gratis; nah ueber pHash (Bild), Shingles (Text),
  Embedding-Cosinus (Audio/Text)
- Klassenbalance live, Warnung unter N Beispielen pro Klasse
- Zweifel-Queue: niedrige Konfidenz, Widerspruch, leere Annotation
- Review-Durchgang nur ueber auto-gelabelte Samples
- Export-Report: wie viele confirmed / suggested / uebersprungen, Balance, Duplikate,
  Gruppen-Leckage-Pruefung

---

## 8. Export

Zielformat waehlen -> Writer -> `generate_dataset_yaml` (vorhanden) -> gruppenbewusster
Split -> Registrierung ueber den bestehenden Import-Pfad. Danach ist es ein ganz
normales FrameTrain-Dataset mit korrekt erkanntem Typ.

Writer: `yolo_bbox`, `coco_json`, `folder_class`, `flat_file` (jsonl/csv),
`audio_transcript`, `pre_split`.

---

## 9. Risiken und offene Punkte

1. **Mikrofon in der Tauri-Webview (macOS):** `getUserMedia` scheitert ohne
   `NSMicrophoneUsageDescription` im Info.plist und passendes Entitlement – und zwar
   still. Empfehlung: Webview-`MediaRecorder` (WebM/Opus) + Konvertierung in Python,
   statt einer neuen Rust-Audio-Abhaengigkeit. Muss frueh an einer echten Release-
   Installation geprueft werden, nicht im Dev-Server.
2. **Web-Erfassung:** robots.txt beachten, Rate-Limit pro Domain, Domain-Allowlist,
   Lizenz mitschreiben. Kein pauschaler "alles herunterladen"-Knopf – der Nutzen
   liegt in URL-Listen, Sitemaps und APIs. Ohne Herkunftsdaten ist ein gesammelter
   Datensatz spaeter unbrauchbar, sobald er das eigene Geraet verlassen soll.
3. **Groesse:** ab ~100k Samples braucht es virtualisierte Listen, Thumbnail-Cache
   und Kompaktierung der JSONL. Vorher nicht optimieren.
4. **Umfang:** drei Modalitaeten gleichzeitig halb fertig ist schlechter als eine
   ganz. Empfehlung fuer den ersten Schritt: **Bild/YOLO**, weil Overlay,
   Labelparser, Klassenfarben und das YOLO-Plugin bereits stehen.
   *Entschieden am 2026-09-16: S1 wird Bild/YOLO.*

---

## 10. Vorgeschlagene Phasen

| Phase | Inhalt | Aufwand |
|---|---|---|
| S1 | Projekt-Ablage, Button, Shell, Import + manuelles Bild-Labeln (Boxen), YOLO-Export | **fertig** |
| S2 | Modellvorschlaege (Stufe 2) — **fertig**; Regeln (Stufe 1) und Zweifel-Queue offen | mittel |
| S3 | Zweite/dritte Modalitaet (Text-Editor, Audio-Aufnahme) | mittel |
| S4 | Assist-Modelle (Stufe 3) ueber den Plugin-Installer | mittel |
| S5 | Aktives Lernen (Stufe 4) + `lab_export_as_dataset` anschliessen | mittel |
| S6 | Web-Erfassung mit Herkunft und Ratenbegrenzung | mittel |

Querschnitt in jeder Phase: de/en vollstaendig, keine Emojis in der UI, Tests
(Frontend/Rust/Python) gruen, Pruefung an einer echten Release-Installation.

---

## 11. Was in S1 tatsaechlich steht

**Backend** `src-tauri/src/studio_manager.rs` (neun Commands, in `main.rs` registriert)
- Projekte anlegen, umbenennen, Klassen ergaenzen, loeschen, auflisten
- Ordner-Import: rekursiv, inhaltsadressiert (sha256), exakte Duplikate werden
  uebersprungen, Bildmasse aus dem Dateikopf (PNG, JPEG, GIF, BMP, WebP) ohne
  Bildbibliothek, Fortschritt als Event
- Vorhandene YOLO-Labels neben einem Bild werden uebernommen (auch Polygone,
  als umschliessende Box), Klassennamen aus classes.txt / obj.names / data.yaml
- Annotation als angehaengtes Ereignis (O(1) je Tastendruck), Kompaktierung ab
  etwa 2000 Ereignissen, kaputte Zeilen werden uebersprungen statt zu blockieren
- Export nach YOLO mit PROVENANCE.csv und DATA_CARD.md, anschliessend
  Registrierung ueber `import_local_dataset` — danach ein ganz normales Dataset

**Frontend** `src/components/studio/`
- `StudioPanel.tsx` Projektliste, Anlegen, Loeschen
- `ImageWorkbench.tsx` Box-Editor: zeichnen, auswaehlen, verschieben, Ecken
  ziehen, Klasse per Taste 1-9, Enter bestaetigt und springt zum naechsten
  offenen Bild, S ueberspringt, Entf loescht die Box. Automatisches Speichern
  nach 400 ms, kein Speichern-Knopf.
- `studioBoxes.ts` reine Umrechnung und Trefferpruefung, getrennt testbar
- Einstieg: Knopf "Datensatz bauen" im Kopf des Dataset-Bereichs; in der
  Seitenleiste bleibt dabei "Datasets" markiert.

**Wiederverwendet statt nachgebaut:** `clientToImagePoint`, `boxFromPoints` aus
`labCorrection.ts`, `classColor` aus `labGroundTruth.ts`, `import_local_dataset`
und `detect_dataset_type` aus `dataset_manager.rs`.

**Gemeinsamer YOLO-Schreiber** `src-tauri/src/yolo_export.rs`: `normalize_box`
(Pixelkanten), `norm_box` (bereits normiert), `label_line` und
`write_yolo_layout`. Das Labor (`lab_export_corrections`) und das Studio
(`write_yolo_export`) benutzen ihn beide — vorher gab es zwei Schreiber mit
demselben Format. Studio-eigen bleiben nur PROVENANCE.csv und DATA_CARD.md.

**Tests:** 19 Rust-Tests (13 Studio, 6 gemeinsamer Exporter) (Bildkopf, YOLO lesen/schreiben, Ereignis-Faltung,
Export wird von `detect_dataset_type` wieder als YOLO erkannt), 28 Frontend-Tests
(Umrechnung, Auswahl, Bestaetigen-Weg, Knopf im Dataset-Bereich).

**Bewusst noch nicht drin:** Vorschlaege (S2), zweite Modalitaet (S3),
Assist-Modelle (S4), aktives Lernen (S5), Web-Erfassung (S6). Der Split beim
Export bleibt Sache des Dataset-Bereichs; `meta.group` wird bereits
mitgeschrieben, ausgewertet wird es erst mit dem gruppenbewussten Split.

**Noch offen in S1:** eine Pruefung an einer echten Release-Installation mit
grossen Ordnern (Hashing-Dauer, fluessiges Blaettern ab etwa 10 000 Bildern).

---

## 12. Was in S2 dazugekommen ist

`studio_suggest` laesst eine trainierte Version ueber alle noch offenen Bilder
laufen. Benutzt wird derselbe Inferenz-Server wie im Labor
(`yolo_inference_server.py`, stdin/stdout) — ein zweiter Weg zum Modell waere
ein zweiter Weg, auf dem etwas anderes herauskommen kann.

- Bestaetigte Bilder werden nie angefasst; der Bericht sagt, wie viele das waren.
- Ein Treffer unterhalb der eingestellten Konfidenz wird verworfen (Regler 5 bis
  95 Prozent, Standard 25).
- Findet das Modell nichts, bleibt das Bild offen statt als "nichts drauf"
  vorgeschlagen zu werden — sonst drueckt man ein uebersehenes Objekt weg.
- Vorschlaege bekommen den Status `suggested` und werden gestrichelt gezeichnet;
  Enter macht daraus eine Bestaetigung.
- Klassenzuordnung ueber die Namen, ohne Ruecksicht auf Gross- und
  Kleinschreibung (`class_index_for`). Am echten Ski-Modell gepruft: es meldet
  Tree, Person, Generallobstacle, im Projekt stehen tree, person,
  generallobstacle — bei genauem Vergleich haette von sechs Projektklassen genau
  eine getroffen. Klassen ohne Entsprechung werden im Bericht genannt und ihre
  Boxen weggelassen, oder auf Wunsch neu angelegt.

Gegen den echten Checkpoint gemessen (13 Modellklassen, 512x512): Sky 0.94,
Tree 0.83, Tree 0.67 werden uebernommen, Offroad 0.72 faellt als unbekannte
Klasse heraus und steht im Bericht.

**Noch offen in S2:** Regeln (Ordner-/Dateiname, Regex) als Stufe 1 und die
Zweifel-Queue ("Modell widerspricht dem bestaetigten Label").

---

## 13. Beta-Stand (1.2.87)

Nach dem ersten echten Einsatz mit dem Ski-Projekt kamen vier Dinge dazu, in
dieser Reihenfolge:

**Klassenliste beim Import.** Labeldateien enthalten nur Zahlen. Lagen im
Quellordner keine Namen (kein classes.txt, keine data.yaml), legte der Import
die IDs stillschweigend auf die Klassenliste des Projekts — aus "Tree" der
Quelle wurde "Ski" im Projekt. Jetzt sieht `studio_inspect_folder` erst nach,
und bringt der Ordner Labels ohne Liste, fragt die App: Liste aus dem Ordner,
Namen aus einem trainierten Modell (`studio_model_classes` liest sie aus dem
Checkpoint), selbst eingeben, oder Labels weglassen. Abgebildet wird ueber
`remap_boxes` immer ueber **Namen**, nie ueber Zahlen. Ohne Liste verweigert
das Backend den Import, statt zu raten.

**Zweifel-Queue.** `studio_review` laesst ein Modell ueber die bereits
bestaetigten Bilder laufen und vergleicht Klassenmengen (`compare_class_sets`:
drei Baeume statt zwei sind kein Widerspruch, ein fehlender Lift schon). Die
Treffer stehen in doubts.json und sind ueber den Filter "Zweifel" erreichbar.
Geaendert wird nichts, nur markiert. Bei uebernommenen Fremdlabels findet das
mehr als jedes neue Label.

**Durchsatz im Editor.** Rueckgaengig (Cmd+Z) ueber einen Stapel je Bild,
"Boxen vom vorigen Bild uebernehmen" (V) fuer aufeinanderfolgende Aufnahmen,
Zoom (+ / − / 0) fuer kleine Objekte.

**Video.** `extract_frames.py` (OpenCV) schreibt jedes n-te Bild eines Videos;
`studio_import_video` nimmt sie mit dem Videonamen als `meta.group` auf. Damit
verdient sich die Gruppe ihren Platz: `assign_splits` teilt beim Export
gruppenbewusst auf, eine Gruppe landet nie in Train und Val gleichzeitig. Der
Export schreibt dann images/<split>/ + labels/<split>/ — das Layout, das der
Dataset-Import als fertig aufgeteilt erkennt.

**Zahlen:** 96 Rust-Tests, 620 Frontend-Tests, 4 Python-Testdateien gruen.

**Was fuer den echten Test noch fehlt:** ein Lauf mit den 463 Bildern (Dauer
des Imports, Fluessigkeit der Warteschlange) und ein Vorschlags-Lauf ueber
mehr als eine Handvoll Bilder.
