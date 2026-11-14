# Engine-Ausbau: Zeitreihen-Vorhersage und Tabellen (Session zeitreihen-tabellen)

Branch: `claude/engine-zeitreihen-tabellen-a0a74c`. Nichts gepusht, keine Versionsnummer geändert, Canvas-Plugin nicht angefasst.

## Entscheidung: Training ohne importiertes Modell

**So entstehen Canvas-Modelle:** Der Synapse-Builder ruft den Rust-Befehl `create_canvas_network_model` auf. Er legt `models/canvas_<id>/` mit `graph_metadata.json` an und trägt Modell und Root-Version in die Datenbank ein. Im Trainings-Panel ist das danach ein ganz normales Modell. `start_training` erkennt es an der ID und lässt den Canvas-Sonderweg laufen.

**Gewählter Weg (geringster Eingriff):** Es gibt einen neuen, eigenständigen Rust-Befehl `create_task_model(task, modelName)` in `model_manager.rs`. Er legt `models/local_<id>/` an mit
- `config.json` = `{"model_type": "ft_time_series" | "ft_tabular", "frametrain_task": …}`,
- `.frametrain_plugin.json` mit der Plugin-Zuordnung (derselbe Mechanismus wie „Plugin zuordnen“ beim Import),

und schreibt Modell plus Root-Version genau wie `import_local_model`. Ab da läuft **alles über den unveränderten Standardweg**: Modellliste, Plugin-Erkennung (`model_type` bzw. Zuordnung), `start_training` (normaler Zweig, `model_path` = Modellordner), Version anlegen, `start_test` (`infer_task_type` liest das `model.json` `{"task": …}`, das beide Plugins beim Export schreiben).

In der UI steht im Import-Dialog der Modellbibliothek ein dritter Reiter **„Neu“** (`src/components/BlankModelPanel.tsx`): Aufgabe wählen, Namen vergeben, „Modell anlegen“.

Verworfene Alternativen:
- **Den Canvas-Sonderweg mitbenutzen** (ID-Präfix, `canvas_graph`): Das hätte `training_manager.rs` und das Canvas-Plugin berührt.
- **Ordner im Frontend über plugin-fs schreiben und dann `import_local_model` aufrufen:** Das braucht Schreibrechte des Webviews auf beliebige Pfade und ist fehleranfälliger.

Die Engine-Plugins brauchen den Modellordner nur als Quelle: leer heißt neu anlegen, `patchtst` mit Gewichten heißt von HF feinabstimmen, eine FrameTrain-Version heißt weitertrainieren.

## Was gebaut wurde

### Zeitreihen (`task_type` `time_series`)
- `train_engine/plugins/time_series/ts_core.py`: gemeinsamer Kern für Train und Test.
- `train_engine/plugins/time_series/plugin.py`, `test_engine/plugins/time_series/plugin.py`.
- **Daten:** CSV, TSV, Parquet, JSONL. Die Zeitspalte wird am Namen oder am Inhalt erkannt, ohne Zeitspalte gilt die Zeilenfolge. Es gibt eine oder mehrere Wertspalten und optional eine Serien-ID. Lücken werden linear interpoliert, an den Rändern fortgeschrieben.
- **Aufteilung immer zeitlich je Serie** (Train → Val → Test am Ende, jeder Abschnitt mindestens eine Vorhersagelänge). Alle Dateien eines Datasets werden zusammengeführt und nach Zeit sortiert, denn der Split-Knopf der App mischt Zeilen zufällig. Trainingsfenster sehen nie einen Wert nach dem Trainingsende. Die Normalisierung je Serie und Kanal kommt nur aus dem Trainingsteil.
- **Modell:** `PatchTSTForPrediction`, entweder neu (Kontext- und Vorhersagelänge, `d_model`, Schichten einstellbar) oder von einem HF-PatchTST. Beim HF-Modell gilt dessen Kontextlänge; der Vorhersagekopf wird bei anderer Länge neu angelegt.
- **Vergleichswert:** letzter Wert bzw. saisonal naiv (`season_length`).
- **Test:** MAE, RMSE, sMAPE (%), MASE (Skala = naiver In-Sample-Fehler des Trainingsteils), jeweils auch für die naive Vorhersage, dazu `mae_skill`. Der Test bewertet den letzten Zeitabschnitt mit derselben Aufteilung wie im Training.
- **Kurve:** `forecast_curve` (erste Serie, erster Kanal: Kontext, echte Werte, Modell, naiv) geht im `test-complete`-Ereignis an die UI.
- **Einzeltest:** Die letzten Werte als Liste eingeben, das Modell liefert die nächsten H Werte. Zu kurze Eingaben werden aufgefüllt, mit Hinweis.

### Tabellen (`task_type` `tabular`)
- `train_engine/plugins/tabular/tabular_core.py`: gemeinsamer Kern.
- `train_engine/plugins/tabular/plugin.py`, `test_engine/plugins/tabular/plugin.py`.
- **Netz:** Zahlen werden mit dem Median aufgefüllt, standardisiert und bekommen einen Fehlt-Indikator. Kategorien werden zu Embeddings; Index 0 steht für fehlend oder unbekannt. Freitext- und ID-Spalten werden ignoriert. Danach folgen MLP-Schichten mit LayerNorm, GELU und Dropout.
- **Ziel und Aufgabe:** Die Zielspalte ist wählbar; leer heißt `label`/`target`/`y`/… oder die letzte Spalte. Die Aufgabe (Klassen oder Zahl) wird automatisch erkannt und lässt sich übersteuern.
- **Aufteilung:** Vorhandene train/val/test-Dateien gelten, sonst wird deterministisch mit Seed geteilt. Der Test nutzt dieselbe Aufteilung, wenn keine Testdatei existiert.
- **Gespeichert neben den Gewichten:** `preprocessing.json` (Skalierung, Kategorie-Tabellen, Klassen, Zielnormierung), `config.json`, `model.json`. `tabular_core.load_bundle()` + `predict_frame()` sind die eine Stelle für Test und künftiges Hosting.
- **Test:** Accuracy und Macro-F1 bzw. MAE, RMSE und R², jeweils mit Vergleichswert (häufigste Klasse bzw. Mittelwert).
- **Einzeltest:** eine Zeile als `spalte=wert; …`, als JSON-Objekt oder als Werte in Spaltenreihenfolge.

### Frontend und Rust
- `src/plugins/time-series/` (detect, index, TestPlugin mit SVG-Kurve, `forecastChart.ts`), `src/plugins/tabular/`.
- `registry.ts`: zwei Imports und zwei Einträge direkt nach Canvas (rein additiv).
- `en.json`/`de.json`: Plugin-Namen, Parameter-Beschriftungen, Kennzahl-Namen, Texte für Test und „Neu“-Dialog (nur hinzugefügt).
- `ModelManager.tsx`: Reiter „Neu“. `BlankModelPanel.tsx` ist neu.
- `model_manager.rs`: `create_task_model` und `blank_task_model` plus Rust-Test; `main.rs` hat eine Zeile mehr im invoke_handler. Kein neues `Command::new`.

## Tests (alle grün)

```
# Kern + Mini-Training (aus src-tauri/python/train_engine)
python3.11 -m unittest plugins.tabular.test_tabular plugins.time_series.test_time_series
Ran 22 tests in 6.2s — OK
python3.11 test_plugin_consistency.py
Ran 4 tests — OK
```

Abgedeckt sind:
- Datenladen: Split-Ordner, Dateinamen, Semikolon-CSV, Parquet, zufällig verteilte Zeilen werden zeitlich sortiert.
- Fehlende Werte: Imputation, Indikator, unbekannte Kategorie, fehlende Spalte, Interpolation, Ränder.
- Zeitliche Aufteilung ohne Leck: Trainingsziele enden vor dem Trainingsende, Val- und Testziele liegen in ihrem Abschnitt, die Normalisierung kommt nur aus dem Trainingsteil, der Fensterinhalt entspricht der Reihe.
- Kennzahlen von Hand nachgerechnet: MAE, RMSE, sMAPE, MASE, Accuracy, Macro-F1, R².
- Mini-Trainings direkt über die Plugin-Klasse.

**Echter Lauf durch beide Engines** (Subprozess `train_engine.py --config` → `test_engine.py --config`, selbst erzeugte Daten, nur CPU, Modell aus leerem Ordner wie nach „Neu“):

```
python3.11 test_tabular_timeseries_e2e.py      Ran 4 tests in 44s — OK
[tabular train] test_accuracy=0.985 baseline=0.505 macro_f1=0.985
[tabular test ] accuracy=0.985 ... baseline_accuracy=0.505
[tabular single] ja (1.00)
[tabular regression test] mae=0.533 baseline_mae=2.862 r2=0.952
[tabular regression single] 10.12 (erwartet ~9.5)
[time_series scratch]    windows=16 mae=0.204 naive_mae=2.936 smape=0.89% mase=0.202 naive_mase=2.901
[time_series pretrained] windows=16 mae=0.287 naive_mae=2.936 smape=1.34% mase=0.283 naive_mase=2.901
```

„pretrained“ simuliert einen HF-Download: ein PatchTST mit 7 Kanälen, Kontext 64 und Vorhersage 24 wird per `save_pretrained` gespeichert und auf 1 Kanal und Vorhersage 12 feinabgestimmt.

```
npm test             Test Files 112 passed, Tests 1208 passed   (neu: src/plugins/__tests__/timeSeriesTabular.test.tsx, 11 Tests)
npx tsc --noEmit     ok
cargo test blank_task_model   1 passed
```

## Nicht belegt / offen
- **Nicht in der laufenden App geklickt:** kein Release, keine Vorschau-App. Der Reiter „Neu“, das Formular mit den Plugin-Parametern und die Kurve sind nur per vitest (jsdom) und `tsc` geprüft. Vor dem Release einmal live: Modell anlegen, trainieren, testen.
- **Kein echter Hugging-Face-Download** von z. B. `ibm-granite/granite-timeseries-patchtst`. Der Weg ist mit einem lokal gespeicherten PatchTST gleicher Struktur geprüft.
- **Hosting/Labor:** Die Vorverarbeitung liegt im Modell und `tabular_core.predict_frame` / `ts_core` sind dafür vorbereitet. `model_host.rs` und Labor kennen die beiden Aufgaben aber noch nicht.
- **KI-Verbindungen (MCP):** Es gibt kein Werkzeug, um ein leeres Modell anzulegen. `training_start` funktioniert, sobald das Modell existiert.
- **Kurve:** Es wird nur Serie 1, Kanal 1 gezeigt; die Kennzahlen gelten für alle Serien und Kanäle.
- **Hinweis im Registry-Text:** Chronos, TimesFM, Moirai und Lag-Llama melden weiter „kein Engine-Plugin“. Das stimmt, nur PatchTST wird unterstützt.
- **Datenmenge:** Die Plugins laden die ganze Tabelle in den Speicher. Für sehr große Dateien (> RAM) gibt es kein Streaming.
