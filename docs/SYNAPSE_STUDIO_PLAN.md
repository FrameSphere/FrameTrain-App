# Synapse Studio: Plan

Stand: 02.11.2026, Basis Version 1.6.13. Status: Kern aller Phasen gebaut, auf `feat/synapse-studio` committet. Live in der Vorschau-App geprueft: Bild, Audio, Text, Mini-GPT, Speichern, PyTorch-Export, Inferenz-Tab, Hosting. Nur in Python getestet: Video, Seq2Seq, U-Net, VAE, GAN, Diffusion, Detektion. Offen: Labor, ONNX-Lauf, Datei-Eingabe im Hosting pruefen, Port-Farben, Budget, Einzelschritt-Ansicht.

## 1. Ziel

Im Synapse Builder baut man ein komplett eigenes KI-Modell: planen, Daten anbinden,
Architektur designen, exportieren, trainieren, ausliefern. Für Bild, Audio, Video, Text,
Tabellen und Mischformen. Was als Baustein fehlt, schreibt man selbst als Code-Node.

Maßstab: Jeder sichtbare Regler wirkt wirklich. Was im Editor geprüft wird, ist exakt das,
was später trainiert wird. Der Export ist normaler PyTorch-Code ohne FrameTrain-Abhängigkeit.

## 2. Warum der heutige Aufbau nicht reicht

| Heute | Folge | Neu |
|---|---|---|
| Ein Eingang `x`, ein Ausgang, ein Loss | Keine Mehrfach-Eingaben, kein VAE, kein Seq2Seq | Benannte Ports, mehrere Ein- und Ausgänge, mehrere Losses |
| Kanten ohne Port-Auswertung | Split, Attention, A/B-Eingänge falsch | Werte je `(Node, Port)` |
| Formen als Text-Tags, doppelt in TS und Python | Geprüft wird nicht, was läuft | Ein Formen-Lauf direkt mit PyTorch |
| Größen von Hand (`inputSize`) | Fehleranfällig, Hauptfrust | Größe `auto` aus der Vorstufe |
| Daten nur als Loader-Node | Tokenizer, Augmentation, Split ohne Wirkung | Eigener Daten-Graph |
| Vorverarbeitung nicht gespeichert | Falsche Vorhersagen nach Training | Vorverarbeitung gehört zum Modell |
| Code-Export nicht erreichbar | Kein echtes Eigentum am Modell | Sauberes PyTorch-Projekt |

## 3. Architektur

Drei Graphen in einem Dokument, ein gemeinsamer Node-Katalog:

- **Daten-Graph** läuft je Beispiel im DataLoader: Quelle → Transformationen → benannte Felder.
- **Modell-Graph** läuft je Batch: Tensoren durch Layer, Blöcke und Code-Nodes.
- **Lern-Graph** verbindet beides: Losses mit Gewicht, Metriken, Optimizer, Rezept.

### 3.1 Node-Katalog: eine Quelle der Wahrheit

Neues Python-Paket `src-tauri/python/synapse/`. Jeder Node ist eine Python-Definition:

```python
@node("conv2d", family="layer", inputs={"in": Image}, outputs={"out": Image})
class Conv2d:
    params = {"out_channels": Int(64), "kernel": Int(3), "in_channels": Auto()}
    def build(p, shapes): return nn.Conv2d(shapes["in"][1], p.out_channels, p.kernel)
    def emit(p, shapes): ...  # Quelltext für den Export
```

Daraus entstehen automatisch:
- die Bibliothek im Frontend (`catalog.generated.json`, ein Test hält die Datei aktuell)
- der Inspector, ohne eigenes UI je Node
- Laufzeit, Formen-Prüfung und Code-Export

Die heutigen 36 Nodes wandern dorthin. Der Katalog wächst auf rund 120 (Abschnitt 6).

### 3.2 Port-Typen

`Vector [B,D]`, `Sequence [B,T,D]`, `Tokens [B,T]`, `Image [B,C,H,W]`, `Volume [B,C,T,H,W]`,
`Waveform [B,C,S]`, `Spectrogram [B,C,F,T]`, `Mask`, `Boxes`, `Label`, `Text`, `Loss`, `Any`.

Jeder Typ hat eine Farbe. Verbindungen werden sofort im Frontend geprüft, ohne Python.
Passende Umwandlungen bietet der Editor als Ein-Klick-Node an, etwa Flatten zwischen
`Image` und `Vector`. Heimliche Umwandlungen wie Mittelwert bei Dense entfallen für neue Graphen.

### 3.3 Live-Prüfung (Formen-Dienst)

Ein dauerhafter Python-Prozess (`synapse/service.py`, Rust: `synapse_service.rs`, async,
`.no_window()`, JSON-Zeilen wie `model_host.rs`) bekommt nach jeder Änderung den Graphen und
führt ihn auf PyTorch-Meta-Tensoren aus. Ergebnis je Node: Formen, aufgelöste `auto`-Werte,
Parameterzahl, Fehlertext. Das Bauen des Modells ist dieselbe Funktion wie im Training.

Gemessen am 02.11.2026 mit torch 2.12 auf diesem Mac:

| Messung | Wert |
|---|---|
| `import torch` | 0,92 s |
| Erster Lauf ViT-Tiny, 77 Nodes | 704 ms |
| Weitere Läufe | 26 ms |
| 365 Nodes, 425 Mio. Parameter | 126 ms, ohne RAM-Belegung der Gewichte |
| Getestete Bausteine (LSTM, GRU, Conv1d/3d, ConvTranspose, Decoder, Einsum u. a.) | 12 von 12 laufen |

Eine gestrichelte Kante oder ein grauer Badge zeigt, solange der Dienst noch rechnet.

### 3.4 IR v2 und Laufzeit

```json
{ "version": 2,
  "data":   { "nodes": [], "edges": [] },
  "model":  { "nodes": [], "edges": [], "inputs": {}, "outputs": {} },
  "learn":  { "losses": [], "metrics": [], "optimizers": [], "recipe": "supervised" },
  "blocks": { "resblock": { "nodes": [], "edges": [] } } }
```

- Kanten verweisen immer auf Quell- und Ziel-Port.
- `forward(batch: dict) -> dict` statt `forward(x)`.
- Blöcke sind Unter-Graphen mit `repeat: N`.
- `migrate_v1()` hebt alte Modelle automatisch an. Die Gewichts-Namen `layers.<id>` bleiben,
  alte `model.pt` laden weiter. Der alte Pfad bleibt, bis die Migration in Tests belegt ist.

### 3.5 Daten-Graph

Quellen nutzen vorhandene `ft_data`-Module wieder:

| Art | Quelle | Wiederverwendet |
|---|---|---|
| Bild, Audio, Video mit Klassen | Ordner, Splits, Parquet | `ft_data.media`, `ft_data.video` |
| Text mit Label, NER | CSV, JSONL, CoNLL | `ft_data.text`, `ft_data.tokens` |
| LLM | Chat, Instruction, Rohtext | `ft_data.llm` |
| Text zu Text | Spalten | `ft_data.seq2seq` |
| Audio mit Transkript | Paare, Metadaten | `ft_data.asr` |
| Bild mit Text | Tabellen, Ordner | `ft_data.image_text` |
| Detektion | YOLO | `ft_data.detection` |
| Bild mit Maske | Bild- und Maskenordner | neu |
| Tabelle | CSV, Parquet | heutiger Canvas-Loader |

Transformationen: Resize, Crop, Flip, Farbe, Normalisieren, Resample, Mel, MFCC,
Frame-Sampling, Tokenizer (eigener BPE/Zeichen/Wort oder vorhandener), Padding, Split.
Zufällige Schritte laufen nur im Training.

**Gelernte Werte** (Mittelwerte, Vokabular, Klassennamen) landen in `preprocess.json` neben
den Gewichten. Inferenz, Labor, Hosting und Export nutzen dieselbe Pipeline.

### 3.6 Lernen

Rezepte als kleine Python-Klassen:

| Rezept | Für |
|---|---|
| `supervised` | Klassifikation, Regression, Multi-Label, Segmentierung, NER, Fusion |
| `language_model` | GPT-artige Modelle: verschobene Ziele, Sampling, Temperatur |
| `seq2seq` | Übersetzung, Zusammenfassung, Audio zu Text |
| `autoencoder` | Autoencoder, VAE |
| `gan` | zwei Optimizer, abwechselnde Schritte |
| `diffusion` | Rausch-Plan, Zeitschritt, Sampling |
| `custom` | eigener `training_step` als Code |

Mehrere Losses mit Gewicht, Metriken als Nodes (Accuracy, F1, IoU, WER, PSNR).
Dazu bestes Modell speichern und Early Stopping, Gerät und Präzision wirken wirklich.
Der Python-Trainings-Loop bleibt im vorhandenen Plugin `canvas`, das Rust-Protokoll,
Dashboard, Analyse und Versionen bleiben unverändert.

### 3.7 Volle Kontrolle

- **Code-Node:** eigene PyTorch-Klasse mit Ports, im Editor geschrieben, in der Live-Prüfung
  mit ausgeführt. Läuft lokal ohne Sandbox wie Dev Train, die UI sagt das beim ersten Mal.
- **Eigene Transformation, Loss, Metrik, Trainingsschritt** nach demselben Muster.
- **Vortrainierter Baustein:** Modelle aus der Bibliothek oder von Hugging Face als Node, mit
  Einfrieren und Ausgangsauswahl.
- **Eigene Blöcke:** Auswahl zu Block zusammenfassen, speichern, wiederverwenden, teilen.

- **Neuron-Ebene:** Eigene Layer entstehen auch ohne Code: `Parameter`-Nodes für Gewichte `w` und
  Bias `b`, Rechen-Nodes für `w·x + b`, Funktions-Node für `f`. Daraus wird ein wiederholbarer Block.
- **Eigener Optimizer und eigene Aktivierung** als Code-Nodes, im selben Muster wie Loss und Metrik.
- **Lernen sichtbar machen:** Im Einzelschritt-Modus zeigt der Canvas je Node Ausgabe, Gradient und
  Gewichtsänderung. Dazu die Reihenfolge Forward, Loss, Backprop, Update für einen Batch.

### 3.7a NeuronEngine: Modelle ohne PyTorch

Stufe 2 der Entwicklung: ein eigenes kleines Framework statt PyTorch. `src-tauri/python/neuron_engine/`
enthält Tensor mit eigener Autograd, Layer, Aktivierungen, Losses, Optimizer und Trainingsschleife
auf NumPy, ohne Abhängigkeit von torch. Jede Datei ist lesbar und wird beim Export mitgeliefert.

Umfang: Linear, Conv2d, Embedding, Attention, Normierung, Aktivierungen, CrossEntropy, MSE, SGD, Adam.
Grenzen: nur CPU, deutlich langsamer, nicht für Code-Nodes und vortrainierte Bausteine. Die Live-Prüfung
sagt je Node, ob er NeuronEngine-fähig ist. Pflichttest: Gradienten und Ergebnis stimmen mit PyTorch
überein (Abweichung unter 1e-5), und ein Mini-Modell lernt messbar.

Nicht Ziel: ein großes Foundation Model. Dafür fehlt die Rechenleistung, nicht die Architektur.

### 3.8 Export

| Format | Inhalt |
|---|---|
| Modellbibliothek | wie heute, dann auch normal über die Trainingsseite trainierbar |
| `.ftmodel` | Graph, Gewichte, Vorverarbeitung, Metadaten |
| PyTorch-Projekt | `model.py`, `data.py`, `train.py`, `infer.py`, `requirements.txt`, Gewichte |
| NeuronEngine-Projekt | dasselbe Modell ganz ohne PyTorch, samt Framework-Quelltext |
| Weitere | safetensors, TorchScript, ONNX (fragt die Paketinstallation an) |

Pflichttest: Der erzeugte Code liefert dasselbe wie die Laufzeit, Abweichung unter 1e-5.

### 3.9 Ausliefern

Ausgangs-Nodes tragen eine Bedeutung: Klassen, Text, Bild, Audio, Boxen, Maske, Embedding,
Zahl. Labor und Hosting wählen damit vorhandene Ansichten aus `LabTaskViews.tsx`.
`canvas_inference_server.py` nimmt alle Eingabearten an.

## 4. Oberfläche

Der Builder bleibt im Navy-Stil (`.ft-synapse`) und gliedert sich in fünf Stationen:

1. **Planen:** Ziel wählen (Eingabe, Ausgabe, Größe), Vorlage oder leer, Budget für diese Hardware.
2. **Daten:** Daten-Graph mit echter Vorschau je Schritt (Bild, Spektrogramm, Tokens).
3. **Bauen:** Canvas mit Formen an den Kanten, Blöcke per Doppelklick, Modell-Check rechts.
4. **Trainieren:** Konsole mit Rezept, Losses, Metriken, Verlauf.
5. **Ausliefern:** Testen, Hosting, Export.

Editor-Grundlagen: Undo/Redo, Kopieren/Einfügen, Mehrfachauswahl, Suche (⌘K), Minimap,
Gruppen, Kommentare. KI-Agent und Coach lesen den generierten Katalog und die Live-Prüfung,
damit sie alle Nodes kennen und ihre Fehler selbst sehen.

`SynapseBuilder.tsx` (1728 Zeilen) wird aufgeteilt in `core/`, `canvas/`, `inspector/`,
`plan/`, `data/`, `learn/`, `deliver/`.

## 5. Bauphasen

Jede Phase ist für sich nutzbar. Gearbeitet wird im eigenen Worktree auf `feat/synapse-studio`
von `main`, die offenen Startseiten-Änderungen bleiben unberührt.

| Phase | Inhalt | Abnahme |
|---|---|---|
| 0 Fundament | Katalog, IR v2, Laufzeit mit Ports, Migration, gefundene Fehler | Alte Modelle laufen gleich, alle Befunde als Test rot, dann grün |
| 1 Live-Editor | Formen-Dienst, `auto`, Port-Typen, Undo, Modell-Check | Mini-ViT ohne eine Größe von Hand, 365 Nodes flüssig |
| 2 Alle Datenarten | Daten-Graph, Vorschau, `preprocess.json` | Je Modalität ein Mini-Modell von Daten bis Vorhersage |
| 3 Lernen | Mehrere Ein-/Ausgänge, Rezepte ohne GAN/Diffusion, Metriken | Segmentierung, Mini-GPT, Seq2Seq, VAE lernen messbar |
| 4 Volle Kontrolle | Code-Nodes, Blöcke, vortrainierte Bausteine | Eigener Node lässt sich prüfen, trainieren, exportieren |
| 5 Export und Ausliefern | PyTorch-Projekt, ONNX, TorchScript, Labor, Hosting | Export läuft ohne FrameTrain in leerer venv |
| 6 Planen und KI | Stationen, Vorlagen, Budget, Agent auf neuem Katalog | Agent baut aus einem Satz ein trainierbares Modell |
| 7 Generativ und Detektion | GAN, Diffusion, Detektionsköpfe | Mini-Modelle erzeugen erkennbare Ergebnisse |
| 8 NeuronEngine | Eigenes Framework ohne PyTorch, Einzelschritt-Ansicht | Gradienten gleich PyTorch, Mini-Modell lernt ohne torch |

## 6. Node-Katalog

- **Daten (ca. 25):** Quellen aus 3.5, Transformationen, Tokenizer, Split, Augmentation
- **Layer (ca. 35):** Linear, Conv1d/2d/3d, ConvTranspose, Depthwise, Pooling, Upsample,
  PixelShuffle, Patch Embed, Embedding, Positions-Encoding, RNN, GRU, LSTM, Attention,
  Cross-Attention, Transformer Encoder/Decoder, Batch/Layer/Group/Instance/RMS-Norm,
  Dropout, DropPath
- **Aktivierung (ca. 12):** ReLU, GELU, SiLU, Mish, PReLU, Softmax, Sigmoid u. a.
- **Tensor (ca. 25):** Add, Multiply, Concat, Stack, Split, Slice, Reshape, Flatten, Permute,
  Squeeze, MatMul, Einsum, Mean, Parameter, Konstante, CLS-Token, Maske
- **Struktur und Lernen (ca. 25):** Block, Repeat, Code, Vortrainiert, Reparameterize, FiLM,
  Losses, Metriken, Optimizer, Scheduler, EMA

## 7. Vorlagen

ResNet, U-Net, ViT, Mini-GPT, BERT-artig, Seq2Seq-Transformer, Audio-CNN, Whisper-artig,
3D-CNN, Video-Transformer, Bild+Text-Fusion, VAE, DCGAN, Mini-Diffusion, Tabellen-MLP.
Jede Vorlage bekommt einen Test, der sie baut und einige Schritte trainiert.

## 8. Tests

- **Python:** Matrix je Node (Bauen, Vorwärts, Rückwärts, Export gleich Laufzeit), je Modalität
  ein Lauf von Daten bis Vorhersage, Migration mit gespeicherten v1-Graphen
- **Rust:** Dienst-Protokoll, Absturz und Neustart des Dienstes
- **Frontend:** Port-Regeln, Katalog aktuell, Undo, Migration
- **Vor jedem Release:** UI im Browser-Pane (mockIPC), Layout per Bild gegen diese Vorschau,
  dann Vorschau-App mit eigener App-ID und echte Mini-Trainings
- **Live:** nur macOS. Windows, Linux und CUDA sind hier nicht testbar.

## 9. Risiken und Grenzen

- **Hardware:** Die Architektur eines großen Modells lässt sich bauen, trainieren lässt es sich
  auf einem Mac nicht. Das Budget warnt, bevor man Stunden verliert.
- **Detektion und Diffusion** sind die schwersten Teile, deshalb zuletzt.
- **Code-Nodes** führen eigenen Python-Code aus. Geteilte Blöcke mit Code brauchen eine Warnung.
- **ONNX** kann nicht jeden Code-Node exportieren. Die App nennt dann den betroffenen Node.
- **Fremde Pakete:** `av`, `onnx`, `onnxscript` fehlen auf diesem Rechner. Sie werden
  erst bei Bedarf über die vorhandene Paketinstallation geholt.
- **Scope:** Das sind 8 Phasen. Ich liefere sie einzeln aus, damit nach jeder Phase etwas läuft.
