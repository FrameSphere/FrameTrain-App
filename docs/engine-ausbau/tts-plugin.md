# Text-to-Speech-Plugin (Session engine-tts)

Branch `claude/engine-tts-c430f6`. Stand: 15.11.2026.

## Kurzfassung

- SpeechT5 (`microsoft/speecht5_tts`) lässt sich auf Audio + Transkript trainieren, testen,
  im Labor ausprobieren und hosten. Ergebnis ist jeweils eine abspielbare WAV-Datei.
- VITS / MMS-TTS (`facebook/mms-tts-*`) ist **nicht trainierbar**, nur Test, Labor und Hosting.
- Kennzahl im Test: Verständlichkeit per Rücktranskription (WER/CER), nur wenn ein
  ASR-Modell lokal liegt. Ohne ASR-Modell entstehen die Audios ohne Kennzahl.
- Alle Tests grün: 42 Python-Tests, 1251 vitest-Tests, 7 betroffene Rust-Tests.
- In der laufenden App wurde nichts geprüft (siehe „Nicht belegt").

## Was gebaut wurde

| Teil | Dateien |
|---|---|
| Gemeinsame Logik | `src-tauri/python/ft_data/tts.py`, `ft_data/tts_default_speaker.py` |
| Train-Plugin | `train_engine/plugins/text_to_speech/` (`plugin.py`, `manifest.json`, Tests) |
| Test-Plugin | `test_engine/plugins/text_to_speech/` (`plugin.py`, `manifest.json`, Tests) |
| Frontend-Plugin | `src/plugins/text-to-speech/` (`detect.ts`, `index.ts`, `TestPlugin.tsx`) |
| Labor / Hosting | Modalität `tts` in `test_engine/model_server.py` |

`ft_data/tts.py` wird von Training, Test und Modell-Server gemeinsam genutzt, damit alle drei
aus demselben Text dasselbe Audio erzeugen: Modellart, Textaufbereitung, Sprecher-Embedding,
Mel-Ziel, Batch-Bau, Vocoder-Suche, Erzeugung, WAV-Ausgabe, Rücktranskription.

Das `speech_recognition`-Plugin wurde nicht angefasst; `ft_data/asr.py` wird nur importiert
(Dataset-Layouts, WER/CER).

### Minimal angefasste gemeinsame Dateien (für das Zusammenführen)

| Datei | Änderung |
|---|---|
| `src/plugins/registry.ts` | Import + Eintrag in `PLUGINS` (vor `speechRecognitionPlugin`); Text des `speecht5`-Eintrags in `KNOWN_UNSUPPORTED` |
| `src/plugins/genericDatasetCompat.ts` | ein `case 'text_to_speech'` |
| `src/plugins/GenericTestPanel.tsx` | `resultKind: 'audio'` (Player im Einzeltest), `audios` im Datensatz-Ergebnis |
| `src/components/TrainingPanel.tsx` | `text_to_speech` in `NON_SEQUENCE_TASKS` |
| `src/components/ModelManager.tsx` | Text zum `pipeline_tag` `text-to-speech` |
| `src/components/studio/studioModels.tsx` | Transkript-Projekte passen auch zu `text_to_speech` |
| `src/components/labTaskSamples.ts` | Aufgabe `tts`, `ttsSamples`, Feld `refAudio` |
| `src/components/LabTaskViews.tsx` | Komponente `GeneratedAudio` |
| `src/components/LaboratoryPanel.tsx` | Feld `refAudio`, Ergebniszweig für `tts` |
| `src/components/labCorrection.ts`, `hosting/hostingModel.ts`, `hosting/HostResultView.tsx` | je ein Zusatz für `tts` |
| `src/locales/de.json`, `en.json` | je 12 neue Schlüssel, nichts geändert |
| `src-tauri/src/plugin_commands.rs` | `packages_for_task("text_to_speech")` |
| `src-tauri/src/test_manager.rs` | `infer_task_type`: `…ForTextToSpeech` / `VitsModel` |
| `src-tauri/src/hosting_manager.rs` | `guess_task`: `tts` |
| `src-tauri/src/model_host.rs` | `infer_timeout_secs("tts") = 180` |
| Bestehende Tests | `modalities.test.ts` (2 Erwartungen), `modelDetection.test.ts` (Beispiel `speecht5` -> `bark`) |

Kein neuer `Command::new` in Rust.

## Abhängigkeiten

**x-vector:** Paket `speechbrain` (geprüft: 1.1.2 mit torch 2.12.1). Es lädt beim ersten Lauf
`speechbrain/spkrec-xvect-voxceleb` (ca. 20 MB) nach `~/.cache/frametrain/`. Das Paket ist
optional und steht deshalb nicht in `packages_for_task`.

**Ausweichweg ohne speechbrain:** Alle Aufnahmen bekommen denselben Standard-Sprechervektor
(`tts_default_speaker.py`), das Modell lernt die Stimme dann nur über seine Gewichte. Das ist
nur bei Daten mit einer einzigen Stimme sinnvoll und wird beim Training so gemeldet.
`speaker_embedding: "xvector"` erzwingt x-vectors und bricht ohne das Paket mit Hinweis ab.

Ein Null-Vektor als Ausweichweg scheidet aus. Gemessen am Basismodell:

| Sprechervektor | Audiolänge für drei verschiedene Sätze | Rücktranskription |
|---|---|---|
| Null-Vektor | 1,12 s / 1,12 s / 1,12 s | „…opens at **night**" (Satzende fehlt) |
| Standardvektor | 1,76 s / 1,66 s / 1,76 s | alle drei Sätze korrekt |

**Vocoder:** `microsoft/speecht5_hifigan` (49 MB) wird nicht mittrainiert. Der Export legt ihn
neben das Modell (`vocoder/`), damit Test, Labor und Hosting ohne Netz laufen. Fehlt er und
ist kein Netz da, wird das Audio mit Griffin-Lim geschätzt (verständlich, aber blechern) und
das wird gemeldet.

**Weitere Pakete:** `librosa`, `soundfile`, `sentencepiece`, `jiwer` (in `packages_for_task`).

## VITS / MMS-TTS: ehrlich geprüft

Nicht mit vertretbarem Aufwand trainierbar. `VitsModel.forward` in transformers 5.12.1 bricht
ab, sobald Labels übergeben werden:

```
raise NotImplementedError("Training of VITS is not supported yet.")
```

Es fehlen Diskriminator, Posterior-Encoder-Loss und die Ausrichtungssuche; das gibt es nur in
externen Repositories mit eigenem Trainingsskript und kompiliertem Zusatzmodul. Deshalb:

- Das Train-Plugin lehnt VITS mit dieser Begründung ab (als „Konfigurationsfehler").
- Test, Labor und Hosting spielen VITS-Modelle ab. Im Manifest steht es getrennt unter
  `inference_only_architectures`.
- Offen: Die Trainingsseite bietet ein VITS-Modell trotzdem an und scheitert erst beim Start.
  `isInferenceOnlyTts()` in `detect.ts` liefert das Signal, um es vorher zu sperren.

## Tests und Ausgaben

### Python: Train-Plugin und `ft_data.tts`

`python3.11 plugins/text_to_speech/test_text_to_speech.py` (aus `train_engine/`):

```
Ran 34 tests in 5.418s
OK
```

Enthalten: Modellart (4 Fälle SpeechT5-Varianten/VITS), Textaufbereitung (Umlaute, Akzente,
unbekannte Zeichen, Textteilung), Sprecher (Standardvektor, Mittelwert, Datei, Ausweichen ohne
speechbrain), Batch-Bau, Filter (zu lange Aufnahmen auslassen statt kappen), WAV, Griffin-Lim,
Kennzahl (7 Fälle), und der **Mini-Trainingslauf**: ein vom Test angelegtes Winz-SpeechT5
läuft auf der CPU durch `setup → load_data → build_model → train → validate → export`, wird
wieder geladen und erzeugt Audio. Geprüft wird auch, dass das Training die Gewichte verändert
und dass der Test-Split nicht ins Training gerät.

### Python: Test-Plugin

`python3.11 plugins/text_to_speech/test_tts_test_plugin.py` (aus `test_engine/`):

```
Ran 8 tests in 0.649s
OK
```

Erzeugung und ASR sind hier Attrappen; geprüft wird, was das Plugin daraus macht (WAV-Datei,
`results.json`, WER über den Korpus, Lauf ohne ASR ohne Kennzahl und ohne Trefferquote,
nicht sprechbarer Satz beendet den Lauf nicht, Obergrenze `max_audios`).

### Python: Konsistenz und Nachbar-Plugin

```
test_plugin_consistency.py                          Ran 4 tests   OK
plugins/speech_recognition/test_speech_recognition.py   Ran 15 tests  OK
```

### vitest

`npm test`:

```
Test Files  112 passed (112)
     Tests  1251 passed (1251)
```

Davon 54 neu in `src/plugins/__tests__/textToSpeech.test.tsx`: Erkennung über `config.json`
und Namen, Reihenfolge in der Registry, Abgleich mit dem Manifest, Dataset-Kompatibilität,
Audio-Player im Einzeltest und im Datensatz-Lauf (mit und ohne ASR), Labor-Samples,
Hosting-Anzeige, Texte in Deutsch und Englisch. `npx tsc --noEmit` ohne Meldung.

### Rust

`cargo test -- aufgabe_wird_ohne_python packages task_type timeout` und
`cargo test -- aufgaben_haben_ihre_gruppe`:

```
test hosting_manager::tests::aufgabe_wird_ohne_python_geschaetzt ... ok
test test_manager::task_type_tests::leitet_aufgabe_aus_dem_versionsordner_ab ... ok
test plugin_commands::dependency_version_tests::aufgaben_haben_ihre_gruppe ... ok
test result: ok. 6 passed; 0 failed   /   test result: ok. 1 passed; 0 failed
```

Die drei genannten Tests wurden um TTS-Fälle erweitert. Die übrigen 282 Rust-Tests wurden
nicht ausgeführt.

## Echter Mini-Lauf durch beide Engines

Mac (Apple Silicon, MPS). Daten: 23 kurze englische Sätze, gesprochen von der macOS-Stimme
„Samantha" (`say`), als Audio + gleichnamige `.txt` in `train/` (16), `val/` (3), `test/` (4).
Basismodell `microsoft/speecht5_tts` (144,4 Mio. Parameter), 6 Schritte, Batch 2.

**Train-Engine** (`train_engine.py --config …`):

| Variante | Train-Loss | Val-Loss | Dauer |
|---|---|---|---|
| ohne speechbrain (Standardvektor) | 0,79 | 0,68 | 26 s |
| mit speechbrain (x-vector je Aufnahme) | 0,65–0,75 | 0,49–0,52 | ähnlich |

Export jeweils: `model.safetensors`, Tokenizer, `speaker_embedding.npy`, `vocoder/`,
`tts_config.json`, Hörprobe `samples/sample.wav` (1,9 s).

**Test-Engine**, Einzeltest: `"Hello, this is my new voice. It was trained in FrameTrain."`
ergibt eine WAV-Datei mit 16 kHz, 16 Bit PCM, 3,68 s.

**Test-Engine**, Datensatz (4 Testsätze, Rücktranskription mit `openai/whisper-tiny` aus dem
lokalen HF-Cache):

```
'The quick brown fox jumps over the lazy dog.' -> 'Quick brown fox jumps over the lazy dog.'  WER 0.11
'Good morning, how are you today?'             -> 'Good morning, how are you today?'          WER 0.0
'Thank you very much for your help.'           -> 'Thank you very much for your help.'        WER 0.0
'The library opens at nine.'                   -> 'The library opens at nine.'                WER 0.0
WER 0.037 | CER 0.031 | 3 von 4 Sätzen wortgleich
```

Beide Modelle (Standardvektor und x-vector) kommen auf dieselben Werte. Mit
`asr_model: "none"`: 2 Audios erzeugt, `asr_model: null`, keine WER/CER, `accuracy` fehlt.

**MMS-TTS** (`facebook/mms-tts-deu`), Einzeltest mit deutschem Satz: 5,18 s Audio.
Whisper-tiny hört „…ein Tiss, der Sprachhausgarde mit Grüßen aus Zahnheim" statt „…ein Test
der Sprachausgabe mit Grüßen aus Zornheim". Das zeigt auch die Grenze der Kennzahl: ein Teil
der Fehler geht aufs Konto des kleinen ASR-Modells.

**Modell-Server** (`model_server.py --model-path …`) mit dem trainierten Modell:
`{"type": "ready", "modality": "tts", "input_kind": "text"}`, eine Textanfrage liefert
`audio_path` und `output_kind: "audio"` (WAV, 2,4 s); leere Anfrage und Dateianfrage werden
mit Meldung abgelehnt.

## Labor, Hosting, CSP

**CSP: keine Änderung nötig.** `tauri.conf.json` erlaubt `asset:` bereits in `media-src`
(für `<audio src>`) und in `connect-src` (für `fetch`). Der Player lädt die Datei per `src`.

Was Labor und Hosting brauchten und bekommen haben:

- Modell-Server: Modalität `tts` (Text rein, WAV-Pfad raus). Ohne diesen Zweig wäre SpeechT5
  als Textklassifikator geladen worden.
- Labor: Player für das Ergebnis, daneben die Aufnahme aus dem Dataset; Samples aus einem
  Audio-Transkript-Dataset sind die Transkripte (nicht die Audiodateien).
- Hosting: Player in der Ergebnisanzeige, eigener Platzhalter und Aufgabenname.
- Erzeugte Dateien liegen unter `<modell>/generated/`, wie bei Text-zu-Bild.

## Nicht belegt

- **Nichts davon wurde in der laufenden App geprüft.** Player im Test-Tab, im Labor und im
  Hosting sind nur durch Komponententests (jsdom) belegt. jsdom spielt kein Audio und kennt
  keine CSP; ob die WAV im Webview wirklich abspielt, ist offen. Das braucht einen Live-Test.
- Die Hosting-API für externe Programme (Rückgabe eines Dateipfads statt Audiodaten) wurde
  nicht betrachtet.
- Kein Training über mehr als 6 Schritte: dass eine Stimme tatsächlich gelernt wird, ist nicht
  gezeigt, nur dass Loss, Export und Wiedergabe funktionieren.
- x-vector-Weg nur in einer separaten Umgebung geprüft; `speechbrain` ist im Python der App
  nicht installiert, und es gibt in der App keinen Knopf, es zu installieren.
- Parquet- und Common-Voice-Datasets laufen über denselben Lader wie die Spracherkennung, sind
  für TTS aber nicht eigens gelaufen.
- Deutsch mit SpeechT5 wurde nicht trainiert, nur die Umschrift (ä -> ae) getestet.
- Windows und CUDA nicht geprüft.
- Griffin-Lim nur im Unittest mit Zufallsdaten und im Mini-Lauf des Winzmodells, nicht mit dem
  echten Modell angehört.

## Offene Punkte für den Orchestrator

1. **Lizenzhinweis:** `tts_default_speaker.py` enthält einen x-vector der Sprecherin „slt" aus
   CMU ARCTIC, entnommen dem Datensatz `Matthijs/cmu-arctic-xvectors` (MIT). Das gehört in
   die Drittanbieter-Hinweise.
2. VITS auf der Trainingsseite sperren (siehe oben).
3. `microsoft/speecht5_asr` landet am Namensteil „asr" bei der Spracherkennung, die SpeechT5
   nicht trainieren kann. Das war schon vorher so und ist hier nicht geändert.
4. `graphify update .` wurde im Worktree nicht ausgeführt (`graphify-out/` ist dort nicht
   ignoriert und soll nicht committet werden); nach dem Zusammenführen im Haupt-Checkout nachholen.

## Wie gut klingt es nach kurzem Training, und was brauchen Nutzer?

**Nach wenigen Schritten (wie im Mini-Lauf):** wie das Basismodell. Verständliches Englisch,
die Stimme ist praktisch unverändert. Das belegt die WER von 0,037, sagt aber nichts über eine
neue Stimme.

**Realistisch für eine eigene Stimme:**

- etwa 30 Minuten bis einige Stunden saubere Aufnahmen **einer** Person. Mit 10 bis 15 Minuten
  verschiebt sich vor allem die Klangfarbe; Betonung und Aussprache bleiben die des Basismodells.
- einzelne Sätze von 1 bis 15 Sekunden, 16 kHz oder mehr, ohne Hall, Musik und Nebengeräusche.
  Längere Aufnahmen lässt das Plugin aus, statt sie zu kappen.
- Text exakt so, wie er gesprochen wird. Zahlen ausschreiben: SpeechT5 kennt keine Ziffern,
  das Plugin lässt sie weg und warnt.
- einige tausend Schritte (Richtwert aus der Praxis: ab etwa 4000 bei Lernrate 1e-5). Auf dem
  Mac sind das Stunden, nicht Minuten.

**Grenzen, die Daten nicht beheben:**

- SpeechT5 ist auf Englisch vortrainiert und kennt nur lateinische Buchstaben ohne Umlaute
  (ä wird zu ae). Deutsch geht, braucht aber deutlich mehr Daten (mehrere Stunden) und behält
  einen hörbaren Akzent. Für fertige deutsche Sprachausgabe ohne Training ist
  `facebook/mms-tts-deu` der einfachere Weg, dort aber ohne eigene Stimme.
- 16 kHz Abtastrate: klingt nach Telefon- bis Radioqualität, nicht nach Studio.
- Die Kennzahl misst nur, ob die Wörter zu verstehen sind. Ob die Stimme natürlich klingt oder
  der Zielperson ähnelt, lässt sich nur durch Anhören beurteilen.
