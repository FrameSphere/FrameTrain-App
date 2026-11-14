# Text-to-Speech-Plugin (Session engine-tts) — ZWISCHENSTAND, nicht fertig

Die Session wurde am Nutzungslimit unterbrochen. Dieser Bericht nennt, was gebaut
und durch echte Laeufe belegt ist, und was noch fehlt.

## Gebaut
- `src-tauri/python/ft_data/tts.py` — gemeinsam fuer Train, Test, Modell-Server: Modellart,
  Textaufbereitung (ae/oe/ue/ss, unbekannte Zeichen melden), x-vector (speechbrain, optional),
  Standard-Sprechervektor (`tts_default_speaker.py`), Mel-Ziel, Collator, Vocoder-Suche,
  Griffin-Lim-Notweg, Erzeugung, WAV, Ruecktranskription (WER/CER).
- Train-Plugin `train_engine/plugins/text_to_speech/` (SpeechT5; VITS wird mit Begruendung abgelehnt).
- Test-Plugin `test_engine/plugins/text_to_speech/` (SpeechT5 und VITS / MMS-TTS).
- Frontend `src/plugins/text-to-speech/` + Registry, Dataset-Kompatibilitaet, Audio-Player in
  `GenericTestPanel` (`resultKind="audio"`), Texte DE/EN.
- Labor/Hosting: Modalitaet `tts` in `model_server.py`, Player in `LabTaskViews`/`HostResultView`,
  Labor-Samples (Transkript als Eingabe, Aufnahme als Referenz). CSP unveraendert:
  `media-src` und `connect-src` erlauben `asset:` bereits.
- Rust: `packages_for_task`, `infer_task_type`, `guess_task`, `infer_timeout_secs` (je eine Zeile + Test).

## Abhaengigkeiten
- x-vector: Paket `speechbrain` (1.1.2 laeuft mit torch 2.12; laedt `speechbrain/spkrec-xvect-voxceleb`).
  Optional. Ohne das Paket gilt fuer alle Aufnahmen ein Standard-Sprechervektor (nur fuer Daten
  mit einer Stimme sinnvoll). Ein Null-Vektor taugt nicht: gemessen bricht das Basismodell damit
  jeden Satz nach 1,12 s ab.
- Vocoder `microsoft/speecht5_hifigan` (49 MB) wird beim Export neben das Modell gelegt.
- VITS / MMS-TTS: nicht trainierbar — `VitsModel.forward` wirft
  `NotImplementedError("Training of VITS is not supported yet.")`. Nur Test/Labor/Hosting.

## Belegt durch echte Laeufe (Mac, MPS, 16 Saetze der macOS-Stimme Samantha, 6 Schritte)
- Train-Engine, Standardvektor: Loss 1,07 -> Val 0,68, 26 s, Export mit Vocoder und Hoerprobe (WAV).
- Train-Engine, x-vector (speechbrain in separater venv): Train 0,65-0,75, Val 0,49-0,52.
- Test-Engine Einzeltest: WAV 16 kHz PCM16, 3,68 s.
- Test-Engine Datensatz (4 Testsaetze, whisper-tiny): WER 0,037, CER 0,031, 3 von 4 Saetzen wortgleich.
- Ohne ASR-Modell (`asr_model: none`): Audios entstehen, keine Kennzahl, keine Trefferquote.
- `facebook/mms-tts-deu` im Einzeltest: 5,2 s Audio; Training wird mit Begruendung abgelehnt.

## NICHT belegt / offen
- Python-Unittests neben dem Plugin (Datenaufbereitung, Kennzahl, Mini-Lauf) sind noch nicht geschrieben.
- vitest: neue Tests fuer `text-to-speech` fehlen; `npm test` wurde nach den Aenderungen NICHT
  ausgefuehrt (drei bestehende Erwartungen zu speecht5 wurden angepasst, ungeprueft).
- `test_plugin_consistency.py` nicht ausgefuehrt.
- Rust: `cargo test` lief beim Abbruch noch im Hintergrund, Ergebnis unbekannt.
- Labor und Hosting nie in der App geprueft (Player, Samples).
- Lizenzhinweis fuer den Standard-Sprechervektor (CMU ARCTIC "slt", Datensatz
  Matthijs/cmu-arctic-xvectors, MIT) fehlt in den Drittanbieter-Hinweisen.

## Klang und Datenbedarf (ehrlich)
Nach wenigen Schritten klingt das Modell wie das Basismodell: verstaendliches Englisch, Stimme
kaum veraendert. Fuer eine erkennbar eigene Stimme sind etwa 30 Minuten bis einige Stunden
saubere Aufnahmen EINER Person noetig (einzelne Saetze, 1-15 s, 16 kHz, ohne Hall/Musik, Text
exakt wie gesprochen, Zahlen ausgeschrieben) und einige tausend Schritte. SpeechT5 ist auf
Englisch vortrainiert und kennt nur lateinische Buchstaben ohne Umlaute; Deutsch braucht
deutlich mehr Daten (mehrere Stunden) und klingt auch dann nach Akzent. Die WER misst nur
Verstaendlichkeit, nicht Natuerlichkeit oder Aehnlichkeit zur Zielstimme.
