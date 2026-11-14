# Engine-Ausbau: Frage-Antwort, Reranker, causal_lm (Session text-qa-reranker)

Branch: `claude/engine-text-qa-reranker-310ccb`. Nicht gepusht, nicht nach main gemergt, keine Versionsnummer geändert.

## Was gebaut wurde

### 1. Extraktives Frage-Antwort (`question_answering`)

| Teil | Datei |
|---|---|
| Daten, Fenster, Nachbearbeitung, SQuAD-Kennzahlen (gemeinsam) | `src-tauri/python/ft_data/qa.py` |
| Train-Plugin (`AutoModelForQuestionAnswering`, HF-Trainer) | `src-tauri/python/train_engine/plugins/question_answering/` |
| Test-Plugin | `src-tauri/python/test_engine/plugins/question_answering/` |
| Frontend | `src/plugins/question-answering/` (id `question-answering`) |

- Daten: SQuAD-JSON v1.1/v2.0 (`data → paragraphs → qas`), HF-SQuAD-Zeilen (`answers = {text, answer_start}`, so liegen squad/squad_v2 als Parquet), flache Tabellen `context/question/answer [answer_start]` (CSV/TSV/JSON/JSONL/Parquet), `answers` auch als Liste oder JSON-Text. Fehlt `answer_start` oder stimmt er nicht, wird die Antwort im Kontext gesucht (erst exakt, dann ohne Groß-/Kleinschreibung, das Vorkommen am nächsten zum Hinweis). Antworten, die nicht im Kontext stehen, werden verworfen und gezählt. Leere Antwort / `is_impossible` = „keine Antwort“ (SQuAD v2). Eigene Spaltennamen über `plugin_config` (`context_column`, `question_column`, `answer_column`, `answer_start_column`).
- Lange Kontexte: Fenster mit `max_seq_length` (Standard 384) und `doc_stride` (Standard 128). Fenster ohne vollständige Antwort lernen „keine Antwort“ (CLS).
- Vorhersage: beste Spanne über alle Fenster (n-best 20, `max_answer_length` 30); mit `allow_no_answer` (auto = an, sobald das Dataset Fragen ohne Antwort hat) gewinnt „keine Antwort“, wenn der CLS-Score die beste Spanne um mehr als `null_threshold` übertrifft.
- Kennzahlen wie im offiziellen SQuAD-Skript: Exact Match und Token-F1 nach Normalisierung, Maximum über alle Referenzen; bei gemischten Daten zusätzlich `has_answer_exact_match`, `has_answer_f1`, `no_answer_accuracy`. Training meldet die Werte vor und nach dem Training (`exact_match_before` …), `accuracy` = Exact Match.
- Einzeltest: Kontext im Textfeld, Frage im zweiten Feld (oder `Frage ||| Kontext`, oder JSON). Ergebnis: Antwort plus Kontext mit markierter Stelle `»…«`, Kandidatenliste; zusätzlich `answer_start`/`answer_end` im Ergebnis (`output_kind: "qa_span"`).

### 2. Reranker / Cross-Encoder (`reranker`)

| Teil | Datei |
|---|---|
| Formate, Gruppen je Anfrage, MRR/nDCG/Accuracy/Spearman | `src-tauri/python/ft_data/rerank.py` |
| Train-Plugin (sentence-transformers `CrossEncoder`, `BinaryCrossEntropyLoss`) | `src-tauri/python/train_engine/plugins/reranker/` |
| Test-Plugin | `src-tauri/python/test_engine/plugins/reranker/` |
| Frontend | `src/plugins/reranker/` (id `reranker`) |

- Daten: Paare mit Score/Label (`query/passage/label`, `sentence1/sentence2/score`, `question/document/relevance` …; Skala 0/1, 0..1, 0..5 usw. wird auf 0..1 gebracht) oder `query + positive [+ negative | negatives (Liste)]`. Fehlen Negative, kommen Passagen anderer Anfragen dazu (`negatives_per_query`, Standard 3; Bewertung bis 9). Validierung wird nach Anfragen getrennt, nicht nach Zeilen.
- Basis: fertiger Reranker (Kopf mit einer Ausgabe wird weitertrainiert) oder beliebiger Encoder (Kopf mit einer Ausgabe neu).
- Kennzahlen je Anfrage über ihre Kandidaten: MRR@10, nDCG@10 (abgestufte Relevanz), Hit@1; bei Score-Paaren zusätzlich Accuracy (Schwelle 0,5) und Spearman. Vorher/nachher im Training.
- Einzeltest: `Anfrage ||| Passage A ||| Passage B` (Rangfolge), `Anfrage ||| Passage` (Relevanz) oder nur die Anfrage → ordnet bis zu 300 Passagen aus dem gewählten Dataset.
- Rust (`model_manager.rs`): ein trainierter Cross-Encoder trägt ebenfalls `modules.json` und wäre beim Import als Sentence-Embedding eingeordnet worden. Neu: `config_sentence_transformers.json` mit `"model_type": "CrossEncoder"` oder `frametrain_rerank.json` → `reranker`, `frametrain_qa.json` → `question-answering`. Mit Rust-Test.

### 3. `causal_lm`: volles Fine-Tuning und Rohtext

- **Speichersperre** (`ft_data/llm.py: full_finetune_block`, Aufruf in `setup()` nur ohne LoRA und ohne Quantisierung, also auch nicht bei DPO mit LoRA): Bedarf = Parameter × 16 Byte (fp32-Gewichte, Gradienten, zwei AdamW-Momente) × 1,25; nutzbar = 70 % des Speichers (CUDA: GPU-Speicher, sonst Arbeitsspeicher, auf Apple Silicon gemeinsam). Die Parameterzahl kommt vor dem Laden aus den safetensors-Köpfen, sonst aus config.json (auch GPT-2-Namen `n_embd`/`n_layer`), sonst aus der Dateigröße. Bei Überschreitung bricht das Training mit Erklärung und dem Hinweis auf LoRA ab. GPT-2 (124M) und SmolLM-135M bleiben frei.
- **Rohtext**: `.txt/.md` und die Spalten `text/content/document/body` gab es schon. Neu: `plugin_config.text_column` für eine frei benannte Spalte und Tabellen mit genau einer Spalte gelten als Rohtext.
- **Perplexität vorher/nachher** im PyTorch-Weg bei Rohtext: Evaluierung vor `train()`, `baseline_perplexity` und `perplexity` in den Abschluss-Metriken, Statuszeile „Perplexitaet: vorher X -> nachher Y“. Der Vorher-Wert wird nicht als Val-Loss der ersten Epoche gemeldet. Der MLX-Weg hatte den Vorher-Wert schon (Iteration 0).
- Frontend: Beschreibung (DE im Plugin, EN/DE in den Locales) und `text_column` in den Standard-Parametern. Name und Standardwerte (LoRA an) unverändert.

### Gemeinsam genutzte Dateien (nur additiv)

`src/plugins/registry.ts` (zwei Imports, zwei Einträge vor `tokenClassificationPlugin`, ein Halbsatz im Hinweistext), `src/plugins/genericDatasetCompat.ts` (zwei `case`), `src/components/TrainingPanel.tsx` (`TEXT_RAM_TASKS` um zwei Einträge), `src/locales/en.json` und `de.json` (eingefügte Zeilen: Aufgabennamen, `datasetCompat.msg.missing.qa`, Testtexte `qa*`/`rerank*`, Kennzahl-Beschriftungen, `plugins.question-answering`, `plugins.reranker`, causal-lm-Beschreibung), `src-tauri/src/plugin_commands.rs` (`packages_for_task` zwei Zeilen), `src-tauri/src/model_manager.rs` (Dateierkennung + Test). Kein neues `Command::new`.

Eine bestehende Erwartung wurde angepasst: `src/plugins/__tests__/tokenEmbedding.test.ts` erwartete für `cross-encoder/ms-marco-MiniLM-L-6-v2` das Plugin `hf-encoder` (es gab keinen Reranker); jetzt `reranker`.

## Tests (mit Ausgabe)

### Python

Neue Testdateien (offline, zufällig initialisierte Mini-Modelle aus `train_engine/plugins/_tiny_models.py`, je Mini-Training wenige Sekunden):
`train_engine/plugins/question_answering/test_question_answering.py` (17), `train_engine/plugins/reranker/test_reranker.py` (13), `train_engine/plugins/causal_lm/test_full_finetune_rawtext.py` (10).

```
$ cd src-tauri/python
$ python3 -m unittest train_engine/plugins/question_answering/test_question_answering.py \
    train_engine/plugins/reranker/test_reranker.py train_engine/plugins/causal_lm/test_full_finetune_rawtext.py \
    ft_data/test_llm.py train_engine/plugins/sentence_embedding/test_sentence_embedding.py
Ran 75 tests in 10.351s
OK
$ cd train_engine && python3 -m unittest test_orchestrator.py test_plugin_consistency.py
Ran 11 tests in 0.845s
OK
```

Abgedeckt: alle Datenformate oben, Startsuche/-korrektur, Verwerfen nicht gefundener Antworten, Fenster mit Stride (Antwort-Token dekodieren zur Antwort), Nachbearbeitung inkl. „keine Antwort“, EM/F1 inkl. v2-Aufteilung, Reranker-Formate/Skalen/Listen-Negative/gezogene Negative/Split nach Anfragen, MRR/nDCG/Hit@1/Accuracy/Spearman mit Handrechnung, Sperre (klein frei, 7B/70B gesperrt, Grenzfall, unbekannt), Sperre in `setup()` vor jedem Laden, GPT-2-Schätzung, je ein Mini-Training durch Train- und Test-Plugin (QA, Reranker) bzw. volles Fine-Tuning auf Rohtext mit sinkender Perplexität. `ft_data/test_llm.py` (bestehende LoRA/DPO/Chat-Datenlogik) bleibt grün.

### Echte Mini-Läufe durch die Engines (`train_engine.py` / `test_engine.py --config`)

Modelle: `google/bert_uncased_L-2_H-128_A-2` (4,4M Parameter) für QA und Reranker, `sshleifer/tiny-gpt2` für causal_lm; Datensätze mit 12–60 Zeilen; Apple MPS.

| Lauf | Ergebnis |
|---|---|
| QA Train (SQuAD-v2-JSON, 50 Fragen, 15 Epochen) | `Vorher -> nachher: EM 0.0% -> 100.0%, F1 22.9% -> 100.0%` |
| QA Test Dataset (flache JSONL ohne `answer_start`) | `12 Fragen | bei 12 Antworten Position im Kontext gesucht`, `exact_match 1.0, f1 1.0` |
| QA Einzeltest | `Greta wohnt seit vielen Jahren in »Oslo«. …`, Konfidenz 0.999; Frage ohne Antwort im Text → `Keine Antwort im Text / No answer in the text` |
| Reranker Train, Tripel | `mrr_at_10 0.786 -> 1.000, ndcg_at_10 0.833 -> 1.000, hit_at_1 0.750 -> 1.000` |
| Reranker Train, Paare mit Label (CSV) | `mrr_at_10 0.500 -> 1.000, …, accuracy 0.500 -> 1.000, spearman -0.447 -> 0.894` |
| Reranker Test Dataset / Einzel / Einzel mit Dataset-Korpus | `mrr_at_10 1.0, ndcg_at_10 1.0`; `Platz 1 (0.986): … owls …`; aus dem Dataset: `Platz 1 (0.986): … sharks …` |
| causal_lm volles FT, `.txt` | `Architektur: gpt2 | Backend: TORCH | volles Fine-Tuning`, `Perplexitaet: vorher 42163.30 -> nachher 33987.73` |
| causal_lm volles FT, CSV mit einer Spalte `zeile` | `Format: text → reiner Text`, `Perplexitaet: vorher 33596.15 -> nachher 26109.52` |
| causal_lm volles FT, 70B-Llama (nur config.json) | `Volles Fine-Tuning ist fuer dieses Modell gesperrt: 78.90 Mrd. Parameter brauchen mit AdamW rund 1470 GB … Nutzbar sind hier etwa 22 GB (70% von 32 GB). LoRA einschalten …` |
| causal_lm LoRA (Gegenprobe) | `Backend: TORCH | LoRA`, trainierbar 0,50 %, Adapter wird zusammengeführt wie bisher |
| causal_lm Test Dataset (Rohtext) | `metrics: {"perplexity": 38280.297}` |

(tiny-gpt2 hat zwei Dimensionen; belegt wird der Abstieg, nicht die Höhe der Perplexität.)

### Frontend (vitest)

Neu: `src/plugins/__tests__/qaReranker.test.ts` (22 Tests): Reranker-Namen → `reranker` und nie `sentence-embedding`; Embedding-Modelle inkl. `msmarco-…` und `multi-qa-MiniLM` bleiben `sentence-embedding`; Basis-Encoder bleiben `hf-encoder`/`xlm-roberta`; Qwen3-Reranker bleibt Decoder; SQuAD-Modelle → `question-answering`, auch über `…ForQuestionAnswering`; Hand-Zuordnung gewinnt; Registrierung vor den generischen Encodern; Texte EN/DE vorhanden und ohne Emojis; Dataset-Prüfung.

```
$ npx vitest run
 Test Files  112 passed (112)
      Tests  1219 passed (1219)
$ npx tsc --noEmit -p tsconfig.json     # ohne Ausgabe, Exit 0
```

### Rust

```
$ cd src-tauri && cargo test
test result: ok. 289 passed; 0 failed; 1 ignored
```
(darin `model_manager::model_type_tests::reranker_und_frage_antwort_werden_an_dateien_erkannt`)

## Was nicht belegt ist

- **Keine Sichtprüfung in der App.** Test-Oberflächen nutzen `GenericTestPanel` (Kontext + Frage-Feld bzw. `|||`-Eingabe); die markierte Antwortstelle erscheint als Text mit `»…«`, nicht als farbige Hervorhebung. Training/Test über die App-Oberfläche, MCP-Werkzeuge und Analyse-Seite wurden nicht live durchgeklickt.
- **Labor / Modell-Server / Hosting** kennen `question_answering` und `reranker` nicht (`src/components/labTaskSamples.ts`, `test_engine/model_server.py` unverändert).
- **Echte Basismodelle** (deepset/roberta-base-squad2, cross-encoder/ms-marco-MiniLM-L-6-v2, BGE-Reranker, SmolLM-135M) wurden nicht heruntergeladen und trainiert; die Läufe nutzen winzige Modelle. Insbesondere ist volles Fine-Tuning von SmolLM-135M nur über die Speicherrechnung (frei) belegt, nicht durch einen Lauf.
- **MLX**: volles Fine-Tuning im MLX-Weg (Entsperren der Blöcke) unterliegt jetzt derselben Sperre, wurde aber nicht laufen gelassen; LoRA/DPO/MLX-Code ist sonst unverändert. Neu ist bei LoRA auf Rohtext (PyTorch) nur die zusätzliche Evaluierung vor dem Training für die Vorher-Perplexität.
- Die Speicherschätzung ist bewusst grob (Aktivierungen pauschal 25 %), Grenzfälle können knapp falsch liegen; es gibt keinen Schalter zum Übergehen der Sperre.
- Rust-Paketprüfung (`packages_for_task`) für die neuen Aufgaben ist nicht eigens getestet.
- Die Test-Engine-Perplexität bei Rohtext rechnet je Zeile (bis 1024 Tokens, ohne EOS) und weicht daher vom Val-Wert des Trainings ab (bestehendes Verhalten).
