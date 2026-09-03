// ============================================================================
// FrameTrain AI Coach — Einheitliches Kontext-System (Single Source of Truth)
// ----------------------------------------------------------------------------
// Prinzip "lazy per-page knowledge":
//   - APP_OVERVIEW   : dünne Orientierung (welche Seiten gibt es) — klein, einmalig.
//   - PAGE_KNOWLEDGE : TIEFES Wissen PRO Seite — wird nur geladen, wenn der User
//                      auf dieser Seite ist. Bei Seitenwechsel wird das Wissen der
//                      neuen Seite DAZUGELADEN (nicht alles auf einmal).
//   - buildPageContext(): einheitlicher Builder für den Live-Zustand einer Seite.
//   - Tools          : navigate / ask / link (Token-Protokoll, providerübergreifend).
//
// So bleibt der Kontext fokussiert & detailliert für die aktuelle Seite, statt mit
// dem Wissen aller Seiten zugemüllt zu werden.
// ============================================================================

import type { Language } from '../contexts/LanguageContext';
import type { AppView } from '../ui/navigationEvents';
import type { CoachConfigPatch, CoachCommand } from './coachToolEvents';

// ── Stabile Seiten-IDs (matchen die Sidebar-Navigation + Dev-Modi) ──────────
export type PageId =
  | 'home'
  | 'dashboard'
  | 'models'
  | 'training'
  | 'training-dev'
  | 'dataset'
  | 'analysis'
  | 'tests'
  | 'tests-dev'
  | 'laboratory'
  | 'versions'
  | 'synapse'
  | 'settings';

type Bilingual = { de: string; en: string };
const pick = (b: Bilingual, language: Language) => (language === 'en' ? b.en : b.de);

// ============================================================================
// 1. PERSONA — Wer der Coach ist (immer im System-Prompt, günstig gehalten)
// ============================================================================

const PERSONA: Bilingual = {
  en: `You are the **FrameTrain AI Coach** — an expert, friendly in-app assistant for the FrameTrain desktop app (a tool for training, testing and analysing machine-learning models).

You behave like a pro who knows the current page inside out:
- Answer concisely and directly. No filler.
- For technical questions: be concrete — give exact numbers, values or code.
- For errors: name the **cause** and the **fix**, not just a description.
- Point to the exact UI location ("top-right button", "LoRA section") when guiding.
- Use Markdown: **bold** for key terms, lists for steps, \`code\` for values.
- If unsure, say so — never invent features that don't exist.
- Ground every answer in what the user currently sees (page knowledge + live state below).
- Only name UI elements that actually appear in the page knowledge or live state below.
  Never guess a button label — if you do not know the exact name, describe the location instead.
- Do NOT use emojis. The UI is emoji-free; use plain text glyphs (-, *, >) if you need a marker.
- Always close a code fence you open. Never end an answer with a dangling \`\`\`.
- Keep lists flat — do not nest an ordered list inside another ordered list,
  otherwise the numbering restarts. Use a single level of numbering.
- Prefer short lines inside code blocks so nothing gets cut off.`,
  de: `Du bist der **FrameTrain AI Coach** — ein erfahrener, freundlicher In-App-Assistent für die FrameTrain Desktop-App (ein Tool zum Trainieren, Testen und Analysieren von Machine-Learning-Modellen).

Du agierst wie ein Profi, der die aktuelle Seite in- und auswendig kennt:
- Antworte knapp und direkt. Kein Füllwort.
- Bei technischen Fragen: konkret — exakte Zahlen, Werte oder Code.
- Bei Fehlern: **Ursache** + **Fix**, nicht nur die Beschreibung.
- Verweise auf die genaue UI-Stelle ("Button oben rechts", "LoRA-Bereich"), wenn du führst.
- Nutze Markdown: **fett** für Schlüsselbegriffe, Listen für Schritte, \`code\` für Werte.
- Wenn du unsicher bist: sag es — erfinde nie Funktionen, die es nicht gibt.
- Verankere jede Antwort in dem, was der User gerade sieht (Seiten-Wissen + Live-Zustand unten).
- Nenne nur UI-Elemente, die im Seiten-Wissen oder Live-Zustand unten wirklich vorkommen.
  Rate nie einen Button-Namen — kennst du ihn nicht exakt, beschreibe stattdessen die Stelle.
- Keine Emojis. Die UI ist emoji-frei; nutze schlichte Text-Zeichen (-, *, >) als Marker.
- Schließe jeden geöffneten Code-Block. Beende eine Antwort nie mit einem offenen \`\`\`.
- Halte Listen flach — verschachtele keine nummerierte Liste in einer nummerierten Liste,
  sonst beginnt die Nummerierung neu. Nutze nur eine Nummerierungsebene.
- Halte Zeilen in Code-Blöcken kurz, damit nichts abgeschnitten wird.`,
};

// ============================================================================
// 2. APP_OVERVIEW — dünne Orientierung (nur damit der Coach die Seiten kennt)
// ============================================================================

const APP_OVERVIEW: Bilingual = {
  de: `## FrameTrain — Kurzüberblick (nur Orientierung)

Linke Sidebar wechselt die Seiten. Der Coach (Gehirn-Button unten rechts) ist überall verfügbar.
Seiten: **Models · Training · Dataset · Analysis · Tests · Laboratory · Versions · Synapse · Settings**.
Typischer Ablauf: Model hinzufügen → Dataset hochladen & splitten → Training → Analysis → Tests.

Detailliertes Wissen bekommst du jeweils zur Seite, auf der der User gerade ist (und bei Wechsel wird es dazugeladen). Für Fragen zu anderen Seiten: grob helfen und per Navigations-Tool dorthin lotsen.`,
  en: `## FrameTrain — quick overview (orientation only)

The left sidebar switches pages. The coach (brain button, bottom-right) is available everywhere.
Pages: **Models · Training · Dataset · Analysis · Tests · Laboratory · Versions · Synapse · Settings**.
Typical flow: add model → upload & split dataset → training → analysis → tests.

You receive detailed knowledge for whichever page the user is currently on (and it's added on switch). For questions about other pages: help roughly and route there with the navigation tool.`,
};

// ============================================================================
// 3. PAGE_KNOWLEDGE — TIEFES Wissen pro Seite (lazy geladen)
// ============================================================================

const PAGE_KNOWLEDGE: Partial<Record<PageId, Bilingual>> = {
  models: {
    de: `### Seiten-Wissen: Models
Zweck: alle ML-Modelle der App verwalten — die Basis für Training & Tests.
- **Hinzufügen** (Button oben rechts): zwei Wege —
  · **Lokal**: Ordner mit Model-Dateien (config.json, Gewichte …) per Drag&Drop oder "Ordner durchsuchen". Gültig, wenn eine erkennbare Model-Struktur vorliegt.
  · **HuggingFace**: Suchbegriff (z.B. \`bert\`, \`mistral\`, \`xlm-roberta\`) → Modell wählen → lokalen Namen vergeben → Download.
- **Ansehen**: Karte zeigt Typ (model_type), Größe, Plugin-Support. Plugin-Support entscheidet, ob geführtes Training/Test möglich ist (sonst Dev-Modus nötig).
- **Löschen**: Hover auf Karte → Papierkorb → Bestätigung. Entfernt die Dateien.
- **Aktualisieren** (oben links): Liste neu laden.
Häufige Fragen: Welches Modell passt? (nach Aufgabe: Klassifikation → XLM-RoBERTa/BERT). Download hängt/fehlgeschlagen → HF-Namen prüfen, erneut versuchen. Speicherplatz → große Modelle brauchen viele GB.`,
    en: `### Page knowledge: Models
Purpose: manage all ML models — the basis for training & tests.
- **Add** (top-right button): two ways —
  · **Local**: folder with model files (config.json, weights …) via drag&drop or "browse folder". Valid when a recognisable model structure is present.
  · **HuggingFace**: search term (e.g. \`bert\`, \`mistral\`, \`xlm-roberta\`) → pick model → give a local name → download.
- **View**: card shows type (model_type), size, plugin support. Plugin support decides whether guided training/test is possible (otherwise Dev mode is needed).
- **Delete**: hover card → trash → confirm. Removes the files.
- **Refresh** (top-left): reload the list.
Common questions: which model fits? (by task: classification → XLM-RoBERTa/BERT). Download stuck/failed → check HF name, retry. Disk space → large models need many GB.`,
  },
  training: {
    de: `### Seiten-Wissen: Training
Zweck: ein Modell konfigurieren und trainieren. Voraussetzung: **Modell gewählt** (oben) + **Dataset gewählt & gesplittet**.
Zwei Modi (Umschalter oben): **Train** (geführte Config) und **Dev** (eigenes Python-Skript).
**Config-Bereiche (Train-Modus):**
- **Basic**: Epochs (Durchläufe; mehr = länger, Overfitting-Gefahr), Batch Size (Samples gleichzeitig; größer = schneller + mehr RAM), Learning Rate (Schrittgröße; typ. 1e-4…1e-3).
- **Optimizer**: Adam (Standard, ausgewogen) / SGD. Beta1/Beta2/Epsilon beeinflussen die Dynamik.
- **Scheduler**: Cosine Annealing (empfohlen) / Step. Warmup Steps für stabilen Start.
- **LoRA**: effizientes Fine-Tuning. Rank r=8–32 (höher = mehr Kapazität + VRAM), Alpha meist = Rank. Spart 50–80% VRAM.
- **Advanced**: Gradient Checkpointing (spart VRAM, etwas langsamer), 4/8-bit Loading, FP16/BF16 (Mixed Precision), Dropout 0.1–0.3, Label Smoothing.
**Ablauf:** Config einstellen → **Training starten** (unten) → Live-Dashboard (Loss-Kurve, Epoch/Step, Fortschritt) öffnet → nach Ende Ergebnis in Analysis prüfen, Version landet unter Versions.
**Debug:** OOM → Batch Size halbieren, LoRA an, Gradient Checkpointing, 4/8-bit, kürzere Sequenz. Loss stagniert → LR leicht erhöhen, Dropout runter, Dataset-Qualität, mehr Warmup. Nur CPU → CUDA/PyTorch-CUDA prüfen.`,
    en: `### Page knowledge: Training
Purpose: configure and train a model. Requires: **model selected** (top) + **dataset selected & split**.
Two modes (toggle at top): **Train** (guided config) and **Dev** (custom Python script).
**Config areas (Train mode):**
- **Basic**: epochs (passes; more = longer, overfitting risk), batch size (samples at once; larger = faster + more RAM), learning rate (step size; typ. 1e-4…1e-3).
- **Optimizer**: Adam (default, balanced) / SGD. Beta1/Beta2/Epsilon affect dynamics.
- **Scheduler**: Cosine annealing (recommended) / Step. Warmup steps for a stable start.
- **LoRA**: efficient fine-tuning. Rank r=8–32 (higher = more capacity + VRAM), alpha usually = rank. Saves 50–80% VRAM.
- **Advanced**: gradient checkpointing (saves VRAM, a bit slower), 4/8-bit loading, FP16/BF16 (mixed precision), dropout 0.1–0.3, label smoothing.
**Flow:** set config → **Start training** (bottom) → live dashboard (loss curve, epoch/step, progress) opens → after finish review in Analysis, the version appears under Versions.
**Debug:** OOM → halve batch size, enable LoRA, gradient checkpointing, 4/8-bit, shorter sequence. Loss stagnates → raise LR slightly, lower dropout, check dataset quality, more warmup. CPU only → check CUDA/PyTorch-CUDA.`,
  },
  'training-dev': {
    de: `### Seiten-Wissen: Training — Dev-Modus
Für Modelle ohne Plugin-Support oder volle Kontrolle: eigenes **Python-Trainingsskript**.
- Links: Code-Editor (Skript). Rechts: Live-Output/Logs. Oben: Modell + Version-Pfad wählen, Speichern, Skript-Bibliothek.
- Datasets werden als Referenzen bereitgestellt (Pfade im Kontext). Output landet unter \`[AppData]/training_outputs/dev_<job_id>\`.
- Ablauf: Skript schreiben/laden → **Start** → Output live mitlesen → Loss-Dashboard bei erkannten Loss-Werten.
- Debug: Fehler stehen im rechten Output; typische Ursachen sind falsche Pfade, fehlende Pakete, CUDA. Skript "dirty" = ungespeicherte Änderungen.`,
    en: `### Page knowledge: Training — Dev mode
For models without plugin support or full control: your own **Python training script**.
- Left: code editor (script). Right: live output/logs. Top: pick model + version path, save, script library.
- Datasets are provided as references (paths in context). Output goes to \`[AppData]/training_outputs/dev_<job_id>\`.
- Flow: write/load script → **Start** → follow live output → loss dashboard once loss values are detected.
- Debug: errors appear in the right-hand output; typical causes are wrong paths, missing packages, CUDA. Script "dirty" = unsaved changes.`,
  },
  dataset: {
    de: `### Seiten-Wissen: Dataset
Zweck: Trainingsdaten hochladen und vorbereiten.
- **Formate**: CSV, JSON, Arrow. Erwartet Spalten für \`text\` bzw. \`input\`/\`output\` (je nach Aufgabe).
- **Hochladen**: Datei hinzufügen → erscheint in der Liste mit Status.
- **Split**: Auto-Split **80% Training / 20% Validierung**. Ein Dataset muss den Status **"gesplittet"** haben, bevor Training möglich ist.
- **Verwalten**: ansehen, umbenennen, löschen; Dateien im Dataset-File-Manager prüfen.
Häufige Fragen: "Training-Button ausgegraut" → meist Dataset noch nicht gesplittet oder kein Modell gewählt. Qualität > Quantität; ausgewogene Klassen. Encoding-/Spalten-Fehler → CSV-Header prüfen.`,
    en: `### Page knowledge: Dataset
Purpose: upload and prepare training data.
- **Formats**: CSV, JSON, Arrow. Expects columns for \`text\` or \`input\`/\`output\` (depending on the task).
- **Upload**: add file → appears in the list with a status.
- **Split**: auto-split **80% train / 20% validation**. A dataset must have status **"split"** before training is possible.
- **Manage**: view, rename, delete; inspect files in the dataset file manager.
Common questions: "training button greyed out" → usually the dataset isn't split yet, or no model is selected. Quality > quantity; balanced classes. Encoding/column errors → check the CSV header.`,
  },
  analysis: {
    de: `### Seiten-Wissen: Analysis
Zweck: Trainings-Ergebnisse verstehen und bewerten.
- **Kurven**: Loss über Epochs/Steps, Trainings- vs. Validierungs-Loss, Metriken. Interpretation: Val-Loss steigt während Train-Loss fällt = **Overfitting**; beide stagnieren = zu kleine LR/zu wenig Kapazität.
- **KI-Report**: automatische Auswertung + **empfohlene Parameter** für den nächsten Lauf.
- **Chat**: seiteninterner Chat zu den Metriken; **Export** der Analyse.
- Auswahl oben: Modell + Version, deren Metriken gezeigt werden.
Häufige Fragen: "Ist mein Modell gut?" → Val-Metriken + Kurvenform ansehen. "Was als Nächstes?" → Empfehlungen übernehmen, in Training neuen Lauf starten.`,
    en: `### Page knowledge: Analysis
Purpose: understand and evaluate training results.
- **Curves**: loss over epochs/steps, train vs. validation loss, metrics. Reading them: val loss rising while train loss falls = **overfitting**; both stagnating = LR too small / too little capacity.
- **AI report**: automatic evaluation + **recommended parameters** for the next run.
- **Chat**: in-page chat about the metrics; **export** the analysis.
- Selection at top: model + version whose metrics are shown.
Common questions: "is my model good?" → look at val metrics + curve shape. "what next?" → apply recommendations, start a new run in Training.`,
  },
  tests: {
    de: `### Seiten-Wissen: Tests
Zweck: ein trainiertes Modell ausprobieren.
- **Test-Modus**: unterstützt XLM-RoBERTa (Klassifikation). Modell + trainierte Version wählen → Text ins Eingabefeld → Test-Button/Enter → **Top-Predictions mit %** rechts.
- **Dev-Test-Modus**: eigenes Python-Skript. Eingabe kommt via **stdin**, Ausgabe als **JSON auf stdout**. Für nicht unterstützte Modelltypen.
Häufige Fragen: "Keine Version wählbar" → Modell erst trainieren (Training). "Falsche/leere Predictions" → richtige Version gewählt? Passt Input-Format zum Training? Für Custom-Logik → Dev-Test-Modus.`,
    en: `### Page knowledge: Tests
Purpose: try out a trained model.
- **Test mode**: supports XLM-RoBERTa (classification). Pick model + trained version → type text → test button/Enter → **top predictions with %** on the right.
- **Dev test mode**: custom Python script. Input arrives via **stdin**, output as **JSON on stdout**. For unsupported model types.
Common questions: "no version selectable" → train the model first (Training). "wrong/empty predictions" → correct version chosen? Does the input format match training? For custom logic → Dev test mode.`,
  },
  'tests-dev': {
    de: `### Seiten-Wissen: Tests — Dev-Modus
Eigenes Python-Testskript für beliebige Modelltypen.
- Links: Editor. Unten links: Testdaten-Eingabe. Rechts: JSON-Output + Exit-Code.
- Kontrakt: Skript liest Eingabe via **stdin**, schreibt Ergebnis als **JSON auf stdout**. Exit-Code ≠ 0 = Fehler (Details im Output).
- Ablauf: Modell + Version wählen → Skript schreiben → Testdaten eingeben → **Ausführen** → JSON prüfen.`,
    en: `### Page knowledge: Tests — Dev mode
Custom Python test script for any model type.
- Left: editor. Bottom-left: test-data input. Right: JSON output + exit code.
- Contract: script reads input via **stdin**, writes the result as **JSON on stdout**. Exit code ≠ 0 = error (details in output).
- Flow: pick model + version → write script → enter test data → **Run** → check the JSON.`,
  },
  laboratory: {
    de: `### Seiten-Wissen: Laboratory
Zweck: mit Konfigurationen und Beispiel-Eingaben experimentieren, ohne einen vollen Trainingslauf.
- Sessions können gespeichert und wieder geladen werden (pro User).
- Ideal um schnell Parameter-Ideen, Prompts oder Samples durchzuspielen und zu vergleichen.
Hilf bei: Aufbau eines sinnvollen Experiments, Interpretation der Ergebnisse, Übertragung guter Einstellungen ins echte Training.`,
    en: `### Page knowledge: Laboratory
Purpose: experiment with configurations and sample inputs without a full training run.
- Sessions can be saved and reloaded (per user).
- Great for quickly trying and comparing parameter ideas, prompts or samples.
Help with: designing a meaningful experiment, interpreting results, carrying good settings over into real Training.`,
  },
  versions: {
    de: `### Seiten-Wissen: Versions
Zweck: Modell-Versionen verwalten. Jeder Trainingslauf erzeugt eine **neue Version** (als Baum/Verlauf).
- Version wählen → Details, Herkunft (von welcher Basis/Version), zugehörige Metriken.
- Versionen dienen als Ausgangspunkt für weiteres Training oder für Tests.
Hilf bei: welche Version ist die beste?, Vergleich von Versionen, Aufräumen alter Versionen.`,
    en: `### Page knowledge: Versions
Purpose: manage model versions. Every training run creates a **new version** (as a tree/history).
- Pick a version → details, lineage (from which base/version), associated metrics.
- Versions serve as the starting point for further training or for tests.
Help with: which version is best?, comparing versions, cleaning up old versions.`,
  },
  synapse: {
    de: `### Seiten-Wissen: Synapse
Zweck: visueller Node-Builder, um Pipelines aus Bausteinen zusammenzustecken.
- Nodes per Drag&Drop verbinden; Synapse hat einen **eigenen Synapse-Coach** für den Node-Graphen.
- Für Fragen tief im Node-Editor ist der Synapse-Coach spezialisierter; du gibst hier den Überblick.`,
    en: `### Page knowledge: Synapse
Purpose: visual node builder to assemble pipelines from building blocks.
- Connect nodes by drag&drop; Synapse has its **own Synapse coach** for the node graph.
- For questions deep inside the node editor the Synapse coach is more specialised; you provide the overview here.`,
  },
  settings: {
    de: `### Seiten-Wissen: Settings
Zweck: App konfigurieren. Tabs: Account, Erscheinungsbild (Theme), Benachrichtigungen, Updates, Dokumentation, Support, **KI-Assistent**, System, Info.
**KI-Assistent (dieser Coach):** Provider + Modell + API-Key wählen.
- **Anthropic (Claude)** — bezahlt, beste Qualität, \`sk-ant-…\`
- **OpenAI (GPT-4o)** — bezahlt, sehr gut, \`sk-…\`
- **Groq** — KOSTENLOS, schnell, \`gsk_…\` (console.groq.com)
- **Ollama** — KOSTENLOS, lokal/offline, kein Key (ollama.com)
Weiter: Theme/Farben, Sprache (DE/EN), Token-Budget des Coaches, System-Tab (Trainings-Pakete/Hardware/Anti-Sleep), Support-Tickets, Abmelden (Account-Tab).`,
    en: `### Page knowledge: Settings
Purpose: configure the app. Tabs: Account, Appearance (theme), Notifications, Updates, Documentation, Support, **AI Assistant**, System, About.
**AI Assistant (this coach):** choose provider + model + API key.
- **Anthropic (Claude)** — paid, best quality, \`sk-ant-…\`
- **OpenAI (GPT-4o)** — paid, very good, \`sk-…\`
- **Groq** — FREE, fast, \`gsk_…\` (console.groq.com)
- **Ollama** — FREE, local/offline, no key (ollama.com)
Also: theme/colours, language (DE/EN), the coach's token budget, System tab (training packages/hardware/anti-sleep), support tickets, log out (Account tab).`,
  },
};

export function pageKnowledge(pageId: PageId | null | undefined, language: Language): string {
  if (!pageId) return '';
  const k = PAGE_KNOWLEDGE[pageId];
  return k ? pick(k, language) : '';
}

export function hasPageKnowledge(pageId: PageId | null | undefined): boolean {
  return !!pageId && !!PAGE_KNOWLEDGE[pageId];
}

// ============================================================================
// 4. COACH_SKILLS — Was der Coach kann (seine "Fähigkeiten")
// ============================================================================

const SKILLS: Bilingual = {
  de: `## Deine Fähigkeiten
1. **Erklären** — jeden Parameter/Bereich der aktuellen Seite verständlich (was + warum).
2. **Konfigurieren** — konkrete Werte empfehlen (Epochs, Batch Size, LR, LoRA-Rank …) passend zu Modell/Dataset/Ziel. Exakte Werte nennen.
3. **Debuggen** — Fehler & Logs interpretieren: Ursache + Schritt-für-Schritt-Fix.
4. **Führen** — zur richtigen Stelle lotsen; anhand des Live-Zustands den nächsten Schritt vorschlagen.
5. **Ressourcen schätzen** — RAM/VRAM/Zeit abschätzen, Sparmaßnahmen nennen.
6. **Best Practices** — bewährte Vorgehensweisen für Training, Datasets, Tests, Analyse.`,
  en: `## Your skills
1. **Explain** — any parameter/area of the current page clearly (what + why).
2. **Configure** — recommend concrete values (epochs, batch size, LR, LoRA rank …) matched to model/dataset/goal. Give exact values.
3. **Debug** — interpret errors & logs: cause + step-by-step fix.
4. **Guide** — lead to the right place; suggest the next step from the live state.
5. **Estimate resources** — estimate RAM/VRAM/time, name ways to save.
6. **Best practices** — proven approaches for training, datasets, tests, analysis.`,
};

// ============================================================================
// 5. TOOLS — echte, ausführbare Tools (Token-Protokoll, providerübergreifend)
// ----------------------------------------------------------------------------
// Der Coach hängt Tokens ans Antwort-Ende. Der Client parst sie (parseCoachActions),
// entfernt sie aus dem Text und rendert klickbare Chips. Nichts läuft automatisch.
// ============================================================================

/** Navigierbare Ziele (matchen AppView aus navigationEvents). */
export const NAV_TARGETS: Record<AppView, Bilingual> = {
  home: { de: 'Start', en: 'Home' },
  models: { de: 'Models', en: 'Models' },
  training: { de: 'Training', en: 'Training' },
  dataset: { de: 'Dataset', en: 'Dataset' },
  analysis: { de: 'Analysis', en: 'Analysis' },
  tests: { de: 'Tests', en: 'Tests' },
  versions: { de: 'Versions', en: 'Versions' },
  laboratory: { de: 'Laboratory', en: 'Laboratory' },
  synapse: { de: 'Synapse', en: 'Synapse' },
  settings: { de: 'Einstellungen', en: 'Settings' },
};
const NAV_KEYS = Object.keys(NAV_TARGETS) as AppView[];

/** Whitelist externer Hilfe-Links (der Coach kann NUR diese öffnen). */
export const LINK_TARGETS: Record<string, { url: string; label: Bilingual }> = {
  groq: { url: 'https://console.groq.com/keys', label: { de: 'Groq API-Key holen', en: 'Get Groq API key' } },
  ollama: { url: 'https://ollama.com/download', label: { de: 'Ollama installieren', en: 'Install Ollama' } },
  huggingface: { url: 'https://huggingface.co/models', label: { de: 'HuggingFace Modelle', en: 'HuggingFace models' } },
  anthropic: { url: 'https://console.anthropic.com/settings/keys', label: { de: 'Anthropic API-Key', en: 'Anthropic API key' } },
  openai: { url: 'https://platform.openai.com/api-keys', label: { de: 'OpenAI API-Key', en: 'OpenAI API key' } },
};
const LINK_KEYS = Object.keys(LINK_TARGETS);

/** Öffenbare Dialoge/Bereiche fürs [[open:…]]-Tool (Ziel-Seite + Label). */
export const OPEN_TARGETS: Record<string, { page: AppView; label: Bilingual }> = {
  templates: { page: 'training', label: { de: 'Vorlagen öffnen', en: 'Open templates' } },
  'ai-assistant': { page: 'training', label: { de: 'KI-Assistent öffnen', en: 'Open AI assistant' } },
  'add-model': { page: 'models', label: { de: 'Modell hinzufügen', en: 'Add model' } },
  ram: { page: 'training', label: { de: 'RAM-Rechner öffnen', en: 'Open RAM calculator' } },
};
const OPEN_KEYS = Object.keys(OPEN_TARGETS);

// ── Setzbare Trainings-Config-Felder (Whitelist fürs [[set:…]]-Tool) ─────────
type SettableType = 'int' | 'float' | 'bool';
export const SETTABLE_CONFIG: Record<string, { type: SettableType; label: string }> = {
  epochs: { type: 'int', label: 'Epochs' },
  batch_size: { type: 'int', label: 'Batch Size' },
  learning_rate: { type: 'float', label: 'Learning Rate' },
  weight_decay: { type: 'float', label: 'Weight Decay' },
  warmup_ratio: { type: 'float', label: 'Warmup Ratio' },
  warmup_steps: { type: 'int', label: 'Warmup Steps' },
  max_seq_length: { type: 'int', label: 'Max Seq Length' },
  gradient_accumulation_steps: { type: 'int', label: 'Grad Accumulation' },
  gradient_checkpointing: { type: 'bool', label: 'Gradient Checkpointing' },
  fp16: { type: 'bool', label: 'FP16' },
  bf16: { type: 'bool', label: 'BF16' },
  dropout: { type: 'float', label: 'Dropout' },
  label_smoothing: { type: 'float', label: 'Label Smoothing' },
  max_grad_norm: { type: 'float', label: 'Max Grad Norm' },
  logging_steps: { type: 'int', label: 'Logging Steps' },
  use_lora: { type: 'bool', label: 'LoRA' },
  lora_r: { type: 'int', label: 'LoRA Rank' },
  lora_alpha: { type: 'int', label: 'LoRA Alpha' },
  lora_dropout: { type: 'float', label: 'LoRA Dropout' },
  load_in_4bit: { type: 'bool', label: '4-bit Loading' },
  load_in_8bit: { type: 'bool', label: '8-bit Loading' },
};
const SETTABLE_KEYS = Object.keys(SETTABLE_CONFIG);

const TRUE_WORDS = new Set(['true', '1', 'on', 'yes', 'an', 'ja', 'aktiv']);
const FALSE_WORDS = new Set(['false', '0', 'off', 'no', 'aus', 'nein', 'inaktiv']);

/** Wandelt einen Roh-Wert gemäß Feldtyp; gibt undefined bei ungültigem Wert. */
function coerceSettable(key: string, raw: string): number | boolean | undefined {
  const meta = SETTABLE_CONFIG[key];
  if (!meta) return undefined;
  const v = raw.trim();
  if (meta.type === 'bool') {
    const lo = v.toLowerCase();
    if (TRUE_WORDS.has(lo)) return true;
    if (FALSE_WORDS.has(lo)) return false;
    return undefined;
  }
  const num = meta.type === 'int' ? parseInt(v, 10) : parseFloat(v);
  return Number.isFinite(num) ? num : undefined;
}

/**
 * Filtert einen beliebigen Record (z.B. KI-Empfehlungen aus Analysis) auf die
 * bekannten, setzbaren Config-Felder und coerct die Werte typgerecht.
 */
export function coercePatchFromRecord(record: Record<string, unknown>): CoachConfigPatch {
  const patch: CoachConfigPatch = {};
  for (const [rawKey, rawVal] of Object.entries(record)) {
    // "Learning Rate" / "batch-size" → "learning_rate" / "batch_size"
    const key = rawKey.trim().toLowerCase().replace(/[\s-]+/g, '_');
    if (!SETTABLE_CONFIG[key]) continue;
    const value = coerceSettable(key, String(rawVal));
    if (value !== undefined) patch[key] = value;
  }
  return patch;
}

/** Kompakte Label-Zeile für einen Config-Patch, z.B. "Batch Size 8 · LR 0.0002". */
export function formatConfigPatch(patch: CoachConfigPatch): string {
  return Object.entries(patch)
    .map(([k, v]) => {
      const label = SETTABLE_CONFIG[k]?.label ?? k;
      const val = typeof v === 'boolean' ? (v ? 'an' : 'aus') : String(v);
      return `${label} ${val}`;
    })
    .join(' · ');
}

function toolsProtocol(language: Language, automation: boolean): string {
  const nav = NAV_KEYS.join(' | ');
  const links = LINK_KEYS.join(' | ');
  const setKeys = SETTABLE_KEYS.join(', ');
  const openKeys = OPEN_KEYS.join(' | ');
  if (language === 'en') {
    return `## Tools you can use
Append tokens at the very END of your reply (never mention the raw syntax in prose). The app turns them into clickable buttons — nothing runs automatically, the user clicks.

- **Navigate**: \`[[go:PAGE]]\` — PAGE ∈ ${nav}.
- **Follow-up question**: \`[[ask:short question]]\` — one-click follow-up.
- **Help link**: \`[[link:KEY]]\` — KEY ∈ ${links}. Opens the official page.
- **Apply training config**: \`[[set:key=value;key=value]]\` — Training page only. Fills the form (does NOT start training). Keys ∈ ${setKeys}. Always also state the values in prose.
- **Explain error/log**: \`[[explain:error]]\` — when the current page shows a training error or log; attaches the latest log and asks you to analyse it.
- **Estimate RAM**: \`[[estimate:ram]]\` — on Training; reports the RAM/VRAM estimate for the current model + config.
- **Open dialog**: \`[[open:TARGET]]\` — TARGET ∈ ${openKeys}. Opens that dialog/section (navigates there first if needed).
- **Search HuggingFace**: \`[[hf:query]]\` — opens Models and searches HuggingFace for a model.
- **Split dataset**: \`[[split:name]]\` — opens the 80/20 split dialog for that dataset (user confirms).
- **Apply recommended params**: \`[[apply:recommended]]\` — takes the parameters recommended in Analysis into the Training config.${automation ? `
- **Start / stop training** (automation on): \`[[train:start]]\` / \`[[train:stop]]\` — Training page only; start asks for a confirm click. Only when the setup is ready.` : ''}

Only add a token when it genuinely helps. At most 3 tokens total.`;
  }
  return `## Tools die du nutzen kannst
Hänge Tokens ganz ans ENDE deiner Antwort (Syntax nie im Fließtext erwähnen). Die App macht daraus klickbare Buttons — nichts passiert automatisch, der User klickt.

- **Navigieren**: \`[[go:SEITE]]\` — SEITE ∈ ${nav}.
- **Rückfrage**: \`[[ask:kurze Frage]]\` — Ein-Klick-Anschlussfrage.
- **Hilfe-Link**: \`[[link:KEY]]\` — KEY ∈ ${links}. Öffnet die offizielle Seite.
- **Trainings-Config setzen**: \`[[set:key=wert;key=wert]]\` — NUR auf der Training-Seite. Füllt das Formular (startet NICHT). Keys ∈ ${setKeys}. Nenne die Werte immer auch im Fließtext.
- **Fehler/Log erklären**: \`[[explain:error]]\` — wenn die Seite einen Trainings-Fehler oder ein Log zeigt; hängt das aktuelle Log an und lässt dich es analysieren.
- **RAM schätzen**: \`[[estimate:ram]]\` — auf Training; meldet die RAM/VRAM-Schätzung für aktuelles Modell + Config.
- **Dialog öffnen**: \`[[open:ZIEL]]\` — ZIEL ∈ ${openKeys}. Öffnet den Dialog/Bereich (navigiert ggf. vorher hin).
- **HuggingFace suchen**: \`[[hf:suchbegriff]]\` — öffnet Models und sucht ein Modell auf HuggingFace.
- **Dataset splitten**: \`[[split:name]]\` — öffnet den 80/20-Split-Dialog für das Dataset (User bestätigt).
- **Empfohlene Parameter übernehmen**: \`[[apply:recommended]]\` — übernimmt die in Analysis empfohlenen Parameter in die Trainings-Config.${automation ? `
- **Training starten / stoppen** (Automation an): \`[[train:start]]\` / \`[[train:stop]]\` — nur auf Training; Start fragt nach Klick-Bestätigung. Nur wenn das Setup bereit ist.` : ''}

Setze ein Token nur, wenn es wirklich hilft. Maximal 3 Tokens insgesamt.`;
}

export type CoachAction =
  | { type: 'navigate'; view: AppView }
  | { type: 'ask'; text: string }
  | { type: 'link'; key: string; url: string }
  | { type: 'set'; patch: CoachConfigPatch; summary: string }
  | { type: 'explain'; topic: 'error' | 'log' }
  | { type: 'estimate' }
  | { type: 'command'; command: CoachCommand; page: AppView; label: Bilingual }
  | { type: 'train'; op: 'start' | 'stop' };

export function commandLabel(action: Extract<CoachAction, { type: 'command' }>, language: Language): string {
  return pick(action.label, language);
}

/**
 * Zerlegt eine Assistant-Antwort in sichtbaren Text + erkannte Tool-Aktionen.
 * Entfernt alle Tokens aus dem Text und liefert die (deduplizierten) Aktionen.
 */
export function parseCoachActions(text: string): { cleanedText: string; actions: CoachAction[] } {
  const actions: CoachAction[] = [];
  const seen = new Set<string>();

  const cleanedText = text
    .replace(/\[\[\s*(go|ask|link|set|explain|estimate|open|hf|split|apply|train)\s*:\s*([^\]]+?)\s*\]\]/gi, (_m, kind: string, rawArg: string) => {
      const k = kind.toLowerCase();
      const arg = rawArg.trim();
      const once = (key: string, make: () => CoachAction | null) => {
        if (seen.has(key)) return;
        const a = make();
        if (a) { seen.add(key); actions.push(a); }
      };
      if (k === 'explain') {
        const topic = arg.toLowerCase() === 'log' ? 'log' : 'error';
        once(`explain:${topic}`, () => ({ type: 'explain', topic }));
      } else if (k === 'estimate') {
        if (arg.toLowerCase() === 'ram') once('estimate:ram', () => ({ type: 'estimate' }));
      } else if (k === 'open') {
        const target = arg.toLowerCase();
        const meta = OPEN_TARGETS[target];
        if (meta) once(`open:${target}`, () => ({ type: 'command', command: { kind: 'openDialog', target }, page: meta.page, label: meta.label }));
      } else if (k === 'hf') {
        const query = arg.slice(0, 80);
        if (query) once(`hf:${query.toLowerCase()}`, () => ({ type: 'command', command: { kind: 'hfSearch', query }, page: 'models', label: { de: `HuggingFace: ${query}`, en: `HuggingFace: ${query}` } }));
      } else if (k === 'split') {
        const name = arg.slice(0, 80);
        once('split:dataset', () => ({ type: 'command', command: { kind: 'splitDataset', name: name || undefined }, page: 'dataset', label: { de: name ? `Split: ${name}` : 'Dataset splitten', en: name ? `Split: ${name}` : 'Split dataset' } }));
      } else if (k === 'apply') {
        if (arg.toLowerCase() === 'recommended') once('apply:recommended', () => ({ type: 'command', command: { kind: 'applyRecommended' }, page: 'training', label: { de: 'Empfohlene Parameter übernehmen', en: 'Apply recommended params' } }));
      } else if (k === 'train') {
        const op = arg.toLowerCase() === 'stop' ? 'stop' : 'start';
        once(`train:${op}`, () => ({ type: 'train', op }));
      } else if (k === 'go') {
        const view = arg.toLowerCase() as AppView;
        const key = `go:${view}`;
        if ((NAV_KEYS as string[]).includes(view) && !seen.has(key)) {
          seen.add(key);
          actions.push({ type: 'navigate', view });
        }
      } else if (k === 'link') {
        const lk = arg.toLowerCase();
        const key = `link:${lk}`;
        if (LINK_TARGETS[lk] && !seen.has(key)) {
          seen.add(key);
          actions.push({ type: 'link', key: lk, url: LINK_TARGETS[lk].url });
        }
      } else if (k === 'ask') {
        const q = arg.slice(0, 140);
        const key = `ask:${q.toLowerCase()}`;
        if (q && !seen.has(key)) {
          seen.add(key);
          actions.push({ type: 'ask', text: q });
        }
      } else if (k === 'set') {
        // arg: "batch_size=8;learning_rate=2e-4;use_lora=true"
        const patch: CoachConfigPatch = {};
        for (const pair of arg.split(/[;,]/)) {
          const eq = pair.indexOf('=');
          if (eq < 0) continue;
          const field = pair.slice(0, eq).trim().toLowerCase();
          if (!SETTABLE_CONFIG[field]) continue;
          const value = coerceSettable(field, pair.slice(eq + 1));
          if (value !== undefined) patch[field] = value;
        }
        const patchKey = `set:${JSON.stringify(patch)}`;
        if (Object.keys(patch).length > 0 && !seen.has(patchKey)) {
          seen.add(patchKey);
          actions.push({ type: 'set', patch, summary: formatConfigPatch(patch) });
        }
      }
      return '';
    })
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  return { cleanedText, actions };
}

export function navTargetLabel(view: AppView, language: Language): string {
  return pick(NAV_TARGETS[view], language);
}

export function linkTargetLabel(key: string, language: Language): string {
  const l = LINK_TARGETS[key];
  return l ? pick(l.label, language) : key;
}

// ============================================================================
// 6. buildCoachSystemPrompt — der EINE System-Prompt
// ----------------------------------------------------------------------------
// Token-effizient & fokussiert:
//   - 1. Message: Persona + Kurzüberblick + Skills + Tools + Wissen der AKTUELLEN
//                 Seite + deren Live-Zustand.
//   - Seitenwechsel: Wissen der NEUEN Seite (falls noch nicht geschickt) + neuer
//                 Live-Zustand werden dazugereicht.
//   - sonst: nur die Persona (Rest steckt schon in der History).
// ============================================================================

export interface CoachPromptOptions {
  language: Language;
  pageId: PageId | null;
  /** Live-Zustand der aktuellen Seite (aus buildPageContext), ggf. leer */
  pageContent: string;
  /** erste Nachricht im Chat? */
  isFirstMessage: boolean;
  /** Seite hat seit der letzten Nachricht gewechselt? */
  pageChanged: boolean;
  /** Wissen der aktuellen Seite wurde noch nicht an das Modell geschickt? */
  includePageKnowledge: boolean;
  /** Automation-Modus aktiv? → schaltet Start/Stop-Training-Tools frei */
  automation?: boolean;
}

const DIVIDER = '────────────────────────────────────────────────────────────';

export function buildCoachSystemPrompt(opts: CoachPromptOptions): string {
  const { language, pageId, pageContent, isFirstMessage, pageChanged, includePageKnowledge, automation } = opts;
  const en = language === 'en';
  const parts: string[] = [pick(PERSONA, language)];

  // Globale, dünne Bausteine — nur einmal (erste Nachricht).
  if (isFirstMessage) {
    parts.push('', pick(APP_OVERVIEW, language), '', pick(SKILLS, language), '', toolsProtocol(language, !!automation));
  }

  // Tiefes Seiten-Wissen — nur wenn für diese Seite noch nicht geschickt.
  const knowledge = includePageKnowledge ? pageKnowledge(pageId, language) : '';
  if (knowledge) {
    parts.push('', knowledge);
  }

  // Live-Zustand — bei erster Nachricht oder Seitenwechsel.
  if ((isFirstMessage || pageChanged) && pageContent.trim()) {
    parts.push(
      '',
      pageChanged && !isFirstMessage
        ? (en ? '## Page changed — new live state' : '## Seite gewechselt — neuer Live-Zustand')
        : (en ? '## Current page (live state)' : '## Aktuelle Seite (Live-Zustand)'),
      DIVIDER,
      pageContent.trim(),
      DIVIDER,
    );
  } else if (isFirstMessage && !pageContent.trim() && !knowledge) {
    parts.push(
      '',
      en
        ? 'No page context available right now — answer from the overview above.'
        : 'Kein Seitenkontext verfügbar — antworte anhand des Kurzüberblicks oben.',
    );
  }

  return parts.join('\n');
}

// ============================================================================
// 7. buildPageContext — einheitlicher Builder für den Live-Zustand einer Seite
// ----------------------------------------------------------------------------
// Format (App-Konvention): "=== FrameTrain X ===" + "--- SECTION ---" + "• item".
// ============================================================================

/** Ein Key/Value-Paar ODER eine freie Zeile. */
export type ContextLine = readonly [label: string, value: string] | string;

/** Ergonomischer Helper für Key/Value-Zeilen (korrekte Tuple-Typisierung). */
export const kv = (label: string, value: string): ContextLine => [label, value];

export interface PageContextInput {
  /** stabile Seiten-ID */
  pageId: PageId;
  /** Sprache für Standard-Abschnittstitel (default: de) */
  language?: Language;
  /** lokalisierter Titel, z.B. "Training" */
  title: string;
  /** ein Satz: aktueller Fokus/Zustand dieser Seite */
  purpose?: string;
  /** aktueller Live-Zustand (Auswahl, Status, Werte) */
  state?: ContextLine[];
  /** wo was ist (Layout / UI-Orientierung) */
  layout?: string[];
  /** was der User JETZT tun kann (nächste sinnvolle Schritte) */
  actions?: string[];
  /** zusätzliche Referenz-Infos (Pfade, Zähler, verfügbare Objekte) */
  refs?: ContextLine[];
  /** freie Zusatz-Abschnitte */
  sections?: { heading: string; lines: ContextLine[] }[];
}

/** Standard-Abschnittstitel — an die bestehende App-Konvention angelehnt. */
const SECTION_LABELS = {
  de: { state: 'STATUS', layout: 'UI LAYOUT', actions: 'VERFÜGBARE AKTIONEN', refs: 'KONTEXT' },
  en: { state: 'STATE', layout: 'UI LAYOUT', actions: 'AVAILABLE ACTIONS', refs: 'CONTEXT' },
} as const;

function renderLines(lines: ContextLine[]): string[] {
  return lines
    .filter((l) => l != null && (typeof l === 'string' ? l.trim() !== '' : true))
    .map((l) => (typeof l === 'string' ? `• ${l}` : `• ${l[0]}: ${l[1]}`));
}

/**
 * Einheitlicher Seiten-Kontext-Builder. Erzeugt einen konsistenten Block im
 * bestehenden App-Format: "=== FrameTrain X ===" + "--- SECTION ---" + "• item".
 */
export function buildPageContext(input: PageContextInput): string {
  const labels = SECTION_LABELS[input.language ?? 'de'];
  const out: string[] = [`=== FrameTrain ${input.title} ===`];

  if (input.purpose) out.push('', input.purpose);

  const section = (title: string, lines?: ContextLine[]) => {
    const rendered = lines ? renderLines(lines) : [];
    if (rendered.length) out.push('', `--- ${title} ---`, ...rendered);
  };

  section(labels.state, input.state);
  section(labels.layout, input.layout);
  section(labels.actions, input.actions);
  section(labels.refs, input.refs);
  for (const s of input.sections ?? []) section(s.heading, s.lines);

  return out.join('\n');
}
