// Smart Knowledge Base with Sections - on-demand loading
// Die KI kann Wissensdocs anfordern statt alles auf einmal zu bekommen

export interface KnowledgeSection {
  id: string;
  title: string;
  keywords: string[];
  content: string;
}

export const KNOWLEDGE_SECTIONS: Record<string, KnowledgeSection> = {
  overview: {
    id: 'overview',
    title: 'FrameTrain Überblick',
    keywords: ['app', 'was ist', 'übersicht', 'struktur', 'features'],
    content: `
# Was ist FrameTrain?

FrameTrain ist eine Desktop-Anwendung zum Trainieren von Machine Learning Modellen mit erweiterten Konfigurationsoptionen, Dataset-Management und Analyse-Tools.

## Verfügbare Funktionen

1. **Models** - Verwalte verfügbare ML-Modelle
2. **Training** - Konfiguriere und starte Model-Training mit erweiterten Einstellungen
3. **Dataset** - Lade und verwalte Trainings-Daten
4. **Analysis** - Analysiere Trainings-Ergebnisse und Metriken
5. **Tests** - Führe Tests mit trainierten Modellen durch
6. **Laboratory** - Experimentiere mit verschiedenen Konfigurationen
7. **Versions** - Verwalte Model-Versionen
8. **Settings** - App-Konfiguration, KI-Provider, Theme
    `
  },

  dataset_management: {
    id: 'dataset_management',
    title: 'Dataset Management',
    keywords: ['dataset', 'datei', 'hinzufügen', 'upload', 'csv', 'json', 'daten', 'format'],
    content: `
# Dataset Management

## Wie man Datasets hinzufügt

### Schritt 1: Zur Dataset-Seite gehen
1. Klick auf "Dataset" in der Sidebar
2. Du siehst die Liste aller verfügbaren Datasets

### Schritt 2: Neue Datei uploaden
1. Klick auf "Upload Dataset" oder "+"-Button
2. Wähle deine Datei (CSV, JSON oder Arrow Format)
3. Die App analysiert automatisch die Struktur
4. Bestätige die Spalten-Zuordnung

### Schritt 3: Spalten Mapping
Die App verlangt folgende Spalten:
- **text** (für reine Text-Daten)
- ODER **input**/**output** (für Frage-Antwort Paare)
- Optional: Tags, Labels, Kategorien

### Nachträglich weitere Dateien hinzufügen
1. Dataset wählen, das du erweitern möchtest
2. Klick auf "Spalten hinzufügen" oder "Merge"
3. Neue Datei auswählen (muss gleiches Spalten-Format haben!)
4. Die Rows werden automatisch kombiniert

### Unterstützte Formate

**CSV**
\`\`\`
text,label
"Das ist mein erster Text","positiv"
"Hier ein zweiter Text","negativ"
\`\`\`

**JSON**
\`\`\`json
[
  {"text": "Erster Text", "label": "positiv"},
  {"text": "Zweiter Text", "label": "negativ"}
]
\`\`\`

## Auto-Split
- 80% Training (für Model-Lernen)
- 20% Validierung (für Qualitätsprüfung)
- Automatisch randomisiert damit keine Bias

## Best Practices
1. Mindestens 100-500 Beispiele für gutes Training
2. Balancierte Daten (nicht nur positive Fälle)
3. Gute Test-Daten in Validierungs-Set
4. Duplikate entfernen bevor du uploadst
    `
  },

  training_config: {
    id: 'training_config',
    title: 'Training Konfiguration',
    keywords: ['training', 'config', 'parameter', 'epochs', 'batch size', 'learning rate', 'lr', 'optimizer'],
    content: `
# Training Konfiguration

## Basic Settings

### Epochs
- **Was**: Wie oft der Algorithmus die KOMPLETTEN Daten durchläuft
- **Effekt**: Höher = längeres Training, potentiell bessere Ergebnisse, aber auch Overfitting-Risiko
- **Typisch**: 3-10 für fine-tuning, 20-50 für von-Grund-auf Training
- **Faustregel**: Je größer dein Dataset, desto weniger Epochs brauchst du

### Batch Size
- **Was**: Wie viele Samples ("Beispiele") gleichzeitig verarbeitet werden
- **Trade-off**: 
  - Größer = schneller Training, aber mehr RAM nötig
  - Kleiner = weniger RAM, aber langsamer Training
- **Typisch**: 8-32 für große Models, 32-64 für kleine Models
- **RAM-Regel**: Batch Size verdoppeln = RAM verdoppelt sich ungefähr

### Learning Rate
- **Was**: Schrittgröße wie schnell das Model lernt
- **Effekt**:
  - Zu hoch = Model macht große Sprünge, kann nicht konvergieren (platzt auseinander)
  - Zu niedrig = Model lernt sehr langsam oder steckt fest
  - Richtig = stetiger Fortschritt
- **Typisch**: 0.00005 - 0.001 (5e-5 bis 1e-3)
- **Faustregel**: Bei LLMs: 2e-4 bis 1e-4, bei kleineren: 1e-4 bis 1e-3

## Optimizer

### Adam (Standard)
- **Bestes für**: Die meisten Fälle, LLMs, Mix aus Performance und Stabilität
- **Parameter**:
  - Beta1 (default 0.9): Exponentieller Durchschnitt der Gradienten
  - Beta2 (default 0.999): Exponentieller Durchschnitt der Quadrate
  - Epsilon (default 1e-8): Numerische Stabilität

### SGD
- **Bestes für**: Spezielle Anwendungen, wenn Adam nicht funktioniert
- **Momentum**: Tyischerweise 0.9 (erinnert sich an vorherige Updates)

## Scheduler (Learning Rate Anpassung)

### Cosine Annealing
- **Was**: Learning Rate sinkt graduell nach Kosinus-Kurve während Training
- **Effekt**: Zuerst schnelles Lernen, später feine Anpassungen
- **Mit Warmup**: Zuerst LOW LR, dann Kosinus runter
- **Empfehlung**: 🏆 Am besten für die meisten Fälle

### Step Scheduler  
- **Was**: Learning Rate wird nach X Steps reduziert
- **Effekt**: Aggressive, diskrete Reduktionen
- **Gamma**: Reduktionsfaktor (0.1 = 10% von vorher)
- **Step Size**: Nach wie vielen Steps reduzieren

### Warmup
- **Was**: Zuerst mit niedriger LR trainieren (z.B. 100-1000 Steps)
- **Warum**: Model stabilisiert sich, vermeidet Divergenz am Anfang
- **Typisch**: 100-10% der Total-Steps

## LoRA (Low-Rank Adaptation)

### Wann verwenden?
- **Große Models** (13B+ Parameter)
- **Begrenzterer VRAM** (sparst 50-80%)
- **Fine-tuning** statt von-Grund-auf Training
- **Schnelleres Training** (50-70% schneller oft)

### Parameter

**Rank (r)**
- Wie viele "Dimensionen" der Anpassungen
- Höher = mehr Freiheitsgrade, aber mehr VRAM
- Typisch: 8, 16, 32
- Faustregel: 8 = minimal, 16 = gut, 32 = viel mehr lernen aber 2x VRAM

**Alpha**
- Skalierungsfaktor der LoRA-Updates
- Üblicherweise gleich wie Rank setzen
- Oder 2x Rank für stärkere Updates

**Target Modules**
- Welche Layer angepasst werden (q_proj, v_proj, up_proj, etc.)
- Standard reicht für die meisten Fälle

### LoRA Effekt auf VRAM
\`\`\`
Ohne LoRA:  70B Model = ~140GB VRAM
Mit LoRA:   70B Model mit Rank 32 = ~20-30GB VRAM
Mit 4bit:   70B Model mit LoRA = ~8-12GB VRAM ⭐
\`\`\`
    `
  },

  advanced_settings: {
    id: 'advanced_settings',
    title: 'Advanced Settings',
    keywords: ['advanced', 'gradient', 'checkpointing', '8bit', '4bit', 'fp16', 'bf16', 'dropout', 'regularization'],
    content: `
# Advanced Settings

## Memory Optimizations

### Gradient Checkpointing
- **Was**: Speichert intermediate Werte nicht, rechnet sie bei Backprop neu
- **Vorteil**: -20-30% VRAM Verbrauch
- **Nachteil**: +40% längeres Training (muss neu rechnen)
- **Wann**: Wenn dir der Speicher zu knapp wird
- **Trade-off**: VRAM sparen für Zeit opfern, aber Training ist möglich

### 8bit Loading
- **Was**: Model wird in 8-bit Integer komprimiert statt 32-bit Float
- **Vorteil**: -75% VRAM (4x weniger)
- **Nachteil**: Kleine Precision-Verluste, funktioniert am besten mit LoRA
- **Wann**: Bei großen Models mit sehr limitiertem VRAM
- **Oft kombiniert mit**: LoRA für bestes Ergebnis

### 4bit Loading
- **Was**: Model wird noch komprimierter in 4-bit komprimiert
- **Vorteil**: -87.5% VRAM (8x weniger!)
- **Nachteil**: Weitere Precision-Verluste, aber meist ok
- **Wann**: 🏆 Beste Balance wenn Speicher sehr knapp (z.B. 13B auf 8GB)
- **Combo**: 4bit + LoRA + Rank 8 = sehr klein mit guten Ergebnissen

## Precision Formats

### FP16 (Float 16-bit)
- **Vorteil**: -50% VRAM, 2-3x schneller auf moderner GPU
- **Nachteil**: Numerische Instabilität möglich bei großen Models
- **Wann**: Solange Training stabil bleibt (Monitor Loss!)

### BF16 (Brain Float 16-bit)
- **Vorteil**: Mehr Stabilität als FP16, fast gleich schnell
- **Nachteil**: Neuere GPUs nötig (A100, H100, RTX 40x)
- **Wann**: 🏆 Wenn GPU es unterstützt, beste Option

### FP32 (Float 32-bit, Standard)
- **Vorteil**: Stabil, vereinheitlicht
- **Nachteil**: Braucht 2x VRAM wie FP16
- **Wann**: Wenn Speicher/Geschwindigkeit egal

## Regularization

### Dropout
- **Was**: Zufällig setzt Prozent der Neuronen während Training auf 0
- **Effekt**: Model verlässt sich nicht zu sehr auf einzelne Neuronen
- **Verhindert**: Overfitting (Model lernt Trainings-Daten zu gut)
- **Typisch**: 0.1 - 0.3 (10-30%)
- **Faustregel**: 0.1 für große Models, 0.3 für kleine

### Label Smoothing
- **Was**: Zielwert nicht auf genau 1.0 sondern etwas niedriger (z.B. 0.9)
- **Effekt**: Model wird weniger "sicher" in seinen Vorhersagen
- **Verhindert**: Overfitting, macht Model robuster
- **Typisch**: 0.01 - 0.1
- **Faustregel**: 0.1 für aggressive Regularisierung

### Weight Decay
- **Was**: Bestraft große Gewichte, favorisiert einfachere Models
- **Effekt**: Gegen Overfitting, bessere Generalisierung
- **Typisch**: 0.0 - 0.01
- **0.01 = Stark**, 0.001 = Schwach

## Gradient Accumulation
- **Was**: Verarbeite mehrere Batches, bevor Gewichte aktualisiert werden
- **Effekt**: Effektive Batch Size wird größer ohne VRAM zu brauchen
- **Beispiel**: Batch 8 + Accumulation 4 = effektiv Batch 32
- **Wann**: Wenn echte große Batches nicht ins VRAM passen
    `
  },

  troubleshooting: {
    id: 'troubleshooting',
    title: 'Fehler & Lösungen',
    keywords: ['fehler', 'error', 'problem', 'oom', 'speicher', 'abstürz', 'crash', 'gpu', 'langsam', 'stagniert'],
    content: `
# Häufige Probleme & Lösungen

## Out of Memory (OOM) Error

### Symptom
Training startet, dann plötzlich Crash mit "out of memory"

### Ursachen & Lösungen (in dieser Reihenfolge)

**1. Batch Size reduzieren** ⭐⭐⭐ (bester Start)
- Aktuell: 32? → Versuch: 16
- Wenn noch OOM: 16 → 8
- Wenn noch OOM: 8 → 4 (bei sehr großen Models)
- Jederzeit sollte funktionieren

**2. Aktiviere LoRA** ⭐⭐⭐
- Wenn nicht aktiviert: Muss an
- Rank auf 8 setzen
- Spart 50-80% speicher sofort!

**3. Aktiviere Gradient Checkpointing** ⭐⭐
- Settings > Advanced > Gradient Checkpointing
- Kostet ~40% längeres Training aber -20-30% VRAM

**4. Aktiviere 4bit Loading** ⭐⭐
- Settings > Advanced > Load in 4bit
- Kombiniert mit LoRA: sehr effizient
- Gutes Trade-off zwischen VRAM und Qualität

**5. Reduziere Max Sequence Length**
- Wenn das Model sehr lange Texte verarbeitet
- 2048 → 1024 oder sogar 512

**6. Verwende BF16/FP16** (falls GPU unterstützt)
- Spart -50% VRAM gegenüber FP32

### Wenn alles oben fehlschlägt:
- Model ist zu groß für deine Hardware
- Versuch ein kleineres Model (7B statt 13B, etc.)
- Oder: Mehrere GPUs brauchen (nicht möglich aktuell in App)

---

## Training ist viel zu langsam

### Symptom
Training dauert extrem lange, Netzwerk langsam

### Lösungen (in Reihenfolge)

**1. Batch Size erhöhen** ⭐⭐⭐ (wenn RAM da)
- Aktuell: 8 → 16 oder 32
- Schneller weil GPU weniger Overhead-Prozesse

**2. Logging Steps erhöhen**
- Aktuell: 10 → 50 (je nach Training-Länge)
- Model schreibt weniger Logs (großer Overhead)

**3. Eval Steps erhöhen**
- Wie oft wird Validierungs-Set geprüft?
- Wenn jede 100 Steps: 500 Steps machen (weniger oft evaluieren)

**4. Gradient Accumulation NICHT verwenden**
- Das ist sogar langsamer (mehr Berechnungen)

**5. GPU Utilization prüfen**
- Ist GPU überhaupt in Benutzung?
- Falls nur CPU: CUDA/GPU-Fehler?

**6. Warmup Steps reduzieren**
- Von 1000 auf 100-500
- Weniger Zeit im Setup-Modus

---

## Loss stagniert / Model lernt nicht

### Symptom
Training läuft, Loss reduziert sich nicht, bleibt gleich

### Lösungen

**1. Learning Rate erhöhen** ⭐⭐⭐
- Aktuell: 5e-5 → Versuch: 1e-4 oder 2e-4
- Wenn Verlust wild springt? LR ist zu hoch
- Dann: Zwischen den zwei Werte midtwegs versuchen

**2. Dropout reduzieren**
- Zu viel Dropout hemmt Lernen
- Von 0.3 → 0.1
- Oder ganz aus (0.0)

**3. Warmup Steps erhöhen**
- Von 100 → 500 oder 1000
- Model braucht mehr Zeit zum "aufwärmen"

**4. Dataset überprüfen**
- Sind die Daten gut qualitativ?
- Sind Duplikate? (reduziert Lern-Signal)
- Genug Variabilität?

**5. Model wechseln**
- Aktueller Model passt vielleicht nicht zu Task
- Versuch anderer Model-Basis

**6. Mehr Epochs**
- Von 5 → 10-20 Epochs
- Model braucht mehr Zeit

---

## GPU wird nicht genutzt (nur CPU)

### Symptom
Task Manager / nvidia-smi zeigt GPU 0%, CPU 100%

### Checks

1. **Überprüfe CUDA Installation**
   - Starte: System-Check in Settings
   - Sollte CUDA + cuDNN zeigen

2. **PyTorch CUDA Support**
   - App-Logs checken ob PyTorch CUDA hat

3. **GPU Treiber**
   - Windows: nvidia-smi in Terminal starten
   - Sollte GPU liste zeigen

4. **Zu viele Prozesse?**
   - Andere Applikationen GPU nutzend?
   - Nur FrameTrain offen lassen

5. **Support kontaktieren**
   - Logs sammeln (Settings > Export Logs)
   - An Support schicken

---

## Model schlägt fehl beim Start

### Symptom
Error: "Model failed to load" oder "CUDA out of memory while loading"

### Lösungen

1. **Zu großes Model für Hardware**
   - 70B Model braucht min. 140GB RAM
   - Versuch: 13B oder 7B Model

2. **Vorher nicht genug Speicher**
   - RAM/VRAM durch andere Prozesse belegt?
   - PC neu starten, nur FrameTrain offen

3. **Model beschädigt**
   - Model neu downloaden/installieren
   - Settings > Models > Reinstall
    `
  },

  performance_tips: {
    id: 'performance_tips',
    title: 'Performance Optimierung',
    keywords: ['schnell', 'performance', 'optimieren', 'tuning', 'bessere ergebnisse', 'accuracy'],
    content: `
# Performance Tipps & Best Practices

## Für schnelleres Training ⚡

1. **LoRA aktivieren** (wenn Model > 7B)
   - Spart 50-70% Trainingszeit
   - RAM auch deutlich sparen

2. **Batch Size erhöhen** (solange VRAM reicht)
   - Doppelte Batch = ~1.5x schneller (nicht 2x)
   - Aber nicht zu aggressiv

3. **Gradient Checkpointing deaktiviert lassen**
   - Das macht Training langsamer!
   - Nur aktivieren bei OOM-Problemen

4. **Validation Steps erhöhen**
   - Standard: jede 100 Steps
   - Besser: jede 500-1000 Steps
   - Spart Zeit proportional

5. **Warmup Steps anpassen**
   - Sehr kleine Werte: 100-500
   - Nicht nötig für fine-tuning

6. **Logging Steps erhöhen**
   - Von 10 auf 50-100
   - Logs schreiben ist Overhead

## Für bessere Ergebnisse 🎯

1. **Learning Rate fein-tunen**
   - Nicht zu hoch (exploding), nicht zu niedrig (stuck)
   - 2e-4 als Startpunkt für LLMs
   - Experiment mit 1e-4, 5e-4

2. **Mehr Epochs** (aber nicht zu viel)
   - 5-10 Epochs für normales fine-tuning
   - 20+ nur wenn Dataset klein
   - Achtung: Nach ~10 Epochs oft nur noch Overfitting

3. **Guts Dataset**
   - Qualität > Quantität
   - 100-500 gute Beispiele > 10000 schlechte
   - Balanciert (nicht nur positive Fälle)

4. **Warmup Steps erhöhen** (für Stabilität)
   - Von 100 → 1000-2000
   - Hilft bei Lern-Stabilität
   - Besonders bei kleineren Datasets

5. **Label Smoothing aktivieren**
   - 0.1 ist gut default
   - Macht Model robuster

6. **Scheduler auswählen**
   - Cosine Annealing 🏆 (am besten)
   - Step am besten für supervised Learning

7. **Dropout nicht zu hoch**
   - 0.1 reicht oft
   - Höher = Model lernt zu wenig

## Experiment-Strategie

### Was ändern wenn Ergebnisse nicht gut?
1. **Learning Rate ±50%** (erste Schritt, schnell zu testen)
2. **Mehr Epochs** (dauert länger, aber sicher besser wenn Dataset klein)
3. **Dropout reduzieren** (0.3 → 0.1)
4. **Warmup erhöhen** (nur wenn Loss wild springt)

### Was nicht ändern (erzeugt nicht sichtbare Verbesserung):
- Beta1/Beta2 des Optimizer (Standard ist gut)
- Batch Norm Parameter
- Zu kleine Änderungen (<10% des Wertes)

---

## Tipps zum Sparen

### VRAM Sparen (Priorität)
1. LoRA aktivieren (50-80% sparen) ⭐⭐⭐
2. Batch Size halbieren (50% sparen) ⭐⭐
3. Gradient Checkpointing (20-30% sparen) ⭐
4. 4bit loading (87.5% sparen!) ⭐⭐

### Geld sparen (bei API-Providers)
- **Groq (kostenlos)** - beste Balance
- **Ollama (kostenlos + offline)** - privat aber CPU-basiert
- Claude/GPT-4o: Nur für komplexe Fragen

### Zeit sparen
1. \`Batch Size 2x\` (wenn möglich) → ~1.5x schneller
2. Validation seltener (500 steps statt 100)
3. LoRA verwenden (50-70% schneller Training)
    `
  },

  model_selection: {
    id: 'model_selection',
    title: 'Model Auswahl',
    keywords: ['model', 'llm', 'welcher model', 'groß', 'klein', 'verfügbar', 'download'],
    content: `
# Model Auswahl Guide

## Verfügbare Models

### Kleine Models (7B Parameter)

**Llama 2 7B**
- Vorteil: Passt auf Laptops (8GB VRAM), schnell
- Nachteil: Schwächer in komplexen Tasks
- Best für: Text-Klassifikation, Sentiment-Analyse
- VRAM: 8-14GB (FP16), 4-8GB mit LoRA

**Mistral 7B**
- Vorteil: Schneller und besser als Llama 2, gutes Preis-Leistungs-Verhältnis
- Nachteil: Nicht so mächtig wie größere
- Best für: Produktionsystem mit begrenzt Ressourcen
- VRAM: 8-14GB (FP16), 4-8GB mit LoRA

### Mittlere Models (13-34B)

**Llama 2 13B**
- Vorteil: Deutlich besser als 7B, noch relativ schnell
- Nachteil: Braucht 16GB+ VRAM
- Best für: Gutes all-around Model
- VRAM: 16-26GB, 8-16GB mit LoRA

**Mistral Medium / Mixtral 8x7B**
- Vorteil: Mixture-of-Experts, sehr gut für komplexe Tasks
- Nachteil: Speicherintensiv
- Best für: Komplexes Reasoning, mehrsprachig
- VRAM: 40GB+, 20GB mit LoRA

### Große Models (70B+)

**Llama 2 70B**
- Vorteil: State-of-the-art auf dem Niveau, sehr intelligent
- Nachteil: Braucht 140GB+ VRAM (praktisch nur Multi-GPU oder Cloud)
- Best für: Komplexe Tasks, wenn Ressourcen kein Limit
- VRAM: 140GB, 20-40GB mit LoRA

## Auswahl nach Anforderung

### Ich brauche schnell Ergebnisse
→ **Mistral 7B** oder **Llama 2 7B**
- Lokal oft in <1GB VRAM möglich
- Training: Die schnellsten

### Ich brauche gute Qualität
→ **Llama 2 13B** oder **Mistral Medium**
- Gute Balance zwischen Qualität und Ressourcen
- Training: 2-5h auf guter GPU bei normaler Config

### Ich brauche Top-Qualität (beste Ergebnisse)
→ **Llama 2 70B** 
- Best-in-class für komplexe Aufgaben
- Training: 10-48h je nach Größe
- ⚠️ Braucht sehr gute Hardware oder Cloud

## RAM Requirements Schnell-Referenz

\`\`\`
Llama 2 7B:  8-14GB (FP16), 4-8GB (LoRA), 2-4GB (4bit + LoRA)
Llama 2 13B: 16-26GB (FP16), 8-14GB (LoRA), 4-8GB (4bit + LoRA)
Mistral 7B:  8-14GB (FP16), 4-8GB (LoRA), 2-4GB (4bit + LoRA)
Mistral Med: 32GB (FP16), 16-20GB (LoRA), 8-12GB (4bit + LoRA)
Llama 2 70B: 140GB (FP16), 20-40GB (LoRA), 8-16GB (4bit + LoRA)
\`\`\`

## Best Practices

1. **Start klein**: 7B testen, später upgraded
2. **Für Production**: 13B ist oft das sweet spot
3. **Budget-Option**: LoRA + 4bit auf 7B = sehr günstig
4. **Top-Qualität**: 13B + LoRA ist oft ausreichend (spart Zeit vs 70B)

## Model Combination Best Practice

**Für schlankes Setup:**
- Modell: 7B (Mistral oder Llama 2)
- LoRA: Rank 8
- 4bit Loading: Yes
- Ergebnis: ~2-4GB VRAM, gut für Laptops

**Für balanciert Setup:**
- Modell: 13B
- LoRA: Rank 16
- 4bit Loading: No (FP32 ok)
- Ergebnis: ~8-16GB VRAM, laptop mit guter GPU

**Für beste Qualität:**
- Modell: 13B oder 70B
- LoRA: Rank 32
- Precision: BF16 wenn GPU unterstützt
- Ergebnis: beste Ergebnisse, braucht gute GPU
    `
  },
};

// Table of Contents für schnellen Überblick
export const KNOWLEDGE_TOC = {
  overview: 'FrameTrain Überblick',
  dataset_management: 'Dataset Management',
  training_config: 'Training Konfiguration',
  advanced_settings: 'Advanced Settings',
  troubleshooting: 'Fehler & Lösungen',
  performance_tips: 'Performance Optimierung',
  model_selection: 'Model Auswahl',
};

// AI System Prompt with Instructions how to use knowledge
export const AI_SYSTEM_PROMPT_WITH_INSTRUCTIONS = `Du bist ein hilfreicher AI-Assistent in der FrameTrain-Anwendung für Machine Learning Training.

## Deine Fähigkeiten

1. **Kontext-Zugang**: Du hast Zugang zu:
   - Aktuell ausgewähltes Model
   - Training-Konfiguration (Epochs, Batch Size, Learning Rate, etc.)
   - System RAM / GPU Verfügbarkeit
   - Aktuelle Seite/View in der App
   - User-Fage

2. **Wissensdatenbank**: Du kannst bei Bedarf auf spezialisierte Dokumentationen zugreifen:
   - App-Überblick
   - Dataset Management
   - Training Parameters
   - Advanced Optimizations
   - Fehlersuche
   - Performance Tipps
   - Model-Auswahl

## Wie du Wissen nutzen sollst

### Option 1: Bekannte Antwort
Wenn die Frage einfach ist (z.B. "Was ist Batch Size?"), antworte direkt mit deinem Wissen.

### Option 2: Dokumentation anfordern  
Wenn die Frage spezifisch und komplex ist, folge diesem Muster:

**Nutzer fragt:** "Wie füge ich datasets nachträglich hinzu?"

**Dein Denken:**
1. Das ist spezifisch (Dataset Management)
2. Der User braucht Schritt-für-Schritt
3. Ich sollte mir relevante Docs ansehen

**Dein Response:**
"Lass mich mal in der Dokumentation nachschauen... *checkt Dataset Management Sektion*... 

Hier ist wie du Datasets hinzufügst:
1. Zur Dataset-Seite gehen
2. ...[antworte basierend auf Dokumentation]..."

## Tone & Style

- Freundlich, hilfreiche, prägnant
- Komplexe Konzepte einfach erklären (keine inneren Monologe für User)
- Gib konkrete Zahlen & Werte an
- Erkläre "warum", nicht nur "wie"
- Deutsch sprechen
- Nutzer sollte actionable Antworten bekommen

## Wichtig

- Nutze den App-Kontext wenn vorhanden
- Gib Warnungen wenn etwas riskant ist (OOM, Performance-Probleme)
- Schlag Alternativen vor "Wenn das nicht funktioniert, versuche stattdessen..."
- Bei kritischen Fragen: "Das war meine Empfehlung basierend auf \${Kontext}, probiere es aus!"
`;

/**
 * Intelligente Dokumentations-Auswahl basierend auf User-Input
 * Das ist das "Extended Thinking" Pattern - wählt die beste Dokumentation
 */
export function getRelevantKnowledge(userMessage: string): KnowledgeSection[] {
  const lowerMessage = userMessage.toLowerCase();
  const selectedSections: KnowledgeSection[] = [];

  // Keywords analysieren und matching sections finden
  Object.values(KNOWLEDGE_SECTIONS).forEach(section => {
    const matchedKeywords = section.keywords.filter(kw =>
      lowerMessage.includes(kw)
    );
    
    // Wenn >= 1 Keyword match = relevante Section
    if (matchedKeywords.length > 0) {
      selectedSections.push(section);
    }
  });

  // Wenn keine Matches: Overview als Fallback
  if (selectedSections.length === 0) {
    return [KNOWLEDGE_SECTIONS.overview];
  }

  return selectedSections;
}

/**
 * Formatiert die ausgewählten Sections für System Prompt
 */
export function formatKnowledgeForContext(sections: KnowledgeSection[]): string {
  if (sections.length === 0) return '';

  const formattedSections = sections
    .map(s => `## ${s.title}\n\n${s.content}`)
    .join('\n\n---\n\n');

  return `\n\n### Relevante Dokumentation:\n\n${formattedSections}`;
}

/**
 * Table of Contents String für User-Referenz
 */
export function getTableOfContents(): string {
  const toc = Object.entries(KNOWLEDGE_TOC)
    .map(([key, title], idx) => `${idx + 1}. ${title}`)
    .join('\n');

  return `Verfügbare Dokumentationen:\n${toc}`;
}
