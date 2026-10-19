// Die Dev-Vorlagen (train.py) sind deutsch kommentiert. In der englischen
// Oberflaeche wirkte das wie ein Uebersetzungsrest — hier werden Kommentare,
// Statusmeldungen und Fehlertexte der Vorlagen ersetzt. Code bleibt unberuehrt:
// ersetzt werden nur ganze Kommentar- bzw. Meldungstexte.
//
// Fuer Deutsch werden die ae/oe/ue-Umschriften in den Kommentaren durch
// Umlaute ersetzt (Python-Dateien sind UTF-8).

const EN: Array<[string | RegExp, string]> = [
  ['# Dieses Template laeuft so wie es ist. Aendere es nach Belieben.', '# This template runs as is. Change it as you like.'],
  ['# Dieses Template läuft so wie es ist. Ändere es nach Belieben.', '# This template runs as is. Change it as you like.'],
  ['# Fortschritt erscheint in FrameTrain, sobald eine JSON-Zeile mit', '# Progress shows up in FrameTrain as soon as a JSON line with'],
  ['# {"type": "progress", ...} auf stdout geschrieben wird - genau das macht', '# {"type": "progress", ...} is written to stdout - which is exactly what'],
  ['# der FrameTrainCallback weiter unten.', '# the FrameTrainCallback below does.'],
  ['# -- Pfade (von FrameTrain als ENV-Vars gesetzt) --------------------------', '# -- Paths (set by FrameTrain as env vars) -------------------------------'],
  ['# -- Dataset laden --------------------------------------------------------', '# -- Load dataset ---------------------------------------------------------'],
  ['# -- Modell & Tokenizer ---------------------------------------------------', '# -- Model & tokenizer ----------------------------------------------------'],
  ['# -- Fortschritt an FrameTrain melden -------------------------------------', '# -- Report progress to FrameTrain ----------------------------------------'],
  ['# -- Speichern ------------------------------------------------------------', '# -- Save -----------------------------------------------------------------'],
  ['"""Sendet ein Event an FrameTrain (eine JSON-Zeile pro Event)."""', '"""Sends an event to FrameTrain (one JSON line per event)."""'],
  ['# Sendet ein Event an FrameTrain (eine JSON-Zeile pro Event).', '# Sends an event to FrameTrain (one JSON line per event).'],
  ['# state.epoch laeuft von 0.0 hoch - 1-basiert melden wie die Engine', '# state.epoch counts up from 0.0 - report 1-based like the engine'],
  ['# -1 = ganzes Dataset. Klein halten beim Ausprobieren.', '# -1 = whole dataset. Keep it small while trying things out.'],
  ['# 0 = kompletten Eval-Split auswerten', '# 0 = evaluate the whole eval split'],
  ['# None = automatisch erkennen', '# None = detect automatically'],
  ['FrameTrain liefert DATASET_PATH als lokalen Ordner.', 'FrameTrain passes DATASET_PATH as a local folder.'],
  ['- sonst: JSON/JSONL/CSV/TSV/Parquet im Split-Layout train/ val/ test/', '- otherwise: JSON/JSONL/CSV/TSV/Parquet in the split layout train/ val/ test/'],
  ['"""Text- und Label-Spalte erraten, falls nicht oben festgelegt."""', '"""Guess the text and label columns unless set above."""'],
  ['# Eval-Split deckeln: eine Auswertung ueber 25.000 Beispiele dauert laenger', '# Cap the eval split: evaluating 25,000 examples takes longer'],
  ['# als das ganze Training mit MAX_STEPS. Zufaellig ziehen, nicht die ersten N —', '# than the whole training with MAX_STEPS. Sample randomly, not the first N —'],
  ['# viele Splits sind nach Label sortiert und ergaeben sonst nur eine Klasse.', '# many splits are sorted by label and would otherwise yield a single class.'],
  ['# Dateien samt Klasse (Ordnername) einsammeln — leere Klassenordner zaehlen nicht.', '# Collect files with their class (folder name) — empty class folders do not count.'],
  ['# FrameTrain stellt dieselbe Dataset-Logik wie im Training bereit: Klassenordner,', '# FrameTrain provides the same dataset logic as in training: class folders,'],
  ['# Klassennamen, damit Labor und Tests spaeter Namen statt 0/1 zeigen', '# Class names, so the laboratory and tests show names instead of 0/1'],
  ['# Adapter mit dem Basismodell zusammenführen, damit Tests und Hosting ein', '# Merge the adapter into the base model so tests and hosting can load a'],
  ['# normales Modell laden können.', '# regular model.'],
  ['"""Eine Zeile als Trainingstext: Chat, Prompt/Antwort oder reiner Text."""', '"""One row as training text: chat, prompt/answer or plain text."""'],
  ['# Trainiert das YOLO-Modell mit Ultralytics. Fortschritt erscheint in FrameTrain,', '# Trains the YOLO model with Ultralytics. Progress shows up in FrameTrain'],
  ['# sobald eine JSON-Zeile mit {"type": "progress", ...} auf stdout geschrieben wird.', '# as soon as a JSON line with {"type": "progress", ...} is written to stdout.'],
  ['# Wertet das YOLO-Modell auf dem Test-Split aus (sonst Val) und schreibt', '# Evaluates the YOLO model on the test split (otherwise val) and writes'],
  ['# den Bericht nach OUTPUT_PATH/results.json.', '# the report to OUTPUT_PATH/results.json.'],
  ['# Ultralytics bekommt die dataset.yaml (DATASET_YAML), nicht den Ordner.', '# Ultralytics gets the dataset.yaml (DATASET_YAML), not the folder.'],
  ['# FrameTrain prueft die yaml vor dem Start: path, train und val stimmen.', '# FrameTrain checks the yaml before starting: path, train and val are valid.'],
  ['# MODEL_PATH ist meist ein Ordner mit einer oder mehreren .pt-Dateien.', '# MODEL_PATH is usually a folder with one or more .pt files.'],
  ['# Fuer Objekterkennung Gewichte ohne -seg/-pose/-cls/-obb bevorzugen.', '# For object detection prefer weights without -seg/-pose/-cls/-obb.'],
  ['# Laeuft nach Training UND Validierung einer Epoche.', '# Runs after training AND validation of an epoch.'],
  ['# Ultralytics feuert nach der finalen Validierung noch einmal', '# Ultralytics fires once more after the final validation'],
  ['message="Daten werden geladen ..."', 'message="Loading data ..."'],
  ['message="Dataset wird geladen ..."', 'message="Loading dataset ..."'],
  ['message="Modell wird geladen ..."', 'message="Loading model ..."'],
  ['message="Training laeuft ..."', 'message="Training ..."'],
  ['message="Training läuft ..."', 'message="Training ..."'],
  ['print("Dataset geladen:", ', 'print("Dataset loaded:", '],
  ['print(f"Modell gespeichert unter ', 'print(f"Model saved to '],
  ['print(f"Bericht gespeichert unter ', 'print(f"Report saved to '],
  ['print(f"Modell geladen von ', 'print(f"Model loaded from '],
  ['print(f"Eval-Split auf {MAX_EVAL} Beispiele begrenzt"', 'print(f"Eval split capped at {MAX_EVAL} examples"'],
  ['print(f"Spalten: text=', 'print(f"Columns: text='],
  [', Klassen={num_labels}"', ', classes={num_labels}"'],
  [' Trainings- / {len(eval_rows)} Eval-Beispiele | ', ' train / {len(eval_rows)} eval examples | '],
  [' Trainings- / {len(eval_items)} Eval-Dateien | Klassen: ', ' train / {len(eval_items)} eval files | classes: '],
  ['print(f"Gewichte:     {weights}"', 'print(f"Weights:      {weights}"'],
  ['print(f"Werte Split \'{split}\' aus"', 'print(f"Evaluating split \'{split}\'"'],
  ['f"DATASET_PATH existiert nicht: ', 'f"DATASET_PATH does not exist: '],
  ['f"Keine Daten-Dateien in {path} gefunden."', 'f"No data files found in {path}."'],
  ['f"Keine Trainingsdaten in {DATASET_PATH} gefunden."', 'f"No training data found in {DATASET_PATH}."'],
  ['f"Quell-/Zielspalte nicht gefunden. Vorhanden: {cols}"', 'f"Source/target column not found. Available: {cols}"'],
  ['f"Keine passende Spalte gefunden. Vorhanden: {cols}"', 'f"No suitable column found. Available: {cols}"'],
  ['f"Text-/Label-Spalte nicht gefunden. Vorhanden: {cols}. "', 'f"Text/label column not found. Available: {cols}. "'],
  ['"Setze TEXT_COL / LABEL_COL oben im Script."', '"Set TEXT_COL / LABEL_COL at the top of the script."'],
  ['f"Keine .pt-Gewichte in {MODEL_PATH}"', 'f"No .pt weights in {MODEL_PATH}"'],
  ['f"Kein Checkpoint in {weights_dir} - das Training hat nichts gespeichert."', 'f"No checkpoint in {weights_dir} - the training saved nothing."'],
  [/f"Label-Spalte '\{label_col\}' hat nur einen Wert - damit laesst sich "\n(\s*)"keine Klassifikation trainieren\."/,
    'f"Label column \'{label_col}\' has a single value - no classifier "\n$1"can be trained on that."'],
  [/Keine (\w+)dateien mit Klassenordnern in/g, 'No $1 files with class folders in'],
  [/HF-Parquet mit (\w+)spalte/g, 'HF Parquet with a $1 column'],
];

const DE_UMLAUTS: Array<[RegExp, string]> = [
  [/laeuft/g, 'läuft'], [/Laeuft/g, 'Läuft'], [/Aendere/g, 'Ändere'], [/aendere/g, 'ändere'],
  [/ueber/g, 'über'], [/laenger/g, 'länger'], [/Zufaellig/g, 'Zufällig'], [/ergaeben/g, 'ergäben'],
  [/zaehlen/g, 'zählen'], [/spaeter/g, 'später'], [/prueft/g, 'prüft'], [/Fuer /g, 'Für '],
  [/laesst/g, 'lässt'],
];

/** Vorlage fuer die Sprache der Oberflaeche aufbereiten. */
export function localizeScript(script: string, language: string): string {
  if (language === 'en') {
    let out = script;
    for (const [from, to] of EN) {
      out = typeof from === 'string' ? out.split(from).join(to) : out.replace(from, to);
    }
    return out;
  }
  // Nur Kommentar- und Meldungszeilen anfassen, nie Bezeichner im Code.
  return script.split('\n').map(line => {
    const isText = /^\s*#/.test(line) || /message="|print\(f?"|raise \w+\(f?"|^\s*f?"/.test(line);
    if (!isText) return line;
    return DE_UMLAUTS.reduce((l, [re, to]) => l.replace(re, to), line);
  }).join('\n');
}
