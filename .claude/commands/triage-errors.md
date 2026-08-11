---
description: Liest FrameTrain-App-Fehler aus dem Manager, clustert sie, baut Fixes auf eigenen Branches, öffnet PRs und lädt Review-Reports in den Manager hoch. Merged NIE nach main.
argument-hint: "[--dry-run] [--limit N]"
allowed-tools: Bash, Read, Edit, Write, Grep, Glob
---

# FrameTrain Auto-Triage

Du bist der Triage-Agent für die FrameTrain-Desktop-App. Dein Job: offene Fehler-Reports
aus dem Manager holen, verstehen, clustern, kategorisieren, für die fixbaren einen
Code-Fix auf einem **eigenen Branch** bauen, einen **PR** öffnen und pro Cluster einen
**Review-Report** in den Manager hochladen. Der Mensch entscheidet im Manager per Button,
ob gemergt wird. **Du mergst niemals selbst und pushst niemals auf `main`.**

## Feste Rahmenbedingungen (nicht abweichen)

- Repo: `FrameSphere/FrameTrain-App`, Base-Branch: `main`.
- Verifikations-Gate: `npm run build` (= `tsc && vite build`). Es gibt kein Test-Script.
- Diese Werte kommen aus Umgebungsvariablen (nie hardcoden, nie loggen):
  - `MANAGER_BASE`  – Default `https://webcontrol-hq-api.karol-paschek.workers.dev`
  - `MANAGER_AUTOMATION_SECRET` – Bearer-Token für die Automation-Endpoints
  - `GITHUB_TOKEN`  – für PR-Erstellung (oder `gh` CLI, falls vorhanden)
- Argumente: `--dry-run` = analysieren + Reports, aber **keine** Branches/PRs/Uploads.
  `--limit N` = höchstens N Fehler-Cluster in diesem Lauf bearbeiten (Default 5).

## Ablauf

### 1. Fehler holen
```bash
: "${MANAGER_BASE:=https://webcontrol-hq-api.karol-paschek.workers.dev}"
curl -s -H "Authorization: Bearer $MANAGER_AUTOMATION_SECRET" \
  "$MANAGER_BASE/api/app-errors/export?site_id=frametrain&limit=200"
```
Antwort: `{ errors: [...], ignore_rules: [...], total_open, filtered_by_ignore }`.
Jeder Fehler hat u.a.: `id`, `error_type`, `title`, `message`, `details`, `logs`,
`config_snapshot`, `screen`, `error_group`, `occurrences`, `created_at`.

Wenn `errors` leer ist → kurzer Abschlussbericht "nichts zu tun", fertig.

### 2. Clustern
Gruppiere Fehler mit gleichem `error_group` (oder klar gleicher Ursache) zu **einem**
Cluster. Ein Cluster = ein Vorschlag = ein Branch/PR. Sortiere Cluster nach Häufigkeit
(`occurrences`, Anzahl Fehler) — die häufigsten zuerst. Bearbeite max. `--limit` Cluster.

### 3. Kategorisieren & Aktion wählen
Bestimme pro Cluster `category` und die Aktion:

| Signal im `error_type` / Stack | category | Aktion |
|---|---|---|
| `synapse:*`, `runtime:*`, UI, generisches TS/React in `src/**` | `ts-react` | **Fix** versuchen |
| `training:*`, `devtrain:*` mit Ursache in `src/**` oder `src-tauri/python/**` | `training` | **Fix** versuchen |
| Ursache in `src-tauri/src/**` (Rust/Tauri) | `rust-report` | **nur Report** (kind=`report`), kein Code-Fix |
| Kein echter Bug (Env-/Netzwerk-/User-Fehler, nicht reproduzierbar, irrelevant) | (passend) | **Ignore-Vorschlag** (kind=`ignore`) |

Untersuche für jede Fix-Entscheidung zuerst den echten Code (Grep/Read), belege die
Ursache. Rate nicht. Wenn du die Ursache im Code nicht sicher findest → `report` statt `fix`.

### 4. Fix bauen (nur category `ts-react` / `training`)
Pro Cluster:
```bash
git fetch origin main
git switch -c auto/fix-<category>-<kebab-kurzbeschreibung>-$(date +%Y%m%d) origin/main
```
- Minimalen, gezielten Fix umsetzen. Keine unnötigen Umbauten, Stil der Umgebung treffen.
- **Build-Gate:** `npm run build` muss grün sein.
  - Grün → `build_status: "passed"`.
  - Rot und nicht schnell lösbar → Branch verwerfen (`git switch main` + Branch löschen),
    Cluster auf **`report`** herabstufen (Analyse + Fix-Idee, aber kein Merge-Kandidat).
- Commit:
```bash
git add -A
git commit -m "fix(<category>): <kurz> [auto-triage]

Behebt App-Fehler-Gruppe <error_group>.
Fehler-IDs: <ids>

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push -u origin HEAD
```
- PR öffnen (nach `main`). Mit `gh` falls vorhanden:
```bash
gh pr create --base main --head "$(git branch --show-current)" \
  --title "Auto-Fix: <titel>" --body "<kurzer PR-Text, Link zu Fehler-IDs>"
```
  Sonst per API:
```bash
curl -s -X POST -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/FrameSphere/FrameTrain-App/pulls \
  -d "{\"title\":\"Auto-Fix: <titel>\",\"head\":\"<branch>\",\"base\":\"main\",\"body\":\"<text>\"}"
```
  Merke dir `pr_number` und `pr_url` (`html_url`) aus der Antwort.

### 5. Report in den Manager hochladen
Pro Cluster **einen** POST. Für `fix` sind `branch` + `pr_number` Pflicht.
```bash
curl -s -X POST -H "Authorization: Bearer $MANAGER_AUTOMATION_SECRET" \
  -H "Content-Type: application/json" \
  "$MANAGER_BASE/api/fix-proposals" -d @- <<'JSON'
{
  "site_id": "frametrain",
  "kind": "fix",
  "category": "ts-react",
  "title": "<kurzer, klarer Titel>",
  "summary": "<1 Satz: was war kaputt, was tut der Fix>",
  "root_cause": "<die belegte Ursache>",
  "report_markdown": "<voller Report, siehe Vorlage unten>",
  "test_steps": "<nummerierte, konkrete manuelle Testschritte>",
  "diff_summary": "<welche Dateien, was geändert>",
  "files_changed": ["src/..."],
  "error_ids": [<ids>],
  "error_group": "<error_group>",
  "branch": "<branch>",
  "base_branch": "main",
  "pr_number": <n>,
  "pr_url": "<html_url>",
  "build_status": "passed",
  "risk": "low|medium|high"
}
JSON
```
Für `kind:"report"` und `kind:"ignore"` gilt dasselbe, nur ohne `branch`/`pr_number`
(bei `ignore` `error_group` mitgeben — daraus wird bei Bestätigung die Ignore-Regel).

### 6. Abschluss
Gib mir am Ende eine kompakte Zusammenfassung:
- Wie viele Fehler / Cluster, wie aufgeteilt (fix / report / ignore).
- Pro Fix: Titel, Branch, PR-Link, Risiko, Build-Status.
- Was besondere Aufmerksamkeit beim Test braucht.

## Report-Vorlage (`report_markdown`)
Schreib ihn so, dass Karol ohne Code-Kontext entscheiden kann:
```
## Was ist passiert
Auf **<screen>** trat <N>× dieser Fehler auf, ausgelöst durch <Aktion des Users>.

## Ursache
<Konkret, mit Datei:Zeile.>

## Was der Fix ändert
<In einfachen Worten. Welche Datei(en), welches Verhalten jetzt.>

## Bitte prüfen
1. <konkreter Schritt in der App>
2. <erwartetes Verhalten>
3. <Regressions-Check, was nicht kaputtgehen darf>

## Risiko
<niedrig/mittel/hoch + warum>
```

## Guardrails
- Niemals auf `main` committen/pushen. Niemals selbst mergen. Nur Branch + PR + Report.
- Secrets nie ausgeben/loggen. Bei fehlendem `MANAGER_AUTOMATION_SECRET` abbrechen und melden.
- Keine Fehler doppelt bearbeiten: der Export liefert nur `triage_status='new'`; sobald ein
  Vorschlag hochgeladen ist, verschwinden die Fehler automatisch aus dem Export.
- Bei `--dry-run`: nur Schritte 1–3 + Report an mich im Chat, **keine** Writes/Pushes/Uploads.
- Konservativ bleiben: im Zweifel `report` statt `fix`. Lieber ein guter Fix als fünf riskante.
