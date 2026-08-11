# FrameTrain Auto-Fix Pipeline – Setup & Betrieb

Automatischer Kreislauf: **App-Fehler → Manager → Triage-Agent baut Fix auf Branch →
Report im Manager → du klickst "Fix übernehmen" → Merge nach `main`.**

```
User nutzt App → Fehler → POST /api/app-errors  (automatisch, global)
      │
Cloud-Routine (nachts) ── /triage-errors ──► Branch auto/fix-* + PR + Report-Upload
      │
Manager-Seite "Auto-Fixes" ── Button "Fix übernehmen" ──► GitHub-Merge nach main
```

## Bausteine (schon gebaut)

**Manager** (`../../../Manager`, Cloudflare Worker + D1):
- `app_errors` erweitert: `error_group`, `triage_status`, `occurrences`, `screen`.
- Tabellen `fix_proposals`, `ignore_rules`.
- Endpoints: `GET /api/app-errors/export` (Bearer), `POST /api/fix-proposals` (Bearer),
  `GET /api/fix-proposals[/:id]`, `POST /api/fix-proposals/:id/approve|reject`,
  `DELETE /api/fix-proposals/:id`, `GET|POST /api/ignore-rules`, `DELETE /api/ignore-rules/:id`.
- Seite `public/fixes.html` ("Auto-Fixes" in der Sidebar) = dein Review-Zentrum.
- Approve bei `kind:"fix"` → mergt PR via GitHub-API. `ignore` → legt Ignore-Regel an.
  `report` → nur als erledigt markieren.

**Desktop-App** (dieses Repo):
- `src/utils/errorReport.ts`: `installGlobalErrorReporting()` meldet ab jetzt **jeden**
  uncaught Fehler + Promise-Rejection automatisch an den Manager (mit Screen + Version).
  In `main.tsx` verdrahtet, Screen-Kontext über `PageContext`.
- `.claude/commands/triage-errors.md`: der Triage-Command (Cloud-Routine **und** manuell).

## Einrichtung – Schritt für Schritt

### 1. Cloudflare / Manager (D1 bleibt, kein Supabase)
```bash
cd ../../../Manager       # in den Manager-Ordner

# Secret 1: gemeinsames Token für die Automation (Export + Upload)
openssl rand -hex 32      # ausgeben, kopieren
npx wrangler secret put AUTOMATION_SECRET     # Wert einfügen

# Secret 2: GitHub-PAT (siehe Schritt 2)
npx wrangler secret put GITHUB_TOKEN

# Worker + Frontend deployen
npx wrangler deploy
npx wrangler pages deploy public --project-name webcontrol-hq
```
> D1-Tabellen werden beim ersten Request automatisch angelegt (`ensure*`-Funktionen).
> `GITHUB_REPO` steht bereits in `wrangler.toml`. `DASHBOARD_PASSWORD` existiert schon.

### 2. GitHub Personal Access Token (fine-grained)
GitHub → Settings → Developer settings → **Fine-grained tokens** → Generate new:
- **Resource owner:** FrameSphere · **Repository access:** nur `FrameSphere/FrameTrain-App`
- **Permissions:** *Contents* = Read and write · *Pull requests* = Read and write
- Token kopieren → als `GITHUB_TOKEN` (Schritt 1) setzen.

Dasselbe Token braucht die Cloud-Routine zum PR-Öffnen (Schritt 4).

### 3. Desktop-App neu bauen & ausliefern
Damit das globale Error-Reporting bei den Usern ankommt, eine neue Version bauen:
```bash
cd -                      # zurück ins desktop-app Repo
npm run tauri:build
```
Version anheben nicht vergessen (`package.json` + `src-tauri/tauri.conf.json`).

### 4. Cloud-Routine = GitHub Actions (läuft ohne dass dein Mac an ist)
Gewählter Weg: der Workflow `.github/workflows/auto-triage.yml`. Er läuft nächtlich
in GitHubs Cloud (Cron `0 1 * * *` = ~03:00 Berlin) und ist manuell über den
Actions-Tab auslösbar (`workflow_dispatch`) — nutze das für den ersten Testlauf.

**Zwei Repo-Secrets** setzen unter GitHub → Repo `FrameTrain-App` → Settings →
Secrets and variables → Actions → *New repository secret*:
- `ANTHROPIC_API_KEY` — API-Key für den headless Claude-Code-Lauf (console.anthropic.com).
  ⚠️ Verursacht API-Kosten pro Lauf (Triage ist klein; ein Lauf = wenige Cents bis ~1 €
  je nach Fehlermenge/Modell).
- `MANAGER_AUTOMATION_SECRET` — **derselbe Wert** wie `AUTOMATION_SECRET` in Cloudflare.

**Kein GitHub-PAT im Workflow nötig:** das eingebaute `GITHUB_TOKEN` von Actions
erledigt `git push` + PR-Öffnen (über den `permissions`-Block im Workflow). Der PAT aus
Schritt 2 wird nur vom Cloudflare-Worker gebraucht (für den Merge-Button).

Erster Test: Actions-Tab → „Auto-Triage FrameTrain Errors" → *Run workflow*. Danach
erscheinen die Vorschläge unter **Auto-Fixes** im Manager.

> Alternative (lokal, nur wenn App offen): der Slash-Command `/triage-errors` manuell,
> mit Secrets im `env`-Block von `.claude/settings.local.json`. Der Aufsetz-Text unten
> beschreibt diesen Weg.

## Betrieb / Review-Loop
1. Routine läuft nachts → neue Vorschläge erscheinen unter **Auto-Fixes** im Manager.
2. Vorschlag öffnen: Report + Testschritte + zugrundeliegende Fehler lesen.
3. Kurz testen → **„Fix übernehmen & mergen"** (oder Ablehnen / Ignorieren bestätigen).
4. Beim Mergen werden die verknüpften Fehler automatisch auf `resolved` gesetzt.
5. Nächster Vorschlag.

Die **Ignore-Liste** (Tab in Auto-Fixes) wächst mit: bestätigte Ignorier-Vorschläge
filtern gleichartige Fehler künftig automatisch aus der Triage.

---

## Aufsetz-Chat Prompt (in neuem Chat einfügen)

> Kopiere alles zwischen den Linien in einen neuen Claude-Code-Chat, der in
> `FrameTrain/desktop-app` läuft.

═══════════════════════════════════════════════════════════════════════
Richte die geplante „Auto-Fix"-Routine für die FrameTrain-Desktop-App ein.

Kontext: In diesem Repo existiert bereits der Slash-Command `/triage-errors`
(`.claude/commands/triage-errors.md`). Er liest offene App-Fehler aus dem Manager
(Cloudflare Worker `webcontrol-hq-api`), clustert & kategorisiert sie, baut für
fixbare TS/React- und Training-Fehler einen Fix auf einem eigenen Branch `auto/fix-*`,
lässt `npm run build` als Gate laufen, öffnet einen PR und lädt pro Cluster einen
Review-Report in den Manager (`POST /api/fix-proposals`). Rust/Tauri-Fehler werden nur
analysiert (`kind:"report"`), Nicht-Bugs als Ignorier-Vorschlag gemeldet. Er mergt NIE
selbst und pusht NIE auf `main` — das macht der Mensch per Button im Manager.

Deine Aufgabe:
1. Lies `.claude/commands/triage-errors.md` vollständig und richte dich exakt danach.
2. Prüfe, dass die nötigen Secrets als Umgebungsvariablen verfügbar sind:
   `MANAGER_AUTOMATION_SECRET` und `GITHUB_TOKEN`. Fehlt eins, sag mir welches — nicht raten.
3. Mach EINEN Testlauf mit `--dry-run` (nur lesen/analysieren, keine Branches/PRs/Uploads)
   und zeig mir das Ergebnis: wie viele Fehler, wie du sie clustern/kategorisieren würdest.
4. Wenn der Dry-Run plausibel ist: richte eine geplante Routine ein, die
   `/triage-errors --limit 5` täglich um 03:00 Europe/Berlin ausführt (nutze die
   schedule-/Routine-Funktion von Claude Code). Base-Branch `main`,
   Repo `FrameSphere/FrameTrain-App`. Übergib der Routine dieselben Secrets.
5. Bestätige mir am Ende: Cadence, welche Fehler-Kategorien angefasst werden
   (ts-react + training = Fix, rust = nur Report), und dass niemals nach `main`
   gemergt oder gepusht wird.

Wichtig: konservativ bleiben (im Zweifel `report` statt `fix`), Secrets nie ausgeben,
und bei Unklarheit lieber nachfragen als etwas Riskantes automatisieren.
═══════════════════════════════════════════════════════════════════════
