# Changelog Automation für FrameTrain Desktop

Diese Vorbereitung macht die Desktop-App für eine Codex-Automation lesbar.
Die Automation soll nicht manuell in Dateien suchen, sondern dieses Script nutzen:

```bash
cd desktop-app
npm run changelog:prepare
```

## Was das Script liefert

Das Script `scripts/changelog-context.mjs` gibt entweder JSON oder Markdown aus.

Standard:

```bash
npm run changelog:prepare
```

JSON mit:

- `source`: immer `desktop-app`
- `version`: aus `package.json`, `tauri.conf.json` und `Cargo.toml`
- `range`: Git-Range für den Änderungszeitraum
- `commitCount`: Anzahl der Commits im Zeitraum
- `commits`: Liste der Commit-Titel mit grober Typisierung
- `files`: betroffene Dateien, gefiltert auf relevante App-Dateien

Markdown:

```bash
CHANGELOG_OUTPUT=markdown npm run changelog:prepare
```

## Empfohlene Automation-Logik

1. Hole den letzten veröffentlichten Changelog-Zeitpunkt oder die letzte changelog-relevante Ref.
2. Setze `CHANGELOG_BASE_REF` auf diese Ref.
3. Führe `npm run changelog:prepare` im `desktop-app/`-Ordner aus.
4. Formatiere aus dem JSON einen Website-Post.
5. Poste an die Website-API `POST /api/status-updates`.

## GitHub Actions Setup

Die fertige Automation läuft als Workflow in `.github/workflows/codex-desktop-changelog.yml`.

Zeitplan:

- alle 2 Tage per `cron`
- zusätzlich manuell per `workflow_dispatch`

Benötigte Secrets:

- `CODEX_API_KEY`: schützt `POST /api/status-updates`
- `WEBSITE_API_URL`: optional, wenn die Website nicht unter der Standard-URL läuft

Wie die Referenz gewählt wird:

- Wenn du beim manuellen Start `base_ref` angibst, wird genau diese Ref benutzt.
- Sonst liest der Workflow den letzten `status-updates`-Eintrag mit
  `source=desktop-app` und `scope=changelog`.
- Falls es noch keinen solchen Eintrag gibt, fällt der Workflow auf den letzten Tag bzw. die aktuelle `HEAD`-Basis zurück.

Bezug zu `./release.sh`:

- `./release.sh` erzeugt neue Tags wie `v1.2.6`.
- Der Changelog-Workflow erkennt diese Versionen automatisch über Git.
- Nach einem Release ist das der natürliche Startpunkt für den nächsten Änderungszeitraum.

## Empfohlene Env-Variablen

- `CHANGELOG_BASE_REF`: Start-Ref für den Zeitraum, z. B. letzter Tag oder letzter Publish-Commit
- `CHANGELOG_HEAD_REF`: optional, standardmäßig `HEAD`
- `CHANGELOG_OUTPUT`: `json` oder `markdown`

## Website-Post

Für die Website kannst du den Output direkt als `StatusUpdate` senden:

```json
{
  "title": "Desktop-App: Änderungen seit dem letzten Changelog",
  "body": "### Was wurde geändert\n- ...",
  "type": "dev",
  "appVersion": "1.2.5",
  "author": "Codex"
}
```

## Gute Praxis

- Verwende die Git-Range als Quelle der Wahrheit, nicht nur Dateidaten.
- Fasse Commits in menschliche Kategorien zusammen.
- Beschränke die Ausgabe auf die wichtigsten Änderungen.
- Poste nur, wenn es neue Desktop-App-Änderungen seit dem letzten Changelog gibt.
