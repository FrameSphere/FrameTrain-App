# Bericht Strang B – Rust-Transport + Einstellungs-Tab "KI-Verbindungen"

Branch: `claude/mcp-transport-3e508b` (nicht gemergt, nicht gepusht, kein Release).
Vertrag: `docs/mcp/CONTRACT.md` (Kopie des verbindlichen Vertrags, Abschnitte 3, 4, 5, 7).

## Was fertig ist

### Rust: `src-tauri/src/mcp/`
| Datei | Inhalt |
|---|---|
| `mod.rs` | `McpState`, `setup()` (aus `main` `.setup`), Tauri-Commands, Claude-Desktop-Config-Merge |
| `bridge.rs` | Manifest-State, Event `mcp://call` an Fenster `main`, Antwort per `mcp_tool_result`, 60-s-Timeout, 15-s-Warten aufs Manifest (zeigt dabei das Hauptfenster), `system_ping` als Fallback, Aufruf-Protokoll (letzte 50) |
| `server.rs` | Streamable HTTP auf `127.0.0.1:<port>/mcp` (Standard 47861), JSON-RPC `initialize` / `notifications/initialized` / `ping` / `tools/list` / `tools/call` (+ leere `resources/list`, `prompts/list`), Bearer-Token, Host-Pruefung (gleiche Funktion wie `hosting_api.rs`), Origin-Pruefung, `GET /health` ohne Token, `GET /mcp` -> 405, `Mcp-Session-Id` nur um den Client-Namen ueber Anfragen zu kennen |
| `stdio.rs` | `--mcp-stdio`: zeilenweise stdin -> HTTP -> stdout; startet die App (macOS `open -g -b <APP_ID>`, sonst eigenes Binary, `.no_window()`), wartet bis 30 s; reicht `Mcp-Session-Id` durch; liest Token bei 401 neu |
| `relay.rs` | Nur bei `remoteEnabled`: Heartbeat 30 s (+ `/manifest` bei `wantManifest`), Long-Poll-Schleife, Jobs durch dieselbe Bridge mit `source: "remote"`, `/result`, Backoff 2 -> 60 s, Generationszaehler zum sauberen Stoppen |
| `config.rs` | `<app_data>/mcp.json` `{ enabled, port, token, deviceId, remoteEnabled }` (Standard: beides aus), Token `ftmcp_<32 hex>`, atomar geschrieben, unter Unix 0600 |

`main()` zweigt bei `--mcp-stdio` als Allererstes ab (vor dotenv, DB, Tauri-Builder) – nichts ausser JSON-RPC geht auf stdout.
**Single-Instance-Plugin:** ist in diesem Projekt nicht eingebunden (`Cargo.toml`), stoert also nicht. Weil vor dem Builder abgezweigt wird, waere es auch sonst nicht beteiligt.

API-Key fuer das Relay: dieselbe Quelle wie der Auto-Login – `app_config_dir/config.json`, Feld `api_key` (siehe `App.tsx` `save_config`/`load_config`). `validate-credentials`/`accept_terms` bekommen ihn vom Frontend, Rust liest ihn hier selbst.

`database::get_data_directory()` ist jetzt `pub(crate)` (fuer den mcp.json-Pfad ohne AppHandle, auch in der Stdio-Bruecke).

### Tauri-Commands (Webview-Seite)
| Command | Args (invoke) | Rueckgabe |
|---|---|---|
| `mcp_register_manifest` | `{ tools: Tool[] }` | Anzahl Werkzeuge inkl. `system_ping` |
| `mcp_tool_result` | `{ callId, result: CallToolResult }` | `true`, wenn die callId noch offen war |
| `mcp_get_status` | – | Config, Server-/Relay-Status, letzte Aufrufe, Claude-Desktop-Status, `exePath` |
| `mcp_save_settings` | `{ enabled, port, remoteEnabled }` | Status |
| `mcp_rotate_token` | – | Status |
| `mcp_connect_claude_desktop` | – | `{ configPath, backupPath, status }`; Fehler `CLAUDE_NOT_FOUND` / `CLAUDE_CONFIG_INVALID: …` |
| `mcp_remote_status` | – | Antwort von `/api/desktop/mcp/status` |
| `mcp_remote_disconnect` | `{ connectionId }` | Antwort von `/api/desktop/mcp/disconnect` |

Event an die Webview: `mcp://call` mit `{ callId, tool, args, source, client: { name, version? } }` (nur an Fenster `main`).

### Einstellungs-Tab
`src/components/settings/AIConnectionsSettings.tsx` + `mcpConnections.ts`, eingehaengt in `Settings.tsx` als Tab `ai-connections` (Icon `Cable`).
Inhalt: Schalter lokaler Zugang, Status/URL, Port, Token (verdeckt, zeigen, kopieren, neu erzeugen mit Rueckfrage im `ModalPortal`),
"Mit Claude Desktop verbinden" (Backup + Merge, schaltet den Endpunkt ein, Haken wenn eingetragen, Hinweis bei fehlendem Claude und bei App-Translocation),
Claude-Code-Befehl, Cursor-JSON, Schalter Fernzugriff mit Relay-Status, Connector-URL, verbundene Clients + letzte Aufrufe aus `mcp_remote_status`, Trennen-Knopf (nur wenn die Website eine `id` liefert),
"Letzte Aufrufe" auf diesem Rechner, "Freigaben je Client" mit Widerrufen. Texte DE+EN unter `mcp.settings.*` und `settings.tabs.aiConnections`. Keine Emojis.

**Freigaben (Strang A):** `src/mcp/approvals.ts` gab es beim Bauen noch nicht. Der Tab laedt es optional per
`import.meta.glob('../../mcp/approvals.ts')` und erwartet:
```ts
listApprovals(): McpApproval[] | Promise<McpApproval[]>      // { client, tool, grantedAt? }
revokeApproval(client: string, tool: string): void | Promise<void>
onApprovalsChanged?(cb: () => void): () => void              // optional
```
Fehlt das Modul oder die Funktionen, zeigt der Tab "Keine dauerhaften Freigaben". Notiz dazu liegt im gemeinsamen Notizordner (`B-approvals-interface.md`).

## Tests
- `cargo test mcp::` – 25 Tests: JSON-RPC (initialize/Versionen, Notifications, ping, tools/list, tools/call, Fehlercodes), HTTP-Schicht (Host, Origin, Token, 401/403/404/405/202, Sitzung, DELETE), echter Socket-Roundtrip, Bridge (Roundtrip, Timeout, Manifest-Warten, unbekanntes Werkzeug), Config (Erzeugen, Teil-/Kaputt-Dateien), Claude-Config-Merge (leer, bestehende Eintraege bleiben, kaputte Datei wird nicht ueberschrieben, Backup), Relay (Backoff, API-Key, Jobs).
- `npx vitest run src/components/__tests__/aiConnections.test.tsx` – Hilfen + Tab (Status, Token verdeckt, Schalter speichern, Remote-Clients/Trennen, Claude-Desktop-Verbinden, fehlendes Claude, Token-Rueckfrage im Portal, Freigaben widerrufen).
- Gesamtlauf (2026-11-04): `cargo test` 261 ok / 1 ignoriert, `cargo check` ok, `npx vitest run` 92 Dateien / 1042 Tests ok, `npx tsc --noEmit` ok.
  (Hinweis: `package.json` hat kein `test`-Skript – `npm test` gibt es nicht, Vitest laeuft per `npx vitest run`.)

### Manuell gelaufen (isolierter Debug-Build, `FRAMETRAIN_APP_ID=com.frametrain.mcptest`, Port 47871)
- Stdio-Bruecke bei **geschlossener** App: startet die App selbst, Antworten auf `initialize`, `tools/list`, `tools/call system_ping` kamen als je eine JSON-Zeile auf stdout, auf stderr nur "starte die App". Gesamt 16,5 s (Start + 15 s Manifest-Warten, da ohne Strang A kein Manifest kommt).
- curl: `/health` 200, `initialize` 200 mit `Mcp-Session-Id`, Notification 202, `tools/list` -> `[system_ping]`, `tools/call models_list` -> `isError` "App startet noch", falscher Token 401, fremder Host 403, fremder Origin 403, `GET /mcp` 405. Socket lauscht nur auf 127.0.0.1.
- Nicht manuell getestet: echtes Claude Desktop (Config-Merge nur per Unit-Test, deine echte Claude-Config wurde nicht angefasst), Relay (Website fehlt).

## Manuell testen

App starten (oder isoliert: `FRAMETRAIN_APP_ID=com.frametrain.mcptest cargo build` – eigene Datenordner), in Einstellungen > KI-Verbindungen den lokalen Zugang einschalten, dann:

```bash
TOKEN=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/Library/Application Support/com.frametrain.desktop/mcp.json')))['token'])")
curl -s http://127.0.0.1:47861/mcp -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"1"}}}'
curl -s http://127.0.0.1:47861/mcp -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
curl -s http://127.0.0.1:47861/mcp -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"system_ping","arguments":{}}}'
```

Stdio-Bruecke per Pipe (App darf dabei geschlossen sein, sie wird gestartet):
```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"pipe","version":"1"}}}' '{"jsonrpc":"2.0","method":"notifications/initialized"}' '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | /Applications/FrameTrain.app/Contents/MacOS/FrameTrain --mcp-stdio
```
Erwartet: genau zwei JSON-Zeilen auf stdout, Meldungen nur auf stderr.

Claude Desktop: Knopf "Mit Claude Desktop verbinden", Claude Desktop beenden/neu oeffnen, in Claude nach FrameTrain-Werkzeugen schauen. Backup liegt neben der Config (`claude_desktop_config.json.bak-frametrain-<zeit>`).

## Offene Punkte
1. **Zusammenspiel mit Strang A** erst nach dem Merge pruefbar: Manifest-Registrierung, `mcp://call`-Listener, Freigabe-Dialog. Bis dahin liefert `tools/list` nur `system_ping`, andere Aufrufe laufen nach 15 s (Manifest fehlt) mit "App startet noch" aus.
2. **Relay gegen Strang C** ungetestet (Website-Routen existieren noch nicht). Feldnamen exakt nach Vertrag Abschnitt 5. `/status` liefert laut Vertrag keine `id` je Verbindung – fuer "Trennen" braucht der Tab sie (`connections[].id`); bitte in Strang C mitliefern oder Vertrag ergaenzen.
3. **Manifest-Warten bei `tools/list`** nur in den ersten 60 s nach App-Start (danach sofort Fallback), damit ein Client nicht bei jeder Liste 15 s haengt, falls die Webview kein Manifest schickt.
4. **Windows:** Release-Binary hat `windows_subsystem = "windows"`. Pipes eines Elternprozesses (Claude Desktop) werden trotzdem geerbt, sollte also funktionieren – auf Windows ungetestet. Claude Desktop aus dem Microsoft Store (MSIX) legt die Config evtl. unter `%LOCALAPPDATA%\Packages\Claude_…\LocalCache\Roaming\Claude` ab; nur `%APPDATA%\Claude` ist abgedeckt.
5. **macOS Ad-hoc-Signatur:** Claude Desktop startet das App-Binary direkt (`command` = Pfad im Bundle). Nach jedem Release aendert sich die Signatur; der Pfad bleibt gleich, das sollte reichen. Liegt die App in einem Translocation-Ordner, warnt der Tab.
6. **Sitzungen** werden nicht persistiert: nach App-Neustart unbekannte `Mcp-Session-Id` wird grosszuegig akzeptiert (Client-Name dann "unknown", bis der Client neu initialisiert). Strenger waere 404 laut Spezifikation.
7. Keine SSE-Streams (Vertrag: nicht noetig); Server-initiierte Nachrichten (`notifications/tools/list_changed`) gibt es daher nicht – `listChanged: false`.
