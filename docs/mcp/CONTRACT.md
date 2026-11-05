# FrameTrain MCP – gemeinsamer Vertrag (v1, 2026-11-04)

Gilt fuer drei parallele Arbeitsstraenge. Jeder Strang haelt sich an diese Schnittstellen,
damit die Teile ohne Nacharbeit zusammenpassen. Aenderungen am Vertrag nur nach Ruecksprache
mit Karol bzw. der Orchestrator-Session.

```
Claude Desktop/Code, Cursor ──stdio/HTTP──▶ [Desktop: lokaler MCP]  ─┐
                                                                       ├─▶ Bridge ─▶ Webview: Tool-Registry (TS)
ChatGPT, Claude.ai ──HTTPS+OAuth──▶ [Website: /api/mcp] ─Relay─▶ [Desktop: Relay-Client] ─┘
```

Die Werkzeuge laufen IMMER in der Desktop-App (lokale GPU, lokale Daten). Die Website ist nur
Auth + Vermittler, sie fuehrt nie selbst Training aus.

---

## 1. Werkzeug-Manifest (Format)

Ein Werkzeug = MCP-Tool-Definition plus FrameTrain-Zusatzfeld:

```json
{
  "name": "datasets_list",
  "title": "Datensaetze auflisten",
  "description": "Englisch, praezise, fuer LLMs geschrieben: was, wann benutzen, was kommt zurueck.",
  "inputSchema": { "type": "object", "properties": { }, "additionalProperties": false },
  "annotations": { "readOnlyHint": true, "destructiveHint": false, "openWorldHint": false },
  "_meta": { "frametrain/domain": "datasets", "frametrain/approval": "none" }
}
```

- `name`: snake_case, Praefix = Domaene (`models_`, `datasets_`, `studio_`, `training_`, `synapse_`,
  `devtrain_`, `lab_`, `tests_`, `analysis_`, `hosting_`, `system_`). Max 64 Zeichen.
- `frametrain/approval`: `none` (lesen) | `once` (erster Aufruf je Client fragt, danach merken) |
  `always` (jedes Mal fragen: loeschen, Training/GPU starten, Dateien ueberschreiben).
- Beschreibungen auf Englisch (beide KI-Plattformen), UI-Texte in der App DE+EN ueber Locales.
- Ergebnis = MCP `CallToolResult`: `{ content: [{type:"text", text}], structuredContent?, isError? }`.
  `structuredContent` immer mitliefern, wenn es Daten sind; `text` ist eine kurze Zusammenfassung.
- Lang laufende Dinge (Training, Download, Test, Export) geben sofort eine `job_id` zurueck,
  Status ueber eigenes `*_status`-Werkzeug. Kein Aufruf blockiert laenger als 60 s.
- Dateipfade: nur innerhalb der FrameTrain-Datenordner oder vom Nutzer per Dialog bestaetigt.

## 2. Werkzeug-Umfang (Ziel, Strang A darf verfeinern)

| Domaene | Werkzeuge (Beispiele) |
|---|---|
| system | `system_info` (Version, GPU/VRAM/RAM, Python-Env-Status), `system_capabilities` |
| models | `models_list`, `models_info`, `models_search_hf`, `models_download_hf` (Job), `models_import_local`, `models_rename`, `models_delete` |
| datasets | `datasets_list`, `datasets_info`, `datasets_files`, `datasets_preview`, `datasets_search_hf`, `datasets_download_hf` (Job), `datasets_import_local`, `datasets_split`, `datasets_delete` |
| studio (Dataset Builder) | `studio_projects_list`, `studio_project_create`, `studio_project_update`, `studio_samples_list`, `studio_add_texts`, `studio_add_image`, `studio_import_folder/_text/_audio/_video`, `studio_fetch_web`, `studio_set_annotation`, `studio_suggest`, `studio_review`, `studio_quality`, `studio_near_duplicates`, `studio_stats`, `studio_export` (-> Datensatz) |
| training | `training_presets`, `training_check_requirements`, `training_estimate_resources`, `training_start`, `training_status`, `training_stop`, `training_history`, `training_rate_config` |
| synapse (eigener Modellbau) | `synapse_node_catalog`, `synapse_graph_get`, `synapse_graph_set` (ganzer Graph als IR), `synapse_graph_edit` (add/remove/connect/update Ops), `synapse_validate` (Shape-Diagnose), `synapse_generate_code`, `synapse_create_model`, `synapse_train` |
| devtrain | `devtrain_context`, `devtrain_script_get`, `devtrain_script_set`, `devtrain_start`, `devtrain_status`, `devtrain_stop`, `devtrain_test_start`, `devtrain_test_stop` |
| lab | `lab_server_start`, `lab_server_status`, `lab_infer`, `lab_sessions`, `lab_stats` |
| tests | `tests_start`, `tests_status`, `tests_results`, `tests_single_input`, `tests_hard_examples_export` |
| analysis | `analysis_runs_compare`, `analysis_metrics`, `analysis_logs` |
| hosting | `hosting_list`, `hosting_load`, `hosting_unload`, `hosting_infer` |

## 3. Desktop-interne Bridge (Rust <-> Webview)  — Strang A + B

- Registrierung: Die Webview ruft beim Start (und bei Aenderung) den Tauri-Command
  `mcp_register_manifest({ tools: Tool[] })` auf. Rust speichert das Manifest im State.
- Aufruf: Rust sendet an das Hauptfenster das Event `mcp://call` mit
  `{ callId: string, tool: string, args: object, source: "local" | "remote", client: { name: string, version?: string } }`.
- Antwort: Webview ruft `mcp_tool_result({ callId, result: CallToolResult })` auf.
- Timeout in Rust: 60 s, danach `isError: true` mit Text "FrameTrain hat nicht rechtzeitig geantwortet".
- Freigabe (Approval) passiert in der Webview (Dialog in `<ModalPortal>`), BEVOR das Werkzeug laeuft.
  Ablehnung -> `isError: true`, Text "Der Nutzer hat die Aktion in FrameTrain abgelehnt."
  "Fuer diesen Client immer erlauben" wird je `client.name` + Werkzeug gespeichert (Einstellungen, widerrufbar).
- Ist kein Hauptfenster geladen: Rust zeigt/erzeugt es und wartet bis 15 s auf das Manifest.
- Fallback ohne Manifest: Rust liefert nur `system_ping` (gibt Version + "App startet noch" zurueck).

## 4. Lokaler MCP-Endpunkt — Strang B

- Streamable HTTP (MCP 2025-06-18), JSON-Antworten (kein SSE noetig), `POST http://127.0.0.1:47861/mcp`.
- Nur 127.0.0.1, Host-Header-Pruefung wie `hosting_api.rs`, `Authorization: Bearer ftmcp_<32 hex>`.
  Token in `<app_data>/mcp.json` (`{ enabled, port, token }`), neu erzeugbar in den Einstellungen.
- Stdio-Bruecke fuer Claude Desktop: das App-Binary selbst mit `--mcp-stdio` gestartet
  liest JSON-RPC zeilenweise von stdin, liest Token/Port aus `mcp.json`, leitet an den Endpunkt weiter,
  schreibt Antworten auf stdout. Laeuft die App nicht: App starten (macOS `open -b <bundle-id>`,
  Windows/Linux eigenes Binary ohne Argument, `.no_window()`), bis 30 s warten.
  Wichtig: in `main()` VOR dem Tauri-Builder abzweigen, keine Fenster, kein Single-Instance-Plugin.
- Einstellungen-Tab "KI-Verbindungen": Schalter lokaler MCP an/aus, Port, Token kopieren/neu,
  Buttons "Mit Claude Desktop verbinden" (schreibt Eintrag `frametrain` in
  `claude_desktop_config.json` mit `command` = App-Binary, `args` = `["--mcp-stdio"]`, vorher Backup),
  "Befehl fuer Claude Code kopieren" (`claude mcp add --transport http frametrain http://127.0.0.1:47861/mcp --header "Authorization: Bearer …"`),
  Cursor-JSON kopieren. Liste "Freigaben je Client" mit Widerrufen. Schalter Fernzugriff (Abschnitt 5).

## 5. Relay Website <-> Desktop — Strang B (Client) + C (Server)

Opt-in: Schalter "Fernzugriff ueber frame-train.com erlauben" (Standard aus). Nur wenn an, laeuft der Relay-Client.
Auth wie bei allen `/api/desktop/*`-Routen: `apiKey` (ft_…) im JSON-Body, `userIdForApiKey()`.
Basis-URL: `PRODUCTION_API_URL` aus `api_config.rs` (https://frame-train.com/api).

| Route | Body | Antwort |
|---|---|---|
| `POST /api/desktop/mcp/heartbeat` | `{ apiKey, deviceId, deviceName, appVersion, manifestHash }` | `{ ok, wantManifest: bool }` |
| `POST /api/desktop/mcp/manifest` | `{ apiKey, deviceId, manifestHash, tools: Tool[] }` | `{ ok }` |
| `POST /api/desktop/mcp/poll` | `{ apiKey, deviceId }` — Long-Poll bis 25 s | `{ jobs: [{ id, tool, args, client: { name }, createdAt }] }` |
| `POST /api/desktop/mcp/result` | `{ apiKey, deviceId, jobId, result: CallToolResult }` | `{ ok }` |
| `POST /api/desktop/mcp/status` | `{ apiKey }` | `{ connections: [{ clientName, createdAt, lastUsedAt }], recentJobs: [{ tool, clientName, status, createdAt }] }` |
| `POST /api/desktop/mcp/disconnect` | `{ apiKey, connectionId }` | `{ ok }` |

- `deviceId`: UUID, einmal erzeugt, in `mcp.json`. Heartbeat alle 30 s. Geraet gilt als online, wenn lastSeen < 90 s.
- Poll laeuft in Schleife direkt nach jeder Antwort; bei Netzfehler Backoff 2→60 s. Jobs gehen durch dieselbe
  Bridge (Abschnitt 3) mit `source: "remote"`, also auch mit Freigabe-Dialog.
- Die Desktop-App zeigt in "KI-Verbindungen" die Daten aus `/status` (verbundene Clients, letzte Aufrufe) –
  das ist der Session-Abgleich mit der Website.

## 6. Remote-MCP auf der Website — Strang C

- Endpunkt `https://frame-train.com/api/mcp` (Streamable HTTP, zustandslos, JSON-Antworten; `GET` -> 405).
  Methoden: `initialize`, `notifications/initialized`, `ping`, `tools/list`, `tools/call`.
- `tools/list`: Manifest des zuletzt online gesehenen Geraets des Nutzers (`_meta`-Felder entfernen,
  `annotations` behalten). Kein Geraet je gesehen -> statisches Fallback-Manifest (`system_status`, das erklaert,
  wie man die App installiert und Fernzugriff einschaltet).
- `tools/call`: Job anlegen (queued), bis 50 s warten (DB-Polling 500 ms), Ergebnis zurueck. Geraet offline ->
  `isError` mit Text "Die FrameTrain-App ist nicht geoeffnet oder Fernzugriff ist aus. Bitte FrameTrain starten
  (Einstellungen > KI-Verbindungen > Fernzugriff)." Timeout -> Job `expired`, Hinweis auf `*_status`-Werkzeuge.
  Jobs aelter als 2 min und nicht abgeholt -> `expired`.
- OAuth 2.1 (fuer Claude.ai Custom Connector + ChatGPT Apps):
  - `/.well-known/oauth-protected-resource` (RFC 9728) und `/.well-known/oauth-authorization-server` (RFC 8414)
  - `POST /api/oauth/register` Dynamic Client Registration (RFC 7591), oeffentliche Clients, PKCE S256 Pflicht
  - `/[locale]/oauth/authorize` Seite: nutzt die normale Website-Anmeldung, zeigt Zustimmung
    ("<Client> moechte FrameTrain auf deinem Rechner steuern"), Abo-Pruefung wie die App
  - `POST /api/oauth/token` (authorization_code, refresh_token; Access 1 h, Refresh 30 Tage, rotierend, gehasht gespeichert)
  - 401 von `/api/mcp` mit `WWW-Authenticate: Bearer resource_metadata="https://frame-train.com/.well-known/oauth-protected-resource"`
  - Redirect-URIs: was der Client bei DCR registriert (https erzwingen ausser localhost/127.0.0.1).
- Prisma-Modelle (Namen verbindlich): `McpDevice`, `McpJob`, `OAuthClient`, `OAuthAuthCode`, `OAuthToken`,
  `McpConnection` (je Nutzer+Client, fuer Anzeige/Widerruf).
- Dashboard-Seite "KI-Verbindungen" auf der Website: App online/offline + Version, verbundene Clients mit Trennen,
  letzte Aufrufe, Anleitungen Claude (Custom Connector-URL kopieren) und ChatGPT.

## 7. Regeln fuer alle Straenge

- Eigener Branch, nichts nach `main` mergen, nicht pushen, kein Release (`release.sh` NICHT ausfuehren).
- Keine Emojis in der UI (lucide-Icons), Modals in `<ModalPortal>`, lang laufende Tauri-Commands `async` +
  `spawn_blocking`, jedes `Command::new` mit `.no_window()`, Texte DE+EN.
- Website: Prisma-Schema + SQL-Migration schreiben, aber NICHT gegen die Produktions-DB ausfuehren.
- Tests: Desktop `npm test` (vitest) + `cargo test`/`cargo check`; Website `npm run build` + Unit-Tests der Protokoll-Logik.
- Am Ende: kurzer Bericht in `docs/mcp/REPORT-<strang>.md` im eigenen Repo: was fertig, was offen, wie testen.
