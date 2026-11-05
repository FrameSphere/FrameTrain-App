# Strang A – Werkzeuge in der Webview (Bericht)

Branch: `claude/mcp-tools-1c9863`. Nicht gemergt, nicht gepusht, kein Release.

## Fertig

- `src/mcp/` mit Typen, Registry, Bridge, Freigabe-Logik, Freigabe-Dialog und **94 Werkzeugen in allen 11 Domaenen**
  (Format nach CONTRACT.md Abschnitt 1: snake_case mit Domaenen-Praefix, englische LLM-Beschreibung,
  JSON-Schema mit `additionalProperties: false`, `annotations`, `_meta` mit `frametrain/domain` + `frametrain/approval`).
- Keine neuen Rust-Commands noetig: alles laeuft ueber bestehende Tauri-Commands. `mcp_tool_commands.rs` gibt es deshalb nicht,
  `cargo check` war nicht noetig (kein Rust angefasst).
- Lange Laeufe (Downloads, Importe, Web-Abruf, Export, Modell-Vorschlaege, Labor-Server) laufen als Hintergrund-Auftrag:
  Ergebnis direkt, wenn nach 20 s fertig, sonst `job_id` fuer `system_job_status` (`src/mcp/jobs.ts`). Downloads geben sofort die job_id.
- Training/Dev Train/Dev Test: Status ueber eigene `*_status`-Werkzeuge; die Bridge schneidet dazu die Lauf-Events mit
  (`src/mcp/runLog.ts`: training-progress/-status/-error/-complete/-finished, dev-training-output, dev-test-output/-complete).
- **Synapse-Zugriffsschicht** `src/components/synapse/synapseGraphStore.ts`: offener Builder meldet sich an (`attachSynapseBuilder`),
  Werkzeuge lesen/schreiben dann direkt in die Leinwand (Konsole zeigt `[MCP] <Client>: …`); bei geschlossener Seite dient der
  Autosave `synapse_autosave_<user>` als Speicher, den der Builder beim Oeffnen ohnehin laedt. `synapse_graph_edit` nutzt den
  Executor des eingebauten Synapse-Agenten (neue Option `visualDelayMs`, Standard unveraendert 320 ms).
  `synapse_train` startet bei offener Seite ueber den Builder (Live-Kurven, Versionsregistrierung), sonst direkt mit
  `start_training` + eigener Versionsregistrierung nach `training-complete`.
- **Dev-Train-Zugriffsschicht** `src/utils/devScriptStore.ts`: offenes Panel = Editor wird direkt gesetzt; sonst Entwurf im
  localStorage, den das Panel beim Oeffnen in den Editor uebernimmt. `devtrain_start` baut die Umgebungsvariablen genau wie das Panel
  (`buildDatasetRefs` + `refsToEnv`, Datensaetze nach Importdatum nummeriert).
- Freigabe: `none` sofort, `once` fragt einmal je Client, `always` jedes Mal; "Fuer <Client> immer erlauben" wird je Client+Werkzeug
  gespeichert (`localStorage: ft_mcp_grants_v1`). **Destruktive Werkzeuge (`*_delete`) fragen immer, ohne Merk-Option.**
  Werkzeuge mit fremden Dateipfaden (Importe, `studio_inspect_source`) sind `always`: der Dialog zeigt den Pfad = Bestaetigung per Dialog.
- Dialog `src/mcp/ApprovalDialog.tsx` in `<ModalPortal>`, lucide-Icons, Escape = Ablehnen, Warteschlange fuer mehrere Anfragen,
  Texte DE+EN unter `mcp.tools.approval.*`. Global in `App.tsx` eingehaengt; dort startet auch die Bridge und meldet den Nutzer.
- Offene Oberflaeche erfaehrt Aenderungen per `window`-Event `ft-mcp-changed` (`{ domain, ... }`); die Studio-Projektliste laedt darauf neu.
- `npm test` (neues Skript `vitest run`): 96 Dateien / 1097 Tests gruen (vorher 91 / 1034). `npx tsc --noEmit` gruen.
  Neue Tests: `src/mcp/__tests__/` manifest, bridge, approval, approvalDialog, domains (jede Domaene mit gemocktem invoke).

## Was Strang B anschliessen muss

1. Rust-Commands `mcp_register_manifest({ tools })` und `mcp_tool_result({ callId, result })` sowie Event `mcp://call`
   an das Hauptfenster – genau wie in CONTRACT.md Abschnitt 3. Die Webview ruft sie abgesichert auf (fehlen sie: Konsolenhinweis, App laeuft).
2. Manifest-Hash: `manifestHash()` in `src/mcp/registry.ts` (FNV-1a ueber das JSON) – Rust darf auch selbst hashen.
3. **Zeitbudget:** Rust wartet 60 s auf das Ergebnis, inklusive Freigabe-Dialog. Die Webview antwortet spaetestens nach 57 s
   (Freigabe max. 30 s, dann Werkzeug-Restzeit, Inline-Wartezeit 20 s). Wenn B den Timeout bei offener Freigabe verlaengern will
   (z. B. Event `mcp://approval-pending`), kann A `APPROVAL_TIMEOUT_MS` anheben – bitte absprechen.
4. Fenster nach vorn: bei noetiger Freigabe ruft die Bridge `hosting_open_main({ view: null })` (existierender Command).
   Ist das Hauptfenster gar nicht geladen, muss B es zeigen/erzeugen (Vertrag Abschnitt 3).
5. Einstellungs-Tab: `src/mcp/approvals.ts` mit genau der Schnittstelle aus B's Notiz:
   `listApprovals()`, `revokeApproval(client, tool)`, `onApprovalsChanged(cb)`, zusaetzlich `revokeClientApprovals(client)`,
   `revokeAllApprovals()`. `McpApproval.kind` = `'once' | 'always'`. Tiefere API: `src/mcp/approval.ts` (`listGrants`, `revokeGrant`, …).
6. Fallback ohne Manifest (`system_ping`): das Webview-Manifest enthaelt ebenfalls `system_ping` (Version + Anmeldestatus), Namen passen.
7. Locales: A hat nur `mcp.tools.*` angelegt (am Dateiende von de.json/en.json). B legt seine Texte bitte unter `mcp.settings.*`
   o. ae. an – beim Mergen beide Bloecke unter dem einen Schluessel `mcp` zusammenfuehren.

## Offene Punkte

- **Nicht live getestet** – nur Unit-Tests mit gemocktem invoke. Vor einem Release die echte App mit Strang B zusammen pruefen
  (siehe Memory: Live-Tests ueber Release-Build). Insbesondere: Argumentnamen echter Commands, `studio_fetch_web`-Optionen,
  `lab_get_server_status`-Felder, Ergebnisformate von `hosting_infer`.
- Training, das per Werkzeug ohne offene Trainingsseite startet, erscheint in `GlobalTrainingProgress` nur so weit, wie diese
  Komponente selbst Events hoert; der React-Kontext der Trainingsseite kennt den Job nicht.
- Offene Studio-Werkbank laedt Samples nach Werkzeug-Aenderungen nicht von selbst neu (nur die Projektliste). Nachruesten:
  `ft-mcp-changed` mit `domain === 'studio'` und passender `projectId` in der Werkbank abonnieren.
- `studio_near_duplicates` fuer Bilder braucht Wahrnehmungs-Hashes, die bisher nur die Oberflaeche berechnet (`missing_hashes`).
- Freigaben liegen im localStorage (je Webview-Profil, nicht je Nutzer). Fuer mehrere Konten auf einem Rechner ggf. pro Nutzer schluesseln.
- Nicht abgebildet: HF-Upload von Versionen (braucht Token), Hosting-Einstellungen/Token (bewusst), Lab-Korrekturen-Export,
  Dataset-Datei-Verschieben, Metrik-Templates. Leicht nachzuruesten nach demselben Muster.

## Wie testen

```bash
npm test
npx tsc --noEmit
npx vitest run src/mcp
```

## Werkzeugliste (94)

| Name | Domaene | Freigabe | Titel |
|---|---|---|---|
| `system_ping` | system | none | Ping |
| `system_info` | system | none | Systeminfo |
| `system_capabilities` | system | none | Faehigkeiten |
| `system_job_status` | system | none | Auftragsstatus |
| `system_jobs_list` | system | none | Auftraege |
| `models_list` | models | none | Modelle auflisten |
| `models_info` | models | none | Modell-Details |
| `models_search_hf` | models | none | Hugging Face durchsuchen |
| `models_hf_files` | models | none | Hugging-Face-Dateien |
| `models_download_hf` | models | once | Modell herunterladen |
| `models_import_local` | models | always | Lokales Modell importieren |
| `models_rename` | models | once | Modell umbenennen |
| `models_delete` | models | always (destruktiv) | Modell loeschen |
| `models_version_delete` | models | always (destruktiv) | Version loeschen |
| `datasets_list` | datasets | none | Datensaetze auflisten |
| `datasets_info` | datasets | none | Datensatz-Details |
| `datasets_files` | datasets | none | Dateien eines Datensatzes |
| `datasets_preview` | datasets | none | Datensatz-Vorschau |
| `datasets_search_hf` | datasets | none | Hugging-Face-Datensaetze suchen |
| `datasets_download_hf` | datasets | once | Datensatz herunterladen |
| `datasets_import_local` | datasets | always | Lokalen Datensatz importieren |
| `datasets_split` | datasets | once | Datensatz aufteilen |
| `datasets_delete` | datasets | always (destruktiv) | Datensatz loeschen |
| `studio_projects_list` | studio | none | Studio-Projekte |
| `studio_project_create` | studio | once | Studio-Projekt anlegen |
| `studio_project_update` | studio | once | Studio-Projekt aendern |
| `studio_project_delete` | studio | always (destruktiv) | Studio-Projekt loeschen |
| `studio_samples_list` | studio | none | Samples auflisten |
| `studio_stats` | studio | none | Projekt-Statistik |
| `studio_add_texts` | studio | once | Texte hinzufuegen |
| `studio_edit_text` | studio | once | Text bearbeiten |
| `studio_add_image` | studio | once | Bild hinzufuegen |
| `studio_delete_samples` | studio | always (destruktiv) | Samples loeschen |
| `studio_inspect_source` | studio | always | Quelle pruefen |
| `studio_import` | studio | always | In Projekt importieren |
| `studio_fetch_web` | studio | once | Aus dem Web holen |
| `studio_set_annotation` | studio | once | Annotation setzen |
| `studio_suggest` | studio | once | Vorschlaege per Modell |
| `studio_review` | studio | once | Pruefung per Modell |
| `studio_near_duplicates` | studio | none | Fast-Dubletten |
| `studio_quality` | studio | none | Qualitaetsbericht |
| `studio_export` | studio | once | Als Datensatz exportieren |
| `studio_model_classes` | studio | none | Klassen eines Modells |
| `training_presets` | training | none | Trainings-Presets |
| `training_check_requirements` | training | none | Voraussetzungen pruefen |
| `training_estimate_resources` | training | none | Speicherbedarf schaetzen |
| `training_rate_config` | training | none | Konfiguration bewerten |
| `training_start` | training | always | Training starten |
| `training_status` | training | none | Trainingsstatus |
| `training_stop` | training | always | Training stoppen |
| `training_history` | training | none | Trainingsverlauf |
| `synapse_node_catalog` | synapse | none | Knoten-Katalog |
| `synapse_blueprints` | synapse | none | Vorlagen |
| `synapse_graph_get` | synapse | none | Graph lesen |
| `synapse_graph_set` | synapse | once | Graph ersetzen |
| `synapse_graph_edit` | synapse | once | Graph bearbeiten |
| `synapse_validate` | synapse | none | Graph pruefen |
| `synapse_generate_code` | synapse | none | PyTorch-Code erzeugen |
| `synapse_models_list` | synapse | none | Gespeicherte Synapse-Modelle |
| `synapse_open_model` | synapse | once | Synapse-Modell laden |
| `synapse_save` | synapse | once | Als Modell speichern |
| `synapse_train` | synapse | always | Synapse-Graph trainieren |
| `synapse_train_status` | synapse | none | Synapse-Trainingsstatus |
| `devtrain_context` | devtrain | none | Dev-Train-Kontext |
| `devtrain_script_get` | devtrain | none | Skript lesen |
| `devtrain_script_set` | devtrain | once | Skript schreiben |
| `devtrain_scripts_list` | devtrain | none | Skript-Bibliothek |
| `devtrain_start` | devtrain | always | Dev Train starten |
| `devtrain_status` | devtrain | none | Dev-Train-Status |
| `devtrain_stop` | devtrain | always | Dev Train stoppen |
| `devtrain_test_start` | devtrain | always | Dev Test starten |
| `devtrain_test_status` | devtrain | none | Dev-Test-Status |
| `devtrain_test_stop` | devtrain | once | Dev Test stoppen |
| `lab_server_start` | lab | once | Labor-Server starten |
| `lab_server_status` | lab | none | Labor-Server-Status |
| `lab_server_stop` | lab | once | Labor-Server stoppen |
| `lab_infer` | lab | none | Einzel-Inferenz |
| `lab_sessions` | lab | none | Labor-Sitzungen |
| `lab_stats` | lab | none | Labor-Statistik |
| `tests_start` | tests | always | Test starten |
| `tests_status` | tests | none | Teststatus |
| `tests_stop` | tests | once | Test stoppen |
| `tests_results` | tests | none | Testergebnisse |
| `tests_history` | tests | none | Testverlauf |
| `tests_single_input` | tests | none | Einzeltest |
| `tests_hard_examples_export` | tests | once | Schwere Beispiele exportieren |
| `analysis_metrics` | analysis | none | Trainingsmetriken |
| `analysis_logs` | analysis | none | Trainingslogs |
| `analysis_version_details` | analysis | none | Versionsdetails |
| `analysis_runs_compare` | analysis | none | Laeufe vergleichen |
| `hosting_list` | hosting | none | Gehostete Modelle |
| `hosting_load` | hosting | once | Modell bereitstellen |
| `hosting_unload` | hosting | once | Modell entladen |
| `hosting_infer` | hosting | none | Gehostetes Modell abfragen |
