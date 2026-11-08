# Synapse: eigene Nodes (Node-Bibliothek)

Stand 2026-11-08: gebaut auf `feat/synapse-node-library`, alle Tests gruen. Live in der Vorschau-App geprueft: Werkstatt mit Probelauf, Speichern, Palette, Node mit zwei Ein- und Ausgaengen verdrahtet, Lernschritt, Training (CSV, 98,5 %), neue Version mit Update-Badge und Uebernehmen. Nicht live: Export/Import ueber den Dateidialog (nur Rust-Tests).

Abweichungen vom Plan:
- Der erste Eingang heisst immer `in` (wie bei allen Nodes), weitere wie ihr Argument. So bleiben alte Kanten gueltig.
- "In Bibliothek speichern" und "Neue Version uebernehmen" sitzen als Knoepfe im Eigenschaften-Panel statt im Rechtsklick-Menue.
- Synapse Magic (eingebauter Assistent) und die MCP-Verbindung kennen die eigenen Nodes: `add_node("custom:<id>")`, Ports und Einstellungen im Kontext, `set_param(code)` liest die Ports sofort, `save_custom_node` legt Nodes in die Bibliothek. Live geprueft: Bau mit Gated Fusion (x und skip getrennt verdrahtet), Einstellungen aendern, neuen Code-Node schreiben, einsetzen und speichern.

## Ziel

Eigene Nodes baut man einmal, gibt ihnen Namen, Icon und Farbe und zieht sie danach in jedem Modell aus der Palette, genau wie eingebaute Nodes. Ein eigener Node ist entweder

- **Code**: eigene PyTorch-Klasse, oder
- **Block**: mehrere Nodes, zu einem Baustein zusammengefasst.

## Was heute fehlt

| Heute | Problem |
|---|---|
| Code-Node hat fest `in` und `out` | keine zwei Zweige, kein `mu`/`logvar` |
| Einstellungen nur im Code | Wer `hidden = 128` ändern will, muss den Code ändern |
| Code-Node und Block leben im Modell | Wiederverwenden heißt kopieren |
| Aussehen fest (`</>`, `▣`) | eigene Nodes sind auf der Fläche nicht unterscheidbar |

## Kernidee: der Code beschreibt den Node

Ports und Einstellungen werden aus dem Code gelesen, ohne extra Formular:

```python
class Node(nn.Module):
    outputs = ("mu", "logvar")            # Ausgänge (fehlt das: ein Ausgang "out")

    def __init__(self, shape, hidden: int = 128, dropout: float = 0.1):
        super().__init__()                # hidden, dropout -> Einstellungen im Panel
        ...

    def forward(self, x, skip=None):      # x, skip -> Eingänge (skip optional)
        ...
        return {"mu": mu, "logvar": logvar}
```

- `forward`-Argumente werden zu Eingängen. Ein Argument mit Default ist optional.
- `__init__`-Argumente mit Default werden zu Einstellungen. Der Typ ergibt sich aus Annotation oder Default, also Zahl, Text, Schalter oder bei `Literal[...]` eine Auswahl.
- `shape` ist bei einem Eingang die Form, bei mehreren ein Dict je Eingang.
- Gibt es mehrere Ausgänge, liefert `forward` ein Dict.

Gelesen wird das in Python mit `inspect`. Neuer Service-Befehl `inspect_code` gibt Ports und Einstellungen zurück, Fehler mit Zeilennummer.

## Speichern

- **Neue SQLite-Tabelle `synapse_custom_nodes`**: id, name, kind (code|block), icon, color, category, description, definition_json, version, updated_at.
- **Rust-Commands:** `list_custom_nodes`, `save_custom_node`, `delete_custom_node`, `export_custom_node` (Datei `.ftnode`, JSON) und `import_custom_node`.
- **Eigenständige Modelle:** Wer einen eigenen Node auf die Fläche zieht, bekommt eine Kopie der Definition ins Modell, plus `libraryId` und `libraryVersion`. Training, Export, Hosting und geteilte Modelle brauchen die Bibliothek also nie.
- **Neue Version in der Bibliothek:** Platzierte Nodes zeigen „Update verfügbar“. Ein Klick übernimmt die neue Version, gleichnamige Einstellungen bleiben erhalten. Nichts ändert sich ungefragt.

## Oberfläche

1. **Palette:** neuer Bereich „Eigene Nodes“ ganz oben, mit Suche. Dazu ein Knopf „Neuer Node“ und Import über `.ftnode`.
2. **Node-Werkstatt** (Dialog über `ModalPortal`):
   - links Name, Icon, Farbe und Beschreibung,
   - Mitte der Code-Editor,
   - rechts die Vorschau: der Node so, wie er auf der Fläche aussieht, mit erkannten Ports und Einstellungen, dazu ein Probelauf mit Beispielform (zum Beispiel `1, 64`, Ausgabeform oder Fehler mit Zeile).
   - Icons sind lucide-Icons, keine Emojis. Farben kommen aus den Kategorie-Farben plus einigen eigenen Tönen.
3. **Speichern aus dem Modell:** Rechtsklick auf einen Code-Node oder Block, dann „In Bibliothek speichern“ öffnet die Werkstatt vorausgefüllt.
4. **Auf der Fläche:** eigener Name, Icon und Farbe, kleines Kennzeichen „eigen“, Badge „Update“, wenn eine neue Version da ist.
5. **Panel:** die erkannten Einstellungen als normale Felder, der Code bleibt aufklappbar darunter.

## Technik je Schicht

- **Python `synapse`:** Ports pro Node statt nur pro Typ (`GNode.inputs`/`outputs`; Code-Nodes lesen sie aus `inspect_code`, alle anderen wie bisher aus dem Spec). Das betrifft `graph.py` (Kanten), `runtime.py` (`_inputs`), `check.py` und `export.py` (Code-Klasse mit Einstellungen als Argumenten, mehrere Ausgänge). `compile_code` übergibt die Einstellungen an `__init__`.
- **Rust:** Tabelle, Migration und fünf Commands in `db_commands.rs`. Dazu der Service-Befehl `inspect_code` in `synapse_service.rs`.
- **Frontend:** `customNodes.ts` (laden, in `NodeDef` umwandeln, Instanz anlegen, Update übernehmen), `NodeWorkshop.tsx`, Palette (`NodeLibrary.tsx`), Node-Komponente (dynamische Handles, Icon, Farbe), `PropertyPanel.tsx` und Kontextmenü.
- **KI und MCP:** `synapse_node_catalog` liefert die eigenen Nodes mit. So kann der Synapse-Assistent sie benutzen.
- **Alte Modelle:** Code-Nodes ohne `outputs`/Extra-Argumente verhalten sich exakt wie heute.

## Reihenfolge

1. Ports und Einstellungen aus dem Code (Python plus Tests): größter Hebel, auch ohne Bibliothek nützlich.
2. Bibliothek in Rust (Tabelle, Commands, `.ftnode`).
3. Palette, Werkstatt, „In Bibliothek speichern“.
4. Versionen und Update-Badge.
5. MCP-Katalog und Assistent.
6. Tests (Python, Rust, Frontend), danach Live-Test in der Vorschau-App.

Gebaut wird auf einem eigenen Branch `feat/synapse-node-library` von `main` in einem eigenen Worktree.

## Bewusst später

- Block-Inhalt auf der Fläche aufklappen und bearbeiten.
- Bibliothek im Konto synchronisieren oder im Markt teilen. Das `.ftnode`-Format ist dafür schon da.
- Sandbox für fremden Code: Import zeigt deshalb vor dem Speichern den Code und den Hinweis, dass er lokal ohne Sandbox läuft.
