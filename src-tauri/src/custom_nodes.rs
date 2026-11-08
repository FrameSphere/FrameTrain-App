// Eigene Nodes des Synapse Builders: Code-Nodes und Bloecke, die man einmal baut
// und in jedem Modell aus der Palette zieht.
//
// Rust kennt den Inhalt nicht: `definition` ist das JSON des Editors (Icon, Farbe,
// Beschreibung, Code oder Block-Graph). Jede inhaltliche Aenderung zaehlt die
// Version hoch; platzierte Nodes merken sich ihre Version und zeigen so "Update".

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
use tauri::State;

use crate::AppState;

pub const FTNODE_FORMAT: &str = "frametrain-node";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CustomNode {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub version: i64,
    pub definition: Value,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomNodeInput {
    pub id: Option<String>,
    pub name: String,
    pub kind: String,
    pub definition: Value,
}

pub fn ensure_table(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS synapse_custom_nodes (
            id              TEXT    PRIMARY KEY,
            user_id         TEXT    NOT NULL,
            name            TEXT    NOT NULL,
            kind            TEXT    NOT NULL,
            version         INTEGER NOT NULL DEFAULT 1,
            definition_json TEXT    NOT NULL,
            updated_at      INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_custom_nodes_user ON synapse_custom_nodes(user_id);",
    )
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn check(input: &CustomNodeInput) -> Result<(), String> {
    if input.name.trim().is_empty() {
        return Err("Der Node braucht einen Namen".into());
    }
    if input.kind != "code" && input.kind != "block" {
        return Err(format!("Unbekannte Node-Art: {}", input.kind));
    }
    if !input.definition.is_object() {
        return Err("Die Definition muss ein Objekt sein".into());
    }
    Ok(())
}

fn row(r: &rusqlite::Row) -> rusqlite::Result<CustomNode> {
    let json: String = r.get(4)?;
    Ok(CustomNode {
        id: r.get(0)?,
        name: r.get(1)?,
        kind: r.get(2)?,
        version: r.get(3)?,
        definition: serde_json::from_str(&json).unwrap_or(Value::Null),
        updated_at: r.get(5)?,
    })
}

const COLUMNS: &str = "id, name, kind, version, definition_json, updated_at";

pub fn list(conn: &Connection, user: &str) -> Result<Vec<CustomNode>, String> {
    let mut stmt = conn
        .prepare(&format!(
            "SELECT {COLUMNS} FROM synapse_custom_nodes WHERE user_id = ?1 ORDER BY name COLLATE NOCASE"
        ))
        .map_err(|e| e.to_string())?;
    let rows = stmt.query_map(params![user], row).map_err(|e| e.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.to_string())
}

pub fn get(conn: &Connection, user: &str, id: &str) -> Result<Option<CustomNode>, String> {
    conn.query_row(
        &format!("SELECT {COLUMNS} FROM synapse_custom_nodes WHERE user_id = ?1 AND id = ?2"),
        params![user, id],
        row,
    )
    .optional()
    .map_err(|e| e.to_string())
}

/// Neu anlegen oder aendern. Gleicher Inhalt laesst die Version stehen.
pub fn save(conn: &Connection, user: &str, input: CustomNodeInput) -> Result<CustomNode, String> {
    check(&input)?;
    let existing = match &input.id {
        Some(id) => get(conn, user, id)?,
        None => None,
    };
    let node = match existing {
        Some(old) => {
            let changed = old.definition != input.definition || old.kind != input.kind || old.name != input.name;
            CustomNode {
                version: if changed { old.version + 1 } else { old.version },
                updated_at: if changed { now_ms() } else { old.updated_at },
                id: old.id,
                name: input.name.trim().to_string(),
                kind: input.kind,
                definition: input.definition,
            }
        }
        None => CustomNode {
            id: input.id.unwrap_or_else(|| format!("cn_{}", uuid::Uuid::new_v4().simple())),
            name: input.name.trim().to_string(),
            kind: input.kind,
            version: 1,
            definition: input.definition,
            updated_at: now_ms(),
        },
    };
    conn.execute(
        "INSERT INTO synapse_custom_nodes (id, user_id, name, kind, version, definition_json, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
         ON CONFLICT(id) DO UPDATE SET name = excluded.name, kind = excluded.kind, version = excluded.version,
             definition_json = excluded.definition_json, updated_at = excluded.updated_at",
        params![node.id, user, node.name, node.kind, node.version, node.definition.to_string(), node.updated_at],
    )
    .map_err(|e| e.to_string())?;
    Ok(node)
}

pub fn delete(conn: &Connection, user: &str, id: &str) -> Result<(), String> {
    conn.execute("DELETE FROM synapse_custom_nodes WHERE user_id = ?1 AND id = ?2", params![user, id])
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// `.ftnode`: dieselben Felder wie in der Bibliothek, ohne Nutzer und Zeitstempel.
pub fn to_file(node: &CustomNode, path: &Path) -> Result<(), String> {
    let doc = serde_json::json!({
        "format": FTNODE_FORMAT, "formatVersion": 1,
        "node": { "name": node.name, "kind": node.kind, "version": node.version, "definition": node.definition },
    });
    let text = serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| format!("{}: {}", path.display(), e))
}

/// Liest eine `.ftnode`-Datei, speichert aber nichts: die Oberflaeche zeigt den Code
/// vor dem Uebernehmen, weil er lokal ohne Sandbox laeuft.
pub fn from_file(path: &Path) -> Result<CustomNodeInput, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let doc: Value = serde_json::from_str(&text).map_err(|_| "Die Datei ist kein gueltiges JSON".to_string())?;
    if doc["format"] != FTNODE_FORMAT {
        return Err("Das ist keine FrameTrain-Node-Datei (.ftnode)".into());
    }
    let node = &doc["node"];
    let input = CustomNodeInput {
        id: None,
        name: node["name"].as_str().unwrap_or_default().to_string(),
        kind: node["kind"].as_str().unwrap_or_default().to_string(),
        definition: node["definition"].clone(),
    };
    check(&input)?;
    Ok(input)
}

// ==================== Tauri-Commands ====================

fn with_db<T>(state: &State<AppState>, f: impl FnOnce(&Connection, &str) -> Result<T, String>) -> Result<T, String> {
    let db = state.db.lock().map_err(|e| format!("DB Lock: {}", e))?;
    let user = db.get_current_user_id().unwrap_or_else(|| "default_user".to_string());
    ensure_table(&db.conn).map_err(|e| e.to_string())?;
    f(&db.conn, &user)
}

#[tauri::command]
pub fn list_custom_nodes(state: State<AppState>) -> Result<Vec<CustomNode>, String> {
    with_db(&state, list)
}

#[tauri::command]
pub fn save_custom_node(state: State<AppState>, node: CustomNodeInput) -> Result<CustomNode, String> {
    with_db(&state, |conn, user| save(conn, user, node))
}

#[tauri::command]
pub fn delete_custom_node(state: State<AppState>, id: String) -> Result<(), String> {
    with_db(&state, |conn, user| delete(conn, user, &id))
}

#[tauri::command]
pub fn export_custom_node(state: State<AppState>, id: String, path: String) -> Result<(), String> {
    let node = with_db(&state, |conn, user| get(conn, user, &id))?.ok_or("Node nicht gefunden")?;
    to_file(&node, Path::new(&path))
}

#[tauri::command]
pub fn read_custom_node_file(path: String) -> Result<CustomNodeInput, String> {
    from_file(Path::new(&path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        ensure_table(&conn).unwrap();
        conn
    }

    fn input(id: Option<&str>, code: &str) -> CustomNodeInput {
        CustomNodeInput {
            id: id.map(String::from),
            name: "Residual MLP".into(),
            kind: "code".into(),
            definition: serde_json::json!({ "code": code, "icon": "layers", "color": "#7F77DD" }),
        }
    }

    #[test]
    fn version_steigt_nur_bei_aenderung() {
        let conn = db();
        let a = save(&conn, "u1", input(None, "v1")).unwrap();
        assert_eq!(a.version, 1);
        assert!(a.id.starts_with("cn_"));
        let same = save(&conn, "u1", input(Some(&a.id), "v1")).unwrap();
        assert_eq!(same.version, 1);
        let b = save(&conn, "u1", input(Some(&a.id), "v2")).unwrap();
        assert_eq!(b.version, 2);
        assert_eq!(list(&conn, "u1").unwrap(), vec![b]);
    }

    #[test]
    fn nutzer_sehen_nur_ihre_nodes() {
        let conn = db();
        let a = save(&conn, "u1", input(None, "x")).unwrap();
        assert!(list(&conn, "u2").unwrap().is_empty());
        delete(&conn, "u2", &a.id).unwrap();
        assert_eq!(list(&conn, "u1").unwrap().len(), 1);
        delete(&conn, "u1", &a.id).unwrap();
        assert!(list(&conn, "u1").unwrap().is_empty());
    }

    #[test]
    fn ungueltige_eingaben_werden_abgelehnt() {
        let conn = db();
        let mut bad = input(None, "x");
        bad.name = "  ".into();
        assert!(save(&conn, "u1", bad).is_err());
        let mut bad = input(None, "x");
        bad.kind = "script".into();
        assert!(save(&conn, "u1", bad).is_err());
    }

    #[test]
    fn ftnode_hin_und_zurueck() {
        let conn = db();
        let a = save(&conn, "u1", input(None, "class Node: ...")).unwrap();
        let dir = std::env::temp_dir().join(format!("ftnode_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("residual.ftnode");
        to_file(&a, &path).unwrap();
        let back = from_file(&path).unwrap();
        assert_eq!(back.id, None);
        assert_eq!(back.name, a.name);
        assert_eq!(back.definition, a.definition);
        std::fs::write(&path, r#"{"format":"anderes"}"#).unwrap();
        assert!(from_file(&path).unwrap_err().contains(".ftnode"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
