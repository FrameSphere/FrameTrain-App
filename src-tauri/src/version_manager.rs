/// Exportiert eine Version in den Downloads-Ordner
#[tauri::command]
pub fn export_model_version(
    app_handle: tauri::AppHandle,
    version_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<String, String> {
    // CRITICAL: Get current user_id for filtering
    let user_id = {
        let db = state.db.lock()
            .map_err(|e| format!("Failed to lock database: {}", e))?;
        db.require_user_id()?.to_string()
    };
    
    let db = state.db.lock()
        .map_err(|e| format!("Failed to lock database: {}", e))?;
    
    // Get version details with user_id check
    let version: (String, String, String) = db.conn.query_row(
        "SELECT v.path, v.version_name, m.name 
         FROM model_versions_new v 
         JOIN models m ON v.model_id = m.id 
         WHERE v.id = ?1 AND v.user_id = ?2",
        rusqlite::params![&version_id, &user_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    ).map_err(|e| format!("Version not found or access denied: {}", e))?;
    
    let (version_path, version_name, model_name) = version;
    
    // Get Downloads folder
    let downloads_dir = app_handle
        .path()
        .download_dir()
        .map_err(|e| format!("Could not get downloads directory: {}", e))?;
    
    // Create export folder name: ModelName_VersionName_timestamp
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let export_folder_name = format!("{}_{}_{}" , 
        model_name.replace(" ", "_"),
        version_name.replace(" ", "_"),
        timestamp
    );
    
    let export_path = downloads_dir.join(&export_folder_name);
    
    println!("[Export] Exporting version {} to {:?}", version_id, export_path);
    
    // Copy version directory to Downloads
    copy_dir_recursive_export(&PathBuf::from(&version_path), &export_path)?;
    
    println!("[Export] ✅ Export completed: {:?}", export_path);
    
    Ok(export_path.to_string_lossy().to_string())
}

fn copy_dir_recursive_export(src: &PathBuf, dst: &PathBuf) -> Result<(), String> {
    if !dst.exists() {
        fs::create_dir_all(dst)
            .map_err(|e| format!("Could not create export directory: {}", e))?;
    }
    
    let entries = fs::read_dir(src)
        .map_err(|e| format!("Could not read source directory: {}", e))?;
    
    for entry in entries {
        let entry = entry.map_err(|e| format!("Entry error: {}", e))?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        
        if src_path.is_dir() {
            copy_dir_recursive_export(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)
                .map_err(|e| format!("Could not copy file: {}", e))?;
        }
    }
    
    Ok(())
}

// Version Manager - Handles model versioning and training history

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tauri::{Manager, State, Emitter};
use crate::database::Database;
use crate::AppState;
use crate::command_ext::{NoWindow, PythonUtf8};

// ============ Data Structures ============

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelWithVersions {
    pub id: String,
    pub name: String,
    pub root_path: String,
    pub version_count: i32,
    pub total_size: i64,
    pub model_type: Option<String>,
    pub last_updated: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelVersion {
    pub id: String,
    pub model_id: String,
    pub version_name: String,
    pub version_number: i32,
    pub path: String,
    pub size_bytes: i64,
    pub file_count: i32,
    pub created_at: String,
    pub is_root: bool,
    pub parent_version_id: Option<String>,
    pub training_metrics: Option<TrainingMetrics>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingMetrics {
    pub final_train_loss: f64,
    pub final_val_loss: Option<f64>,
    pub total_epochs: i32,
    pub total_steps: i32,
    pub best_epoch: Option<i32>,
    pub training_duration_seconds: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelWithVersionTree {
    pub id: String,
    pub name: String,
    pub versions: Vec<VersionTreeItem>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VersionTreeItem {
    pub id: String,
    pub name: String,
    pub is_root: bool,
    pub version_number: i32,
}

// ============ Helper Functions ============

fn calculate_directory_size(path: &Path) -> Result<(i64, i32), String> {
    let mut total_size: i64 = 0;
    let mut file_count: i32 = 0;

    fn visit_dirs(dir: &Path, total_size: &mut i64, file_count: &mut i32) -> Result<(), String> {
        if dir.is_dir() {
            for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
                let entry = entry.map_err(|e| e.to_string())?;
                let path = entry.path();
                if path.is_dir() {
                    visit_dirs(&path, total_size, file_count)?;
                } else {
                    if let Ok(metadata) = fs::metadata(&path) {
                        *total_size += metadata.len() as i64;
                        *file_count += 1;
                    }
                }
            }
        }
        Ok(())
    }

    visit_dirs(path, &mut total_size, &mut file_count)?;
    Ok((total_size, file_count))
}

// ============ Database Extensions ============

impl Database {
    pub fn create_version_tables(&self) -> Result<(), String> {
        // CRITICAL: Disable foreign key constraints
        self.conn.execute("PRAGMA foreign_keys = OFF", [])
            .map_err(|e| format!("Failed to disable foreign keys: {}", e))?;
        
        // 1. Create models table first (base table)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                base_model TEXT,
                model_path TEXT,
                status TEXT NOT NULL DEFAULT 'created',
                user_id TEXT NOT NULL DEFAULT 'default_user',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name)
            )",
            [],
        ).map_err(|e| format!("Failed to create models table: {}", e))?;
        
        // 2. Create model_versions_new table with user_id
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS model_versions_new (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                version_name TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                file_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                is_root INTEGER NOT NULL DEFAULT 0,
                parent_version_id TEXT,
                user_id TEXT NOT NULL DEFAULT 'default_user'
            )",
            [],
        ).map_err(|e| format!("Failed to create model_versions_new table: {}", e))?;

        // 3. Create training_metrics_new table with user_id
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS training_metrics_new (
                id TEXT PRIMARY KEY,
                version_id TEXT NOT NULL UNIQUE,
                final_train_loss REAL NOT NULL,
                final_val_loss REAL,
                total_epochs INTEGER NOT NULL,
                total_steps INTEGER NOT NULL,
                best_epoch INTEGER,
                training_duration_seconds INTEGER,
                created_at TEXT NOT NULL,
                user_id TEXT NOT NULL DEFAULT 'default_user'
            )",
            [],
        ).map_err(|e| format!("Failed to create training_metrics_new table: {}", e))?;

        // Migration: Add user_id to existing tables if not present
        self.migrate_version_tables_user_id()?;

        // Create indices
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_versions_model ON model_versions_new(model_id)",
            [],
        ).map_err(|e| format!("Failed to create index: {}", e))?;

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_version ON training_metrics_new(version_id)",
            [],
        ).map_err(|e| format!("Failed to create index: {}", e))?;
        
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_versions_user ON model_versions_new(user_id)",
            [],
        ).map_err(|e| format!("Failed to create index: {}", e))?;
        
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_user ON training_metrics_new(user_id)",
            [],
        ).map_err(|e| format!("Failed to create index: {}", e))?;

        Ok(())
    }
    
    fn migrate_version_tables_user_id(&self) -> Result<(), String> {
        // Check if user_id column exists in model_versions_new
        let has_user_id: i32 = self.conn.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('model_versions_new') WHERE name='user_id'",
            [],
            |row| row.get(0)
        ).unwrap_or(0);
        
        if has_user_id == 0 {
            println!("[Migration] Adding user_id to model_versions_new");
            self.conn.execute(
                "ALTER TABLE model_versions_new ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default_user'",
                [],
            ).ok(); // Ignore if already exists
        }
        
        // Check training_metrics_new
        let has_user_id_metrics: i32 = self.conn.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('training_metrics_new') WHERE name='user_id'",
            [],
            |row| row.get(0)
        ).unwrap_or(0);
        
        if has_user_id_metrics == 0 {
            println!("[Migration] Adding user_id to training_metrics_new");
            self.conn.execute(
                "ALTER TABLE training_metrics_new ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default_user'",
                [],
            ).ok(); // Ignore if already exists
        }
        
        Ok(())
    }

    pub fn get_models_with_versions(&self) -> Result<Vec<ModelWithVersions>, String> {
        println!("[Version] get_models_with_versions called");
        
        // CRITICAL: Get current user_id for filtering
        let user_id = self.require_user_id()?;
        println!("[Version] Filtering for user_id: {}", user_id);
        
        // First, let's see ALL models in the database for this user
        let total_models: i32 = self.conn.query_row(
            "SELECT COUNT(*) FROM models WHERE user_id = ?1",
            [user_id],
            |row| row.get(0),
        ).unwrap_or(0);
        println!("[Version] Total models in database for this user: {}", total_models);
        
        // List all model IDs for this user
        let mut id_stmt = self.conn.prepare("SELECT id, name FROM models WHERE user_id = ?1").unwrap();
        let ids = id_stmt.query_map([user_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        }).unwrap();
        
        println!("[Version] All models in DB for this user:");
        for id in ids {
            if let Ok((model_id, model_name)) = id {
                println!("[Version]   - {} ({})", model_name, model_id);
            }
        }
        
        let mut stmt = self.conn.prepare(
            "SELECT 
                m.id,
                m.name,
                COALESCE(m.model_path, '') as model_path,
                m.created_at,
                (SELECT COUNT(*) FROM model_versions_new WHERE model_id = m.id AND user_id = ?1) as version_count,
                (SELECT MAX(created_at) FROM model_versions_new WHERE model_id = m.id AND user_id = ?1) as last_version
            FROM models m
            WHERE m.user_id = ?1
            ORDER BY m.created_at DESC"
        ).map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let models = stmt.query_map([user_id], |row| {
            let id: String = row.get(0)?;
            let name: String = row.get(1)?;
            let model_path: String = row.get(2)?;
            let created_at: String = row.get(3)?;
            let version_count: i32 = row.get(4)?;
            let last_updated: Option<String> = row.get(5).ok();
            
            println!("[Version] Found model: {} ({}) with {} versions at path: {}", name, id, version_count, model_path);

            // Calculate total size (only if path exists and is not empty)
            let (total_size, _) = if !model_path.is_empty() {
                let path = PathBuf::from(&model_path);
                if path.exists() {
                    println!("[Version]   - Path exists, calculating size...");
                    calculate_directory_size(&path).unwrap_or((0, 0))
                } else {
                    println!("[Version]   - ⚠️  Path does NOT exist!");
                    (0, 0)
                }
            } else {
                println!("[Version]   - No path set");
                (0, 0)
            };

            Ok(ModelWithVersions {
                id,
                name,
                root_path: model_path,
                version_count,
                total_size,
                model_type: None, // Could be extracted from config.json
                last_updated: last_updated.unwrap_or(created_at),
            })
        }).map_err(|e| format!("Failed to query models: {}", e))?;

        let mut result: Vec<ModelWithVersions> = models.collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to collect models: {}", e))?;
        
        // CRITICAL: Filter out models where the directory does NOT exist
        println!("[Version] Filtering models - checking if directories exist...");
        result.retain(|model| {
            if model.root_path.is_empty() {
                println!("[Version]   - Removing {} (no path)", model.id);
                return false;
            }
            
            let path = PathBuf::from(&model.root_path);
            let exists = path.exists();
            
            if !exists {
                println!("[Version]   - Removing {} (path does not exist: {})", model.id, model.root_path);
            } else {
                println!("[Version]   - Keeping {} (path exists)", model.id);
            }
            
            exists
        });
        
        println!("[Version] Returning {} models (after filtering)", result.len());
        Ok(result)
    }

    pub fn get_model_version_details(&self, model_id: &str) -> Result<Vec<ModelVersion>, String> {
        // CRITICAL: Get current user_id for filtering
        let user_id = self.require_user_id()?;
        println!("[Version] get_model_version_details for model {} and user {}", model_id, user_id);
        
        let mut stmt = self.conn.prepare(
            "SELECT 
                v.id,
                v.model_id,
                v.version_name,
                v.version_number,
                v.path,
                v.size_bytes,
                v.file_count,
                v.created_at,
                v.is_root,
                v.parent_version_id,
                tm.final_train_loss,
                tm.final_val_loss,
                tm.total_epochs,
                tm.total_steps,
                tm.best_epoch,
                tm.training_duration_seconds
            FROM model_versions_new v
            LEFT JOIN training_metrics_new tm ON v.id = tm.version_id AND tm.user_id = ?2
            WHERE v.model_id = ?1 AND v.user_id = ?2
            ORDER BY v.is_root DESC, v.version_number DESC"
        ).map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let versions = stmt.query_map([model_id, user_id], |row| {
            let training_metrics = if let Ok(loss) = row.get::<_, f64>(10) {
                Some(TrainingMetrics {
                    final_train_loss: loss,
                    final_val_loss: row.get(11).ok(),
                    total_epochs: row.get(12)?,
                    total_steps: row.get(13)?,
                    best_epoch: row.get(14).ok(),
                    training_duration_seconds: row.get(15).ok(),
                })
            } else {
                None
            };

            Ok(ModelVersion {
                id: row.get(0)?,
                model_id: row.get(1)?,
                version_name: row.get(2)?,
                version_number: row.get(3)?,
                path: row.get(4)?,
                size_bytes: row.get(5)?,
                file_count: row.get(6)?,
                created_at: row.get::<_, String>(7)?,
                is_root: row.get::<_, i32>(8)? != 0,
                parent_version_id: row.get(9).ok(),
                training_metrics,
            })
        }).map_err(|e| format!("Failed to query versions: {}", e))?;

        versions.collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to collect versions: {}", e))
    }

    pub fn delete_version(&self, version_id: &str) -> Result<(), String> {
        // CRITICAL: Get current user_id for filtering
        let user_id = self.require_user_id()?;
        
        // Check if it's a root version AND belongs to current user
        let is_root: i32 = self.conn.query_row(
            "SELECT is_root FROM model_versions_new WHERE id = ?1 AND user_id = ?2",
            [version_id, user_id],
            |row| row.get(0)
        ).map_err(|e| format!("Version not found or access denied: {}", e))?;

        if is_root != 0 {
            return Err("Cannot delete root version".to_string());
        }

        // Get the path before deleting
        let path: String = self.conn.query_row(
            "SELECT path FROM model_versions_new WHERE id = ?1 AND user_id = ?2",
            [version_id, user_id],
            |row| row.get(0)
        ).map_err(|e| format!("Failed to get version path: {}", e))?;

        // Delete from database (training_metrics will be handled separately)
        self.conn.execute(
            "DELETE FROM model_versions_new WHERE id = ?1 AND user_id = ?2",
            [version_id, user_id],
        ).map_err(|e| format!("Failed to delete version from database: {}", e))?;
        
        // Also delete training metrics for this version
        self.conn.execute(
            "DELETE FROM training_metrics_new WHERE version_id = ?1 AND user_id = ?2",
            [version_id, user_id],
        ).ok(); // Ignore errors if no metrics exist

        // Delete directory from filesystem
        if Path::new(&path).exists() {
            fs::remove_dir_all(&path)
                .map_err(|e| format!("Failed to delete version directory: {}", e))?;
        }

        Ok(())
    }

    pub fn rename_version(&self, version_id: &str, new_name: &str) -> Result<(), String> {
        // CRITICAL: Get current user_id for filtering
        let user_id = self.require_user_id()?;
        
        self.conn.execute(
            "UPDATE model_versions_new SET version_name = ?1 WHERE id = ?2 AND user_id = ?3",
            [new_name, version_id, user_id],
        ).map_err(|e| format!("Failed to rename version: {}", e))?;

        Ok(())
    }

    pub fn get_models_with_version_tree(&self) -> Result<Vec<ModelWithVersionTree>, String> {
        // CRITICAL: Use user-filtered models
        let models = self.list_models()
            .map_err(|e| format!("Failed to list models: {}", e))?;
        let mut result = Vec::new();

        for model in models {
            let versions = self.get_model_version_details(&model.id)?;
            let version_items: Vec<VersionTreeItem> = versions.into_iter().map(|v| {
                VersionTreeItem {
                    id: v.id,
                    name: v.version_name,
                    is_root: v.is_root,
                    version_number: v.version_number,
                }
            }).collect();

            result.push(ModelWithVersionTree {
                id: model.id,
                name: model.name,
                versions: version_items,
            });
        }

        Ok(result)
    }

    pub fn create_root_version(&self, model_id: &str, model_path: &str) -> Result<String, String> {
        use uuid::Uuid;
        
        // CRITICAL: Get current user_id
        let user_id = self.require_user_id()?;
        
        let version_id = Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        
        // Calculate size and file count
        let (size_bytes, file_count) = calculate_directory_size(Path::new(model_path))
            .unwrap_or((0, 0));

        self.conn.execute(
            "INSERT INTO model_versions_new 
             (id, model_id, version_name, version_number, path, size_bytes, file_count, created_at, is_root, parent_version_id, user_id)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rusqlite::params![
                version_id,
                model_id,
                "Original",
                0,
                model_path,
                size_bytes,
                file_count,
                now,
                1, // is_root = true
                Option::<String>::None,
                user_id
            ],
        ).map_err(|e| format!("Failed to create root version: {}", e))?;

        Ok(version_id)
    }

    fn get_all_models(&self) -> Result<Vec<crate::model_manager::ModelInfo>, String> {
        // CRITICAL: Filter by current user_id
        let user_id = self.require_user_id()?;
        
        let mut stmt = self.conn.prepare(
            "SELECT id, name, description, model_path, created_at 
             FROM models 
             WHERE user_id = ?1
             ORDER BY created_at DESC"
        ).map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let models = stmt.query_map([user_id], |row| {
            let created_at_str: String = row.get(4)?;
            let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(|_| chrono::Utc::now());

            // model_path dient als local_path (tatsächlicher Pfad auf der Festplatte)
            let model_path: Option<String> = row.get(3).ok();
            
            Ok(crate::model_manager::ModelInfo {
                id: row.get(0)?,
                name: row.get(1)?,
                source: row.get::<_, Option<String>>(2)?.unwrap_or_else(|| "local".to_string()),
                source_path: model_path.clone(),
                local_path: model_path.unwrap_or_default(),
                size_bytes: 0,
                file_count: 0,
                created_at,
                model_type: None,
                plugin_override: None,
            })
        }).map_err(|e| format!("Failed to query models: {}", e))?;

        models.collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to collect models: {}", e))
    }
}

// ============ Tauri Commands ============

#[tauri::command]
pub fn list_models_with_versions(state: State<AppState>) -> Result<Vec<ModelWithVersions>, String> {
    let db = state.db.lock().map_err(|e| format!("Failed to lock database: {}", e))?;
    
    // Ensure version tables exist
    db.create_version_tables()?;
    
    db.get_models_with_versions()
}

#[tauri::command]
pub fn list_model_versions(model_id: String, state: State<AppState>) -> Result<Vec<ModelVersion>, String> {
    let db = state.db.lock().map_err(|e| format!("Failed to lock database: {}", e))?;
    
    // Ensure version tables exist
    db.create_version_tables()?;
    
    db.get_model_version_details(&model_id)
}

#[tauri::command]
pub fn delete_model_version(version_id: String, state: State<AppState>) -> Result<(), String> {
    let db = state.db.lock().map_err(|e| format!("Failed to lock database: {}", e))?;
    db.delete_version(&version_id)
}

#[tauri::command]
pub fn rename_model_version(version_id: String, new_name: String, state: State<AppState>) -> Result<(), String> {
    let db = state.db.lock().map_err(|e| format!("Failed to lock database: {}", e))?;
    db.rename_version(&version_id, &new_name)
}

#[tauri::command]
pub fn list_models_with_version_tree(state: State<AppState>) -> Result<Vec<ModelWithVersionTree>, String> {
    let db = state.db.lock().map_err(|e| format!("Failed to lock database: {}", e))?;
    
    // Ensure version tables exist
    db.create_version_tables()?;
    
    db.get_models_with_version_tree()
}

#[tauri::command]
pub fn get_version_path_for_ui(version_id: String, state: State<AppState>) -> Result<String, String> {
    let db = state.db.lock().map_err(|e| format!("Failed to lock database: {}", e))?;
    
    db.get_version_path(&version_id)
}

/// Laedt eine Version als Modell-Repository zu HuggingFace hoch.
///
/// Der Upload laeuft ueber `huggingface_hub` in genau dem Python, mit dem auch
/// trainiert wird (`resolve_python`) — dieselbe Umgebung, die das Paket schon
/// mitbringt. Token und Pfade gehen als Umgebungsvariablen an ein Skript in
/// einer Temp-Datei, damit der Token nicht in der Prozessliste sichtbar wird.
#[derive(Clone, Serialize)]
struct HfUploadProgress {
    version_id: String,
    /// 0–100; -1 bedeutet "unbestimmt" (laeuft, aber noch keine Prozentzahl)
    percent: f64,
    /// "starting" | "uploading" | "done" | "error"
    phase: String,
    /// Menschlich lesbarer Status, z. B. der Dateiname der gerade laeuft
    message: String,
    /// Xet-Upload (alter create_commit-Pfad): fertig verarbeitete / gesamte Dateien
    files_done: Option<i64>,
    files_total: Option<i64>,
    /// pipelined_upload: verarbeitete / gesamte Bytes
    bytes_done: Option<i64>,
    bytes_total: Option<i64>,
    /// Geglaettetes Tempo in Bytes pro Sekunde
    speed: Option<f64>,
}

/// Zusatzangaben einer Fortschrittszeile — je nach Upload-Pfad unterschiedlich gefuellt.
#[derive(Debug, Default, Clone, PartialEq)]
struct ProgressDetail {
    files: Option<(i64, i64)>,
    bytes: Option<(i64, i64)>,
    speed: Option<f64>,
}

/// Eine stdout-Zeile des Upload-Skripts.
#[derive(Debug, PartialEq)]
enum UploadLine {
    /// {"progress": {"percent": .., "desc"?, "files_done"?, "files_total"?,
    ///               "bytes_done"?, "bytes_total"?, "speed"?}}
    Progress { percent: f64, desc: String, detail: ProgressDetail },
    /// {"phase": "create_repo"} — Abschnitt ohne Prozentangabe
    Phase(String),
    /// {"ok": ..} — Endergebnis (die komplette Zeile)
    Result(String),
    /// Alles andere (Ausgaben von Bibliotheken usw.)
    Other,
}

fn parse_upload_line(line: &str) -> UploadLine {
    let trimmed = line.trim();
    if !trimmed.starts_with('{') {
        return UploadLine::Other;
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return UploadLine::Other;
    };
    if let Some(p) = v.get("progress") {
        let percent = p.get("percent").and_then(|x| x.as_f64()).unwrap_or(-1.0);
        let desc = p.get("desc").and_then(|x| x.as_str()).unwrap_or("").to_string();
        let pair = |a: &str, b: &str| match (
            p.get(a).and_then(|x| x.as_i64()),
            p.get(b).and_then(|x| x.as_i64()),
        ) {
            (Some(d), Some(t)) if t > 0 => Some((d, t)),
            _ => None,
        };
        let detail = ProgressDetail {
            files: pair("files_done", "files_total"),
            bytes: pair("bytes_done", "bytes_total"),
            speed: p.get("speed").and_then(|x| x.as_f64()).filter(|s| s.is_finite() && *s >= 0.0),
        };
        UploadLine::Progress { percent, desc, detail }
    } else if let Some(phase) = v.get("phase").and_then(|x| x.as_str()) {
        UploadLine::Phase(phase.to_string())
    } else if v.get("ok").is_some() {
        UploadLine::Result(trimmed.to_string())
    } else {
        UploadLine::Other
    }
}

/// Protokoll des letzten Uploads (ohne Token) mit Zeitstempeln — damit sich
/// nachvollziehen laesst, WANN welche Zeile kam, falls der Fortschritt wieder
/// nicht ankommt. Schreibt sofort durch, die Datei ist auch mitten im Lauf aktuell.
struct TraceLog {
    file: std::sync::Mutex<Option<fs::File>>,
    t0: std::time::Instant,
}

impl TraceLog {
    fn open(path: Option<&Path>) -> Self {
        let file = path.and_then(|p| {
            if let Some(dir) = p.parent() {
                let _ = fs::create_dir_all(dir);
            }
            fs::File::create(p).ok()
        });
        TraceLog { file: std::sync::Mutex::new(file), t0: std::time::Instant::now() }
    }

    fn line(&self, tag: &str, text: &str) {
        if let Ok(mut guard) = self.file.lock() {
            if let Some(f) = guard.as_mut() {
                let _ = writeln!(f, "[{:>8.2}s] {} {}", self.t0.elapsed().as_secs_f64(), tag, text);
            }
        }
    }
}

fn emit_hf_progress(
    app: &tauri::AppHandle,
    version_id: &str,
    percent: f64,
    phase: &str,
    message: String,
    detail: ProgressDetail,
) {
    let _ = app.emit("hf-upload-progress", HfUploadProgress {
        version_id: version_id.to_string(),
        percent,
        phase: phase.to_string(),
        message,
        files_done: detail.files.map(|f| f.0),
        files_total: detail.files.map(|f| f.1),
        bytes_done: detail.bytes.map(|b| b.0),
        bytes_total: detail.bytes.map(|b| b.1),
        speed: detail.speed,
    });
}

#[tauri::command]
pub async fn upload_model_version_huggingface(
    app_handle: tauri::AppHandle,
    version_id: String,
    repo_id: String,
    token: String,
    private: bool,
    state: tauri::State<'_, crate::AppState>,
) -> Result<String, String> {
    let token = token.trim().to_string();
    let repo_id = repo_id.trim().to_string();
    if token.is_empty() {
        return Err("Kein HuggingFace-Token angegeben".to_string());
    }
    if repo_id.is_empty() {
        return Err("Kein Repository-Name angegeben".to_string());
    }

    // Versionspfad mit user_id-Pruefung ermitteln (wie beim lokalen Export)
    let user_id = {
        let db = state.db.lock()
            .map_err(|e| format!("Failed to lock database: {}", e))?;
        db.require_user_id()?.to_string()
    };
    let version_path: String = {
        let db = state.db.lock()
            .map_err(|e| format!("Failed to lock database: {}", e))?;
        db.conn.query_row(
            "SELECT path FROM model_versions_new WHERE id = ?1 AND user_id = ?2",
            rusqlite::params![&version_id, &user_id],
            |row| row.get(0),
        ).map_err(|e| format!("Version not found or access denied: {}", e))?
    };

    if !Path::new(&version_path).exists() {
        return Err(format!("Versionsordner nicht gefunden: {}", version_path));
    }

    // Der eigentliche Upload blockiert (Python-Prozess, Netzwerk). Als synchroner
    // Command lief er im Main-Thread und fror die UI ein (Beachball, keine Bar).
    // Darum in einen Blocking-Thread auslagern — der Main-Thread bleibt frei und
    // die Fortschritts-Events kommen live an.
    tauri::async_runtime::spawn_blocking(move || {
        run_hf_upload(app_handle, version_id, repo_id, token, private, version_path)
    })
    .await
    .map_err(|e| format!("Upload-Task fehlgeschlagen: {}", e))?
}

/// Blockierender Teil des HuggingFace-Uploads: Python starten, Fortschritt streamen,
/// Ergebnis parsen. Laeuft in einem Blocking-Thread (spawn_blocking), damit der
/// Main-Thread frei bleibt.
fn run_hf_upload(
    app_handle: tauri::AppHandle,
    version_id: String,
    repo_id: String,
    token: String,
    private: bool,
    version_path: String,
) -> Result<String, String> {
    let python = crate::python_env::resolve_python();

    // Skript in eine Temp-Datei — nichts landet auf der Kommandozeile
    let script = r#"import os, json, sys
import time as _time

# Fortschritt selbst melden statt die tqdm-Ausgabe zu parsen: huggingface_hub
# zeichnet seine Balken mit tqdm (die hf-Klasse erbt von tqdm.std.tqdm). update()
# der Basisklasse abfangen und je ganzem Prozent eine JSON-Zeile auf stdout
# schreiben — das Backend liest stdout zeilenweise live mit.
#
# Mit hf_xet (Standard in huggingface_hub 1.x) gibt es zwei Gesamtbalken:
#   "Processing Files (a / b)" — ueber alle Bytes; Chunking, Dedup und Upload
#                                laufen parallel -> das ist der Hauptbalken.
#   "New Data Upload"          — nur tatsaechlich neue Bytes nach Dedup; bei schon
#                                bekannten Modellen fast 0 -> ignorieren, sonst
#                                springt der Balken zwischen zwei Werten.
# Die Dateibalken setzen .n direkt (ohne update) und tauchen hier nicht auf.
# Ohne hf_xet (klassischer LFS-Pfad) laeuft je Datei ein Balken mit dem Dateinamen.
import re as _re
_FILES_RE = _re.compile(r"\((\d+)\s*/\s*(\d+)\)")

def _install_progress():
    try:
        from tqdm import tqdm as _T
    except Exception:
        return
    _orig = _T.update
    def _upd(self, n=1):
        r = _orig(self, n)
        try:
            if not self.total:
                return r
            desc = (self.desc or "").strip()
            if desc.endswith(":"):
                desc = desc[:-1].strip()
            if desc.startswith("New Data Upload"):
                return r
            pct = min(100.0, self.n * 100.0 / self.total)
            ip = int(pct)
            now = _time.monotonic()
            # Tempo je Balken (geglaettet), nur bei Byte-Balken sinnvoll
            if getattr(self, "unit", "") == "B":
                prev = getattr(self, "_ft_prev", None)
                if prev is None:
                    self._ft_prev = (now, self.n)
                    self._ft_speed = 0.0
                elif now - prev[0] >= 0.5:
                    rate = max(0.0, (self.n - prev[1]) / (now - prev[0]))
                    old = getattr(self, "_ft_speed", 0.0)
                    self._ft_speed = rate if old == 0 else 0.3 * rate + 0.7 * old
                    self._ft_prev = (now, self.n)
            # Neues ganzes Prozent sofort, sonst hoechstens jede Sekunde (Tempo/Restzeit)
            if getattr(self, "_ft_last", -1) == ip and now - getattr(self, "_ft_last_emit", 0.0) < 1.0:
                return r
            self._ft_last = ip
            self._ft_last_emit = now
            info = {"percent": pct}
            if getattr(self, "unit", "") == "B":
                info["bytes_done"] = int(self.n)
                info["bytes_total"] = int(self.total)
                info["speed"] = getattr(self, "_ft_speed", 0.0)
            if desc.startswith("Processing Files"):
                m = _FILES_RE.search(desc)
                if m:
                    info["files_done"] = int(m.group(1))
                    info["files_total"] = int(m.group(2))
            else:
                info["desc"] = desc
            print(json.dumps({"progress": info}), flush=True)
        except Exception:
            pass
        return r
    _T.update = _upd

# Abschnitte ohne Prozentangabe melden, damit die UI nicht minutenlang nur
# "Wird vorbereitet" zeigt und das Protokoll zeigt, wo die Zeit bleibt.
def _phase(name):
    print(json.dumps({"phase": name}), flush=True)

# huggingface_hub >= 1.x laedt mit hf_xet ueber pipelined_upload (_upload_pipeline).
# Dessen _LiveDisplay zeichnet nur, wenn stderr ein Terminal ist, und reicht sonst
# (Loglevel WARNING) hf_xet GAR KEINEN Fortschritts-Callback durch — in der App ist
# stderr eine Pipe, also kam nie Fortschritt an. Eigene Anzeige: immer aktiv, zeichnet
# nie auf stderr, meldet Bytes, Tempo und Prozent als JSON auf stdout.
def _install_pipeline_progress():
    import time as _time
    try:
        import huggingface_hub._upload_pipeline as _up
        _Base = _up._LiveDisplay
    except Exception:
        return  # aeltere huggingface_hub ohne Pipeline -> der tqdm-Patch greift

    # Eingriff so klein wie moeglich: nur "aktiv, aber nie aufs Terminal zeichnen".
    # Ohne TTY zeichnet die Basisklasse nichts; ihr Render-Thread schreibt nur
    # logger.info, das beim Standard-Loglevel stumm bleibt.
    class _FTDisplay(_Base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._tty = False
            self._active = True
            self._ft_commits = {}
            self._ft_last_pct = -1
            self._ft_last_emit = 0.0
            self._ft_prev = None
            self._ft_speed = 0.0

        def new_xet_callback(self):
            inner = super().new_xet_callback()
            key = object()
            def cb(group_report, item_reports):
                if inner is not None:
                    inner(group_report, item_reports)
                # Unsere Meldung darf den Upload nie gefaehrden — aendert eine kuenftige
                # huggingface_hub-Version Interna, faellt die UI nur auf "unbestimmt" zurueck.
                try:
                    # Mehrere Upload-Commits koennen gleichzeitig laufen; ihre Zaehler sind
                    # je Commit kumulativ -> je Commit merken und aufsummieren.
                    with self._lock:
                        self._ft_commits[key] = (group_report.total_bytes_completed, group_report.total_bytes)
                        done = sum(v[0] for v in self._ft_commits.values())
                        total = sum(v[1] for v in self._ft_commits.values())
                    self._ft_report(done, total)
                except Exception:
                    pass
            return cb

        def _ft_report(self, done, total):
            if total <= 0:
                return
            now = _time.monotonic()
            if self._ft_prev is None:
                self._ft_prev = (now, done)
            elif now - self._ft_prev[0] >= 0.5:
                rate = max(0.0, (done - self._ft_prev[1]) / (now - self._ft_prev[0]))
                self._ft_speed = rate if self._ft_speed == 0 else 0.3 * rate + 0.7 * self._ft_speed
                self._ft_prev = (now, done)
            pct = min(100.0, done * 100.0 / total)
            ip = int(pct)
            # Neues ganzes Prozent sofort, sonst hoechstens jede Sekunde (Tempo/Restzeit)
            if ip == self._ft_last_pct and now - self._ft_last_emit < 1.0:
                return
            self._ft_last_pct = ip
            self._ft_last_emit = now
            print(json.dumps({"progress": {
                "percent": pct, "bytes_done": int(done), "bytes_total": int(total),
                "speed": self._ft_speed,
            }}), flush=True)

    _up._LiveDisplay = _FTDisplay

# Fortschritt ist Beiwerk: scheitert die Installation, laeuft der Upload trotzdem.
for _install in (_install_progress, _install_pipeline_progress):
    try:
        _install()
    except Exception:
        pass

try:
    _phase("connect")
    from huggingface_hub import HfApi
    try:
        import huggingface_hub as _hh
        from huggingface_hub.utils._runtime import is_xet_available as _xa
        print(json.dumps({"info": {"hf_hub": _hh.__version__, "xet": bool(_xa()), "python": sys.version.split()[0]}}), flush=True)
    except Exception:
        pass
    api = HfApi(token=os.environ["FT_HF_TOKEN"])
    repo_id = os.environ["FT_HF_REPO"].strip()
    private = os.environ.get("FT_HF_PRIVATE") == "1"
    # Ohne Namensraum ("yolo8n") wuerde create_repo unter dem eigenen Account
    # anlegen (deinname/yolo8n), der Upload aber woertlich "yolo8n" suchen -> 404.
    # Darum den Account aus dem Token ergaenzen, wenn kein "/" angegeben ist.
    if "/" not in repo_id:
        me = api.whoami()
        user = me.get("name") if isinstance(me, dict) else None
        if not user:
            raise RuntimeError("Konnte den HuggingFace-Account zum Token nicht ermitteln.")
        repo_id = user + "/" + repo_id
    _phase("create_repo")
    created = api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    # Die von create_repo zurueckgegebene ID ist die kanonische — die fuer den Upload nehmen.
    canonical = getattr(created, "repo_id", None) or repo_id
    _phase("upload")
    api.upload_folder(folder_path=os.environ["FT_HF_FOLDER"], repo_id=canonical, repo_type="model")
    print(json.dumps({"ok": True, "url": "https://huggingface.co/" + canonical}))
except ImportError:
    print(json.dumps({"ok": False, "error": "huggingface_hub ist in dieser Python-Umgebung nicht installiert (pip install huggingface_hub)."}))
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)}))
"#;

    let tmp = std::env::temp_dir()
        .join(format!("ft_hf_upload_{}.py", uuid::Uuid::new_v4()));
    fs::write(&tmp, script)
        .map_err(|e| format!("Temp-Datei konnte nicht geschrieben werden: {}", e))?;

    println!("[HF-Upload] Lade Version {} nach {} hoch", version_id, repo_id);

    // "Los geht's" — die UI zeigt sofort einen (unbestimmten) Balken
    emit_hf_progress(&app_handle, &version_id, -1.0, "starting", String::new(), ProgressDetail::default());

    let log_path = app_handle.path().app_log_dir().ok().map(|d| d.join("hf-upload-last.log"));
    let trace = std::sync::Arc::new(TraceLog::open(log_path.as_deref()));
    trace.line("INFO", &format!("python={} repo={} ordner={}", python, repo_id, version_path));
    if let Ok(entries) = fs::read_dir(&version_path) {
        for e in entries.flatten() {
            let size = e.metadata().map(|m| m.len()).unwrap_or(0);
            trace.line("INFO", &format!("datei {} ({} Bytes)", e.file_name().to_string_lossy(), size));
        }
    }

    let mut child = Command::new(&python)
        .no_window()
        .python_utf8()
        .arg(tmp.to_string_lossy().to_string())
        .env("FT_HF_TOKEN", &token)
        .env("FT_HF_REPO", &repo_id)
        .env("FT_HF_FOLDER", &version_path)
        .env("FT_HF_PRIVATE", if private { "1" } else { "0" })
        // tqdm muss aktiv sein, sonst ruft niemand update() auf
        .env("HF_HUB_DISABLE_PROGRESS_BARS", "0")
        // hf_transfer (Rust) meldet Fortschritt anders — Standard-Upload erzwingen
        .env("HF_HUB_ENABLE_HF_TRANSFER", "0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            let _ = fs::remove_file(&tmp);
            format!("Python konnte nicht gestartet werden: {}", e)
        })?;

    // stderr: fuer Fehlermeldungen sammeln und mit Zeitstempel protokollieren.
    // tqdm aktualisiert per '\r' — darum an '\r' und '\n' zerlegen. Eigener Thread,
    // damit eine volle stderr-Pipe den Prozess nicht blockiert.
    let stderr = child.stderr.take();
    let trace_err = trace.clone();
    let stderr_handle = std::thread::spawn(move || {
        let mut collected = String::new();
        let Some(se) = stderr else { return collected };
        let mut reader = BufReader::new(se);
        let mut buf = [0u8; 8192];
        let mut seg: Vec<u8> = Vec::new();
        let mut last = String::new();
        let mut logged = 0usize;
        let flush_seg = |seg: &mut Vec<u8>, collected: &mut String, last: &mut String, logged: &mut usize| {
            if seg.is_empty() { return; }
            let s = String::from_utf8_lossy(seg).trim().to_string();
            seg.clear();
            if s.is_empty() || s == *last { return; }
            if *logged < 3000 {
                trace_err.line("ERR", &s);
                *logged += 1;
            }
            // Fuer eine eventuelle Fehlermeldung nur die letzten ~8 KB behalten
            collected.push_str(&s);
            collected.push('\n');
            if collected.len() > 8192 {
                let cut = collected.len() - 8192;
                let cut = (cut..collected.len()).find(|i| collected.is_char_boundary(*i)).unwrap_or(0);
                collected.drain(..cut);
            }
            *last = s;
        };
        loop {
            match reader.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    for &b in &buf[..n] {
                        if b == b'\r' || b == b'\n' {
                            flush_seg(&mut seg, &mut collected, &mut last, &mut logged);
                        } else {
                            seg.push(b);
                        }
                    }
                }
            }
        }
        flush_seg(&mut seg, &mut collected, &mut last, &mut logged);
        collected
    });

    // stdout live zeilenweise: Fortschritt/Phase -> Event, {"ok": ...} -> Ergebnis
    let mut result_line: Option<String> = None;
    let mut last_emitted: i32 = -2;
    let mut progress_lines = 0usize;
    if let Some(so) = child.stdout.take() {
        for line in BufReader::new(so).lines() {
            let line = match line { Ok(l) => l, Err(_) => break };
            match parse_upload_line(&line) {
                UploadLine::Progress { percent, desc, detail } => {
                    progress_lines += 1;
                    trace.line("OUT", line.trim());
                    // Python drosselt schon (je Prozent bzw. hoechstens 1/s) — jede Zeile
                    // weiterreichen, damit Tempo und Restzeit live bleiben.
                    let rounded = percent.round() as i32;
                    if rounded != last_emitted && rounded % 10 == 0 {
                        println!("[HF-Upload] {}%", rounded);
                    }
                    last_emitted = rounded;
                    emit_hf_progress(&app_handle, &version_id, percent, "uploading", desc, detail);
                }
                UploadLine::Phase(phase) => {
                    trace.line("OUT", line.trim());
                    println!("[HF-Upload] Phase: {}", phase);
                    emit_hf_progress(&app_handle, &version_id, -1.0, &phase, String::new(), ProgressDetail::default());
                }
                UploadLine::Result(r) => {
                    trace.line("OUT", "{\"ok\": ...} (Ergebnis)");
                    result_line = Some(r);
                }
                UploadLine::Other => {
                    if !line.trim().is_empty() {
                        trace.line("OUT", line.trim());
                    }
                }
            }
        }
    }

    let stderr_str = stderr_handle.join().unwrap_or_default();
    let status = child.wait();
    let _ = fs::remove_file(&tmp);
    trace.line("INFO", &format!("prozess beendet: {:?}, fortschrittszeilen={}", status.map(|s| s.code()), progress_lines));
    println!(
        "[HF-Upload] {} Fortschrittszeilen empfangen. Protokoll: {}",
        progress_lines,
        log_path.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "-".into())
    );

    let json_line = result_line.ok_or_else(|| {
        emit_hf_progress(&app_handle, &version_id, -1.0, "error", String::new(), ProgressDetail::default());
        format!("Unerwartete Antwort vom Upload.\nstderr: {}", stderr_str.trim())
    })?;

    let parsed: serde_json::Value = serde_json::from_str(json_line.trim())
        .map_err(|e| format!("Antwort nicht lesbar: {}", e))?;

    if parsed.get("ok").and_then(|b| b.as_bool()).unwrap_or(false) {
        let url = parsed.get("url").and_then(|u| u.as_str()).unwrap_or("").to_string();
        println!("[HF-Upload] ✅ fertig: {}", url);
        emit_hf_progress(&app_handle, &version_id, 100.0, "done", String::new(), ProgressDetail::default());
        Ok(url)
    } else {
        emit_hf_progress(&app_handle, &version_id, -1.0, "error", String::new(), ProgressDetail::default());
        Err(parsed.get("error").and_then(|e| e.as_str())
            .unwrap_or("Upload fehlgeschlagen").to_string())
    }
}

#[tauri::command]
pub fn list_version_files(path: String) -> Result<Vec<serde_json::Value>, String> {
    let entries = fs::read_dir(&path).map_err(|e| e.to_string())?;
    let mut files = Vec::new();

    for entry in entries.flatten() {
        let meta = entry.metadata().map_err(|e| e.to_string())?;
        files.push(serde_json::json!({
            "name": entry.file_name().to_string_lossy().to_string(),
            "path": entry.path().to_string_lossy().to_string(),
            "size_bytes": if meta.is_file() { meta.len() } else { 0 },
            "is_dir": meta.is_dir(),
        }));
    }

    Ok(files)
}

#[cfg(test)]
mod hf_upload_tests {
    use super::*;

    #[test]
    fn xet_fortschritt_mit_dateizahl() {
        let l = r#"{"progress": {"percent": 42.5, "files_done": 1, "files_total": 4}}"#;
        assert_eq!(
            parse_upload_line(l),
            UploadLine::Progress { percent: 42.5, desc: String::new(), detail: ProgressDetail { files: Some((1, 4)), ..Default::default() } }
        );
    }

    #[test]
    fn lfs_fortschritt_mit_dateiname() {
        let l = r#"{"progress": {"percent": 7.0, "desc": "model.safetensors"}}"#;
        assert_eq!(
            parse_upload_line(l),
            UploadLine::Progress { percent: 7.0, desc: "model.safetensors".into(), detail: ProgressDetail::default() }
        );
    }

    #[test]
    fn dateizahl_ohne_gesamt_wird_verworfen() {
        let l = r#"{"progress": {"percent": 3.0, "files_done": 0, "files_total": 0}}"#;
        assert_eq!(
            parse_upload_line(l),
            UploadLine::Progress { percent: 3.0, desc: String::new(), detail: ProgressDetail::default() }
        );
    }

    #[test]
    fn pipeline_fortschritt_mit_bytes_und_tempo() {
        // So meldet _FTDisplay im pipelined_upload (hf_xet, huggingface_hub 1.x)
        let l = r#"{"progress": {"percent": 50.0, "bytes_done": 7863375, "bytes_total": 15728640, "speed": 524225.5}}"#;
        assert_eq!(
            parse_upload_line(l),
            UploadLine::Progress {
                percent: 50.0,
                desc: String::new(),
                detail: ProgressDetail { files: None, bytes: Some((7863375, 15728640)), speed: Some(524225.5) },
            }
        );
    }

    #[test]
    fn unsinniges_tempo_wird_verworfen() {
        let l = r#"{"progress": {"percent": 1.0, "bytes_done": 1, "bytes_total": 2, "speed": -5}}"#;
        let UploadLine::Progress { detail, .. } = parse_upload_line(l) else { panic!("keine Fortschrittszeile") };
        assert_eq!(detail.speed, None);
    }

    #[test]
    fn phase_und_ergebnis() {
        assert_eq!(parse_upload_line(r#"{"phase": "create_repo"}"#), UploadLine::Phase("create_repo".into()));
        let ok = r#"{"ok": true, "url": "https://huggingface.co/a/b"}"#;
        assert_eq!(parse_upload_line(ok), UploadLine::Result(ok.into()));
        let err = r#"{"ok": false, "error": "401"}"#;
        assert_eq!(parse_upload_line(err), UploadLine::Result(err.into()));
    }

    #[test]
    fn fremde_ausgaben_werden_ignoriert() {
        assert_eq!(parse_upload_line("Processing Files (0 / 1): 12%|#"), UploadLine::Other);
        assert_eq!(parse_upload_line(r#"{"info": {"xet": true}}"#), UploadLine::Other);
        assert_eq!(parse_upload_line("{kaputt"), UploadLine::Other);
        assert_eq!(parse_upload_line(""), UploadLine::Other);
    }
}
