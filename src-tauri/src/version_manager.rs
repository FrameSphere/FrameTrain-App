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
use std::path::{Path, PathBuf};
use tauri::{Manager, State};
use crate::database::Database;
use crate::AppState;

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
                created_at: row.get::<_, String>(7)?, // ✅ FIXED: Explizit String statt DateTime
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
            // Parse created_at string to DateTime
            let created_at_str: String = row.get(4)?;
            let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(|_| chrono::Utc::now());
            
            Ok(crate::model_manager::ModelInfo {
                id: row.get(0)?,
                name: row.get(1)?,
                source: row.get::<_, Option<String>>(2)?.unwrap_or_else(|| "local".to_string()),  // description als source
                source_path: row.get(3).ok(),  // model_path
                size_bytes: 0,  // Nicht in dieser Tabelle
                file_count: 0,  // Nicht in dieser Tabelle
                created_at,
                model_type: None,  // Nicht in dieser Tabelle
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
