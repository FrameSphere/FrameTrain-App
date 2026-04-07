// Database Module für FrameTrain Desktop App
// SQLite Integration für lokale Datenverwaltung

use rusqlite::{Connection, params, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub base_model: Option<String>,
    pub model_path: Option<String>,
    pub status: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelVersion {
    pub id: String,
    pub model_id: String,
    pub version: i32,
    pub version_path: Option<String>,
    pub status: String,
    pub created_at: String,
    pub completed_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub id: String,
    pub version_id: String,
    pub dataset_path: String,
    pub epochs: i32,
    pub batch_size: i32,
    pub learning_rate: f64,
    pub optimizer: String,
    pub loss_function: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub id: Option<i64>,
    pub version_id: String,
    pub epoch: i32,
    pub loss: f64,
    pub accuracy: Option<f64>,
    pub val_loss: Option<f64>,
    pub val_accuracy: Option<f64>,
    pub timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Dataset {
    pub id: String,
    pub name: String,
    pub file_path: String,
    pub file_type: String,
    pub size_bytes: Option<i64>,
    pub rows_count: Option<i32>,
    pub columns_count: Option<i32>,
    pub validated: bool,
    pub created_at: String,
}

pub struct Database {
    pub(crate) conn: Connection,
    current_user_id: Option<String>,
}

impl Database {
    /// Erstellt oder öffnet die SQLite Datenbank
    pub fn new(db_path: PathBuf) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        
        // CRITICAL: Disable foreign key constraints globally
        conn.execute("PRAGMA foreign_keys = OFF", [])?;
        
        // Initialisiere Schema
        Self::init_schema(&conn)?;
        
        // Run user isolation migration
        Self::run_user_isolation_migration(&conn)?;
        
        Ok(Database { 
            conn,
            current_user_id: None,
        })
    }
    
    /// Initialisiert das Datenbank-Schema
    fn init_schema(conn: &Connection) -> Result<()> {
        let schema = r#"
-- FrameTrain Desktop App Database Schema

-- Models Table
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    base_model TEXT,
    model_path TEXT,
    status TEXT NOT NULL DEFAULT 'created',
    user_id TEXT NOT NULL DEFAULT 'default_user',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Model Versions Table
CREATE TABLE IF NOT EXISTS model_versions (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    version_path TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    user_id TEXT NOT NULL DEFAULT 'default_user',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT
);

-- Training Configs Table
CREATE TABLE IF NOT EXISTS training_configs (
    id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL,
    dataset_path TEXT NOT NULL,
    epochs INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    learning_rate REAL NOT NULL,
    optimizer TEXT NOT NULL,
    loss_function TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Training Metrics Table
CREATE TABLE IF NOT EXISTS training_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    loss REAL NOT NULL,
    accuracy REAL,
    val_loss REAL,
    val_accuracy REAL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Datasets Table
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    size_bytes INTEGER,
    rows_count INTEGER,
    columns_count INTEGER,
    validated BOOLEAN NOT NULL DEFAULT 0,
    user_id TEXT NOT NULL DEFAULT 'default_user',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Test Results Table
CREATE TABLE IF NOT EXISTS test_results (
    id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL,
    total_samples INTEGER NOT NULL,
    correct_predictions INTEGER NOT NULL,
    accuracy REAL NOT NULL,
    average_loss REAL,
    average_inference_time REAL NOT NULL,
    results_json TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT 'default_user',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- App Config Table
CREATE TABLE IF NOT EXISTS app_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Create Indexes
CREATE INDEX IF NOT EXISTS idx_models_user_id ON models(user_id);
CREATE INDEX IF NOT EXISTS idx_model_versions_model_id ON model_versions(model_id);
CREATE INDEX IF NOT EXISTS idx_model_versions_user_id ON model_versions(user_id);
CREATE INDEX IF NOT EXISTS idx_training_metrics_version_id ON training_metrics(version_id);
CREATE INDEX IF NOT EXISTS idx_datasets_user_id ON datasets(user_id);
CREATE INDEX IF NOT EXISTS idx_test_results_version_id ON test_results(version_id);
CREATE INDEX IF NOT EXISTS idx_test_results_user_id ON test_results(user_id);
        "#;
        conn.execute_batch(schema)?;
        Ok(())
    }
    
    /// Führt die User-Isolation Migration aus
    fn run_user_isolation_migration(conn: &Connection) -> Result<()> {
        // Prüfe ob Migration bereits durchgeführt wurde
        let migration_check: Result<i32> = conn.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='user_id'",
            [],
            |row| row.get(0)
        );
        
        if let Ok(count) = migration_check {
            if count > 0 {
                // Migration bereits durchgeführt - nur Indexes erstellen
                let _ = conn.execute("CREATE INDEX IF NOT EXISTS idx_models_user_id ON models(user_id)", []);
                let _ = conn.execute("CREATE INDEX IF NOT EXISTS idx_model_versions_user_id ON model_versions(user_id)", []);
                let _ = conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_user_id ON datasets(user_id)", []);
                let _ = conn.execute("CREATE INDEX IF NOT EXISTS idx_test_results_user_id ON test_results(user_id)", []);
                return Ok(());
            }
        }
        
        // Führe Migration aus - mit Fehlerbehandlung für bereits existierende Spalten
        // Add user_id columns
        let _ = conn.execute("ALTER TABLE models ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default_user'", []);
        let _ = conn.execute("ALTER TABLE model_versions ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default_user'", []);
        let _ = conn.execute("ALTER TABLE datasets ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default_user'", []);
        let _ = conn.execute("ALTER TABLE test_results ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default_user'", []);
        
        // Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_models_user_id ON models(user_id)", [])?;
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model_versions_user_id ON model_versions(user_id)", [])?;
        conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_user_id ON datasets(user_id)", [])?;
        conn.execute("CREATE INDEX IF NOT EXISTS idx_test_results_user_id ON test_results(user_id)", [])?;
        
        Ok(())
    }
    
    /// Setzt die aktuelle User-ID für alle Datenbankoperationen
    pub fn set_current_user(&mut self, user_id: String) {
        self.current_user_id = Some(user_id);
    }
    
    /// Entfernt die aktuelle User-ID (Logout)
    pub fn clear_current_user(&mut self) {
        self.current_user_id = None;
    }
    
    /// Gibt die aktuelle User-ID zurück oder Fehler
    pub fn require_user_id(&self) -> std::result::Result<&String, String> {
        self.current_user_id.as_ref()
            .ok_or_else(|| "Kein Benutzer angemeldet".to_string())
    }
    
    /// Gibt die aktuelle User-ID zurück (für externe Verwendung)
    pub fn get_current_user_id(&self) -> Option<String> {
        self.current_user_id.clone()
    }
    
    // ==================== MODELS ====================
    
    pub fn create_model(&self, model: &Model) -> Result<()> {
        let user_id = self.require_user_id().map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::new(std::io::ErrorKind::PermissionDenied, e))))?;
        
        self.conn.execute(
            "INSERT INTO models (id, name, description, base_model, model_path, status, user_id) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                &model.id,
                &model.name,
                &model.description,
                &model.base_model,
                &model.model_path,
                &model.status,
                user_id
            ],
        )?;
        Ok(())
    }
    
    pub fn get_model(&self, id: &str) -> Result<Model> {
        let user_id = self.require_user_id().map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::new(std::io::ErrorKind::PermissionDenied, e))))?;
        
        self.conn.query_row(
            "SELECT id, name, description, base_model, model_path, status, created_at, updated_at 
             FROM models WHERE id = ?1 AND user_id = ?2",
            params![id, user_id],
            |row| {
                Ok(Model {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    description: row.get(2)?,
                    base_model: row.get(3)?,
                    model_path: row.get(4)?,
                    status: row.get(5)?,
                    created_at: row.get(6)?,
                    updated_at: row.get(7)?,
                })
            },
        )
    }
    
    pub fn list_models(&self) -> Result<Vec<Model>> {
        let user_id = self.require_user_id().map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::new(std::io::ErrorKind::PermissionDenied, e))))?;
        
        let mut stmt = self.conn.prepare(
            "SELECT id, name, description, base_model, model_path, status, created_at, updated_at 
             FROM models WHERE user_id = ?1 ORDER BY created_at DESC"
        )?;
        
        let models = stmt.query_map(params![user_id], |row| {
            Ok(Model {
                id: row.get(0)?,
                name: row.get(1)?,
                description: row.get(2)?,
                base_model: row.get(3)?,
                model_path: row.get(4)?,
                status: row.get(5)?,
                created_at: row.get(6)?,
                updated_at: row.get(7)?,
            })
        })?;
        
        models.collect()
    }
    
    pub fn update_model_status(&self, id: &str, status: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE models SET status = ?1 WHERE id = ?2",
            params![status, id],
        )?;
        Ok(())
    }
    
    pub fn delete_model(&self, id: &str) -> Result<()> {
        self.conn.execute("DELETE FROM models WHERE id = ?1", params![id])?;
        Ok(())
    }
    
    // ==================== MODEL VERSIONS ====================
    
    pub fn create_version(&self, version: &ModelVersion) -> Result<()> {
        let user_id = self.require_user_id().map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::new(std::io::ErrorKind::PermissionDenied, e))))?;
        
        self.conn.execute(
            "INSERT INTO model_versions (id, model_id, version, version_path, status, user_id) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                &version.id,
                &version.model_id,
                version.version,
                &version.version_path,
                &version.status,
                user_id
            ],
        )?;
        Ok(())
    }
    
    pub fn get_model_versions(&self, model_id: &str) -> Result<Vec<ModelVersion>> {
        let user_id = self.require_user_id().map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::new(std::io::ErrorKind::PermissionDenied, e))))?;
        
        let mut stmt = self.conn.prepare(
            "SELECT id, model_id, version, version_path, status, created_at, completed_at 
             FROM model_versions WHERE model_id = ?1 AND user_id = ?2 ORDER BY version DESC"
        )?;
        
        let versions = stmt.query_map(params![model_id, user_id], |row| {
            Ok(ModelVersion {
                id: row.get(0)?,
                model_id: row.get(1)?,
                version: row.get(2)?,
                version_path: row.get(3)?,
                status: row.get(4)?,
                created_at: row.get(5)?,
                completed_at: row.get(6)?,
            })
        })?;
        
        versions.collect()
    }
    
    pub fn get_next_version_number(&self, model_id: &str) -> Result<i32> {
        let version: Option<i32> = self.conn.query_row(
            "SELECT MAX(version) FROM model_versions WHERE model_id = ?1",
            params![model_id],
            |row| row.get(0),
        )?;
        
        Ok(version.unwrap_or(0) + 1)
    }
    
    // ==================== TRAINING CONFIG ====================
    
    pub fn save_training_config(&self, config: &TrainingConfig) -> Result<()> {
        self.conn.execute(
            "INSERT INTO training_configs 
             (id, version_id, dataset_path, epochs, batch_size, learning_rate, optimizer, loss_function) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                &config.id,
                &config.version_id,
                &config.dataset_path,
                config.epochs,
                config.batch_size,
                config.learning_rate,
                &config.optimizer,
                &config.loss_function
            ],
        )?;
        Ok(())
    }
    
    pub fn get_training_config(&self, version_id: &str) -> Result<TrainingConfig> {
        self.conn.query_row(
            "SELECT id, version_id, dataset_path, epochs, batch_size, learning_rate, optimizer, loss_function 
             FROM training_configs WHERE version_id = ?1",
            params![version_id],
            |row| {
                Ok(TrainingConfig {
                    id: row.get(0)?,
                    version_id: row.get(1)?,
                    dataset_path: row.get(2)?,
                    epochs: row.get(3)?,
                    batch_size: row.get(4)?,
                    learning_rate: row.get(5)?,
                    optimizer: row.get(6)?,
                    loss_function: row.get(7)?,
                })
            },
        )
    }
    
    // ==================== TRAINING METRICS ====================
    
    pub fn save_training_metric(&self, metric: &TrainingMetrics) -> Result<()> {
        self.conn.execute(
            "INSERT INTO training_metrics 
             (version_id, epoch, loss, accuracy, val_loss, val_accuracy) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                &metric.version_id,
                metric.epoch,
                metric.loss,
                metric.accuracy,
                metric.val_loss,
                metric.val_accuracy
            ],
        )?;
        Ok(())
    }
    
    pub fn get_training_metrics(&self, version_id: &str) -> Result<Vec<TrainingMetrics>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, version_id, epoch, loss, accuracy, val_loss, val_accuracy, timestamp 
             FROM training_metrics WHERE version_id = ?1 ORDER BY epoch ASC"
        )?;
        
        let metrics = stmt.query_map(params![version_id], |row| {
            Ok(TrainingMetrics {
                id: Some(row.get(0)?),
                version_id: row.get(1)?,
                epoch: row.get(2)?,
                loss: row.get(3)?,
                accuracy: row.get(4)?,
                val_loss: row.get(5)?,
                val_accuracy: row.get(6)?,
                timestamp: row.get(7)?,
            })
        })?;
        
        metrics.collect()
    }
    
    // ==================== DATASETS ====================
    
    pub fn save_dataset(&self, dataset: &Dataset) -> Result<()> {
        let user_id = self.require_user_id().map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::new(std::io::ErrorKind::PermissionDenied, e))))?;
        
        self.conn.execute(
            "INSERT INTO datasets 
             (id, name, file_path, file_type, size_bytes, rows_count, columns_count, validated, user_id) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                &dataset.id,
                &dataset.name,
                &dataset.file_path,
                &dataset.file_type,
                dataset.size_bytes,
                dataset.rows_count,
                dataset.columns_count,
                dataset.validated,
                user_id
            ],
        )?;
        Ok(())
    }
    
    pub fn list_datasets(&self) -> Result<Vec<Dataset>> {
        let user_id = self.require_user_id().map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::new(std::io::ErrorKind::PermissionDenied, e))))?;
        
        let mut stmt = self.conn.prepare(
            "SELECT id, name, file_path, file_type, size_bytes, rows_count, columns_count, validated, created_at 
             FROM datasets WHERE user_id = ?1 ORDER BY created_at DESC"
        )?;
        
        let datasets = stmt.query_map(params![user_id], |row| {
            Ok(Dataset {
                id: row.get(0)?,
                name: row.get(1)?,
                file_path: row.get(2)?,
                file_type: row.get(3)?,
                size_bytes: row.get(4)?,
                rows_count: row.get(5)?,
                columns_count: row.get(6)?,
                validated: row.get(7)?,
                created_at: row.get(8)?,
            })
        })?;
        
        datasets.collect()
    }
    
    // ==================== TEST RESULTS ====================
    
    pub fn save_test_result(
        &self,
        version_id: &str,
        total_samples: i32,
        correct_predictions: i32,
        accuracy: f64,
        average_loss: f64,
        average_inference_time: f64,
        results_json: &str,
    ) -> Result<String> {
        let user_id = self.require_user_id().map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::new(std::io::ErrorKind::PermissionDenied, e))))?;
        
        let id = format!("test_{}", uuid::Uuid::new_v4());
        let created_at = chrono::Utc::now().to_rfc3339();
        
        self.conn.execute(
            "INSERT INTO test_results (
                id, version_id, total_samples, correct_predictions, 
                accuracy, average_loss, average_inference_time, 
                results_json, created_at, user_id
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                id,
                version_id,
                total_samples,
                correct_predictions,
                accuracy,
                average_loss,
                average_inference_time,
                results_json,
                created_at,
                user_id
            ],
        )?;
        
        Ok(id)
    }
    
    pub fn get_test_results_for_version(&self, version_id: &str) -> Result<Vec<String>> {
        let user_id = self.require_user_id().map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::new(std::io::ErrorKind::PermissionDenied, e))))?;
        
        let mut stmt = self.conn.prepare(
            "SELECT results_json FROM test_results 
             WHERE version_id = ?1 AND user_id = ?2 
             ORDER BY created_at DESC"
        )?;
        
        let results = stmt.query_map(params![version_id, user_id], |row| {
            row.get(0)
        })?;
        
        results.collect()
    }
    
    // ==================== APP CONFIG ====================
    
    pub fn set_config(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO app_config (key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
        Ok(())
    }
    
    pub fn get_config(&self, key: &str) -> Result<String> {
        self.conn.query_row(
            "SELECT value FROM app_config WHERE key = ?1",
            params![key],
            |row| row.get(0),
        )
    }

    // ==================== HELPER METHODS FOR VERSION_MANAGER ====================
    
    pub fn execute_sql(&self, sql: &str, params: &[&dyn rusqlite::ToSql]) -> std::result::Result<usize, String> {
        self.conn.execute(sql, params)
            .map_err(|e| e.to_string())
    }

    pub fn query_row_string(&self, sql: &str, params: &[&dyn rusqlite::ToSql]) -> std::result::Result<String, String> {
        self.conn.query_row(sql, params, |row| row.get(0))
            .map_err(|e| e.to_string())
    }

    pub fn query_row_i32(&self, sql: &str, params: &[&dyn rusqlite::ToSql]) -> std::result::Result<i32, String> {
        self.conn.query_row(sql, params, |row| row.get(0))
            .map_err(|e| e.to_string())
    }

    pub fn prepare_statement(&self, sql: &str) -> std::result::Result<rusqlite::Statement, String> {
        self.conn.prepare(sql)
            .map_err(|e| e.to_string())
    }
}

/// Gibt den Pfad zur Datenbank zurück
pub fn get_database_path() -> PathBuf {
    let data_dir = get_data_directory();
    data_dir.join("frametrain.db")
}

fn get_data_directory() -> PathBuf {
    // CRITICAL: Use the correct Tauri app identifier from tauri.conf.json
    // identifier: "com.frametrain.desktop2"
    
    #[cfg(target_os = "windows")]
    {
        let appdata = std::env::var("APPDATA").unwrap_or_else(|_| String::from("."));
        PathBuf::from(appdata).join("com.frametrain.desktop2")
    }
    
    #[cfg(target_os = "macos")]
    {
        let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
        PathBuf::from(home).join("Library/Application Support/com.frametrain.desktop2")
    }
    
    #[cfg(target_os = "linux")]
    {
        let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
        PathBuf::from(home).join(".local/share/com.frametrain.desktop2")
    }
}
