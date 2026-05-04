// Initialize root versions for all existing models
use rusqlite::Connection;
use std::fs;
use std::path::PathBuf;

pub fn initialize_root_versions_for_all_models(db_path: &std::path::Path, _models_dir: &std::path::Path) -> Result<(), String> {
    let conn = Connection::open(db_path)
        .map_err(|e| format!("Database error: {}", e))?;
    
    // CRITICAL: Disable foreign key constraints to avoid FK errors
    conn.execute("PRAGMA foreign_keys = OFF", [])
        .map_err(|e| format!("Failed to disable foreign keys: {}", e))?;
    println!("[INIT] Foreign key constraints disabled");
    
    // CRITICAL: Create ALL required tables first!
    println!("[INIT] Creating ALL required tables if they don't exist...");
    
    // 1. Create models table first (base table)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            base_model TEXT,
            model_path TEXT,
            status TEXT NOT NULL DEFAULT 'created',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name)
        )",
        [],
    ).map_err(|e| format!("Failed to create models table: {}", e))?;
    println!("[INIT] ✅ models table created/verified");
    
    // 2. Create model_versions_new table (WITHOUT foreign key)
    conn.execute(
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
            parent_version_id TEXT
        )",
        [],
    ).map_err(|e| format!("Failed to create model_versions_new table: {}", e))?;
    println!("[INIT] ✅ model_versions_new table created/verified");

    // 3. Create training_metrics_new table (WITHOUT foreign key)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS training_metrics_new (
            id TEXT PRIMARY KEY,
            version_id TEXT NOT NULL UNIQUE,
            final_train_loss REAL NOT NULL,
            final_val_loss REAL,
            total_epochs INTEGER NOT NULL,
            total_steps INTEGER NOT NULL,
            best_epoch INTEGER,
            training_duration_seconds INTEGER,
            created_at TEXT NOT NULL
        )",
        [],
    ).map_err(|e| format!("Failed to create training_metrics_new table: {}", e))?;
    println!("[INIT] ✅ training_metrics_new table created/verified");

    // Create indices
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_versions_model ON model_versions_new(model_id)",
        [],
    ).map_err(|e| format!("Failed to create index: {}", e))?;

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_metrics_version ON training_metrics_new(version_id)",
        [],
    ).map_err(|e| format!("Failed to create index: {}", e))?;
    
    println!("[INIT] ✅ Version tables created/verified");
    
    // CRITICAL: Migrate existing versions with wrong model_id (local_*)
    println!("[INIT] Checking for versions with local_* model_ids...");
    
    let migrate_count: i32 = conn.query_row(
        "SELECT COUNT(*) FROM model_versions_new WHERE model_id LIKE 'local_%'",
        [],
        |row| row.get(0),
    ).unwrap_or(0);
    
    if migrate_count > 0 {
        println!("[INIT] Found {} versions with local_* IDs, migrating...", migrate_count);
        
        // Get all versions with local_* IDs
        let mut stmt = conn.prepare(
            "SELECT id, model_id, version_name FROM model_versions_new WHERE model_id LIKE 'local_%'"
        ).map_err(|e| format!("Failed to prepare migration query: {}", e))?;
        
        let versions_to_migrate: Vec<(String, String, String)> = stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        }).map_err(|e| format!("Failed to query versions: {}", e))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to collect versions: {}", e))?;
        
        drop(stmt); // Release the statement before updating
        
        for (version_id, old_model_id, version_name) in versions_to_migrate {
            // Extract model name from version_name (e.g., "mt5-base v1" -> "mt5-base")
            let model_name = version_name.split(" v").next().unwrap_or(&version_name);
            
            // Find the correct model_id by name
            let correct_model_id: Result<String, _> = conn.query_row(
                "SELECT id FROM models WHERE name = ?1",
                [model_name],
                |row| row.get(0),
            );
            
            if let Ok(new_model_id) = correct_model_id {
                println!("[INIT] Migrating version {} from {} to {}", version_id, old_model_id, new_model_id);
                
                conn.execute(
                    "UPDATE model_versions_new SET model_id = ?1 WHERE id = ?2",
                    rusqlite::params![&new_model_id, &version_id],
                ).map_err(|e| format!("Failed to update version: {}", e))?;
                
                println!("[INIT] ✅ Migrated version {}", version_id);
            } else {
                println!("[INIT] ⚠️  Could not find model with name '{}' for version {}", model_name, version_id);
            }
        }
        
        println!("[INIT] ✅ Migration completed");
    } else {
        println!("[INIT] No versions need migration");
    }
    
    println!("[INIT] Checking for models without root versions...");
    
    // Get all models WITH their user_id
    let mut stmt = conn.prepare("SELECT id, name, model_path, user_id FROM models")
        .map_err(|e| format!("Failed to prepare statement: {}", e))?;
    
    let models = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, Option<String>>(3)?,  // user_id might be NULL for old data
        ))
    }).map_err(|e| format!("Failed to query models: {}", e))?;
    
    for model in models {
        let (model_id, model_name, model_path, user_id_opt) = model.map_err(|e| format!("Row error: {}", e))?;
        
        // Skip if no user_id (shouldn't happen but be safe)
        let user_id = match user_id_opt {
            Some(uid) => uid,
            None => {
                println!("[INIT] ⚠️  Model {} has no user_id, skipping root version creation", model_id);
                continue;
            }
        };
        
        // Check if root version already exists
        let has_root: i32 = conn.query_row(
            "SELECT COUNT(*) FROM model_versions_new WHERE model_id = ? AND is_root = 1",
            [&model_id],
            |row| row.get(0),
        ).unwrap_or(0);
        
        if has_root == 0 {
            println!("[INIT] Creating root version for model: {}", model_name);
            
            // Create root version
            let version_id = format!("ver_{}", uuid::Uuid::new_v4().to_string().replace("-", "")[..12].to_string());
            let now = chrono::Utc::now().to_rfc3339();
            
            // Calculate size and file count
            let model_path_buf = PathBuf::from(&model_path);
            let (size_bytes, file_count) = if model_path_buf.exists() {
                calculate_dir_size(&model_path_buf).unwrap_or((0, 0))
            } else {
                (0, 0)
            };
            
            conn.execute(
                "INSERT INTO model_versions_new (id, model_id, version_name, version_number, path, size_bytes, file_count, created_at, is_root, parent_version_id, user_id) 
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                rusqlite::params![
                    &version_id,
                    &model_id,
                    "Original",
                    0,
                    &model_path,
                    size_bytes,
                    file_count,
                    &now,
                    1, // is_root = true
                    Option::<String>::None,
                    &user_id,  // CRITICAL: Include user_id
                ],
            ).map_err(|e| format!("Failed to create root version: {}", e))?;
            
            println!("[INIT] ✅ Created root version for {}", model_name);
        }
    }
    
    Ok(())
}

fn calculate_dir_size(path: &std::path::Path) -> Result<(i64, i32), String> {
    let mut total_size: i64 = 0;
    let mut file_count: i32 = 0;
    
    fn visit_dirs(dir: &std::path::Path, total_size: &mut i64, file_count: &mut i32) -> Result<(), String> {
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
