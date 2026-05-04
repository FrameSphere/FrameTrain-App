// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod ml_backend;
mod database;
mod db_commands;
mod auth;
mod api_config;
mod user_manager;
mod model_manager;
mod dataset_manager;
mod training_manager;
mod dev_trainer;
mod version_manager;
mod init_versions;
mod analysis_manager;
mod test_manager;
mod plugin_commands;
mod power_manager;
mod laboratory_manager;

use std::fs;
use std::sync::{Arc, Mutex};
use database::{Database, get_database_path};
use tauri::{Manager, Emitter};

pub struct AppState {
    db: Mutex<Database>,
}

#[tauri::command]
fn verify_api_key(api_key: String) -> Result<bool, String> {
    if api_key.starts_with("ft_") && api_key.len() > 20 {
        Ok(true)
    } else {
        Err("Ungültiger API-Key".to_string())
    }
}

#[tauri::command]
fn save_config(app_handle: tauri::AppHandle, _api_key: String, config: String) -> Result<(), String> {
    let config_dir = app_handle.path().app_config_dir()
        .map_err(|e| format!("Konnte Config-Verzeichnis nicht finden: {}", e))?;
    fs::create_dir_all(&config_dir)
        .map_err(|e| format!("Konnte Config-Verzeichnis nicht erstellen: {}", e))?;
    let config_path = config_dir.join("config.json");
    fs::write(config_path, config)
        .map_err(|e| format!("Konnte Config nicht speichern: {}", e))?;
    Ok(())
}

#[tauri::command]
fn load_config(app_handle: tauri::AppHandle) -> Result<String, String> {
    let config_dir = app_handle.path().app_config_dir()
        .map_err(|e| format!("Konnte Config-Verzeichnis nicht finden: {}", e))?;
    let config_path = config_dir.join("config.json");
    if !config_path.exists() {
        return Err("Keine Konfiguration gefunden".to_string());
    }
    fs::read_to_string(config_path)
        .map_err(|e| format!("Konnte Config nicht lesen: {}", e))
}

#[tauri::command]
fn get_app_data_dir(app_handle: tauri::AppHandle) -> Result<String, String> {
    app_handle.path().app_data_dir()
        .map_err(|e| format!("Konnte App-Daten-Verzeichnis nicht finden: {}", e))
        .map(|p| p.to_string_lossy().to_string())
}

#[tauri::command]
fn clear_config(app_handle: tauri::AppHandle) -> Result<(), String> {
    let config_dir = app_handle.path().app_config_dir()
        .map_err(|e| format!("Konnte Config-Verzeichnis nicht finden: {}", e))?;
    let config_path = config_dir.join("config.json");
    if config_path.exists() {
        fs::remove_file(config_path)
            .map_err(|e| format!("Konnte Config nicht löschen: {}", e))?;
    }
    Ok(())
}

#[tauri::command]
fn force_quit_app(app_handle: tauri::AppHandle) {
    println!("[App] Force quit requested by user");
    app_handle.exit(0);
}

fn main() {
    #[cfg(debug_assertions)]
    {
        if let Err(e) = dotenvy::dotenv() {
            println!("ℹ️  No .env file found ({}). Using default API configuration.", e);
        } else {
            println!("✅ .env file loaded for local development");
        }
    }

    let db_path = get_database_path();
    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent).expect("Konnte Datenbank-Verzeichnis nicht erstellen");
    }

    let db = Database::new(db_path.clone()).expect("Konnte Datenbank nicht öffnen");

    println!("[INIT] Creating version tables...");
    match db.create_version_tables() {
        Ok(_) => {
            println!("[INIT] ✅ Version tables created successfully");
            let models_dir = db_path.parent().unwrap().join("models");
            if let Err(e) = init_versions::initialize_root_versions_for_all_models(&db_path, &models_dir) {
                eprintln!("[INIT] ⚠️  Warning: Could not initialize root versions: {}", e);
            }
        },
        Err(e) => {
            eprintln!("[INIT] ❌ Failed to create version tables: {}", e);
        }
    }

    let context = tauri::generate_context!();

    tauri::Builder::default()
        .manage(AppState { db: Mutex::new(db) })
        .manage(Arc::new(Mutex::new(training_manager::TrainingState::default())))
        .manage(Arc::new(Mutex::new(test_manager::TestState::default())))
        .manage(Arc::new(Mutex::new(laboratory_manager::LabState::default())))
        .manage(std::sync::Mutex::new(power_manager::PowerState::default()))
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            verify_api_key,
            save_config,
            load_config,
            get_app_data_dir,
            clear_config,
            force_quit_app,
            auth::validate_credentials,
            user_manager::login_user,
            user_manager::logout_user,
            user_manager::is_user_logged_in,
            model_manager::get_models_directory,
            model_manager::list_models,
            model_manager::import_local_model,
            model_manager::delete_model,
            model_manager::get_model_info,
            model_manager::validate_model_directory,
            model_manager::get_directory_size,
            model_manager::search_huggingface_models,
            model_manager::get_huggingface_model_files,
            model_manager::download_huggingface_model,
            dataset_manager::list_datasets_for_model,
            dataset_manager::list_test_datasets_for_model,
            dataset_manager::list_all_datasets,
            dataset_manager::import_local_dataset,
            dataset_manager::delete_dataset,
            dataset_manager::split_dataset,
            dataset_manager::search_huggingface_datasets,
            dataset_manager::get_huggingface_dataset_files,
            dataset_manager::download_huggingface_dataset,
            dataset_manager::get_dataset_filter_options,
            dataset_manager::get_dataset_files,
            dataset_manager::read_dataset_file,
            dataset_manager::move_dataset_files,
            dataset_manager::delete_dataset_files,
            dataset_manager::add_files_to_dataset,
            dataset_manager::split_dataset_in_half,
            dataset_manager::validate_image_label_folders,
            dataset_manager::import_structured_dataset,
            training_manager::get_training_presets,
            training_manager::rate_training_config,
            training_manager::start_training,
            training_manager::stop_training,
            dev_trainer::start_dev_training,
            training_manager::get_current_training,
            training_manager::get_training_history,
            training_manager::delete_training_job,
            training_manager::check_training_requirements,
            training_manager::get_system_ram_gb,
            training_manager::get_model_ram_info,
            version_manager::list_models_with_versions,
            version_manager::list_model_versions,
            version_manager::delete_model_version,
            version_manager::rename_model_version,
            version_manager::list_models_with_version_tree,
            version_manager::get_version_path_for_ui,
            version_manager::list_version_files,
            version_manager::export_model_version,
            analysis_manager::get_training_metrics,
            analysis_manager::get_version_details,
            analysis_manager::get_training_logs,
            analysis_manager::save_training_metrics,
            analysis_manager::save_training_logs,
            analysis_manager::update_training_progress,
            analysis_manager::save_ai_analysis_report,
            analysis_manager::get_ai_analysis_report,
            analysis_manager::delete_ai_analysis_report,
            analysis_manager::get_training_full_data,
            training_manager::save_metrics_template,
            training_manager::get_metrics_templates,
            training_manager::delete_metrics_template,
            test_manager::start_test,
            test_manager::test_single_input,
            test_manager::stop_test,
            test_manager::get_current_test,
            test_manager::get_active_test_job,
            test_manager::get_test_history,
            test_manager::get_test_results_for_version,
            test_manager::export_hard_examples,
            plugin_commands::get_available_plugins,
            plugin_commands::check_first_launch,
            plugin_commands::install_plugins,
            plugin_commands::handle_plugin_approval,
            power_manager::enable_prevent_sleep,
            power_manager::disable_prevent_sleep,
            power_manager::get_prevent_sleep_status,
            laboratory_manager::lab_load_sample,
            laboratory_manager::lab_run_inference,
            laboratory_manager::lab_save_session,
            laboratory_manager::lab_get_sessions,
            laboratory_manager::lab_delete_session,
            laboratory_manager::lab_export_as_dataset,
            laboratory_manager::lab_get_stats,
            laboratory_manager::lab_start_model_server,
            laboratory_manager::lab_infer_sample,
            laboratory_manager::lab_stop_model_server,
            laboratory_manager::lab_get_server_status,
            laboratory_manager::run_lab_script_sample,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.emit("app-close-requested", ());
            }
        })
        .run(context)
        .expect("Fehler beim Starten der Tauri-Anwendung");
}
