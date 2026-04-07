#!/bin/bash

# FrameTrain Complete Cleanup Script
# Löscht ALLE gespeicherten Daten (Modelle, Datasets, Datenbank, Logs, Training-Outputs)
# Updated: Löscht BOTH OLD und NEW App Data Directories

set -e  # Exit on error

echo "================================================"
echo "  FrameTrain Complete Cleanup Script"
echo "================================================"
echo ""
echo "⚠️  WARNUNG: Dieses Script löscht ALLE Daten!"
echo ""
echo "Folgende Daten werden gelöscht:"
echo "  - Alle Modelle und Versionen (OLD + NEW)"
echo "  - Alle Datasets"
echo "  - Die gesamte Datenbank"
echo "  - Alle Training-Outputs und Logs"
echo "  - Alle Checkpoints"
echo "  - Metadata und Konfigurationen"
echo ""
read -p "Möchtest du wirklich fortfahren? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Abgebrochen."
    exit 0
fi

echo ""
echo "🧹 Starte Cleanup..."
echo ""

# Detect OS and set app data directories (BOTH old and new)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    APP_DATA_DIR_NEW="$HOME/Library/Application Support/com.frametrain.desktop2"
    APP_DATA_DIR_OLD="$HOME/Library/Application Support/com.frametrain.desktop"
    echo "📍 macOS erkannt"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    APP_DATA_DIR_NEW="$HOME/.local/share/com.frametrain.desktop2"
    APP_DATA_DIR_OLD="$HOME/.local/share/com.frametrain.desktop"
    echo "📍 Linux erkannt"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash)
    APP_DATA_DIR_NEW="$APPDATA/com.frametrain.desktop2"
    APP_DATA_DIR_OLD="$APPDATA/com.frametrain.desktop"
    echo "📍 Windows erkannt"
else
    echo "❌ Unbekanntes Betriebssystem: $OSTYPE"
    exit 1
fi

echo "📂 App Data Directory (NEW): $APP_DATA_DIR_NEW"
echo "📂 App Data Directory (OLD): $APP_DATA_DIR_OLD"
echo ""

# Check if directories exist
if [ ! -d "$APP_DATA_DIR_NEW" ] && [ ! -d "$APP_DATA_DIR_OLD" ]; then
    echo "⚠️  Keine App Data Directories gefunden."
    echo "    Nichts zu löschen!"
    exit 0
fi

# Clean function that handles both directories
clean_both_dirs() {
    local subpath=$1
    local name=$2
    
    # Clean NEW directory
    if [ -d "$APP_DATA_DIR_NEW/$subpath" ]; then
        echo "🗑️  Lösche $name (NEW)..."
        rm -rf "$APP_DATA_DIR_NEW/$subpath"
        echo "   ✅ $name (NEW) gelöscht"
    else
        echo "   ⏭️  $name (NEW) existiert nicht"
    fi
    
    # Clean OLD directory
    if [ -d "$APP_DATA_DIR_OLD/$subpath" ]; then
        echo "🗑️  Lösche $name (OLD)..."
        rm -rf "$APP_DATA_DIR_OLD/$subpath"
        echo "   ✅ $name (OLD) gelöscht"
    else
        echo "   ⏭️  $name (OLD) existiert nicht"
    fi
}

clean_file_both() {
    local filepath=$1
    local name=$2
    
    # Clean NEW directory
    if [ -f "$APP_DATA_DIR_NEW/$filepath" ]; then
        echo "🗑️  Lösche $name (NEW)..."
        rm -f "$APP_DATA_DIR_NEW/$filepath"
        echo "   ✅ $name (NEW) gelöscht"
    else
        echo "   ⏭️  $name (NEW) existiert nicht"
    fi
    
    # Clean OLD directory
    if [ -f "$APP_DATA_DIR_OLD/$filepath" ]; then
        echo "🗑️  Lösche $name (OLD)..."
        rm -f "$APP_DATA_DIR_OLD/$filepath"
        echo "   ✅ $name (OLD) gelöscht"
    else
        echo "   ⏭️  $name (OLD) existiert nicht"
    fi
}

# 1. Delete Models Directory
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Modelle"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
clean_both_dirs "models" "Modelle-Ordner"
echo ""

# 2. Delete Training Outputs
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Training-Outputs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
clean_both_dirs "training_outputs" "Training-Outputs"
echo ""

# 3. Delete Checkpoints
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. Checkpoints"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
clean_both_dirs "checkpoints" "Checkpoints"
echo ""

# 4. Delete Datasets (if stored separately)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. Datasets"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
clean_both_dirs "datasets" "Datasets-Ordner"
echo ""

# 5. Delete Database
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. Datenbank"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
clean_file_both "frametrain.db" "Datenbank (frametrain.db)"
clean_file_both "frametrain.db-shm" "Datenbank-Shared Memory"
clean_file_both "frametrain.db-wal" "Datenbank-Write Ahead Log"
echo ""

# 6. Delete Logs
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. Logs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
clean_both_dirs "logs" "Logs-Ordner"
clean_file_both "training_jobs.json" "Training Jobs JSON"
echo ""

# 7. Delete Metadata and Config
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7. Metadata & Konfiguration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
clean_file_both "models_metadata.json" "Models Metadata"
clean_file_both "app_config.json" "App Konfiguration"
clean_both_dirs "cache" "Cache-Ordner"
clean_both_dirs "temp" "Temp-Ordner"
echo ""

# 8. Calculate space freed and clean up empty directories
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "8. Aufräumen & Statistiken"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Clean NEW directory if empty
if [ -d "$APP_DATA_DIR_NEW" ]; then
    remaining_files_new=$(find "$APP_DATA_DIR_NEW" -type f 2>/dev/null | wc -l || echo "0")
    remaining_files_new=$(echo $remaining_files_new | tr -d ' ')
    
    if [ "$remaining_files_new" -eq 0 ]; then
        echo "🗑️  Lösche leeren App Data Ordner (NEW)..."
        rmdir "$APP_DATA_DIR_NEW" 2>/dev/null || true
        echo "   ✅ App Data Ordner (NEW) gelöscht"
    else
        remaining_size_new=$(du -sh "$APP_DATA_DIR_NEW" 2>/dev/null | cut -f1 || echo "0B")
        echo "📊 Verbleibende Daten (NEW): $remaining_size_new ($remaining_files_new Dateien)"
    fi
else
    echo "✅ App Data Ordner (NEW) komplett gelöscht"
fi

# Clean OLD directory if empty
if [ -d "$APP_DATA_DIR_OLD" ]; then
    remaining_files_old=$(find "$APP_DATA_DIR_OLD" -type f 2>/dev/null | wc -l || echo "0")
    remaining_files_old=$(echo $remaining_files_old | tr -d ' ')
    
    if [ "$remaining_files_old" -eq 0 ]; then
        echo "🗑️  Lösche leeren App Data Ordner (OLD)..."
        rmdir "$APP_DATA_DIR_OLD" 2>/dev/null || true
        echo "   ✅ App Data Ordner (OLD) gelöscht"
    else
        remaining_size_old=$(du -sh "$APP_DATA_DIR_OLD" 2>/dev/null | cut -f1 || echo "0B")
        echo "📊 Verbleibende Daten (OLD): $remaining_size_old ($remaining_files_old Dateien)"
    fi
else
    echo "✅ App Data Ordner (OLD) komplett gelöscht"
fi

echo ""
echo "================================================"
echo "  ✅ Cleanup abgeschlossen!"
echo "================================================"
echo ""
echo "Alle FrameTrain-Daten wurden erfolgreich entfernt."
echo "Du kannst die App jetzt neu starten für einen"
echo "frischen Start."
echo ""
