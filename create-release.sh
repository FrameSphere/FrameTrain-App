#!/bin/bash

# FrameTrain Desktop App - Release Script
# Automatisiert den kompletten Release-Prozess

set -e

echo "================================================"
echo "  FrameTrain Desktop App - Release Script"
echo "================================================"
echo ""

# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DESKTOP_DIR="$SCRIPT_DIR"

cd "$DESKTOP_DIR"

# 1. Get current version
CURRENT_VERSION=$(node -p "require('./package.json').version")
echo "📦 Aktuelle Version: $CURRENT_VERSION"
echo ""

# 2. Ask for new version
read -p "Neue Version (z.B. 1.0.1): " NEW_VERSION

if [ -z "$NEW_VERSION" ]; then
    echo "❌ Keine Version angegeben. Abbruch."
    exit 1
fi

echo ""
echo "🔄 Bereite Release v$NEW_VERSION vor..."
echo ""

# 3. Confirm
read -p "Version v$NEW_VERSION erstellen? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "❌ Abgebrochen."
    exit 0
fi

echo ""

# 4. Update version in package.json
echo "📝 Update package.json..."
node -e "
const fs = require('fs');
const pkg = JSON.parse(fs.readFileSync('./package.json', 'utf8'));
pkg.version = '$NEW_VERSION';
fs.writeFileSync('./package.json', JSON.stringify(pkg, null, 2) + '\n');
"
echo "   ✅ package.json aktualisiert"

# 5. Update version in tauri.conf.json
echo "📝 Update tauri.conf.json..."
node -e "
const fs = require('fs');
const config = JSON.parse(fs.readFileSync('./src-tauri/tauri.conf.json', 'utf8'));
config.version = '$NEW_VERSION';
fs.writeFileSync('./src-tauri/tauri.conf.json', JSON.stringify(config, null, 2) + '\n');
"
echo "   ✅ tauri.conf.json aktualisiert"

# 6. Run cleanup (optional)
echo ""
read -p "Cleanup-Script ausführen? (yes/no): " run_cleanup
if [ "$run_cleanup" == "yes" ]; then
    if [ -f "./cleanup-for-release.sh" ]; then
        echo "🧹 Führe Cleanup aus..."
        chmod +x ./cleanup-for-release.sh
        ./cleanup-for-release.sh
    else
        echo "⚠️  cleanup-for-release.sh nicht gefunden"
    fi
fi

# 7. Git status
echo ""
echo "📊 Git Status:"
git status

# 8. Commit changes
echo ""
read -p "Änderungen committen? (yes/no): " commit_changes
if [ "$commit_changes" == "yes" ]; then
    git add package.json src-tauri/tauri.conf.json
    git commit -m "chore: bump version to $NEW_VERSION"
    echo "   ✅ Changes committed"
fi

# 9. Create and push tag
echo ""
read -p "Git Tag v$NEW_VERSION erstellen und pushen? (yes/no): " create_tag
if [ "$create_tag" == "yes" ]; then
    echo "🏷️  Erstelle Tag v$NEW_VERSION..."
    git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"
    
    echo "📤 Pushe zu GitHub..."
    git push origin main
    git push origin "v$NEW_VERSION"
    
    echo ""
    echo "================================================"
    echo "  ✅ Release v$NEW_VERSION wurde erstellt!"
    echo "================================================"
    echo ""
    echo "🔗 GitHub Actions werden jetzt automatisch:"
    echo "   1. Die App für alle Plattformen bauen"
    echo "   2. GitHub Release erstellen"
    echo "   3. Installer hochladen"
    echo ""
    echo "📊 Verfolge den Build hier:"
    echo "   https://github.com/FrameTrain/FrameTrain/actions"
    echo ""
    echo "📦 Release wird verfügbar sein unter:"
    echo "   https://github.com/FrameTrain/FrameTrain/releases/tag/v$NEW_VERSION"
    echo ""
else
    echo ""
    echo "⚠️  Tag wurde NICHT erstellt."
    echo "   Führe manuell aus:"
    echo "   git tag -a v$NEW_VERSION -m 'Release v$NEW_VERSION'"
    echo "   git push origin main"
    echo "   git push origin v$NEW_VERSION"
fi

echo ""
echo "🎉 Fertig!"
echo ""
