#!/bin/bash
# FrameTrain - Icon Generator (RGBA kompatibel für Tauri)
# Erstellt Icons direkt in TrueColor + Alpha, kompatibel mit Tauri 2

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🎨 Generiere FrameTrain Icons..."

# Prüfe ob ImageMagick installiert ist
if ! command -v magick &> /dev/null; then
    echo "❌ ImageMagick nicht gefunden!"
    echo ""
    echo "Installiere ImageMagick:"
    echo "  macOS:   brew install imagemagick"
    echo "  Ubuntu:  sudo apt-get install imagemagick"
    echo "  Windows: choco install imagemagick"
    exit 1
fi

# Basis-Icon erstellen (512x512)
echo "📐 Erstelle Basis-Icon (512x512)..."
magick -size 512x512 canvas:none \
    -fill '#6366f1' -draw "rectangle 0,0 512,512" \
    -gravity center \
    -pointsize 200 \
    -font Arial-Bold \
    -fill white \
    -annotate +0+0 'FT' \
    base-icon.png

# PNG Icons generieren (RGBA)
echo "📦 Generiere PNG Icons (32x32, 128x128, 256x256)..."
for SIZE in 32 128 256; do
    magick base-icon.png -resize "${SIZE}x${SIZE}" -type TrueColorAlpha "${SIZE}x${SIZE}.png"
done

# High-DPI 128x128@2x
magick base-icon.png -resize 256x256 -type TrueColorAlpha "128x128@2x.png"

# macOS .icns generieren
echo "🍎 Generiere macOS Icon (.icns)..."
mkdir -p icon.iconset
for SIZE in 16 32 128 256 512; do
    # normale und @2x
    magick base-icon.png -resize "${SIZE}x${SIZE}" -type TrueColorAlpha "icon.iconset/icon_${SIZE}x${SIZE}.png"
    magick base-icon.png -resize "$((SIZE*2))x$((SIZE*2))" -type TrueColorAlpha "icon.iconset/icon_${SIZE}x${SIZE}@2x.png"
done

iconutil -c icns icon.iconset -o icon.icns
rm -rf icon.iconset

# Windows .ico generieren
echo "🪟 Generiere Windows Icon (.ico)..."
magick base-icon.png -define icon:auto-resize=256,128,96,64,48,32,16 -type TrueColorAlpha icon.ico

# Aufräumen
rm base-icon.png

echo ""
echo "✅ Icons erfolgreich generiert!"
echo ""
echo "📂 Generierte Dateien:"
ls -lh *.png *.icns *.ico

echo ""
echo "🚀 Bereit für: npm run tauri