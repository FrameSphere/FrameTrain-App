# 🎨 FrameTrain Icons

Dieser Ordner enthält die Icons für die Desktop-App.

## 📋 Benötigte Dateien

- `32x32.png` - Kleines Icon
- `128x128.png` - Standard Icon
- `128x128@2x.png` - Retina Icon (256x256)
- `icon.icns` - macOS Icon
- `icon.ico` - Windows Icon

## 🚀 Quick Start

### Option 1: Automatisch mit Bash (macOS/Linux)

```bash
chmod +x generate-placeholder.sh
./generate-placeholder.sh
```

**Voraussetzung:** ImageMagick
```bash
brew install imagemagick  # macOS
sudo apt-get install imagemagick  # Linux
```

### Option 2: Python Script (alle Plattformen)

```bash
pip install Pillow
python3 generate-icons.py
```

### Option 3: Online Tool

1. Gehe zu: https://icon.kitchen/
2. Lade dein Logo hoch (PNG, mindestens 512x512px)
3. Wähle "Tauri" als Platform
4. Download und entpacke hier

### Option 4: Eigene Icons

Platziere deine Icons einfach hier mit den korrekten Namen:

```
icons/
├── 32x32.png           # 32x32 Pixel
├── 128x128.png         # 128x128 Pixel
├── 128x128@2x.png      # 256x256 Pixel (Retina)
├── icon.icns           # macOS Bundle
└── icon.ico            # Windows Bundle
```

## ✅ Verifikation

Prüfe ob alle Icons vorhanden sind:

```bash
ls -lh
```

Du solltest sehen:
- ✅ 5 Dateien (.png, .icns, .ico)
- ✅ Alle mindestens 1 KB groß

## 🧪 Test

Nach Icon-Generierung lokal testen:

```bash
cd ../..  # Zurück zu desktop-app/
npm run tauri:build
```

Der Build sollte ohne Icon-Fehler durchlaufen.

## 📝 Hinweise

- **Placeholder:** Die generierten Icons sind einfache Platzhalter mit "FT" Text
- **Production:** Für Production solltest du ein professionelles Logo verwenden
- **Format:** PNG mit transparentem Hintergrund wird empfohlen
- **Größe:** Originalgröße mindestens 512x512px für beste Qualität

## 🎨 Design-Tipps

Ein gutes App-Icon sollte:
- ✅ Einfach und erkennbar sein
- ✅ Bei kleinen Größen (32px) noch lesbar sein
- ✅ Konsistente Farben haben
- ✅ Transparent sein oder einheitlichen Hintergrund haben
- ✅ Das Produkt/Brand repräsentieren

## 🔧 Probleme?

**"ImageMagick not found"**
```bash
brew install imagemagick  # macOS
```

**"Pillow not found"**
```bash
pip install Pillow
```

**"iconutil: command not found"**
- Normal auf Windows/Linux
- `.iconset/` Ordner bleibt erhalten
- Manuell zu `.icns` konvertieren oder auf macOS laufen lassen
