#!/usr/bin/env python3
"""
FrameTrain - Icon Generator (Python Version)
Generiert Placeholder-Icons ohne ImageMagick
Benötigt nur Pillow (PIL)
"""

from PIL import Image, ImageDraw, ImageFont
import os
import sys

def create_base_icon(size=512, color="#6366f1", text="FT"):
    """Erstellt das Basis-Icon mit RGBA (Alpha)"""
    img = Image.new('RGBA', (size, size), color=color + "FF")  # volle Deckkraft
    draw = ImageDraw.Draw(img)
    
    # Versuche eine Font zu laden
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size // 3)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size // 3)
        except:
            font = ImageFont.load_default()
    
    # Text zentrieren
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((size - text_width) // 2, (size - text_height) // 2)
    
    draw.text(position, text, fill=(255,255,255,255), font=font)
    
    return img

def generate_png_icons(base_img, output_dir):
    """Generiert PNG Icons in RGBA"""
    sizes = {
        '32x32.png': 32,
        '128x128.png': 128,
        '128x128@2x.png': 256
    }
    
    for filename, size in sizes.items():
        resized = base_img.resize((size, size), Image.Resampling.LANCZOS)
        resized = resized.convert("RGBA")  # <-- zwingt RGBA
        resized.save(os.path.join(output_dir, filename))
        print(f"✅ {filename} erstellt (RGBA)")

def generate_ico(base_img, output_dir):
    """Generiert Windows .ico"""
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    icon_images = []
    
    for size in sizes:
        resized = base_img.resize(size, Image.Resampling.LANCZOS)
        icon_images.append(resized)
    
    # Speichere als .ico
    ico_path = os.path.join(output_dir, 'icon.ico')
    icon_images[0].save(
        ico_path,
        format='ICO',
        sizes=sizes
    )
    print(f"✅ icon.ico erstellt")

def generate_icns(base_img, output_dir):
    """Generiert macOS .icns (vereinfachte Version)"""
    # Hinweis: Echte .icns benötigen iconutil (macOS)
    # Hier erstellen wir die benötigten PNGs
    
    sizes = [
        (16, 'icon_16x16.png'),
        (32, 'icon_16x16@2x.png'),
        (32, 'icon_32x32.png'),
        (64, 'icon_32x32@2x.png'),
        (128, 'icon_128x128.png'),
        (256, 'icon_128x128@2x.png'),
        (256, 'icon_256x256.png'),
        (512, 'icon_256x256@2x.png'),
        (512, 'icon_512x512.png'),
        (1024, 'icon_512x512@2x.png'),
    ]
    
    iconset_dir = os.path.join(output_dir, 'icon.iconset')
    os.makedirs(iconset_dir, exist_ok=True)
    
    for size, filename in sizes:
        resized = base_img.resize((size, size), Image.Resampling.LANCZOS)
        resized.save(os.path.join(iconset_dir, filename))
    
    print(f"✅ icon.iconset/ erstellt")
    
    # Versuche iconutil zu nutzen (nur macOS)
    if sys.platform == 'darwin':
        import subprocess
        try:
            subprocess.run([
                'iconutil', '-c', 'icns',
                iconset_dir,
                '-o', os.path.join(output_dir, 'icon.icns')
            ], check=True)
            print(f"✅ icon.icns erstellt")
            
            # Aufräumen
            import shutil
            shutil.rmtree(iconset_dir)
        except subprocess.CalledProcessError:
            print("⚠️  iconutil fehlgeschlagen - icon.iconset/ bleibt erhalten")
    else:
        print("⚠️  iconutil nicht verfügbar (nur macOS)")
        print("   → icon.iconset/ manuell zu .icns konvertieren")

def main():
    print("🎨 FrameTrain Icon Generator (Python)")
    print("=" * 50)
    print()
    
    # Prüfe ob Pillow installiert ist
    try:
        from PIL import __version__
        print(f"✅ Pillow {__version__} gefunden")
    except ImportError:
        print("❌ Pillow nicht installiert!")
        print()
        print("Installation:")
        print("  pip install Pillow")
        sys.exit(1)
    
    # Aktuelles Verzeichnis
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"📂 Arbeitsverzeichnis: {script_dir}")
    print()
    
    # Basis-Icon erstellen
    print("📐 Erstelle Basis-Icon (512x512)...")
    base_img = create_base_icon(size=512, color="#6366f1", text="FT")
    print("✅ Basis-Icon erstellt")
    print()
    
    # PNG Icons
    print("📦 Generiere PNG Icons...")
    generate_png_icons(base_img, script_dir)
    print()
    
    # Windows .ico
    print("🪟 Generiere Windows Icon...")
    generate_ico(base_img, script_dir)
    print()
    
    # macOS .icns
    print("🍎 Generiere macOS Icon...")
    generate_icns(base_img, script_dir)
    print()
    
    # Fertig
    print("=" * 50)
    print("✅ Icons erfolgreich generiert!")
    print()
    print("📂 Generierte Dateien:")
    for filename in os.listdir(script_dir):
        if filename.endswith(('.png', '.ico', '.icns')) or filename.endswith('.iconset'):
            filepath = os.path.join(script_dir, filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                print(f"   {filename:30} ({size:,} bytes)")
            else:
                print(f"   {filename:30} (directory)")
    print()
    print("🚀 Bereit für: npm run tauri:build")

if __name__ == "__main__":
    main()
