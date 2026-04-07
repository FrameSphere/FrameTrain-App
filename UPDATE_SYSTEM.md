# FrameTrain Desktop App - Auto-Update System

## 🎯 Wie es funktioniert

FrameTrain verwendet ein **einfaches, unsigniertes Update-System**, das perfekt für Open-Source-Projekte ohne teure Code-Signing-Zertifikate ist.

## ⚙️ Technische Details

### Update-Check Mechanismus

1. **Beim App-Start** prüft die App automatisch auf Updates
2. **GitHub Releases API** wird abgefragt: `https://api.github.com/repos/KarolP-tech/FrameTrain/releases/latest`
3. **Versionsvergleich**: Aktuelle Version vs. neueste GitHub Release Version
4. **Dialog erscheint** wenn Update verfügbar ist
5. **User klickt "View Release"** → Browser öffnet GitHub Release-Seite
6. **Manueller Download & Installation** des Installers

### Komponenten

- **UpdateChecker.tsx**: React-Komponente für Update-Check und Dialog
- **tauri.conf.json**: Kein Tauri Updater Plugin (entfernt)
- **release-desktop-app.yml**: GitHub Action für automatische Releases

### Unterstützte Plattformen

| Platform | Installer Format | Auto-Update |
|----------|-----------------|-------------|
| macOS | `.dmg` | ✅ Via GitHub |
| Windows | `.exe` / `.msi` | ✅ Via GitHub |
| Linux | `.AppImage` / `.deb` | ✅ Via GitHub |

## 🚀 Release-Prozess

### Automatischer Build & Release

```bash
# 1. Version in package.json & tauri.conf.json erhöhen
# 2. Tag erstellen und pushen
git tag v1.0.29
git push origin v1.0.29

# 3. GitHub Action läuft automatisch:
#    - Buildet für macOS, Windows, Linux
#    - Erstellt GitHub Release
#    - Uploaded alle Installer
```

### Was wird erstellt

```
Release v1.0.29
├── FrameTrain.2_1.0.29_aarch64.dmg (macOS Installer)
├── FrameTrain.2_1.0.29_x64-setup.exe (Windows Installer)
├── FrameTrain.2_1.0.29_x64_en-US.msi (Windows Installer)
├── FrameTrain.2_1.0.29_amd64.AppImage (Linux Installer)
└── FrameTrain.2_1.0.29_amd64.deb (Linux Installer)
```

## ✅ Vorteile

- ✅ **Keine Code-Signing-Kosten** (Apple Developer, Windows Certificate)
- ✅ **Einfach zu implementieren**
- ✅ **Transparent** - User sieht GitHub Release
- ✅ **Sicher** - Downloads direkt von GitHub
- ✅ **Automatisch** - GitHub Actions baut alles

## ⚠️ Wichtig

### Keine Signatur-Validierung

- Die Installer sind **nicht signiert**
- macOS: User muss **"Rechtsklick → Öffnen"** beim ersten Start
- Windows: User muss **"Trotzdem ausführen"** klicken
- Linux: Keine Probleme (chmod +x für AppImage)

### User Experience

1. App zeigt Dialog: **"Update v1.0.29 verfügbar!"**
2. User klickt **"View Release"**
3. Browser öffnet GitHub Release-Seite
4. User lädt passenden Installer herunter
5. User installiert Update manuell

## 🔮 Zukünftige Verbesserungen

Wenn du später signierte Updates möchtest:

1. **Apple Developer Account** ($99/Jahr) für macOS Signing
2. **Windows Code Signing Certificate** (~$100-300/Jahr)
3. Tauri Updater Plugin mit Signierung aktivieren
4. Private Keys in GitHub Secrets speichern

Dann würde die App automatisch im Hintergrund updaten, ohne User-Interaktion.

## 📚 Weitere Infos

- [Tauri Updater Docs](https://tauri.app/v1/guides/distribution/updater/)
- [GitHub Releases API](https://docs.github.com/en/rest/releases)
- [Code Signing Overview](https://www.electronjs.org/docs/latest/tutorial/code-signing)
