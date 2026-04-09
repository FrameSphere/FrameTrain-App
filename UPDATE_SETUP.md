# Auto-Update Setup für FrameTrain Desktop App

## 🔐 WICHTIG: Update-Signing (Vor dem ersten Release!)

Das Auto-Update-System erfordert kryptografische Signaturen für Sicherheit.

### 1️⃣ Signing-Keys generieren (EINMALIG)

```bash
cd /Users/karol/Desktop/Laufende_Projekte/FrameTrain/desktop-app2/src-tauri

# Generiere Key-Pair
cargo tauri signer generate
```

**Output:**
```
Your secret key (KEEP THIS PRIVATE!):
dW50cnVzdGVkIGNvbW1lbnQ6IHJzaWduIGVuY3J5cHRlZCBzZWNyZXQga2V5...

Your public key (ADD TO tauri.conf.json):
dW50cnVzdGVkIGNvbW1lbnQ6IG1pbmlzaWduIHB1YmxpYyBrZXk6IEU2REVD...
```

### 2️⃣ Keys speichern

#### Private Key (GEHEIM HALTEN!)
Speichere den **Private Key** sicher:

**Option A: Lokale Datei (für lokales Testen)**
```bash
echo "DEIN_PRIVATE_KEY" > ~/.tauri/frametrain.key
chmod 600 ~/.tauri/frametrain.key
```

**Option B: GitHub Secret (für CI/CD)** ✅ Empfohlen
1. Gehe zu: https://github.com/KarolP-tech/FrameTrain/settings/secrets/actions
2. Klicke "New repository secret"
3. Name: `TAURI_SIGNING_PRIVATE_KEY`
4. Value: Dein Private Key
5. Speichern

#### Public Key
Füge den **Public Key** in `tauri.conf.json` ein:

```json
{
  "plugins": {
    "updater": {
      "pubkey": "HIER_DEINEN_PUBLIC_KEY_EINFÜGEN"
    }
  }
}
```

### 3️⃣ GitHub Actions für Signing konfigurieren

Die Workflow-Datei `.github/workflows/release-desktop-app.yml` muss aktualisiert werden:

```yaml
- name: Build Tauri App
  env:
    TAURI_SIGNING_PRIVATE_KEY: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY }}
    TAURI_SIGNING_PRIVATE_KEY_PASSWORD: ${{ secrets.TAURI_SIGNING_PASSWORD }}
  run: |
    cd desktop-app2
    npm run tauri build
```

**Hinweis:** Wenn dein Private Key ein Passwort hat, füge auch `TAURI_SIGNING_PASSWORD` als Secret hinzu.

### 4️⃣ Lokales Bauen mit Signing

```bash
cd /Users/karol/Desktop/Laufende_Projekte/FrameTrain/desktop-app2

# Setze Private Key als Environment Variable
export TAURI_SIGNING_PRIVATE_KEY="DEIN_PRIVATE_KEY"

# Baue die App (wird automatisch signiert)
npm run tauri build
```

**Ergebnis:**
```
src-tauri/target/release/bundle/
├── macos/
│   ├── FrameTrain 2.app.tar.gz        # Signiertes Update-Paket
│   └── FrameTrain 2.app.tar.gz.sig    # Signatur
├── latest.json                         # Update-Manifest
└── latest.json.sig                     # Manifest-Signatur
```

---

## 🚀 Wie Updates funktionieren

### Für den User:
1. App startet
2. Prüft automatisch auf Updates (im Hintergrund)
3. Zeigt Modal: "Update verfügbar"
4. User klickt "Jetzt updaten"
5. Download läuft (mit Progress-Bar)
6. App startet automatisch neu
7. ✅ Neue Version läuft

### Technisch:
1. App ruft `https://github.com/KarolP-tech/FrameTrain/releases/latest/download/latest.json` ab
2. Vergleicht Version in `latest.json` mit aktueller App-Version
3. Lädt Update-Paket (`.app.tar.gz` / `.msi` / `.AppImage`) herunter
4. Verifiziert Signatur mit Public Key
5. Installiert Update
6. Startet App neu

---

## 📦 GitHub Release-Struktur

Jeder Release **MUSS** diese Dateien enthalten:

```
v1.0.7
├── FrameTrain.2_1.0.7_aarch64.dmg        # macOS Installer (Download-Seite)
├── FrameTrain.2_1.0.7_x64_en-US.msi      # Windows Installer
├── FrameTrain.2_1.0.7_amd64.AppImage     # Linux Installer
├── FrameTrain 2.app.tar.gz                # macOS Update-Paket
├── FrameTrain 2.app.tar.gz.sig            # macOS Update-Signatur
├── FrameTrain 2_1.0.7_x64-setup.nsis.zip  # Windows Update-Paket
├── FrameTrain 2_1.0.7_x64-setup.nsis.zip.sig
├── latest.json                            # Update-Manifest ⚠️ WICHTIG
└── latest.json.sig                        # Manifest-Signatur
```

### ⚠️ KRITISCH: `latest.json`

Diese Datei wird **automatisch** von Tauri erstellt beim Build:

```json
{
  "version": "v1.0.7",
  "notes": "Release notes here",
  "pub_date": "2024-12-11T15:30:00Z",
  "platforms": {
    "darwin-aarch64": {
      "signature": "...",
      "url": "https://github.com/.../FrameTrain 2.app.tar.gz"
    },
    "windows-x86_64": {
      "signature": "...",
      "url": "https://github.com/.../setup.nsis.zip"
    }
  }
}
```

**Tauri baut diese Datei automatisch** - du musst nichts manuell machen! ✅

---

## ✅ Testing

### 1. Lokaler Test (ohne Signing):
```bash
cd desktop-app2
npm run tauri dev
```

→ Update-Check wird ausgeführt, aber Fehler wenn kein Public Key gesetzt

### 2. Production Test (mit Signing):
1. Baue Version `1.0.6` mit Signing
2. Erstelle Release auf GitHub
3. Baue Version `1.0.7` mit Signing
4. Starte `1.0.6` App
5. → Update-Modal sollte erscheinen
6. Installiere Update
7. → App startet neu mit `1.0.7`

---

## 🔒 Sicherheits-Best-Practices

✅ **DO:**
- Private Key **NIE** in Git committen
- Private Key in GitHub Secrets speichern
- Jedes Release signieren
- Public Key in `tauri.conf.json` hinterlegen

❌ **DON'T:**
- Private Key teilen
- Private Key in Code oder Logs loggen
- Releases ohne Signatur deployen (in Production)

---

## 🐛 Troubleshooting

### "Failed to check for updates"
- Prüfe Internet-Verbindung
- Prüfe ob GitHub Release existiert
- Prüfe ob `latest.json` im Release vorhanden ist

### "Signature verification failed"
- Public Key falsch in `tauri.conf.json`
- Release wurde mit anderem Private Key signiert
- Signatur-Datei fehlt im Release

### "Update already installed"
- Version in `latest.json` ist gleich oder älter
- Markiere GitHub Release als "latest"

### Update erscheint nicht
- Version in `tauri.conf.json` muss kleiner sein als Release-Version
- Cache-Problem: Warte 1 Minute (GitHub API Cache)

---

## 📝 Checklist für jeden Release

- [ ] Version in `tauri.conf.json` erhöht
- [ ] `TAURI_SIGNING_PRIVATE_KEY` in GitHub Secrets gesetzt
- [ ] Build mit Signing erfolgreich
- [ ] `latest.json` + Signatur im Release vorhanden
- [ ] Installer-Dateien im Release vorhanden
- [ ] Release als "latest" markiert
- [ ] Update-Test durchgeführt

---

## 🎯 Nächste Schritte

1. **Jetzt:** Keys generieren und speichern
2. **Dann:** Public Key in `tauri.conf.json` einfügen
3. **Build:** `npm run tauri build`
4. **Release:** v1.0.7 erstellen
5. **Test:** v1.0.6 App starten → Update sollte erscheinen

---

🎉 **Fertig!** Deine App hat jetzt professionelles Auto-Update!
