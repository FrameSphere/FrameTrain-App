# FrameTrain v1.0.8 - Changelog & Fixes

## ✅ 1. Zwei neue Theme-Farben hinzugefügt

**Neue Themes:**
- **Light Gray**: Helles neutrales Grau-Theme
- **Pure White**: Klares weißes Light-Mode Theme

**Datei**: `src/contexts/ThemeContext.tsx`
**Status**: ✅ Implementiert

---

## ⚠️ 2-4. Version-Auswahl: Neueste Version vorauswählen

**Problem**: Aktuell wird die "Original/Root"-Version vorausgewählt
**Lösung**: Automatisch die neueste Version (höchste version_number) auswählen

**Betroffene Dateien:**
- `src/components/TrainingPanel.tsx`
- `src/components/AnalysisPanel.tsx`
- `src/components/TestPanel.tsx`

**Erforderliche Änderung** (in allen 3 Dateien):

### Suche nach:
```typescript
// Aktueller Code (findet Root-Version)
const rootVersion = versions.find(v => v.is_root);
setSelectedVersion(rootVersion?.id || '');
```

### Ersetze mit:
```typescript
// Neuer Code (findet neueste Version)
const sortedVersions = [...versions].sort((a, b) => b.version_number - a.version_number);
const newestVersion = sortedVersions[0];
setSelectedVersion(newestVersion?.id || '');
```

**Status**: ⚠️ Manuelle Änderung erforderlich (Dateien zu groß für automatische Bearbeitung)

---

## 📋 5. Python Plugins prüfen

**Zu prüfende Ordner:**
- `src-tauri/python/train_engine/plugins/`
- `src-tauri/python/test_engine/plugins/`

**Erforderliche Plugins:**
- `plugin_vision_test.py` - Vision/Bild-Klassifikation
- `plugin_audio_test.py` - Audio-Verarbeitung
- `plugin_detection_test.py` - Objekt-Erkennung
- `plugin_segmentation_test.py` - Bild-Segmentierung

**Status**: 🔍 Wird geprüft

---

## 🐛 6. Test Predictions Anzeige reparieren

**Problem**: Predictions-Tabelle zeigt keine Daten ("0 / 0")
**Ursache**: Daten werden nicht korrekt aus der Datenbank geladen oder angezeigt

**Betroffene Dateien:**
- `src/components/TestPanel.tsx`
- Backend: `src-tauri/src/test_manager.rs`

**Zu prüfen:**
1. Werden Predictions korrekt in DB gespeichert?
2. Wird `test_results` Table korrekt abgefragt?
3. Wird JSON korrekt geparst?

**Status**: 🔍 Wird analysiert

---

## 🚀 7. Auto-Update System

**Status**: ✅ Implementiert und bereit für Test

**Signing Keys**: ✅ Bereits in GitHub Secrets und tauri.conf.json

**Workflow**: ✅ Konfiguriert für signierte Releases

**Test-Plan**:
1. v1.0.7 installieren
2. v1.0.8 Release erstellen (mit allen Fixes)
3. v1.0.7 App starten
4. Update-Modal sollte erscheinen
5. Update installieren
6. App startet neu mit v1.0.8

---

## 📦 Release Checklist v1.0.8

- [x] Theme-System erweitert (Light Gray, Pure White)
- [ ] Version-Auswahl auf neueste Version (Training/Analyse/Test)
- [ ] Python Plugins überprüft und vervollständigt
- [ ] Test Predictions Anzeige repariert
- [ ] Version auf 1.0.8 erhöht (tauri.conf.json)
- [ ] Changelog aktualisiert
- [ ] Git commit & push
- [ ] GitHub Workflow starten
- [ ] Auto-Update testen

---

## 🔧 Nächste Schritte

1. **Manuelle Code-Änderungen** für Punkte 2-4 (Version-Auswahl)
2. **Plugin-Ordner prüfen** und fehlende Plugins erstellen
3. **Test Predictions Bug** analysieren und fixen
4. **Version bumpen** auf 1.0.8
5. **Release erstellen** und Auto-Update testen
