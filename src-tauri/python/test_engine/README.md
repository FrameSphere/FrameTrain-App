# FrameTrain v2 - Universal Testing Engine

## 🎯 Überblick

Dies ist ein **modulares, plugin-basiertes Test-System** für alle Modalitäten.

### ✅ Unterstützte Modalitäten:

| Modalität | Plugin | Metriken | Status |
|-----------|--------|----------|--------|
| **Text/NLP** | Built-in | Accuracy, Loss, BLEU | ✅ |
| **Vision** | plugin_vision_test.py | Accuracy, Top-5, Confidence | ✅ |
| **Detection** | plugin_detection_test.py | mAP@50, mAP@75, Precision, Recall | ✅ |
| **Audio** | plugin_audio_test.py | WER, CER, Accuracy | ✅ |

---

## 🚀 Verwendung

### **Basis-Verwendung:**
```bash
# 1. Config erstellen
cat > test_config.json <<EOF
{
  "model_path": "/path/to/trained/model",
  "dataset_path": "/path/to/test/dataset",
  "output_path": "/path/to/output",
  "batch_size": 8,
  "max_samples": null
}
EOF

# 2. Test ausführen
python proto_test_engine.py --config test_config.json
```

### **Mit Auto-Install:**
```bash
export FRAMETRAIN_AUTO_INSTALL=true
python proto_test_engine.py --config test_config.json
```

---

## 📊 Beispiel-Configs

### **Text/NLP Testing:**
```json
{
  "model_path": "/models/gpt2_finetuned",
  "dataset_path": "/data/text_test",
  "output_path": "/output/test_results",
  "batch_size": 16,
  "max_samples": 100
}
```

**Dataset-Struktur:**
```
dataset/
└── test/
    ├── data.txt
    ├── data.csv
    └── data.jsonl
```

---

### **Vision Testing (Classification):**
```json
{
  "model_path": "/models/resnet50_trained",
  "dataset_path": "/data/cats_dogs_test",
  "output_path": "/output/vision_test",
  "batch_size": 32
}
```

**Dataset-Struktur:**
```
dataset/
└── test/
    ├── cat/
    │   ├── cat1.jpg
    │   └── cat2.jpg
    └── dog/
        ├── dog1.jpg
        └── dog2.jpg
```

---

### **Detection Testing (YOLO):**
```json
{
  "model_path": "/models/yolov8n.pt",
  "dataset_path": "/data/detection_test",
  "output_path": "/output/detection_test",
  "batch_size": 8
}
```

**Dataset-Struktur:**
```
dataset/
└── test/
    ├── images/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── labels/
        ├── img1.txt  (YOLO format)
        └── img2.txt
```

---

### **Audio Testing (Whisper):**
```json
{
  "model_path": "openai/whisper-base",
  "dataset_path": "/data/audio_test",
  "output_path": "/output/audio_test",
  "batch_size": 4
}
```

**Dataset-Struktur:**
```
dataset/
└── test/
    ├── audio1.wav
    ├── audio1.txt  (transcript)
    ├── audio2.wav
    └── audio2.txt
```

---

## 📈 Output Format

### **Gespeicherte Dateien:**
```
output/
└── test_results.json
```

### **test_results.json Struktur:**
```json
{
  "accuracy": 95.5,
  "total_samples": 100,
  "correct_predictions": 95,
  "incorrect_predictions": 5,
  "average_loss": 0.123,
  "average_inference_time": 0.05,
  "predictions": [
    {
      "sample_id": 0,
      "predicted": "cat",
      "expected": "cat",
      "is_correct": true,
      "confidence": 0.98,
      "inference_time": 0.045
    }
  ],
  "per_class_metrics": {
    "cat": {"accuracy": 96.0, "total": 50},
    "dog": {"accuracy": 95.0, "total": 50}
  }
}
```

---

## 🔌 Plugin-Entwicklung

### **Neues Plugin erstellen:**

```python
# plugins/plugin_mymodal_test.py

"""
My Modal Test Plugin
====================

MANIFEST:
{
    "name": "My Modal Test Plugin",
    "description": "Testing for My Modal models",
    "modality": "mymodal",
    "required": ["torch", "mylib"],
    "optional": [],
    "python": "3.8"
}
"""

from proto_test_engine import BaseTestLoader, Modality, TEST_REGISTRY

class MyModalTestLoader(BaseTestLoader):
    def load_model(self):
        # Load your model
        pass
    
    def load_test_data(self):
        # Load test data
        return test_data
    
    def test_sample(self, sample):
        # Test one sample
        return results
    
    def compute_metrics(self, all_results):
        # Compute final metrics
        return metrics

# Register
TEST_REGISTRY.register_test_loader(Modality.MYMODAL, MyModalTestLoader)
```

---

## 🎨 Modalität Detection

Das System erkennt automatisch die Modalität:

1. **Aus Model-Pfad:**
   - `resnet50` → Vision
   - `yolov8` → Detection
   - `whisper` → Audio

2. **Aus Dataset:**
   - `.jpg/.png` → Vision
   - `.jpg + labels/` → Detection
   - `.wav/.mp3` → Audio
   - `.txt/.json` → Text

---

## 📝 Metriken pro Modalität

### **Text:**
- Accuracy
- Average Loss
- Inference Time

### **Vision:**
- Accuracy
- Top-5 Accuracy
- Per-Class Accuracy
- Average Confidence

### **Detection:**
- mAP@50
- mAP@75
- Precision
- Recall
- F1 Score
- IoU

### **Audio:**
- WER (Word Error Rate)
- CER (Character Error Rate)
- Accuracy (1 - WER)

---

## 🔧 Debugging

### **Debug-Modus aktivieren:**
```bash
export FRAMETRAIN_DEBUG=true
python proto_test_engine.py --config test_config.json
```

### **Unterstützte Modalitäten auflisten:**
```bash
python proto_test_engine.py --list-supported
```

---

## 🐛 Troubleshooting

### **Problem: "No module named 'timm'"**
```bash
pip install timm
# Oder mit Auto-Install:
export FRAMETRAIN_AUTO_INSTALL=true
```

### **Problem: "No test data found"**
- Prüfe dass `test/` oder `val/` Ordner existiert
- Prüfe Datei-Extensions (.jpg, .txt, .wav, etc.)

### **Problem: "Model loading failed"**
- Prüfe `model_path` in Config
- Für YOLO: Muss `.pt` Datei sein
- Für Vision: Kann timm model name oder Checkpoint sein

---

## 🎯 Integration mit FrameTrain

Das Test-System ist vollständig integriert:

1. **Frontend:** TestPanel ruft Test-Engine auf
2. **Backend:** Rust startet Python-Prozess
3. **Output:** JSON messages via stdout
4. **Ergebnisse:** Werden in DB gespeichert

---

## 📚 Weitere Informationen

- **Training System:** `../proto_train_engine.py`
- **Plugin System:** Nutzt gleiche Architektur wie Training
- **Dependency Manager:** Automatische Installation von Paketen

---

## ✅ Status: PRODUCTION READY

Das Test-System ist einsatzbereit für:
- ✅ Text/NLP Modelle
- ✅ Vision Modelle (Classification)
- ✅ Detection Modelle (YOLO)
- ✅ Audio Modelle (Whisper)

**Next:** Weitere Plugins für Segmentation, Time Series, etc.
