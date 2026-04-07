#!/usr/bin/env python3
"""
Quick Demo: Test Engine mit Plugin-System
==========================================

Demonstriert das neue plugin-basierte Test-System.
"""

import os
import sys
import json
from pathlib import Path

print("="*70)
print("PROTO TEST ENGINE - PLUGIN-BASED TESTING DEMO")
print("="*70)

def demo_1_list_supported():
    """Demo: Zeige unterstützte Modalitäten"""
    print("\n" + "="*70)
    print("DEMO 1: Unterstützte Modalitäten")
    print("="*70)
    
    from test_engine import TEST_REGISTRY
    
    supported = TEST_REGISTRY.list_supported()
    
    print("\n✅ Unterstützte Test-Modalitäten:")
    for modality in supported:
        print(f"  - {modality}")
    
    print("\n💡 Neue Modalitäten können via Plugins hinzugefügt werden!")


def demo_2_check_plugins():
    """Demo: Prüfe verfügbare Plugins"""
    print("\n" + "="*70)
    print("DEMO 2: Verfügbare Test-Plugins")
    print("="*70)
    
    plugin_dir = Path(__file__).parent / "plugins"
    
    if not plugin_dir.exists():
        print("❌ Plugin-Ordner nicht gefunden")
        return
    
    plugins = list(plugin_dir.glob("plugin_*_test.py"))
    
    print(f"\n📦 Gefundene Plugins ({len(plugins)}):")
    for plugin in plugins:
        print(f"  - {plugin.name}")
    
    if not plugins:
        print("\n⚠️  Keine Plugins gefunden")


def demo_3_modality_detection():
    """Demo: Modalität Detection"""
    print("\n" + "="*70)
    print("DEMO 3: Modalität Detection")
    print("="*70)
    
    from test_engine import ModalityDetector, Modality
    
    test_cases = [
        ("resnet50", "Model-Pfad"),
        ("yolov8n.pt", "Model-Pfad"),
        ("whisper-base", "Model-Pfad"),
        ("/data/text", "Dataset-Pfad"),
    ]
    
    print("\n🔍 Test verschiedener Pfade:\n")
    
    for path, source in test_cases:
        if source == "Model-Pfad":
            modality, metadata = ModalityDetector.detect_from_model(path)
        else:
            modality, metadata = ModalityDetector.detect_from_dataset(path)
        
        print(f"  {path:<30} → {modality.value}")


def demo_4_config_examples():
    """Demo: Zeige Config-Beispiele"""
    print("\n" + "="*70)
    print("DEMO 4: Config-Beispiele")
    print("="*70)
    
    examples = {
        "Text Testing": {
            "model_path": "/models/gpt2_finetuned",
            "dataset_path": "/data/text_test",
            "output_path": "/output/results",
            "batch_size": 16
        },
        "Vision Testing": {
            "model_path": "resnet50",
            "dataset_path": "/data/images_test",
            "output_path": "/output/results",
            "batch_size": 32
        },
        "Detection Testing": {
            "model_path": "/models/yolov8.pt",
            "dataset_path": "/data/detection_test",
            "output_path": "/output/results",
            "batch_size": 8
        }
    }
    
    for name, config in examples.items():
        print(f"\n📝 {name}:")
        print(json.dumps(config, indent=2))


def demo_5_comparison():
    """Demo: Vergleich Alt vs Neu"""
    print("\n" + "="*70)
    print("DEMO 5: Vergleich: test_engine.py vs test_engine.py")
    print("="*70)
    
    comparison = """
    | Feature                  | test_engine.py | test_engine.py |
    |--------------------------|----------------|----------------------|
    | Text/NLP                 | ✅             | ✅                   |
    | Vision                   | ❌             | ✅                   |
    | Detection                | ❌             | ✅                   |
    | Audio                    | ❌             | ✅                   |
    | Plugin-System            | ❌             | ✅                   |
    | Auto-Dependency Install  | ❌             | ✅                   |
    | Task-specific Metrics    | ⚠️  Partial    | ✅                   |
    | Erweiterbar              | ❌             | ✅                   |
    """
    
    print(comparison)
    
    print("\n🎯 Fazit:")
    print("  • test_engine.py: Nur Text/NLP")
    print("  • test_engine.py: Alle Modalitäten via Plugins")


def demo_6_usage_example():
    """Demo: Verwendungs-Beispiel"""
    print("\n" + "="*70)
    print("DEMO 6: Verwendungs-Beispiel")
    print("="*70)
    
    example = """
# 1. Config erstellen
cat > test_config.json <<EOF
{
  "model_path": "/models/resnet50_trained",
  "dataset_path": "/data/test_images",
  "output_path": "/output/test_results",
  "batch_size": 32
}
EOF

# 2. Test ausführen
python test_engine.py --config test_config.json

# 3. Mit Auto-Install
export FRAMETRAIN_AUTO_INSTALL=true
python test_engine.py --config test_config.json

# 4. Ergebnisse prüfen
cat /output/test_results/test_results.json
    """
    
    print(example)


def main():
    """Run all demos"""
    
    demos = [
        ("Unterstützte Modalitäten", demo_1_list_supported),
        ("Verfügbare Plugins", demo_2_check_plugins),
        ("Modalität Detection", demo_3_modality_detection),
        ("Config-Beispiele", demo_4_config_examples),
        ("Vergleich Alt vs Neu", demo_5_comparison),
        ("Verwendungs-Beispiel", demo_6_usage_example),
    ]
    
    print("\nVerfügbare Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    
    print("\nOptionen:")
    print("  all   - Alle Demos ausführen")
    print("  1-6   - Einzelne Demo ausführen")
    print("  q     - Beenden")
    
    while True:
        choice = input("\nDemo auswählen: ").strip().lower()
        
        if choice == 'q':
            print("\n👋 Tschüss!")
            break
        
        elif choice == 'all':
            for name, demo_func in demos:
                try:
                    demo_func()
                except Exception as e:
                    print(f"\n❌ Demo fehlgeschlagen: {e}")
                    import traceback
                    traceback.print_exc()
        
        elif choice.isdigit() and 1 <= int(choice) <= len(demos):
            try:
                demos[int(choice) - 1][1]()
            except Exception as e:
                print(f"\n❌ Demo fehlgeschlagen: {e}")
                import traceback
                traceback.print_exc()
        
        else:
            print("❌ Ungültige Auswahl")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Abgebrochen!")
        sys.exit(0)
