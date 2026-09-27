"""Schnelle Pruefungen fuer text_to_image_lora — ohne Netz und ohne Modell.

Geprueft wird, was beim ersten echten Lauf am ehesten schiefgeht: welche
Captions zu welchem Bild gehoeren, DreamBooth ohne Captions, Zuschnitt,
Schrittzahl, und dass Test-Plugin und Labor ein exportiertes Modell als
Diffusion-Modell erkennen.

Aufruf:  python3 -m pytest plugins/text_to_image_lora/test_text_to_image_lora.py   (aus train_engine/)
"""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))                       # train_engine/
sys.path.insert(0, str(HERE.parents[3]))                       # python/ (ft_data)
sys.path.append(str(HERE.parents[3] / "test_engine"))       # model_server

from PIL import Image  # noqa: E402

from ft_data.diffusion import (  # noqa: E402
    LORA_INFO_FILE, LORA_WEIGHTS_FILE, is_diffusion_dir, pipeline_kind, read_lora_info,
    unsupported_message,
)
from ft_data.image_text import evaluation_samples, resolve_image_text  # noqa: E402
from plugins.text_to_image_lora.plugin import crop_box, training_steps  # noqa: E402


def img(path: Path, size=(40, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (200, 10, 10)).save(path)


class CaptionDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_txt_neben_dem_bild_ist_die_caption(self):
        for i in range(3):
            img(self.root / f"a{i}.png")
            (self.root / f"a{i}.txt").write_text(f"bild nummer {i}\n", encoding="utf-8")
        s = resolve_image_text(self.root, "caption", val_fraction=0.0)
        self.assertEqual(len(s.train), 3)
        self.assertEqual({x.answer for x in s.train}, {"bild nummer 0", "bild nummer 1", "bild nummer 2"})
        self.assertEqual(s.val, [])  # Diffusion: nichts abtrennen

    def test_metadata_jsonl_mit_unterordner(self):
        img(self.root / "img" / "x.png")
        img(self.root / "img" / "y.png")
        (self.root / "metadata.jsonl").write_text(
            json.dumps({"file_name": "img/x.png", "text": "ein roter kreis"}) + "\n"
            + json.dumps({"file_name": "img/y.png", "text": "noch einer"}) + "\n", encoding="utf-8")
        s = resolve_image_text(self.root, "caption", val_fraction=0.0)
        self.assertEqual(sorted(x.answer for x in s.train), ["ein roter kreis", "noch einer"])
        self.assertTrue(all(x.image.exists() for x in s.train))

    def test_metadata_csv_mit_caption_spalte(self):
        img(self.root / "p.jpg")
        (self.root / "metadata.csv").write_text("file_name,caption\np.jpg,ein foto\n", encoding="utf-8")
        s = resolve_image_text(self.root, "caption", val_fraction=0.0)
        self.assertEqual([x.answer for x in s.train], ["ein foto"])

    def test_dreambooth_nur_bilder_plus_instance_prompt(self):
        for i in range(4):
            img(self.root / f"hund_{i}.jpg")
        s = resolve_image_text(self.root, "caption", instance_prompt="a photo of sks dog", val_fraction=0.0)
        self.assertEqual(len(s.train), 4)
        self.assertEqual({x.answer for x in s.train}, {"a photo of sks dog"})

    def test_ohne_caption_und_ohne_instance_prompt_klare_meldung(self):
        img(self.root / "a.png")
        with self.assertRaises(ValueError) as ctx:
            resolve_image_text(self.root, "caption", val_fraction=0.0)
        self.assertIn("instance_prompt", str(ctx.exception))

    def test_eigene_beispielbilder_zaehlen_nicht_als_daten(self):
        img(self.root / "a.png")
        (self.root / "a.txt").write_text("echt", encoding="utf-8")
        img(self.root / "samples" / "after_1.png")
        s = resolve_image_text(self.root, "caption", instance_prompt="x", val_fraction=0.0)
        self.assertEqual(len(s.train), 1)

    def test_test_split_fuer_den_testlauf(self):
        img(self.root / "train" / "a.png"); (self.root / "train" / "a.txt").write_text("t", encoding="utf-8")
        img(self.root / "test" / "b.png"); (self.root / "test" / "b.txt").write_text("u", encoding="utf-8")
        samples, split = evaluation_samples(self.root, "caption")
        self.assertEqual(split, "test")
        self.assertEqual([s.answer for s in samples], ["u"])


class ZuschnittUndSchritteTest(unittest.TestCase):
    def test_kuerzere_seite_auf_aufloesung_mittig(self):
        w, h, left, top = crop_box(800, 400, 256, True, random.Random(0))
        self.assertEqual(h, 256)
        self.assertEqual(w, 512)
        self.assertEqual((left, top), (128, 0))

    def test_zufaelliger_ausschnitt_bleibt_im_bild(self):
        rng = random.Random(1)
        for _ in range(50):
            w, h, left, top = crop_box(300, 900, 128, False, rng)
            self.assertLessEqual(left + 128, w)
            self.assertLessEqual(top + 128, h)

    def test_schrittzahl(self):
        self.assertEqual(training_steps(10, 2, 1, 3, -1), 15)
        self.assertEqual(training_steps(10, 4, 2, 1, 0), 2)      # 3 Batches / 2 aufgerundet
        self.assertEqual(training_steps(10, 2, 1, 3, 100), 100)  # max_steps gewinnt


class ModellErkennungTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_pipeline_klassen(self):
        self.assertEqual(pipeline_kind("StableDiffusionPipeline"), "sd")
        self.assertEqual(pipeline_kind("StableDiffusionXLPipeline"), "sdxl")
        self.assertIsNone(pipeline_kind("FluxPipeline"))
        self.assertIn("FLUX", unsupported_message("FluxPipeline"))

    def test_lora_export_und_pipeline_gelten_als_diffusion(self):
        import model_server

        lora = self.root / "lora"
        lora.mkdir()
        (lora / LORA_INFO_FILE).write_text(json.dumps({"base_model_path": "/x"}), encoding="utf-8")
        (lora / LORA_WEIGHTS_FILE).write_bytes(b"0")
        pipe = self.root / "pipe"
        pipe.mkdir()
        (pipe / "model_index.json").write_text('{"_class_name": "StableDiffusionPipeline"}', encoding="utf-8")
        hf = self.root / "hf"
        hf.mkdir()
        (hf / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")

        self.assertTrue(is_diffusion_dir(lora))
        self.assertEqual(read_lora_info(lora)["base_model_path"], "/x")
        self.assertTrue(model_server.is_diffusion_model(lora))
        self.assertTrue(model_server.is_diffusion_model(pipe))
        self.assertFalse(model_server.is_diffusion_model(hf))
        self.assertEqual(model_server.INPUT_KIND["text_to_image"], "text")



class WeightVariantTest(unittest.TestCase):
    def test_nur_fp16_dateien_brauchen_variant(self):
        import tempfile
        from pathlib import Path
        from ft_data.diffusion import weight_variant
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "unet").mkdir()
            (Path(d) / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")
            self.assertEqual(weight_variant(Path(d)), "fp16")
            (Path(d) / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
            self.assertIsNone(weight_variant(Path(d)))

if __name__ == "__main__":
    unittest.main()
