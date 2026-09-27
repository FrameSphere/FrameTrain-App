"""Schnelle Pruefungen fuer vision_language — ohne Netz und ohne Modell.

Datenformate (JSONL, Chat-Format, ShareGPT, .txt, Klassenordner), Kennzahlen,
LoRA-Zielschichten (Bild-Encoder bleibt aussen vor), Labels nur auf der
Antwort, Einzel-Eingabe "pfad ||| frage" und die Modalitaet im Modell-Server.

Aufruf:  python3 -m pytest plugins/vision_language/test_vision_language.py   (aus train_engine/)
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))                            # train_engine/
sys.path.insert(0, str(HERE.parents[3]))                            # python/ (ft_data)
sys.path.append(str(HERE.parents[3] / "test_engine"))            # model_server
sys.path.append(str(HERE.parents[3] / "test_engine" / "plugins"))

from PIL import Image  # noqa: E402

from ft_data.image_text import (  # noqa: E402
    normalise_answer, parse_messages, resolve_image_text, row_to_sample, text_scores,
)
from ft_data import vlm  # noqa: E402


def img(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), (0, 0, 255)).save(path)


class ZeilenTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        img(self.root / "images" / "a.png")

    def test_frage_und_antwort(self):
        s = row_to_sample({"image": "images/a.png", "question": "Farbe?", "answer": "blau"}, [self.root], "vlm")
        self.assertEqual((s.prompt, s.answer), ("Farbe?", "blau"))

    def test_prompt_und_caption(self):
        s = row_to_sample({"image": "images/a.png", "prompt": "Beschreibe", "caption": "ein Quadrat"},
                          [self.root], "vlm")
        self.assertEqual((s.prompt, s.answer), ("Beschreibe", "ein Quadrat"))

    def test_nur_text_ist_die_antwort(self):
        s = row_to_sample({"file_name": "images/a.png", "text": "ein blaues Quadrat"}, [self.root], "vlm")
        self.assertEqual((s.prompt, s.answer), ("", "ein blaues Quadrat"))

    def test_vqa_liste_von_antworten(self):
        s = row_to_sample({"image": "images/a.png", "question": "?", "answers": ["blau", "dunkelblau"]},
                          [self.root], "vlm")
        self.assertEqual(s.answer, "blau")

    def test_chat_format_mit_bildteil(self):
        row = {"messages": [
            {"role": "user", "content": [{"type": "image", "image": "images/a.png"},
                                         {"type": "text", "text": "Welche Farbe?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "blau"}]}]}
        s = row_to_sample(row, [self.root], "vlm")
        self.assertEqual((s.prompt, s.answer, s.image.name), ("Welche Farbe?", "blau", "a.png"))

    def test_sharegpt_mit_platzhalter(self):
        row = {"image": "images/a.png", "conversations": [
            {"from": "human", "value": "<image>\nWas siehst du?"}, {"from": "gpt", "value": "Ein Quadrat."}]}
        s = row_to_sample(row, [self.root], "vlm")
        self.assertEqual((s.prompt, s.answer), ("Was siehst du?", "Ein Quadrat."))

    def test_fehlendes_bild_wird_uebersprungen(self):
        self.assertIsNone(row_to_sample({"image": "fehlt.png", "answer": "x"}, [self.root], "vlm"))

    def test_parse_messages_nimmt_ersten_dialogschritt(self):
        q, a, _ = parse_messages([{"role": "system", "content": "sys"},
                                  {"role": "user", "content": "F1"}, {"role": "assistant", "content": "A1"},
                                  {"role": "user", "content": "F2"}, {"role": "assistant", "content": "A2"}])
        self.assertEqual((q, a), ("F1", "A1"))


class LayoutTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_split_ordner_mit_jsonl(self):
        for split, n in (("train", 4), ("val", 2)):
            rows = []
            for i in range(n):
                img(self.root / split / "images" / f"{i}.png")
                rows.append(json.dumps({"image": f"images/{i}.png", "question": "F", "answer": str(i)}))
            (self.root / split / "data.jsonl").write_text("\n".join(rows), encoding="utf-8")
        s = resolve_image_text(self.root, "vlm")
        self.assertEqual((len(s.train), len(s.val)), (4, 2))

    def test_klassenordner_antwort_ist_der_ordnername(self):
        for cls in ("rot", "gruen_hell"):
            for i in range(5):
                img(self.root / cls / f"{i}.png")
        s = resolve_image_text(self.root, "vlm", val_fraction=0.2)
        answers = {x.answer for x in s.train + s.val}
        self.assertEqual(answers, {"rot", "gruen hell"})
        self.assertEqual(len(s.val), 2)  # 20 % von 10, aus train abgetrennt

    def test_bilder_mit_txt_fuer_captioning(self):
        for i in range(3):
            img(self.root / f"{i}.png")
            (self.root / f"{i}.txt").write_text(f"text {i}", encoding="utf-8")
        s = resolve_image_text(self.root, "vlm", val_fraction=0.0)
        self.assertEqual(sorted(x.answer for x in s.train), ["text 0", "text 1", "text 2"])

    def test_hf_parquet_mit_bildbytes_wird_entpackt(self):
        import io
        import pyarrow as pa
        import pyarrow.parquet as pq

        def png():
            buf = io.BytesIO()
            Image.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="PNG")
            return buf.getvalue()

        table = pa.table({
            "image": [{"bytes": png(), "path": None} for _ in range(3)],
            "question": ["Farbe?"] * 3,
            "answer": ["rot", "rot", "rot"],
        })
        (self.root / "data").mkdir()
        pq.write_table(table, self.root / "data" / "train-00000-of-00001.parquet")
        pq.write_table(table, self.root / "train.parquet")
        s = resolve_image_text(self.root, "vlm", val_fraction=0.0)
        self.assertEqual(len(s.train), 3)
        self.assertEqual({(x.prompt, x.answer) for x in s.train}, {("Farbe?", "rot")})
        self.assertTrue(all(x.image.exists() for x in s.train))
        # Zweiter Aufruf nutzt den Cache (gleiche Signatur).
        self.assertEqual(len(resolve_image_text(self.root, "vlm", val_fraction=0.0).train), 3)

    def test_leerer_ordner_klare_meldung(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_image_text(self.root, "vlm")
        self.assertIn("JSONL", str(ctx.exception))


class KennzahlenTest(unittest.TestCase):
    def test_exact_match_ignoriert_gross_und_satzzeichen(self):
        self.assertEqual(normalise_answer("  Rot. "), "rot")
        sc = text_scores(["Rot.", "blau"], ["rot", "gruen"])
        self.assertAlmostEqual(sc["exact_match"], 0.5)

    def test_rouge_l_mit_umlauten(self):
        sc = text_scores(["ein grünes quadrat"], ["ein grünes dreieck"])
        self.assertAlmostEqual(sc["rougeL"], 2 / 3, places=3)


class LoraZieleTest(unittest.TestCase):
    def test_nur_sprachmodell_ohne_bild_encoder(self):
        import torch.nn as nn

        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(4, 4) for _ in range(4))

        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_model = nn.ModuleDict({"attn": Attn()})
                self.connector = nn.Linear(4, 4)
                self.text_model = nn.ModuleDict({"attn": Attn()})
                self.lm_head = nn.Linear(4, 10)

        names = vlm.find_lora_targets(Toy())
        self.assertEqual(sorted(names), ["text_model.attn.k_proj", "text_model.attn.o_proj",
                                         "text_model.attn.q_proj", "text_model.attn.v_proj"])
        with_vision = vlm.find_lora_targets(Toy(), train_vision=True)
        self.assertEqual(len(with_vision), 8)

    def test_blip_namen_query_key_value(self):
        import torch.nn as nn

        class Self(nn.Module):
            def __init__(self):
                super().__init__()
                self.query, self.key, self.value = (nn.Linear(4, 4) for _ in range(3))

        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.text_decoder = nn.ModuleDict({"self": Self()})

        self.assertEqual(len(vlm.find_lora_targets(Toy())), 3)

    def test_gemeinsamer_praefix(self):
        import torch
        self.assertEqual(vlm._common_prefix(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 9, 9])), 2)
        self.assertEqual(vlm._common_prefix(torch.tensor([1, 2]), torch.tensor([1, 2, 5])), 2)


class EingabeUndServerTest(unittest.TestCase):
    def test_pfad_und_frage_in_einem_feld(self):
        from vision_language.plugin import split_single_input

        self.assertEqual(split_single_input("/a/b.png ||| Welche Farbe?"), ("/a/b.png", "Welche Farbe?"))
        self.assertEqual(split_single_input('"/a/b.png"', "Frage aus Feld"), ("/a/b.png", "Frage aus Feld"))
        # Die Frage aus dem eigenen Feld gewinnt gegen die Kurzform.
        self.assertEqual(split_single_input("/a.png ||| alt", "neu"), ("/a.png", "neu"))

    def test_vlm_statt_seq2seq_im_modell_server(self):
        import model_server

        smol = {"model_type": "idefics3", "architectures": ["Idefics3ForConditionalGeneration"]}
        blip = {"model_type": "blip", "architectures": ["BlipForConditionalGeneration"]}
        t5 = {"model_type": "t5", "architectures": ["T5ForConditionalGeneration"]}
        whisper = {"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}
        self.assertEqual(model_server.detect_modality(smol), "vlm")
        self.assertEqual(model_server.detect_modality(blip), "vlm")
        self.assertEqual(model_server.detect_modality(t5), "seq2seq")
        self.assertEqual(model_server.detect_modality(whisper), "asr")
        self.assertEqual(model_server.INPUT_KIND["vlm"], "image")

    def test_clip_ist_kein_vlm(self):
        self.assertFalse(vlm.is_vlm_config({"model_type": "clip", "architectures": ["CLIPModel"]}))


if __name__ == "__main__":
    unittest.main()
