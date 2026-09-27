"""Tests fuer ft_data.llm: Formaterkennung, Splits, Maskierung, Bewertung — ohne Netz.

Aufruf:  python3 -m ft_data.test_llm   (aus src-tauri/python/)
"""
import json
import tempfile
import unittest
from pathlib import Path

from ft_data import llm as D


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


class _CharTok:
    """Minimal-Tokenizer: ein Token je Zeichen, mit offset_mapping."""
    eos_token_id = 0
    eos_token = "<e>"
    bos_token = None
    chat_template = None

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        out = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        s = "".join(f"<{m['role']}>{m['content']}" + ("|" if m["role"] == "assistant" else "") for m in messages)
        return s + ("<assistant>" if add_generation_prompt else "")


class FormatTest(unittest.TestCase):
    def test_detect_format(self):
        for cols, fmt in [(["messages"], "messages"), (["conversations"], "messages"),
                          (["prompt", "completion"], "pair"), (["instruction", "input", "output"], "pair"),
                          (["Question", "Answer"], "pair"), (["text"], "text")]:
            self.assertEqual(D.detect_format(cols)[0], fmt, cols)

    def test_alpaca_input_wird_angehaengt(self):
        fmt, cols = D.detect_format(["instruction", "input", "output"])
        ex = D.row_to_example({"instruction": "Fasse zusammen.", "input": "Langer Text", "output": "Kurz"}, fmt, cols)
        self.assertEqual(ex.messages[0]["content"], "Fasse zusammen.\n\nLanger Text")
        self.assertEqual(ex.reference(), "Kurz")

    def test_sharegpt_rollen(self):
        ex = D.row_to_example({"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hallo"}]},
                              "messages", ("conversations",))
        self.assertEqual([m["role"] for m in ex.messages], ["user", "assistant"])

    def test_unbekanntes_format_nennt_den_ausweg(self):
        with self.assertRaisesRegex(ValueError, "prompt_column"):
            D.detect_format(["foo", "bar"])

    def test_override_spalten(self):
        self.assertEqual(
            D.detect_format(["frage_x", "antwort_y"], {"prompt_column": "frage_x", "response_column": "antwort_y"}),
            ("pair", ("frage_x", "antwort_y")))


class SplitTest(unittest.TestCase):
    def test_splits_aus_dateinamen_ohne_beipackzettel(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            row = {"prompt": "a", "completion": "b"}
            _jsonl(root / "train.jsonl", [row] * 12)
            _jsonl(root / "validation.jsonl", [row] * 3)
            _jsonl(root / "test.jsonl", [row] * 2)
            (root / "PROVENANCE.csv").write_text("x,y\n1,2\n", encoding="utf-8")
            data = D.load_examples(str(root))
            self.assertEqual((len(data.train), len(data.val), len(data.test)), (12, 3, 2))

    def test_val_wird_abgetrennt_im_test_nicht(self):
        with tempfile.TemporaryDirectory() as d:
            _jsonl(Path(d) / "daten.jsonl", [{"prompt": f"p{i}", "completion": "c"} for i in range(40)])
            data = D.load_examples(d)
            self.assertEqual((len(data.train), len(data.val)), (36, 4))
            self.assertEqual(len(D.load_examples(d, val_fraction=0).train), 40)

    def test_text_und_chat_gemischt_wird_abgelehnt(self):
        with tempfile.TemporaryDirectory() as d:
            _jsonl(Path(d) / "a.jsonl", [{"prompt": "a", "completion": "b"}])
            (Path(d) / "b.txt").write_text("Fliesstext", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mischt"):
                D.load_examples(d)

    def test_textdatei_wird_in_absaetze_gepackt(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "wissen.txt").write_text(
                "\n\n".join("Absatz %d " % i + "x" * 300 for i in range(20)), encoding="utf-8")
            data = D.load_examples(d)
            self.assertEqual(data.kind, "text")
            self.assertTrue(len(data.train) > 1 and all(e.text for e in data.train))


class TokenTest(unittest.TestCase):
    def test_loss_nur_auf_antworten(self):
        ex = D.Example(messages=[{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"},
                                 {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "B2"}])
        t = D.tokenize_example(_CharTok(), ex, 512)[0]
        self.assertEqual("".join(chr(i) for i, l in zip(t.input_ids, t.labels) if l != -100), "A1|B2|")
        self.assertTrue(t.masked_ok)
        self.assertEqual(chr(t.input_ids[t.first_target]), "A")

    def test_abgeschnittene_antwort_wird_verworfen(self):
        ex = D.Example(messages=[{"role": "user", "content": "x" * 50}, {"role": "assistant", "content": "y"}])
        self.assertEqual(D.tokenize_example(_CharTok(), ex, 20), [])

    def test_text_wird_in_fenster_geteilt(self):
        items = D.tokenize_example(_CharTok(), D.Example(text="a" * 25), 10)
        self.assertEqual([len(t.input_ids) for t in items], [10, 10, 6])


class ScoreTest(unittest.TestCase):
    def test_bewertung(self):
        s = D.score_generations(["Abteilung: KONTO", "falsch"], ["abteilung:  konto", "richtig"])
        self.assertEqual(s["exact_match"], 0.5)
        self.assertAlmostEqual(D.rouge_l("a b c d", "a b x d"), 0.75)


if __name__ == "__main__":
    unittest.main(verbosity=2)
