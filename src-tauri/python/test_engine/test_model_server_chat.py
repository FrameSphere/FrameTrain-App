"""Chat-Verlauf fuer das Hosting: build_chat_messages im Modell-Server."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_server import build_chat_messages  # noqa: E402


def test_einzelne_frage_mit_systemprompt():
    assert build_chat_messages("Sei knapp.", "Hallo", None) == [
        {"role": "system", "content": "Sei knapp."},
        {"role": "user", "content": "Hallo"},
    ]


def test_verlauf_bleibt_in_reihenfolge():
    hist = [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]
    out = build_chat_messages("", "C", hist)
    assert [m["content"] for m in out] == ["A", "B", "C"]


def test_frage_wird_nicht_doppelt_angehaengt():
    hist = [{"role": "user", "content": "A"}]
    assert build_chat_messages("", "A", hist) == [{"role": "user", "content": "A"}]


def test_eigener_systemprompt_im_verlauf_hat_vorrang():
    hist = [{"role": "system", "content": "eigen"}]
    out = build_chat_messages("training", "x", hist)
    assert [m["content"] for m in out] == ["eigen", "x"]


def test_unbrauchbare_eintraege_fallen_weg():
    hist = [{"role": "tool", "content": "x"}, {"role": "user", "content": ""}, "kaputt",
            {"role": "user", "content": ["bild"]}]
    assert build_chat_messages("", "ok", hist) == [{"role": "user", "content": "ok"}]
