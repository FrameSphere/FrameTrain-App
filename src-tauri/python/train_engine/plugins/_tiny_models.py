"""Winzige, zufaellig initialisierte Modelle fuer Plugin-Tests (offline, Sekunden).

Kein Plugin (keine manifest.json) — nur ein Helfer fuer test_*.py.
"""
import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

_TRAIN = Path(__file__).resolve().parents[1]          # train_engine
_PY = _TRAIN.parent                                    # src-tauri/python
for p in (str(_TRAIN), str(_PY)):
    if p not in sys.path:
        sys.path.insert(0, p)

WORDS = ("wo wohnt arbeitet lebt in berlin paris rom wien oslo prag anna ben clara david eva felix "
         "die stadt ist gross klein schoen wer was wie alt the a an of cats dogs rain tea coffee "
         "tell me about facts passage explains this is and are interesting info on all more").split()


def bert_dir(path: Path, hidden: int = 32, layers: int = 2) -> Path:
    """Mini-BERT mit eigenem WordPiece-Vokabular."""
    from transformers import BertConfig, BertModel, BertTokenizerFast
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", ",", "?", "!"] + sorted(set(WORDS))
    vocab += [f"##{c}" for c in "abcdefghijklmnopqrstuvwxyz"] + list("abcdefghijklmnopqrstuvwxyz0123456789")
    (path / "vocab.txt").write_text("\n".join(dict.fromkeys(vocab)) + "\n", encoding="utf-8")
    # transformers 5 liest vocab_file im Konstruktor nicht mehr — Tokenizer
    # ueber `tokenizers` bauen und als tokenizer.json speichern.
    from tokenizers import BertWordPieceTokenizer
    wp = BertWordPieceTokenizer(str(path / "vocab.txt"), lowercase=True)
    tok = BertTokenizerFast(tokenizer_object=wp._tokenizer, unk_token="[UNK]", sep_token="[SEP]",
                            pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]", do_lower_case=True)
    tok.model_max_length = 128
    tok.save_pretrained(str(path))
    cfg = BertConfig(vocab_size=len(tok), hidden_size=hidden, num_hidden_layers=layers,
                     num_attention_heads=2, intermediate_size=hidden * 2, max_position_embeddings=128)
    BertModel(cfg).save_pretrained(str(path))
    return path


def gpt2_dir(path: Path, n_embd: int = 32) -> Path:
    """Mini-GPT-2 mit Zeichen-Vokabular (Byte-Level-BPE aus wenigen Saetzen)."""
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator([" ".join(WORDS)] * 4, vocab_size=300, min_frequency=1,
                            special_tokens=["<|endoftext|>"])
    bpe.save_model(str(path))
    tok = GPT2TokenizerFast(tokenizer_object=bpe._tokenizer, bos_token="<|endoftext|>",
                            eos_token="<|endoftext|>", unk_token="<|endoftext|>")
    tok.save_pretrained(str(path))
    cfg = GPT2Config(vocab_size=len(tok), n_embd=n_embd, n_layer=2, n_head=2, n_positions=128,
                     bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    GPT2LMHeadModel(cfg).save_pretrained(str(path))
    return path


def use_train_core() -> None:
    """Train- und Test-Engine haben je ein Paket `core`. Laufen Tests beider in
    einem Prozess, muss vor einem Train-Plugin wieder das der Train-Engine gelten."""
    for k in [k for k in sys.modules if k == "core" or k.startswith("core.")]:
        mod_file = getattr(sys.modules[k], "__file__", "") or ""
        if str(_TRAIN) not in mod_file:
            sys.modules.pop(k)
    while str(_TRAIN) in sys.path:
        sys.path.remove(str(_TRAIN))
    sys.path.insert(0, str(_TRAIN))


def load_plugin(engine: str, name: str):
    """plugins/<name>/plugin.py als eigenes Modul (alle heissen plugin.py)."""
    root = _TRAIN if engine == "train" else _PY / "test_engine"
    if engine == "train":
        use_train_core()
    if engine == "test":
        for p in (str(root),):
            if p not in sys.path:
                sys.path.insert(0, p)
    f = root / "plugins" / name / "plugin.py"
    spec = importlib.util.spec_from_file_location(f"_ft_{engine}_{name}", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_training(name: str, cfg: dict):
    """setup -> export wie der Orchestrator; Rueckgabe (metrics, export_pfad, Meldungen)."""
    mod = load_plugin("train", name)
    from core.config import TrainingConfig
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        plugin = mod.Plugin(TrainingConfig.from_dict(cfg))
        for step in ("setup", "load_data", "build_model", "train"):
            if getattr(plugin, step)() is False:
                raise RuntimeError(f"{step} lieferte False:\n{out.getvalue()[-2000:]}")
        metrics = plugin.validate()
        path = plugin.export()
    msgs = [json.loads(l) for l in out.getvalue().splitlines() if l.startswith("{\"type\"")]
    return metrics, path, msgs


def run_test(name: str, cfg: dict):
    """Test-Plugin setup + run_*; Rueckgabe (complete-Daten, Meldungen)."""
    test_core = str(_PY / "test_engine")
    # core der Test-Engine hat Vorrang (TestConfig/TestProtocol)
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "core" or k.startswith("core.")}
    sys.path.insert(0, test_core)
    try:
        from core.config import TestConfig
        mod = load_plugin("test", name)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            plugin = mod.Plugin(TestConfig.from_dict(cfg))
            plugin.setup()
            (plugin.run_single if cfg.get("mode") == "single" else plugin.run_dataset)()
    finally:
        sys.path.remove(test_core)
        for k in [k for k in sys.modules if k == "core" or k.startswith("core.")]:
            sys.modules.pop(k)
        sys.modules.update(saved)
    msgs = [json.loads(l) for l in out.getvalue().splitlines() if l.startswith("{\"type\"")]
    done = [m["data"] for m in msgs if m["type"] == "complete"]
    return (done[-1] if done else None), msgs


def base_cfg(model: Path, data: Path, out: Path, task: str, **kw) -> dict:
    cfg = {"model_path": str(model), "dataset_path": str(data), "output_path": str(out),
           "task_type": task, "epochs": 2, "batch_size": 4, "learning_rate": 1e-3,
           "max_seq_length": 64, "eval_strategy": "epoch", "logging_steps": 1,
           "warmup_ratio": 0.0, "save_steps": 1000, "plugin_config": {}}
    cfg.update(kw)
    return cfg
