"""
ft_data.llm – Daten fuer das LLM-Fine-Tuning (causal_lm).

Gemeinsam fuer beide Backends (PyTorch/peft und MLX) und fuer das Test-Plugin,
damit Training und Test dieselbe Eingabe sehen:

  - Dateien finden und in Splits sortieren (train/val/test),
  - jede Zeile auf eine Form bringen: Chat (``messages``) oder reiner Text,
  - Chat-Template anwenden und den Loss auf die Antworten beschraenken.

Erkannte Formate je Zeile
  {"messages": [{"role": "user", "content": ...}, {"role": "assistant", ...}]}
  {"conversations": [{"from": "human", "value": ...}, {"from": "gpt", ...}]}  (ShareGPT)
  {"prompt": ..., "completion": ...}                (OpenAI / MLX)
  {"instruction": ..., "input": ..., "output": ...} (Alpaca)
  {"question": ..., "answer": ...} / {"query": ..., "response": ...} u. a.
  {"text": ...}                                     (weiteres Vortraining)
  .txt/.md-Dateien                                   (weiteres Vortraining)
  {"prompt": ..., "chosen": ..., "rejected": ...}   (Praeferenzen → DPO)
    chosen/rejected auch als Chat-Liste; der gemeinsame Anfang ist dann der Prompt.
"""
from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TABLE_EXTS = (".jsonl", ".json", ".csv", ".tsv", ".parquet")
TEXT_EXTS = (".txt", ".md")
DATA_EXTS = TABLE_EXTS + TEXT_EXTS

# Paare (Eingabe, Antwort) in der Reihenfolge, in der sie gesucht werden.
# Die erste Spalte darf fehlen, wenn `input` dabei ist (Alpaca).
PAIR_COLUMNS: Sequence[Tuple[str, str]] = (
    ("prompt", "completion"),
    ("instruction", "output"),
    ("instruction", "response"),
    ("question", "answer"),
    ("query", "response"),
    ("input", "output"),
    ("input", "target"),
    ("source", "target"),
    ("user", "assistant"),
    ("frage", "antwort"),
    ("eingabe", "ausgabe"),
)
TEXT_COLUMNS = ("text", "content", "document", "body")

ROLE_ALIASES = {
    "human": "user", "user": "user", "prompter": "user",
    "gpt": "assistant", "assistant": "assistant", "bot": "assistant", "model": "assistant",
    "system": "system",
}

# Fuer Basismodelle ohne eigenes Chat-Template. Wird beim Export in den
# Tokenizer geschrieben, damit Test, Labor und Ollama dasselbe Format nutzen,
# mit dem trainiert wurde.
FALLBACK_CHAT_TEMPLATE = (
    "{% if bos_token and bos_token != eos_token %}{{ bos_token }}{% endif %}"
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}### System:\n{{ m['content'] }}\n\n"
    "{% elif m['role'] == 'user' %}### User:\n{{ m['content'] }}\n\n"
    "{% elif m['role'] == 'assistant' %}### Assistant:\n{{ m['content'] }}{{ eos_token }}\n\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}### Assistant:\n{% endif %}"
)


@dataclass
class Example:
    """Eine Trainingszeile: entweder ein Chat oder reiner Text."""
    messages: Optional[List[Dict[str, str]]] = None
    text: Optional[str] = None
    # Nur bei Praeferenzdaten (DPO): die schlechtere Antwort. messages endet
    # dann mit der besseren (chosen).
    rejected: Optional[str] = None

    @property
    def is_chat(self) -> bool:
        return self.messages is not None

    def prompt_messages(self) -> List[Dict[str, str]]:
        """Alles bis vor die letzte Antwort — das, was im Test eingegeben wird."""
        msgs = self.messages or []
        last = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=len(msgs))
        return msgs[:last]

    def reference(self) -> Optional[str]:
        """Die letzte Antwort — die Referenz fuer Exact Match / ROUGE-L."""
        for m in reversed(self.messages or []):
            if m["role"] == "assistant":
                return m["content"]
        return None


@dataclass
class LoadedData:
    train: List[Example]
    val: List[Example]
    test: List[Example] = field(default_factory=list)
    kind: str = "chat"          # "chat" | "text" | "preference"
    source_format: str = ""     # fuer die Statusmeldung
    notes: List[str] = field(default_factory=list)


# ─── Dateien ──────────────────────────────────────────────────────────────────

def split_of(path: Path) -> Optional[str]:
    """train/val/test aus Ordner- oder Dateiname (train.jsonl, validation-000.parquet)."""
    for part in [path.stem.lower()] + [p.lower() for p in reversed(path.parts[:-1])]:
        token = re.split(r"[-_.]", part)[0]
        if token in ("train", "training"):
            return "train"
        if token in ("val", "valid", "validation", "dev", "eval"):
            return "val"
        if token in ("test", "testing"):
            return "test"
    return None


def find_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in DATA_EXTS else []
    files = [f for f in sorted(root.rglob("*"))
             if f.is_file() and f.suffix.lower() in DATA_EXTS
             and not any(p.startswith(".") for p in f.relative_to(root).parts)]
    # Beipackzettel der Werkstatt (PROVENANCE.csv, EXPORT_REPORT.md, README)
    # sind keine Trainingsdaten.
    skip = {"provenance", "export_report", "readme", "license", "metadata"}
    return [f for f in files if f.stem.lower() not in skip]


def _read_table(path: Path) -> List[Dict[str, Any]]:
    ext = path.suffix.lower()
    if ext == ".jsonl":
        rows = []
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path.name}, Zeile {n}: kein gueltiges JSON ({e.msg}).")
        return rows
    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            # {"data": [...]} oder {"train": [...]} — die erste Liste nehmen
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        return [r for r in data if isinstance(r, dict)]
    if ext in (".csv", ".tsv"):
        with open(path, encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh, delimiter="\t" if ext == ".tsv" else ","))
    if ext == ".parquet":
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Nicht unterstuetztes Format: {path.name}")


def _read_text_file(path: Path, chunk_chars: int = 2000) -> List[Dict[str, Any]]:
    """Fliesstext in Absatz-Pakete bis ~chunk_chars Zeichen."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    rows, buf = [], ""
    for p in paragraphs:
        if buf and len(buf) + len(p) > chunk_chars:
            rows.append({"text": buf})
            buf = ""
        buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        rows.append({"text": buf})
    return rows


# ─── Normalisierung ───────────────────────────────────────────────────────────

def _as_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v != v:   # NaN aus Pandas
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _norm_messages(raw: Any) -> Optional[List[Dict[str, str]]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, list):
        return None
    out = []
    for m in raw:
        if not isinstance(m, dict):
            return None
        role = ROLE_ALIASES.get(str(m.get("role", m.get("from", ""))).lower())
        content = m.get("content", m.get("value"))
        if role is None or content is None:
            continue
        if isinstance(content, list):
            # OpenAI-Content-Teile: nur Textteile zaehlen
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        out.append({"role": role, "content": str(content)})
    return out if any(m["role"] == "assistant" for m in out) else None


def detect_format(columns: Iterable[str], overrides: Optional[Dict[str, Any]] = None) -> Tuple[str, Tuple[str, ...]]:
    """Welche Spalten tragen Eingabe und Antwort?

    Rueckgabe: ("messages", (spalte,)) | ("pair", (prompt, antwort[, input])) | ("text", (spalte,))
    """
    cols = {c.lower(): c for c in columns}
    ov = overrides or {}
    p_col, r_col = ov.get("prompt_column"), ov.get("response_column")
    if p_col and r_col:
        missing = [c for c in (p_col, r_col) if c not in columns]
        if missing:
            raise ValueError(f"Spalte(n) {missing} aus plugin_config nicht gefunden. Vorhanden: {sorted(columns)}")
        return "pair", (p_col, r_col)
    if "chosen" in cols and "rejected" in cols:
        prompt = next((cols[k] for k in ("prompt", "question", "instruction", "query", "input") if k in cols), "")
        return "preference", (prompt, cols["chosen"], cols["rejected"])
    for key in ("messages", "conversations", "conversation", "chat"):
        if key in cols:
            return "messages", (cols[key],)
    for p, r in PAIR_COLUMNS:
        if r in cols and (p in cols or (p == "instruction" and "input" in cols)):
            extra = (cols["input"],) if p == "instruction" and "input" in cols else ()
            prompt = cols.get(p, cols.get("input"))
            if extra and prompt == extra[0]:
                extra = ()
            return "pair", (prompt, cols[r], *extra)
    for t in TEXT_COLUMNS:
        if t in cols:
            return "text", (cols[t],)
    raise ValueError(
        "Konnte das Datenformat nicht erkennen. Gefundene Spalten: "
        f"{sorted(columns)}.\nErwartet z. B. 'messages' (Chat), 'prompt'/'completion', "
        "'instruction'/'output' (Alpaca), 'question'/'answer' oder 'text'.\n"
        "Andere Namen lassen sich im Plugin-Parameter prompt_column / response_column setzen."
    )


def _split_preference(prompt_raw: Any, chosen_raw: Any, rejected_raw: Any):
    """(prompt_messages, chosen_text, rejected_text) aus Text- oder Chat-Form."""
    ch, rj = _norm_messages(chosen_raw), _norm_messages(rejected_raw)
    if ch and rj:
        # Chat-Form (z. B. Anthropic HH): alles vor der letzten Antwort ist der Prompt.
        ctx = ch[:-1] if ch[-1]["role"] == "assistant" else ch
        return ctx, ch[-1]["content"], rj[-1]["content"]
    prompt_msgs = _norm_messages(prompt_raw) if isinstance(prompt_raw, (list, str)) else None
    if not prompt_msgs:
        text = _as_text(prompt_raw).strip()
        prompt_msgs = [{"role": "user", "content": text}] if text else []
    return prompt_msgs, _as_text(chosen_raw).strip(), _as_text(rejected_raw).strip()


def row_to_example(row: Dict[str, Any], fmt: str, cols: Tuple[str, ...],
                   system_prompt: str = "") -> Optional[Example]:
    if fmt == "preference":
        ctx, chosen, rejected = _split_preference(row.get(cols[0]) if cols[0] else None,
                                                  row.get(cols[1]), row.get(cols[2]))
        if not ctx or not chosen or not rejected or chosen == rejected:
            return None
        if system_prompt and ctx[0]["role"] != "system":
            ctx = [{"role": "system", "content": system_prompt}] + ctx
        return Example(messages=ctx + [{"role": "assistant", "content": chosen}], rejected=rejected)
    if fmt == "messages":
        msgs = _norm_messages(row.get(cols[0]))
        if not msgs:
            return None
        if system_prompt and msgs[0]["role"] != "system":
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        return Example(messages=msgs)
    if fmt == "pair":
        prompt = _as_text(row.get(cols[0])).strip()
        answer = _as_text(row.get(cols[1])).strip()
        if len(cols) > 2:
            extra = _as_text(row.get(cols[2])).strip()
            if extra:
                prompt = f"{prompt}\n\n{extra}" if prompt else extra
        if not prompt or not answer:
            return None
        msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
        if system_prompt:
            msgs.insert(0, {"role": "system", "content": system_prompt})
        return Example(messages=msgs)
    text = _as_text(row.get(cols[0])).strip()
    return Example(text=text) if text else None


def load_examples(dataset_path: str, plugin_config: Optional[Dict[str, Any]] = None,
                  seed: int = 42, val_fraction: float = 0.1) -> LoadedData:
    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset nicht gefunden: {root}")
    files = find_files(root)
    if not files:
        raise ValueError(
            f"Keine Datendateien in '{root}' gefunden (erwartet: {', '.join(DATA_EXTS)})."
        )
    pc = plugin_config or {}
    system_prompt = str(pc.get("system_prompt") or "")
    by_split: Dict[str, List[Example]] = {"train": [], "val": [], "test": []}
    formats, notes, skipped = set(), [], 0

    for f in files:
        rows = _read_text_file(f) if f.suffix.lower() in TEXT_EXTS else _read_table(f)
        if not rows:
            continue
        fmt, cols = detect_format(rows[0].keys(), pc) if f.suffix.lower() not in TEXT_EXTS else ("text", ("text",))
        formats.add(fmt)
        split = split_of(f.relative_to(root) if root.is_dir() else Path(f.name)) or "train"
        for row in rows:
            ex = row_to_example(row, fmt, cols, system_prompt)
            if ex is None:
                skipped += 1
                continue
            by_split[split].append(ex)

    if len(formats & {"text", "preference"}) and len(formats) > 1:
        raise ValueError(
            "Das Dataset mischt verschiedene Arten (Chat/Frage-Antwort, Praeferenzpaare, reiner Text). "
            "Bitte eine Art pro Dataset verwenden."
        )
    if skipped:
        notes.append(f"{skipped} Zeilen ohne Eingabe oder Antwort uebersprungen.")

    train = by_split["train"] or by_split["test"]
    if not train:
        raise ValueError("Keine verwertbare Zeile gefunden — alle Zeilen sind leer.")
    val = by_split["val"]
    test = by_split["test"] if by_split["train"] else []
    if not val and val_fraction > 0:
        rng = random.Random(seed)
        idx = list(range(len(train)))
        rng.shuffle(idx)
        n_val = max(1, min(200, int(round(len(train) * val_fraction)))) if len(train) >= 10 else 0
        if n_val:
            val_idx = set(idx[:n_val])
            val = [train[i] for i in sorted(val_idx)]
            train = [train[i] for i in range(len(train)) if i not in val_idx]
            notes.append(f"Kein Validierungs-Split gefunden — {n_val} Beispiele abgetrennt.")
        else:
            # Bei einer Handvoll Beispielen lieber auf den Trainingsdaten messen
            # als gar nicht — die Zahl ist dann nur ein Lernnachweis.
            val = list(train)
            notes.append("Sehr kleines Dataset — Validierung auf den Trainingsdaten (nur Lernnachweis).")

    kind = "text" if formats == {"text"} else "preference" if formats == {"preference"} else "chat"
    return LoadedData(train=train, val=val, test=test, kind=kind,
                      source_format="/".join(sorted(formats)), notes=notes)


# ─── Chat-Template & Tokenisierung ────────────────────────────────────────────

def ensure_chat_template(tokenizer) -> str:
    """Setzt das Ersatz-Template, wenn das Modell keins mitbringt.

    Rueckgabe: "model" (eigenes Template) oder "fallback".
    """
    if getattr(tokenizer, "chat_template", None):
        return "model"
    tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
    return "fallback"


def _merge_system(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Fuer Templates ohne System-Rolle (Gemma): System vor die erste Nutzerfrage."""
    if not messages or messages[0]["role"] != "system":
        return messages
    sys_text, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
    for m in rest:
        if m["role"] == "user":
            m["content"] = f"{sys_text}\n\n{m['content']}"
            break
    return rest


def render_chat(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=add_generation_prompt)
    except Exception as exc:
        if messages and messages[0]["role"] == "system" and "system" in str(exc).lower():
            return tokenizer.apply_chat_template(_merge_system(messages), tokenize=False,
                                                 add_generation_prompt=add_generation_prompt)
        raise


def _ids(tokenizer, text: str) -> List[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


@dataclass
class Tokenized:
    input_ids: List[int]
    labels: List[int]          # -100 = kein Loss
    first_target: int          # erste Position mit Loss (fuer MLX: ein Offset)
    masked_ok: bool = True     # False = Template nicht praefix-stabil, ganzer Text trainiert


def tokenize_example(tokenizer, ex: Example, max_len: int) -> List[Tokenized]:
    """Tokenisiert eine Zeile. Chat: Loss nur auf den Antworten.

    Reiner Text wird in Fenster der Laenge max_len geteilt (mehrere Eintraege),
    Chats werden hinten gekappt — die Antwort steht dort am Ende.
    """
    eos = tokenizer.eos_token_id
    if not ex.is_chat:
        ids = _ids(tokenizer, ex.text or "")
        if eos is not None and (not ids or ids[-1] != eos):
            ids.append(eos)
        out = []
        for start in range(0, len(ids), max_len):
            window = ids[start:start + max_len]
            if len(window) >= 2:
                out.append(Tokenized(window, list(window), 0))
        return out

    msgs = ex.messages or []
    full_text = render_chat(tokenizer, msgs, add_generation_prompt=False)
    # Antwortbereiche als Zeichenpositionen. Auf Token-Ebene waere der Vergleich
    # nicht stabil: BPE verschmilzt "\n" am Ende des Prompts mit dem ersten
    # Antwortwort ("\nHi"), der Prompt allein tokenisiert dann anders.
    spans: List[Tuple[int, int]] = []
    ok = True
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prefix = render_chat(tokenizer, msgs[:i], add_generation_prompt=True) if i else ""
        upto = render_chat(tokenizer, msgs[:i + 1], add_generation_prompt=False)
        if not (full_text.startswith(prefix) and full_text.startswith(upto)) or len(upto) <= len(prefix):
            ok = False
            break
        spans.append((len(prefix), len(upto)))

    try:
        enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
    except Exception:
        # Langsame Tokenizer kennen kein offset_mapping — dann ganz trainieren.
        enc, offsets, ok = tokenizer(full_text, add_special_tokens=False), None, False
    full = list(enc["input_ids"])
    if ok and offsets is not None:
        # Ein Token zaehlt zur Antwort, sobald es in einen Antwortbereich hineinragt.
        labels = [tid if any(s < e and ts < e and te > s for s, e in spans) else -100
                  for tid, (ts, te) in zip(full, offsets)]
    else:
        labels = list(full)
    first = next((i for i, l in enumerate(labels) if l != -100), len(full))
    if len(full) > max_len:
        full, labels = full[:max_len], labels[:max_len]
    if all(l == -100 for l in labels):
        return []   # Antwort komplett abgeschnitten — nichts zu lernen
    return [Tokenized(full, labels, min(first, len(full)), ok)]


@dataclass
class PreferencePair:
    chosen: Tokenized
    rejected: Tokenized


def tokenize_preferences(tokenizer, examples: List[Example], max_len: int) -> Tuple[List[PreferencePair], Dict[str, int]]:
    """Fuer DPO: beide Antworten mit demselben Prompt, Loss-Maske nur auf der Antwort."""
    out: List[PreferencePair] = []
    stats = {"examples": len(examples), "dropped": 0, "unmasked": 0, "truncated": 0}
    for ex in examples:
        prompt = ex.prompt_messages()
        good = tokenize_example(tokenizer, Example(messages=ex.messages), max_len)
        bad = tokenize_example(tokenizer, Example(messages=prompt + [{"role": "assistant", "content": ex.rejected or ""}]), max_len)
        if not good or not bad:
            stats["dropped"] += 1
            continue
        if not (good[0].masked_ok and bad[0].masked_ok):
            stats["unmasked"] += 1
        out.append(PreferencePair(good[0], bad[0]))
    return out, stats


def tokenize_all(tokenizer, examples: List[Example], max_len: int) -> Tuple[List[Tokenized], Dict[str, int]]:
    out: List[Tokenized] = []
    stats = {"examples": len(examples), "dropped": 0, "unmasked": 0, "truncated": 0}
    for ex in examples:
        items = tokenize_example(tokenizer, ex, max_len)
        if not items:
            stats["dropped"] += 1
        for t in items:
            if not t.masked_ok:
                stats["unmasked"] += 1
            if len(t.input_ids) >= max_len and ex.is_chat:
                stats["truncated"] += 1
        out.extend(items)
    return out, stats


# ─── Bewertung erzeugter Antworten ────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def exact_match(pred: str, ref: str) -> bool:
    return _norm(pred) == _norm(ref)


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F1 ueber Woerter (LCS) — ohne Zusatzpaket."""
    a, b = _norm(pred).split(), _norm(ref).split()
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def score_generations(preds: List[str], refs: List[str]) -> Dict[str, float]:
    if not refs:
        return {}
    n = len(refs)
    return {
        "exact_match": sum(exact_match(p, r) for p, r in zip(preds, refs)) / n,
        "rougeL": sum(rouge_l(p, r) for p, r in zip(preds, refs)) / n,
    }
