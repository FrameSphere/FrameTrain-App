#!/usr/bin/env python3
"""Lädt die Skripte aus manifest.json in die Open Library (frame-train.com) hoch.

Aufruf (aus diesem Ordner):
    FRAMETRAIN_API_KEY=<dein API-Key> python3 upload.py --author "<Community-Name>"
    python3 upload.py --author "<Community-Name>" --dry-run     # zeigt nur, was gesendet würde

Der API-Key steht in der FrameTrain-App unter Einstellungen. Er wird nur aus der
Umgebungsvariable gelesen und nirgends gespeichert. Hochgeladene Skripte sind
ungeprüft, bis sie im Manager freigegeben werden. Ein Skript, dessen Name unter
diesem Autor schon in der Bibliothek steht, wird übersprungen.
"""
import argparse, json, os, sys, urllib.error, urllib.request
from pathlib import Path

API = "https://frame-train.com/api/library/scripts"
HERE = Path(__file__).resolve().parent


def request(method, url, key=None, payload=None):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, method=method, headers=headers,
                                 data=json.dumps(payload).encode() if payload is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            return res.status, res.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def existing(script_type, key):
    status, body = request("GET", f"{API}?script_type={script_type}", key)
    if status != 200:
        return []
    data = json.loads(body)
    return data if isinstance(data, list) else data.get("scripts", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", required=True, help="Community-Name, unter dem die Skripte erscheinen")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="nur Skripte, deren Dateiname diesen Text enthält")
    args = ap.parse_args()
    key = os.environ.get("FRAMETRAIN_API_KEY", "").strip()
    if not key and not args.dry_run:
        sys.exit("FRAMETRAIN_API_KEY ist nicht gesetzt.")
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    known = {}
    failed = 0
    for entry in manifest:
        if args.only and args.only not in entry["file"]:
            continue
        payload = {k: entry[k] for k in ("name", "description", "model_type", "task_type", "framework", "script_type", "tags")}
        payload.update(author=args.author.strip(), verified=False,
                       script=(HERE / entry["file"]).read_text(encoding="utf-8"))
        label = f"[{entry['script_type']}] {entry['name']}"
        if args.dry_run:
            print(f"würde senden: {label} ({len(payload['script'])} Zeichen, {payload['model_type']} / {payload['task_type']} / {payload['framework']})")
            continue
        st = entry["script_type"]
        if st not in known:
            known[st] = {(s.get("name"), s.get("author")) for s in existing(st, key)}
        if (payload["name"], payload["author"]) in known[st]:
            print(f"übersprungen (schon vorhanden): {label}")
            continue
        status, body = request("POST", API, key, payload)
        if 200 <= status < 300:
            print(f"hochgeladen: {label}")
        else:
            failed += 1
            print(f"FEHLER {status}: {label}\n   {body[:300]}")
    if failed:
        sys.exit(f"{failed} Skript(e) nicht hochgeladen.")


if __name__ == "__main__":
    main()
