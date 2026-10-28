#!/usr/bin/env python3
"""Generate latest.json for the Tauri updater from the assets of a release.

Usage (set env vars before calling):
  VERSION      e.g. 1.6.11 (required)
  PUB_DATE     ISO timestamp
  ASSETS_FILE  JSON array of release assets, as printed by
               `gh release view <tag> --json assets --jq .assets`
  SIGS_DIR     folder with the downloaded *.sig files (default ./sigs)
  OUTPUT_PATH  where to write latest.json (default latest.json)

A platform only goes into latest.json when its installer AND the matching
<installer>.sig are both uploaded to the release and the signature could be
read. Missing platforms are reported as warnings (one failed matrix job must
not block the update for the others). If no platform is complete, the script
exits with code 1 and writes no latest.json — otherwise every updater would
be offered download links that do not exist.
"""
import json
import os
import sys

# Updater key -> installer suffixes, in order of preference.
PLATFORMS = [
    ("darwin-aarch64", [".app.tar.gz"]),
    ("windows-x86_64", [".exe", ".msi"]),
    ("linux-x86_64",   [".AppImage"]),
]


def get(name, default=""):
    return os.environ.get(name, default).strip()


def is_uploaded(asset):
    # "state" is "open" while an upload is still running or was aborted.
    return asset.get("state", "uploaded") == "uploaded" and asset.get("size", 1) > 0


def read_sig(sigs_dir, sig_name):
    path = os.path.join(sigs_dir, sig_name)
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def select_platforms(assets, sigs_dir):
    """Return (platforms, problems): complete updater entries and why others are missing."""
    by_name = {a["name"]: a for a in assets if is_uploaded(a)}
    platforms, problems = {}, []

    for key, suffixes in PLATFORMS:
        reasons = []
        for suffix in suffixes:
            installers = sorted(n for n in by_name if n.lower().endswith(suffix.lower()))
            for name in installers:
                sig_name = name + ".sig"
                if sig_name not in by_name:
                    reasons.append(f"{name}: {sig_name} fehlt im Release")
                    continue
                sig = read_sig(sigs_dir, sig_name)
                if not sig:
                    reasons.append(f"{name}: {sig_name} nicht lesbar oder leer")
                    continue
                platforms[key] = {"url": by_name[name]["url"], "signature": sig}
                break
            if key in platforms:
                break
        if key not in platforms:
            problems.append(f"{key}: " + ("; ".join(reasons) if reasons else "kein Installer im Release"))

    return platforms, problems


def main():
    version = get("VERSION")
    if not version:
        print("ERROR: VERSION env var is required", file=sys.stderr)
        return 1

    assets_file = get("ASSETS_FILE")
    if not assets_file:
        print("ERROR: ASSETS_FILE env var is required", file=sys.stderr)
        return 1
    with open(assets_file) as f:
        assets = json.load(f) or []

    platforms, problems = select_platforms(assets, get("SIGS_DIR", "sigs"))

    for p in problems:
        print(f"::warning title=latest.json ohne Plattform::{p}")

    if not platforms:
        print(
            f"::error title=latest.json nicht erzeugt::Release v{version} hat fuer keine "
            "Plattform Installer und Signatur. Alle Builds sind gescheitert oder haben nichts "
            "hochgeladen. latest.json wird weder ins Releases-Repo noch an die Bruecke geladen, "
            "damit kein Updater ein Update ohne Installer angeboten bekommt."
        )
        return 1

    latest = {
        "version":  version,
        "notes":    f"FrameTrain Desktop App v{version}",
        "pub_date": get("PUB_DATE"),
        "platforms": platforms,
    }

    output_path = os.environ.get("OUTPUT_PATH", "latest.json")
    with open(output_path, "w") as f:
        json.dump(latest, f, indent=2)
        f.write("\n")

    print(f"✅ latest.json generated ({len(platforms)} von {len(PLATFORMS)} Plattformen):")
    print(json.dumps(latest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
