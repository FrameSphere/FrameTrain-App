#!/usr/bin/env python3
"""Generate latest.json for Tauri updater from environment variables.

Usage (set env vars before calling):
  VERSION, PUB_DATE, WIN_URL, MAC_ARM_URL, LINUX_URL,
  WIN_SIG, MAC_ARM_SIG, LINUX_SIG
"""
import json
import os
import sys

def get(name, default=""):
    return os.environ.get(name, default).strip()

version    = get("VERSION")
pub_date   = get("PUB_DATE")
notes      = f"FrameTrain Desktop App v{version}"

win_url    = get("WIN_URL")
mac_url    = get("MAC_ARM_URL")
linux_url  = get("LINUX_URL")
win_sig    = get("WIN_SIG")
mac_sig    = get("MAC_ARM_SIG")
linux_sig  = get("LINUX_SIG")

if not version:
    print("ERROR: VERSION env var is required", file=sys.stderr)
    sys.exit(1)

def make_entry(url, sig):
    entry = {"url": url}
    if sig:
        entry["signature"] = sig
    return entry

platforms = {}
if win_url:
    platforms["windows-x86_64"] = make_entry(win_url, win_sig)
if mac_url:
    platforms["darwin-aarch64"] = make_entry(mac_url, mac_sig)
if linux_url:
    platforms["linux-x86_64"]   = make_entry(linux_url, linux_sig)

if not platforms:
    print("WARNING: No platform URLs found — latest.json will have empty platforms", file=sys.stderr)

latest = {
    "version":  version,
    "notes":    notes,
    "pub_date": pub_date,
    "platforms": platforms,
}

output_path = os.environ.get("OUTPUT_PATH", "latest.json")
with open(output_path, "w") as f:
    json.dump(latest, f, indent=2)
    f.write("\n")

print("✅ latest.json generated:")
print(json.dumps(latest, indent=2))
