#!/bin/bash

# ============================================================
# FrameTrain Release Script
# Usage: ./release.sh <version> "<commit message>"
# Example: ./release.sh 1.0.34 "Fix: dataset loading for HF parquet files"
# ============================================================

set -e

# --- Argument check ---
if [ -z "$1" ] || [ -z "$2" ]; then
  echo ""
  echo "Usage:   ./release.sh <version> \"<commit message>\""
  echo "Example: ./release.sh 1.0.34 \"Fix: dataset loading for HF parquet files\""
  echo ""
  exit 1
fi

VERSION="$1"
COMMIT_MSG="$2"
TAG="v${VERSION}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "============================================"
echo "  FrameTrain Release: ${TAG}"
echo "  Commit: ${COMMIT_MSG}"
echo "============================================"
echo ""

# --- Update all version fields via Python (macOS-compatible) ---
echo "-> Updating versions to ${VERSION} in config files..."

python3 << PYEOF
import json, sys

def update_json(path, updater):
    with open(path, 'r') as f:
        data = json.load(f)
    updater(data)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    print(f"   Updated: {path}")

version = "${VERSION}"
base = "${SCRIPT_DIR}"

# package.json
update_json(f"{base}/package.json", lambda d: d.update({"version": version}))

# package-lock.json
def update_lock(d):
    d["version"] = version
    if "packages" in d and "" in d["packages"]:
        d["packages"][""]["version"] = version
update_json(f"{base}/package-lock.json", update_lock)

# tauri.conf.json
update_json(f"{base}/src-tauri/tauri.conf.json", lambda d: d.update({"version": version}))
PYEOF

# --- Stage all changes ---
echo "-> Staging all changes..."
git -C "${SCRIPT_DIR}" add .

# --- Commit (skip if nothing to commit) ---
echo "-> Committing: \"${COMMIT_MSG}\""
if git -C "${SCRIPT_DIR}" diff --cached --quiet; then
  echo "   (nothing to commit, skipping)"
else
  git -C "${SCRIPT_DIR}" commit -m "${COMMIT_MSG}"
fi

# --- Push to main ---
echo "-> Pushing to origin/main..."
git -C "${SCRIPT_DIR}" push origin main

# --- Delete old tag (local + remote, silently) ---
echo "-> Removing old tag ${TAG} if it exists..."
git -C "${SCRIPT_DIR}" tag -d "${TAG}" 2>/dev/null || true
git -C "${SCRIPT_DIR}" push origin ":refs/tags/${TAG}" 2>/dev/null || true

# --- Create new annotated tag ---
echo "-> Creating annotated tag ${TAG}..."
git -C "${SCRIPT_DIR}" tag -a "${TAG}" -m "Release ${TAG}"

# --- Push tag (triggers GitHub Actions) ---
echo "-> Pushing tag ${TAG} to origin..."
git -C "${SCRIPT_DIR}" push origin "${TAG}"

echo ""
echo "Done! Release ${TAG} pushed."
echo "GitHub Actions should now build and deploy the release."
echo ""
