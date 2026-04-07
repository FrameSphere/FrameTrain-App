#!/bin/bash
# Script to fix version selection in Training, Analysis, and Test panels
# Automatically selects the newest version instead of the original/root version

echo "🔧 Fixing version selection to always select newest version..."

cd "$(dirname "$0")"

# The fix: When versions are loaded, sort by version_number and select the highest
# This needs to be applied in 3 files:
# - TrainingPanel.tsx
# - AnalysisPanel.tsx  
# - TestPanel.tsx

# Pattern to find: Where versions are loaded and selectedVersion is set
# We need to sort versions by version_number (descending) and select first

echo "⚠️  Manual changes required in:"
echo "  1. src/components/TrainingPanel.tsx"
echo "  2. src/components/AnalysisPanel.tsx"
echo "  3. src/components/TestPanel.tsx"
echo ""
echo "Find where 'selectedVersion' is set after loading versions."
echo "Change from:"
echo "  const rootVersion = versions.find(v => v.is_root);"
echo "  setSelectedVersion(rootVersion?.id || '');"
echo ""
echo "To:"
echo "  // Select newest version (highest version_number)"
echo "  const sortedVersions = [...versions].sort((a, b) => b.version_number - a.version_number);"
echo "  const newestVersion = sortedVersions[0];"
echo "  setSelectedVersion(newestVersion?.id || '');"
echo ""
echo "This ensures the NEWEST version is always pre-selected."
