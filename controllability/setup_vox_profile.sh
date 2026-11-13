#!/usr/bin/env bash
# Check out the Vox-Profile model code that score_controllability.py imports, at the commit we used.
# Usage: bash controllability/setup_vox_profile.sh [target_dir]   (default controllability/vox-profile-release)
set -euo pipefail
DEST="${1:-$(cd "$(dirname "$0")" && pwd)/vox-profile-release}"
COMMIT=85100e60844a3f324a139e24fb9225aa6d8e45d1
[ -d "$DEST/.git" ] || git clone https://github.com/tiantiaf0627/vox-profile-release.git "$DEST"
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$COMMIT"
echo "vox-profile-release at $(git -C "$DEST" rev-parse --short HEAD) in $DEST"
