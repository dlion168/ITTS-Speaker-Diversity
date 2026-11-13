#!/usr/bin/env bash
# Create the generation environments as named conda environments.
#   bash envs/setup_generation_envs.sh [all|moss|parler|qwen|vox|voxcpm2|capspeech ...]
# VoxCPM2 and CapSpeech-NAR need one more step after this script; see envs/README.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALL=(moss parler qwen vox voxcpm2 capspeech)
if [[ $# -eq 0 || "$1" == all ]]; then
  targets=("${ALL[@]}")
else
  targets=("$@")
fi
for target in "${targets[@]}"; do
  case "$target" in moss|parler|qwen|vox|voxcpm2|capspeech) ;;
    *) echo "Unknown generation environment: $target (choose from: ${ALL[*]})" >&2; exit 2;; esac
done
command -v conda >/dev/null || { echo 'conda is required' >&2; exit 2; }
for target in "${targets[@]}"; do
  conda env create -f "$HERE/itts_${target}.yml"
done
