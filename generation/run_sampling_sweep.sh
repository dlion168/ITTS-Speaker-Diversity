#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${GENERATION_DRIVER_PYTHON:-python}" "$HERE/run_generation.py" sampling_sweep "$@"
