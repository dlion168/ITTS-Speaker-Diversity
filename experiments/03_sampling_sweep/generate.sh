#!/usr/bin/env bash
# Paper §4.2: generate the audio that run.sh --from generated evaluates: partition 1's 10,000 texts
# for 5 values x 3 seeds (17, 42, 99) of temperature (MOSS-VoiceGenerator) or CFG scale
# (VoxInstruct, VoxCPM2).
#   bash experiments/03_sampling_sweep/generate.sh [--dry-run] [--limit N] [--batch-size B]
# Writes $ITTS_DATA_ROOT/outputs/param_sweep/<system>/<temp|cfg>_<value>/seed_<s>/. Options go to
# generation/run_sampling_sweep.sh; SYSTEMS selects the systems (only those three are swept).
# Needs the generation environments and model weights (generation/README.md).
source "$(dirname "$0")/../common.sh"
generate sampling_sweep "$@"
