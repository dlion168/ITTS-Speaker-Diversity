#!/usr/bin/env bash
# Paper §4.1: generate the audio that run.sh --from generated evaluates: 3 partitions x 10,000
# Emilia texts per system, neutral instruction.
#   bash experiments/02_overall_diversity/generate.sh [--dry-run] [--limit N] [--batch-size B]
# Writes $ITTS_DATA_ROOT/outputs/overall/<system>/sample{1,2,3}/. Options go to
# generation/run_overall.sh; SYSTEMS selects the systems. Needs the generation environments and
# model weights (generation/README.md). Settings: see experiments/common.sh.
source "$(dirname "$0")/../common.sh"
generate overall "$@"
