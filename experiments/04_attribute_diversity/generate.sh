#!/usr/bin/env bash
# Paper §5: generate the attribute audio that run.sh --from generated evaluates (also used by
# 05_controllability): 3,000 utterances per single-attribute value and 6,000 per gender x age cell.
#   bash experiments/04_attribute_diversity/generate.sh [--dry-run] [--limit N] [--batch-size B]
# Writes $ITTS_DATA_ROOT/outputs/attributes/<system>/<category>/. Options go to
# generation/run_attributes.sh; SYSTEMS selects the systems. Needs the generation environments and
# model weights (generation/README.md).
source "$(dirname "$0")/../common.sh"
generate attributes "$@"
