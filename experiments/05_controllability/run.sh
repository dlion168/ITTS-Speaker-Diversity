#!/usr/bin/env bash
# Paper §3.5, §5, Appendices B to D: does each utterance have the requested attribute?
# Scores all 7 systems x 5 categories with the four classifiers and writes
# results/controllability/summary.json (recall and confusion per system and category).
#   bash experiments/05_controllability/run.sh --from generated|audio
# (Needs audio; there is no --from embeddings.) --from generated scores the audio of
# experiments/04_attribute_diversity/generate.sh. Accent scoring loads Whisper-Large-V3: ~12 GB per
# worker, 10 workers by default (NPROC; see controllability/run_scoring.sh).
source "$(dirname "$0")/../common.sh"
parse_from "$@"
need_python "$VOX_ENV"
if [ "$FROM" = embeddings ]; then
    echo "controllability scores audio; use --from audio or --from generated" >&2; exit 2
elif [ "$FROM" = audio ]; then
    need_python "$EMBED_ENV"
    "$PY" embeddings/download_released_audio.py "${HF_ARGS[@]}" --systems $(hf_splits) --configs attribute gender_x_age --out_root "$DATA"
else
    dirs=()
    for sys in "${SYSTEMS[@]}"; do
        for category in gender age accent emotion gender_x_age; do
            dirs+=("$DATA/outputs/attributes/$sys/$category")
        done
    done
    need_generated experiments/04_attribute_diversity/generate.sh "${dirs[@]}"
fi

[ -d controllability/vox-profile-release ] || [ -n "${VOX_PROFILE_DIR:-}" ] || bash controllability/setup_vox_profile.sh
"$VOX_ENV/bin/python" controllability/download_checkpoints.py
"$VOX_ENV/bin/python" controllability/make_manifests.py --root "$DATA/outputs/attributes" --models "${SYSTEMS[@]}"
PY="$VOX_ENV/bin/python" MODELS="${SYSTEMS[*]}" WAV_ROOT="$DATA/outputs/attributes" SHARD_DIR="$DATA/controllability" \
    SUMMARY="$RESULTS_DIR/controllability/summary.json" bash controllability/run_scoring.sh
