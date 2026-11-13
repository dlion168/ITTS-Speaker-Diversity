#!/usr/bin/env bash
# Paper §4.1: speaker diversity under a neutral instruction, compared with real speech.
#   bash experiments/02_overall_diversity/run.sh --from generated|audio|embeddings
# --from generated evaluates the audio of generate.sh. Writes results/overall/summary_diversity.csv.
# Settings: see experiments/common.sh.
#
# Released data (see README.md): overall audio and embeddings for all seven systems. Real-speech
# recordings are not redistributed, only their embeddings, so the audio paths also need REAL_AUDIO (common.sh).
source "$(dirname "$0")/../common.sh"
parse_from "$@"
need_python "$EMBED_ENV"
EMB="$DATA/embeddings_overall"
AUDIO="$DATA/outputs/overall"

if [ "$FROM" = embeddings ]; then
    "$PY" embeddings/download_released.py "${HF_ARGS[@]}" --configs overall real_emilia --out_root "$DATA"
else
    need_python "$NEMO_ENV"
    need_real_audio
    if [ "$FROM" = audio ]; then
        "$PY" embeddings/download_released_audio.py "${HF_ARGS[@]}" --systems $(hf_splits) --configs overall --out_root "$DATA"
    else
        dirs=()
        for sys in "${SYSTEMS[@]}"; do dirs+=("$AUDIO/$sys/sample"{1,2,3}); done
        need_generated experiments/02_overall_diversity/generate.sh "${dirs[@]}"
    fi
    for sys in "${SYSTEMS[@]}"; do
        for k in 1 2 3; do
            d="$AUDIO/$sys/sample$k"
            if [ -d "$d" ]; then embed_dir "$d" "$sys" "$EMB"; else echo "[skip] no audio in $d"; fi
        done
    done
    embed_dir "$REAL_AUDIO/set_1" real "$EMB"
fi

"$PY" experiments/02_overall_diversity/summarize.py --emb_root "$EMB" --systems "${SYSTEMS[@]}" \
    --out "$RESULTS_DIR/overall/summary_diversity.csv"
