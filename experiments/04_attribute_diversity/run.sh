#!/usr/bin/env bash
# Paper §5 and Appendix E: within-value Vendi and MPCD with 95% half-sampling confidence intervals
# for 7 systems x 24 attribute values (gender 2, age 4, accent 6, emotion 4, gender x age 8).
#   bash experiments/04_attribute_diversity/run.sh --from generated|audio|embeddings
# --from generated evaluates the audio of generate.sh. Writes results/attributes/{vendi,mpcd}_ci{,_by_encoder}.csv (+ the draws as .npz).
# The CI step needs a GPU (10,000 subsets per value).
source "$(dirname "$0")/../common.sh"
parse_from "$@"
need_python "$EMBED_ENV"
EMB="$DATA/embeddings_attributes"

if [ "$FROM" = embeddings ]; then
    "$PY" embeddings/download_released.py "${HF_ARGS[@]}" --configs attribute gender_x_age --out_root "$DATA"
else
    need_python "$NEMO_ENV"
    if [ "$FROM" = audio ]; then
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
    tasks=()
    for sys in "${SYSTEMS[@]}"; do tasks+=(--task "$sys/all"); done
    EMBEDDINGS_OUTPUT_ROOT="$EMB" bash embeddings/run_embedding_subset.sh --worker-id run "${tasks[@]}"
fi

"$PY" metrics/half_sampling_ci.py --emb_root "$EMB" --systems "${SYSTEMS[@]}" --out_dir "$RESULTS_DIR/attributes"
