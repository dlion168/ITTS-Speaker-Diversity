#!/usr/bin/env bash
# Paper §4.2: diversity across sampling temperature (MOSS-VoiceGenerator) and CFG scale
# (VoxInstruct, VoxCPM2), 5 values x 3 seeds x 10,000 utterances, plus the real-speech baseline.
#   bash experiments/03_sampling_sweep/run.sh --from generated|audio|embeddings
# --from generated evaluates the audio of generate.sh. Writes results/sampling_sweep/param_sweep_seeded_diversity.csv and results/overall/real_vendi_n3000.csv.
source "$(dirname "$0")/../common.sh"
parse_from "$@"
need_python "$EMBED_ENV"
EMB="$DATA/embeddings_param_sweep"
REAL="$DATA/embeddings_overall/real"
sweep_systems=()   # the selected SYSTEMS that are in the sweep
for s in "${SYSTEMS[@]}"; do
    case "$s" in
        moss-voicegen|voxcpm2|voxinstruct) sweep_systems+=("$s") ;;
        parler-mini|parler-large|capspeech|qwen3-voicedesign) ;;
        *) echo "unknown paper system: $s" >&2; exit 2 ;;
    esac
done
[ "${#sweep_systems[@]}" -gt 0 ] || { echo "SYSTEMS has none of moss-voicegen voxcpm2 voxinstruct" >&2; exit 2; }

if [ "$FROM" = embeddings ]; then
    "$PY" embeddings/download_released.py "${HF_ARGS[@]}" --configs sampling_sweep real_emilia --out_root "$DATA"
else
    need_python "$NEMO_ENV"
    if [ "$FROM" = audio ]; then
        sweep=()
        for s in $(hf_splits); do case "$s" in moss_voicegen|voxcpm2|voxinstruct) sweep+=("$s") ;; esac; done
        "$PY" embeddings/download_released_audio.py "${HF_ARGS[@]}" --configs sampling_sweep \
            --systems "${sweep[@]}" --out_root "$DATA"
    fi
    tasks=()
    dirs=()
    for s in "${sweep_systems[@]}"; do
        system_tasks=0
        for d in "$DATA/outputs/param_sweep/$s"/*/seed_*; do
            if [ -d "$d" ]; then
                dirs+=("$d")
                tasks+=(--task "${d#"$DATA/outputs/param_sweep/"}")
                system_tasks=$((system_tasks + 1))
            fi
        done
        [ "$system_tasks" -gt 0 ] || { echo "no sweep seed directories for selected system $s under $DATA/outputs/param_sweep" >&2; exit 2; }
    done
    # WAVs are in seed leaves, not in the model parent directory. Validate
    # every selected leaf before extracting any embeddings.
    need_generated experiments/03_sampling_sweep/generate.sh "${dirs[@]}"
    PARAM_SWEEP_EMBEDDINGS_ROOT="$EMB" bash embeddings/run_embedding_param_sweep.sh --worker-id run "${tasks[@]}"
    if [ ! -d "$REAL/set_1" ]; then
        need_real_audio
        embed_dir "$REAL_AUDIO/set_1" real "$DATA/embeddings_overall"
    fi
fi

"$PY" experiments/03_sampling_sweep/summarize.py --sweep_root "$EMB" --real_root "$REAL" --out_dir "$RESULTS_DIR" \
    --systems "${sweep_systems[@]}"
