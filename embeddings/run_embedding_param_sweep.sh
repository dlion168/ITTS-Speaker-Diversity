#!/usr/bin/env bash
# Extract speaker embeddings for the sampling-parameter sweep (paper §4.2).
# Task format: --task MODEL/PARAM/SEED, e.g. moss-voicegen/temp_0.5/seed_17, voxinstruct/cfg_1.0/seed_42.
# WAVs are read from $WAV_ROOT/MODEL/PARAM/SEED/*.wav; embeddings are written to
# $OUTPUT_ROOT/MODEL/PARAM/SEED/<encoder>/ (the extractor appends the WAV folder name, SEED,
# to --tts_model MODEL/PARAM).
set -Eeuo pipefail

cd "$(dirname "$0")/.."                    # repo root

ROOT="$PWD"
DATA_ROOT="${ITTS_DATA_ROOT:-$ROOT/data}"
WAV_ROOT="${PARAM_SWEEP_WAV_ROOT:-$DATA_ROOT/outputs/param_sweep}"
OUTPUT_ROOT="${PARAM_SWEEP_EMBEDDINGS_ROOT:-$DATA_ROOT/embeddings_param_sweep}"
EMBED_ENV="${EMBED_ENV:-$ROOT/envs/itts_embed}"
NEMO_ENV="${NEMO_ENV:-$ROOT/envs/itts_nemo}"
DEVICE="${EMB_DEVICE:-cuda:0}"
BATCH_SIZE="${EMB_BATCH_SIZE:-256}"
RESNET_BATCH="${EMB_RESNET_BATCH_SIZE:-16}"
ECAPA_BATCH="${EMB_ECAPA_BATCH_SIZE:-16}"
WORKERS="${EMB_WORKERS:-8}"
WORKER_ID=""
TASKS=()

ALL_EMBEDDERS=(
    speechbrain_resnet
    speechbrain_ecapa
    wespeaker
    modelscope_eres2net
    modelscope_campplus
    nemo_titanet
    modelscope_eres2netv2
    wespeaker_voxblink2_resnet152_16k
)

declare -A EMBED_DIM=(
    [speechbrain_resnet]=256
    [speechbrain_ecapa]=192
    [wespeaker]=256
    [modelscope_eres2net]=192
    [modelscope_campplus]=192
    [nemo_titanet]=192
    [modelscope_eres2netv2]=192
    [wespeaker_voxblink2_resnet152_16k]=256
)

LOG_DIR="$OUTPUT_ROOT/logs"
LOCK_ROOT="$OUTPUT_ROOT/.locks"
CURRENT_LOCK=""
cleanup() {
    if [ -n "$CURRENT_LOCK" ]; then
        rmdir -- "$CURRENT_LOCK" 2>/dev/null || echo "[worker] WARN: could not remove owned lock $CURRENT_LOCK" >&2
    fi
    return 0   # An empty lock must not turn an otherwise successful EXIT into status 1.
}
trap cleanup EXIT

usage() {
    cat <<'EOF'
Usage:
  bash embeddings/run_embedding_param_sweep.sh --worker-id ID --task MODEL/PARAM/SEED [--task ...]

Environment: ITTS_DATA_ROOT, PARAM_SWEEP_WAV_ROOT (default $ITTS_DATA_ROOT/outputs/param_sweep),
PARAM_SWEEP_EMBEDDINGS_ROOT (default $ITTS_DATA_ROOT/embeddings_param_sweep), EMBED_ENV, NEMO_ENV.

Examples:
  bash embeddings/run_embedding_param_sweep.sh --worker-id gpu0 \
    --task moss-voicegen/temp_0.5/seed_17 --task moss-voicegen/temp_1.0/seed_17
  bash embeddings/run_embedding_param_sweep.sh --worker-id gpu1 \
    --task voxinstruct/cfg_0.5/seed_42
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --worker-id) WORKER_ID="${2:?--worker-id needs a value}"; shift 2 ;;
        --task) TASKS+=("${2:?--task needs MODEL/PARAM}"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[worker] Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[ -n "$WORKER_ID" ] || { echo "[worker] --worker-id is required" >&2; exit 2; }
[ "${#TASKS[@]}" -gt 0 ] || { echo "[worker] at least one --task is required" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$LOCK_ROOT"

is_complete() {
    local wav_dir="$1" output_dir="$2" dim="$3"
    local emb_file="$output_dir/embeddings.npy"
    [ -f "$emb_file" ] || return 1
    "$EMBED_ENV/bin/python" - "$wav_dir" "$emb_file" "$dim" <<'PY'
import sys, numpy as np
from pathlib import Path
wav_dir, emb_file, dim = sys.argv[1], sys.argv[2], int(sys.argv[3])
wavs = list(Path(wav_dir).glob("*.wav")) + list(Path(wav_dir).glob("*.WAV"))
n_wavs = len(wavs)
try:
    emb = np.load(emb_file)
    valid = emb.shape == (n_wavs, dim)
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

run_embedder() {
    local model="$1" param="$2" embedder="$3"
    local wav_dir="$WAV_ROOT/$model/$param"
    local output_dir="$OUTPUT_ROOT/$model/$param/$embedder"
    local dim="${EMBED_DIM[$embedder]}"
    local py="$EMBED_ENV/bin/python"
    local model_batch="$BATCH_SIZE"
    local tag="${model//\//__}__${param}__${embedder}"     # MODEL may contain a slash
    local lock_dir="$LOCK_ROOT/$tag.lock"
    local log="$LOG_DIR/extract_$tag.log"

    if is_complete "$wav_dir" "$output_dir" "$dim"; then
        printf '[worker:%s] SKIP valid %s/%s/%s\n' "$WORKER_ID" "$model" "$param" "$embedder"
        return
    fi
    if ! mkdir "$lock_dir" 2>/dev/null; then
        printf '[worker:%s] SKIP locked %s/%s/%s\n' "$WORKER_ID" "$model" "$param" "$embedder"
        return
    fi
    CURRENT_LOCK="$lock_dir"

    [ "$embedder" = nemo_titanet ] && py="$NEMO_ENV/bin/python"
    [ "$embedder" = speechbrain_resnet ] && model_batch="$RESNET_BATCH"
    [ "$embedder" = speechbrain_ecapa ] && model_batch="$ECAPA_BATCH"

    printf '===== [worker:%s] %s/%s/%s start %s =====\n' "$WORKER_ID" "$model" "$param" "$embedder" "$(date '+%F %T')" | tee -a "$log"

    "$py" -u embeddings/extract_embeddings.py \
        --wav_dir "$wav_dir" \
        --tts_model "$model" \
        --output_dir "$OUTPUT_ROOT" \
        --models "$embedder" \
        --device "$DEVICE" \
        --batch_size "$model_batch" \
        --num_workers "$WORKERS" >> "$log" 2>&1

    if is_complete "$wav_dir" "$output_dir" "$dim"; then
        printf '===== [worker:%s] %s/%s/%s verified %s =====\n' "$WORKER_ID" "$model" "$param" "$embedder" "$(date '+%F %T')" | tee -a "$log"
    else
        echo "[worker:$WORKER_ID] ERROR: output is incomplete for $model/$param/$embedder" >&2
        return 1
    fi

    rmdir -- "$lock_dir"
    CURRENT_LOCK=""
}

for task in "${TASKS[@]}"; do
    model="${task%/*}"
    param="${task##*/}"
    wav_dir="$WAV_ROOT/$model/$param"

    if [ ! -d "$wav_dir" ]; then
        echo "[worker:$WORKER_ID] WARN: WAV dir not found: $wav_dir, skipping" >&2
        continue
    fi

    wav_count=$(find "$wav_dir" -name "*.wav" 2>/dev/null | wc -l)
    echo "[worker:$WORKER_ID] Processing $model/$param ($wav_count WAVs)"

    for embedder in "${ALL_EMBEDDERS[@]}"; do
        run_embedder "$model" "$param" "$embedder"
    done
done

echo "[worker:$WORKER_ID] DONE $(date '+%F %T')"
