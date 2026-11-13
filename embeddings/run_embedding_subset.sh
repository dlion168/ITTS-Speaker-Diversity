#!/usr/bin/env bash
# Extract embeddings for an explicit set of <tts-model>/<category> tasks.
# Tasks can be split across machines because each embedder writes to an
# independent output directory; a shared mkdir lock prevents accidental overlap.
set -Eeuo pipefail

cd "$(dirname "$0")/.."                    # repo root
ROOT="$PWD"
DATA_ROOT="${ITTS_DATA_ROOT:-$ROOT/data}"
WAV_ROOT="${ATTRIBUTES_OUTPUT_ROOT:-$DATA_ROOT/outputs/attributes}"
OUTPUT_ROOT="${EMBEDDINGS_OUTPUT_ROOT:-$DATA_ROOT/embeddings_attributes}"
EMBED_ENV="${EMBED_ENV:-$ROOT/envs/itts_embed}"
NEMO_ENV="${NEMO_ENV:-$ROOT/envs/itts_nemo}"
DEVICE="${EMB_DEVICE:-cuda:0}"
# Large batches for batch-capable embedders. The extractor splits
# variable-duration batches on OOM, while ECAPA retains its stricter limit below.
BATCH_SIZE="${EMB_BATCH_SIZE:-256}"
WORKERS="${EMB_WORKERS:-8}"
WORKER_ID=""
DRY_RUN=0
TASKS=()
EMBEDDERS=()

# The five attribute folders of paper §5 (gender_x_age is §5.6).
ALL_CATEGORIES=(gender age accent emotion gender_x_age)
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

usage() {
    cat <<'EOF'
Usage:
  bash embeddings/run_embedding_subset.sh --worker-id ID --task TTS/CATEGORY [--task ...]

CATEGORY is one of gender, age, accent, emotion, gender_x_age, or all.
Each task runs all eight embedders unless one or more --embedder options are given.
Several workers (e.g. one per GPU or host) can share one OUTPUT_ROOT: each
(task, embedder) is claimed with a lock directory, and finished outputs are skipped.

Environment:
  ITTS_DATA_ROOT            default <repo>/data
  ATTRIBUTES_OUTPUT_ROOT    WAVs, <root>/<tts>/<category>/*.wav  (default $ITTS_DATA_ROOT/outputs/attributes)
  EMBEDDINGS_OUTPUT_ROOT    default $ITTS_DATA_ROOT/embeddings_attributes
  EMBED_ENV, NEMO_ENV       conda prefixes of itts_embed and itts_nemo (default <repo>/envs/<name>)
  EMB_DEVICE, EMB_BATCH_SIZE, EMB_RESNET_BATCH_SIZE, EMB_ECAPA_BATCH_SIZE, EMB_WORKERS

Examples:
  bash embeddings/run_embedding_subset.sh --worker-id gpu0 \
    --task capspeech/all --task voxcpm2/all
  bash embeddings/run_embedding_subset.sh --worker-id gpu1 \
    --task qwen3-voicedesign/accent --embedder nemo_titanet
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --worker-id) WORKER_ID="${2:?--worker-id needs a value}"; shift 2 ;;
        --task) TASKS+=("${2:?--task needs TTS/CATEGORY}"); shift 2 ;;
        --embedder) EMBEDDERS+=("${2:?--embedder needs a model name}"); shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[worker] Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[ -n "$WORKER_ID" ] || { echo "[worker] --worker-id is required" >&2; exit 2; }
[ "${#TASKS[@]}" -gt 0 ] || { echo "[worker] at least one --task is required" >&2; exit 2; }
[ -d "$WAV_ROOT" ] || { echo "[worker] Missing WAV root: $WAV_ROOT" >&2; exit 2; }
[ -x "$EMBED_ENV/bin/python" ] || { echo "[worker] Missing embedding Python: $EMBED_ENV/bin/python" >&2; exit 2; }
[ -x "$NEMO_ENV/bin/python" ] || { echo "[worker] Missing NeMo Python: $NEMO_ENV/bin/python" >&2; exit 2; }

if [ "${#EMBEDDERS[@]}" -eq 0 ]; then
    EMBEDDERS=("${ALL_EMBEDDERS[@]}")
fi
for embedder in "${EMBEDDERS[@]}"; do
    case " $embedder " in
        " speechbrain_resnet "|" speechbrain_ecapa "|" wespeaker "|" modelscope_eres2net "|" modelscope_campplus "|" nemo_titanet "|" modelscope_eres2netv2 "|" wespeaker_voxblink2_resnet152_16k ") ;;
        *) echo "[worker] Unsupported embedder: $embedder" >&2; exit 2 ;;
    esac
done

declare -A EXPECTED_ROWS=(
    [gender]=6000 [age]=12000 [accent]=18000 [emotion]=12000 [gender_x_age]=48000
)
declare -A EMBED_DIM=(
    [speechbrain_resnet]=256 [speechbrain_ecapa]=192 [wespeaker]=256
    [modelscope_eres2net]=192 [modelscope_campplus]=192 [nemo_titanet]=192
    [modelscope_eres2netv2]=192 [wespeaker_voxblink2_resnet152_16k]=256
)

LOG_DIR="$OUTPUT_ROOT/logs"
LOCK_ROOT="$OUTPUT_ROOT/.locks"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$LOCK_ROOT"
export MODELSCOPE_LOG_LEVEL=40
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,max_split_size_mb:256
# WeSpeaker downloads through urllib and this environment's OpenSSL default
# points at a non-existent prefix. Use certifi for both urllib and requests.
CERTIFI_BUNDLE="$($EMBED_ENV/bin/python -c 'import certifi; print(certifi.where())')"
export SSL_CERT_FILE="${SSL_CERT_FILE:-$CERTIFI_BUNDLE}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$CERTIFI_BUNDLE}"

CURRENT_LOCK=""
cleanup() {
    [ -z "$CURRENT_LOCK" ] || rm -rf "$CURRENT_LOCK"
}
trap cleanup EXIT INT TERM

# Kept as a separate function so a completed output is the source of truth on
# restart; there is no mutable shared scheduler state to recover.
is_complete() {
    local wav_dir="$1" output_dir="$2" expected_rows="$3" expected_dim="$4"
    "$EMBED_ENV/bin/python" - "$wav_dir" "$output_dir" "$expected_rows" "$expected_dim" <<'PY'
import sys
from pathlib import Path
import numpy as np

wav_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
expected_rows = int(sys.argv[3])
expected_dim = int(sys.argv[4])
npy_path = output_dir / "embeddings.npy"
names_path = output_dir / "filenames.txt"
if not npy_path.is_file() or not names_path.is_file():
    raise SystemExit(1)
try:
    embeddings = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    names = names_path.read_text(encoding="utf-8").splitlines()
    wav_names = {path.stem for path in wav_dir.glob("*.wav")} | {path.stem for path in wav_dir.glob("*.WAV")}
    valid = (
        embeddings.shape == (expected_rows, expected_dim)
        and len(names) == expected_rows
        and len(set(names)) == len(names)
        and set(names) == wav_names
        and np.isfinite(embeddings).all()
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

run_embedder() {
    local tts="$1" category="$2" embedder="$3"
    local wav_dir="$WAV_ROOT/$tts/$category"
    local output_dir="$OUTPUT_ROOT/$tts/$category/$embedder"
    local expected="${EXPECTED_ROWS[$category]}"
    local dim="${EMBED_DIM[$embedder]}"
    local py="$EMBED_ENV/bin/python"
    local model_batch="$BATCH_SIZE"
    local lock_dir="$LOCK_ROOT/${tts}__${category}__${embedder}.lock"
    local log="$LOG_DIR/extract_${tts}_${category}_${embedder}.log"

    if is_complete "$wav_dir" "$output_dir" "$expected" "$dim"; then
        printf '[worker:%s] SKIP valid %s/%s/%s\n' "$WORKER_ID" "$tts" "$category" "$embedder"
        return
    fi
    if ! mkdir "$lock_dir" 2>/dev/null; then
        printf '[worker:%s] SKIP locked %s/%s/%s\n' "$WORKER_ID" "$tts" "$category" "$embedder"
        return
    fi
    CURRENT_LOCK="$lock_dir"
    printf 'worker=%s host=%s pid=%s started=%s\n' "$WORKER_ID" "$(hostname)" "$$" "$(date -Is)" > "$lock_dir/owner"

    if is_complete "$wav_dir" "$output_dir" "$expected" "$dim"; then
        printf '[worker:%s] SKIP valid-after-lock %s/%s/%s\n' "$WORKER_ID" "$tts" "$category" "$embedder"
        rm -rf "$lock_dir"
        CURRENT_LOCK=""
        return
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '[worker:%s] DRY-RUN %s/%s/%s\n' "$WORKER_ID" "$tts" "$category" "$embedder"
        rm -rf "$lock_dir"
        CURRENT_LOCK=""
        return
    fi

    [ "$embedder" = nemo_titanet ] && py="$NEMO_ENV/bin/python"
    # SpeechBrain batches are zero-padded to the longest clip; some systems produce very
    # long outliers, so keep these two batches small to bound peak VRAM.
    [ "$embedder" = speechbrain_resnet ] && model_batch="${EMB_RESNET_BATCH_SIZE:-16}"
    # ECAPA's attentive pooling can hit PyTorch's 32-bit index limit when
    # variable-duration WAVs are padded in batches of 64.
    [ "$embedder" = speechbrain_ecapa ] && model_batch="${EMB_ECAPA_BATCH_SIZE:-16}"
    printf '===== [worker:%s] %s/%s/%s start %s =====\n' "$WORKER_ID" "$tts" "$category" "$embedder" "$(date '+%F %T')" | tee -a "$log"
    "$py" -u embeddings/extract_embeddings.py \
        --wav_dir "$wav_dir" \
        --tts_model "$tts" \
        --output_dir "$OUTPUT_ROOT" \
        --models "$embedder" \
        --device "$DEVICE" \
        --batch_size "$model_batch" \
        --num_workers "$WORKERS" >> "$log" 2>&1
    if ! is_complete "$wav_dir" "$output_dir" "$expected" "$dim"; then
        echo "[worker:$WORKER_ID] INVALID output after $tts/$category/$embedder; see $log" >&2
        return 1
    fi
    printf '===== [worker:%s] %s/%s/%s verified %s =====\n' "$WORKER_ID" "$tts" "$category" "$embedder" "$(date '+%F %T')" | tee -a "$log"
    rm -rf "$lock_dir"
    CURRENT_LOCK=""
}

expanded_tasks=()
for task in "${TASKS[@]}"; do
    case "$task" in
        */*) tts="${task%/*}"; category="${task##*/}" ;;
        *) echo "[worker] Invalid task '$task'; expected TTS/CATEGORY" >&2; exit 2 ;;
    esac
    [ -n "$tts" ] || { echo "[worker] Empty TTS name in '$task'" >&2; exit 2; }
    if [ "$category" = all ]; then
        for category in "${ALL_CATEGORIES[@]}"; do expanded_tasks+=("$tts/$category"); done
    elif [ -n "${EXPECTED_ROWS[$category]+x}" ]; then
        expanded_tasks+=("$tts/$category")
    else
        echo "[worker] Invalid category '$category' in '$task'" >&2
        exit 2
    fi
done

for task in "${expanded_tasks[@]}"; do
    tts="${task%/*}"
    category="${task##*/}"
    wav_dir="$WAV_ROOT/$tts/$category"
    expected="${EXPECTED_ROWS[$category]}"
    [ -d "$wav_dir" ] || { echo "[worker] Missing WAV directory: $wav_dir" >&2; exit 1; }
    shopt -s nullglob
    wavs=("$wav_dir"/*.wav "$wav_dir"/*.WAV)
    shopt -u nullglob
    if [ "${#wavs[@]}" -ne "$expected" ]; then
        echo "[worker] $tts/$category has ${#wavs[@]}/$expected WAVs; refusing incomplete task" >&2
        exit 1
    fi
    for embedder in "${EMBEDDERS[@]}"; do
        run_embedder "$tts" "$category" "$embedder"
    done
done

echo "[worker:$WORKER_ID] DONE $(date '+%F %T')"
