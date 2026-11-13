# Shared settings and helpers for experiments/*/{generate,run}.sh. Source it; do not run it.
#
# generate.sh (02, 03, 04) generates the audio with generation/ into $ITTS_DATA_ROOT/outputs/.
# run.sh evaluates; it takes --from generated | audio | embeddings:
#   generated   embed / score the audio generate.sh wrote
#   audio       download the released audio from Hugging Face, embed / score it
#   embeddings  download the released speaker embeddings and only compute the metrics
#
# Environment (defaults in brackets):
#   ITTS_DATA_ROOT   large intermediate files: audio, embeddings, shard outputs  [<repo>/data]
#   RESULTS_DIR      reproduced outputs [<repo>/results/reproductions/run-<UTC>-<pid>]
#   EMBED_ENV        conda prefix of itts_embed (envs/itts_embed.lock.txt)       [<repo>/envs/itts_embed]
#   NEMO_ENV         conda prefix of itts_nemo (envs/itts_nemo.lock.txt)         [<repo>/envs/itts_nemo]
#   VOX_ENV          conda prefix of vox_profile (envs/vox_profile.lock.txt)     [<repo>/envs/vox_profile]
#   HF_REPO / HF_REVISION   released dataset                                     [dlion168/itts-speaker-diversity / configs/release_assets.json]
#   HF_LOCAL_DIR     an already downloaded snapshot of the dataset; skips the download
#   SYSTEMS          space-separated subset of the 7 systems (folder names)    [all 7]
#   HF_TOKEN         the dataset is gated: accept its terms on Hugging Face, then use a read token
#   REAL_AUDIO       real Emilia reference recordings, <dir>/set_{1,2,3}/*.wav   [$ITTS_DATA_ROOT/real_world_data]
#                    (fetched by data/emilia/download_reference_audio.py; not redistributed)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$ROOT"
export ITTS_DATA_ROOT="${ITTS_DATA_ROOT:-$ROOT/data}"
DATA="$ITTS_DATA_ROOT"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/reproductions/run-$(date -u '+%Y%m%dT%H%M%S%N')-$$}"
REAL_AUDIO="${REAL_AUDIO:-$DATA/real_world_data}"
EMBED_ENV="${EMBED_ENV:-$ROOT/envs/itts_embed}"
NEMO_ENV="${NEMO_ENV:-$ROOT/envs/itts_nemo}"
VOX_ENV="${VOX_ENV:-$ROOT/envs/vox_profile}"
export EMBED_ENV NEMO_ENV
HF_REPO="${HF_REPO:-dlion168/itts-speaker-diversity}"
if [ "$HF_REPO" = dlion168/itts-speaker-diversity ]; then
    # Dataset revision of configs/release_assets.json; HF_REVISION overrides it.
    HF_REVISION="${HF_REVISION:-6fbfb3a69abdc4543a4989809bf085661ab836f4}"
fi
HF_ARGS=(--repo "$HF_REPO")
[ -n "${HF_REVISION:-}" ] && HF_ARGS+=(--revision "$HF_REVISION")
[ -n "${HF_LOCAL_DIR:-}" ] && HF_ARGS=(--local_dir "$HF_LOCAL_DIR")
PY="$EMBED_ENV/bin/python"

# The seven systems (local folder names, as in results/).
ALL_SYSTEMS=(moss-voicegen parler-large parler-mini capspeech qwen3-voicedesign voxinstruct voxcpm2)
read -r -a SYSTEMS <<< "${SYSTEMS:-${ALL_SYSTEMS[*]}}"

hf_splits() {   # Hugging Face split names of the selected SYSTEMS
    local s
    for s in "${SYSTEMS[@]}"; do
        case "$s" in
            moss-voicegen) echo moss_voicegen ;; parler-large) echo parler_large ;;
            parler-mini) echo parler_mini ;; capspeech) echo capspeech_nar ;;
            qwen3-voicedesign) echo qwen3_tts ;; voxinstruct|voxcpm2) echo "$s" ;;
            *) echo "unknown system: $s" >&2; return 1 ;;
        esac
    done
}
ENCODERS=(speechbrain_resnet speechbrain_ecapa wespeaker modelscope_eres2net modelscope_campplus
          nemo_titanet modelscope_eres2netv2 wespeaker_voxblink2_resnet152_16k)

FROM=""
parse_from() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --from) FROM="${2:?--from needs generated, audio or embeddings}"; shift 2 ;;
            -h|--help) sed -n '2,/^source /p' "$RUN_SH" | sed '$d'; exit 0 ;;
            *) echo "unknown option: $1" >&2; exit 2 ;;
        esac
    done
    case "$FROM" in
        generated|audio|embeddings) ;;
        scratch) echo "Generation and evaluation are separate: run this experiment's generate.sh, then run.sh --from generated (see generation/NEW_RUNS.md)." >&2; exit 2 ;;
        *) echo "usage: $RUN_SH --from generated|audio|embeddings" >&2; exit 2 ;;
    esac
}

need_python() {
    [ -x "$1/bin/python" ] || { echo "missing environment $1 (see envs/README.md)" >&2; exit 2; }
}

need_real_audio() {
    [ -d "$REAL_AUDIO/set_1" ] && return
    echo "no real reference audio in $REAL_AUDIO/set_1: run data/emilia/download_reference_audio.py" >&2
    echo "(gated Emilia subset, needs HF_TOKEN) and set REAL_AUDIO to its output folder" >&2
    exit 2
}

generate() {   # generate <experiment> [options for generation/run_<experiment>.sh]
    local kind="$1"; shift
    if [ ! -f "$ROOT/generation/run_$kind.sh" ]; then
        echo "generation/run_$kind.sh is not in the repo yet (generation module)." >&2
        echo "Use run.sh --from audio to start from the released audio instead." >&2
        exit 2
    fi
    bash "$ROOT/generation/run_$kind.sh" --models "${SYSTEMS[@]}" "$@"
}

need_generated() {   # need_generated <generate.sh> <dir>...: audio that generate.sh writes
    # generate.sh writes WAVs into subfolders (attributes/<system>/<category>/,
    # param_sweep/<system>/<value>/seed_<seed>/), so look below each <dir>, not only in it.
    local gen="$1" d; shift
    for d in "$@"; do
        if [ -d "$d" ] && [ -n "$(find "$d" -type f -name '*.wav' -size +0c -print -quit)" ]; then
            continue
        fi
        echo "no generated audio in $d: run $gen first, or use --from audio" >&2
        exit 2
    done
}

# embed_dir <wav_dir> <tts_name> <out_root>: all eight paper encoders, TitaNet in its own env.
# Batch sizes match the ones behind the released embeddings.
embed_dir() {
    local wav_dir="$1" tts="$2" out_root="$3" enc py batch
    export SSL_CERT_FILE="${SSL_CERT_FILE:-$("$PY" -c 'import certifi; print(certifi.where())')}"
    for enc in "${ENCODERS[@]}"; do
        py="$EMBED_ENV/bin/python"; batch=256
        [ "$enc" = nemo_titanet ] && py="$NEMO_ENV/bin/python"
        case "$enc" in speechbrain_*) batch=16 ;; esac
        "$py" -u embeddings/extract_embeddings.py --wav_dir "$wav_dir" --tts_model "$tts" \
            --output_dir "$out_root" --models "$enc" --batch_size "$batch"
    done
}
