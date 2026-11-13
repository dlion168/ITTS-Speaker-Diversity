#!/usr/bin/env bash
# Score attribute controllability (paper §3.5, §5) for every system in data/outputs/attributes/.
#
# The five scoreable categories are run as separate GROUPS so each worker process only
# loads the classifiers that group needs (the accent group is the only one that pays for
# Whisper-Large-V3). Within a group, NPROC processes share the one GPU via
# `--shard i/NPROC`; merge_shards.py recombines them into identical statistics afterwards.
#
#   group          folders/model   clips/model   classifiers
#   accent         accent          18000         accent WavLM + Whisper (ensemble)
#   age            age             12000         age Vox-Profile regressor (tiantiaf/wavlm-large-age-sex), binned
#   gender         gender           6000         gender Vox-Profile age_sex (sex head)
#   gender_x_age   gender_x_age    48000         age + gender, both from the one Vox-Profile age_sex model
#   emotion        emotion         12000         emotion2vec+ large (funasr; no vox-profile weights)
#
# Each folder needs a controllability_manifest.csv (make_manifests.py) and the accent group needs
# checkpoint/accent_{wavlm,whisper}/best.pt (download_checkpoints.py).
#
# Usage:
#   bash controllability/run_scoring.sh [group ...]                 # default: all five
#   bash controllability/run_scoring.sh accent
#   SCORING_RUN_DIR=/original/run SHARDS="1 2 3" RESUME=1 bash controllability/run_scoring.sh accent
#
# Environment: PY (python of the vox_profile env), WAV_ROOT, SHARD_DIR (per-shard outputs),
# SUMMARY (merged result), MODELS, NPROC, BATCH, WORKERS, SHARDS, RESUME.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"       # controllability/
ROOT="$(cd "$HERE/.." && pwd)"
DATA_ROOT="${ITTS_DATA_ROOT:-$ROOT/data}"
PY="${PY:-$ROOT/envs/vox_profile/bin/python}"
WAV_ROOT="${WAV_ROOT:-$DATA_ROOT/outputs/attributes}"
RESULTS="${SHARD_DIR:-$DATA_ROOT/controllability}"
# Resolve the default summary beneath the isolated run after preparing it.
SUMMARY="${SUMMARY:-}"
# results/controllability/summary.json was produced with NPROC=10, BATCH=8 (~12 GB per worker);
# keep both to reproduce it exactly.
NPROC="${NPROC:-10}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-8}"
# Re-run only these shard indices, e.g. SHARDS="1 2". NPROC must stay the same,
# because the shard partition is rows[i::NPROC].
SHARDS="${SHARDS:-}"
RESUME="${RESUME:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MODELS="${MODELS:-capspeech moss-voicegen parler-large parler-mini qwen3-voicedesign voxcpm2 voxinstruct}"

# NB: not `GROUPS` — that is a read-only bash builtin array of the caller's group ids.
RUN_GROUPS=("$@")
[ ${#RUN_GROUPS[@]} -eq 0 ] && RUN_GROUPS=(accent age gender gender_x_age emotion)

attrs_for() {
  case "$1" in
    accent)       echo "accent" ;;
    age)          echo "age" ;;
    gender)       echo "gender" ;;
    gender_x_age) echo "age,gender" ;;
    emotion)      echo "emotion" ;;
    *) echo "unknown group: $1" >&2; return 1 ;;
  esac
}

[ -x "$PY" ] || { echo "missing interpreter: $PY (set PY to the vox_profile env's python; see envs/README.md)" >&2; exit 1; }
if [ -n "$SHARDS" ] && [ "$RESUME" != 1 ]; then
    echo "SHARDS is retry-only; use RESUME=1 with the original SCORING_RUN_DIR" >&2; exit 2
fi
if [ "$RESUME" = 1 ]; then
    [ -n "${SCORING_RUN_DIR:-}" ] || { echo "RESUME=1 requires SCORING_RUN_DIR, the folder of the run to resume." >&2; exit 2; }
else
    SCORING_RUN_DIR="${SCORING_RUN_DIR:-$RESULTS/run-$(date -u '+%Y%m%dT%H%M%S%N')-$$}"
fi
read -r -a selected_models <<< "$MODELS"
plan_args=(--run-dir "$SCORING_RUN_DIR" --wav-root "$WAV_ROOT" --models "${selected_models[@]}"
           --groups "${RUN_GROUPS[@]}" --nproc "$NPROC" --batch "$BATCH" --workers "$WORKERS")
[ "${SCORING_ALLOW_PARTIAL_INPUTS:-0}" = 1 ] && plan_args+=(--allow-partial-inputs)
[ "$RESUME" = 1 ] && plan_args+=(--resume)
"$PY" "$HERE/scoring_run.py" "${plan_args[@]}"
SUMMARY="${SUMMARY:-$SCORING_RUN_DIR/summary.json}"
for index in ${SHARDS:-$(seq 0 $((NPROC - 1)))}; do
    [[ "$index" =~ ^[0-9]+$ ]] && [ "$index" -lt "$NPROC" ] || { echo "Invalid selected shard: $index" >&2; exit 2; }
done
LOGS="$SCORING_RUN_DIR/logs"
mkdir -p "$LOGS"
echo "SCORING_RUN_DIR=$SCORING_RUN_DIR"
total_fail=0

for group in "${RUN_GROUPS[@]}"; do
  attrs="$(attrs_for "$group")" || exit 1

  folders=()
  for m in $MODELS; do
    d="$WAV_ROOT/$m/$group"
    if [ -f "$d/controllability_manifest.csv" ]; then
      folders+=("$d")
    else
      echo "missing selected manifest: $m/$group" >&2; exit 2
    fi
  done
  [ ${#folders[@]} -gt 0 ] || { echo "No selected scoring folders" >&2; exit 2; }

  shard_list="$SHARDS"
  [ -z "$shard_list" ] && shard_list="$(seq 0 $((NPROC - 1)))"
  resume_flag=""
  [ "$RESUME" = "1" ] && resume_flag="--resume"

  echo "=== group $group  attrs=$attrs  folders=${#folders[@]}  shards=[$(echo $shard_list | tr '\n' ' ')]/$NPROC  batch=$BATCH ==="
  start=$SECONDS
  pids=(); idx=()
  for i in $shard_list; do
    "$PY" "$HERE/score_controllability.py" "${folders[@]}" \
        --csv controllability_manifest.csv \
        --attrs "$attrs" --shard "$i/$NPROC" --accent ensemble --age voxprofile $resume_flag \
        --batch "$BATCH" --workers "$WORKERS" \
        --out_dir "$SCORING_RUN_DIR" > "$LOGS/${group}.shard${i}.log" 2>&1 &
    pids+=($!); idx+=("$i")
  done

  fail=0
  for k in "${!pids[@]}"; do
    wait "${pids[$k]}" || { echo "[error] $group shard ${idx[$k]} failed -> $LOGS/${group}.shard${idx[$k]}.log" >&2; fail=$((fail + 1)); }
  done
  echo "=== group $group done in $((SECONDS - start))s (failed shards: $fail) ==="
  total_fail=$((total_fail + fail))
done

echo
[ "$total_fail" -eq 0 ] || { echo "Scoring failed ($total_fail workers); no summary was replaced. Preserve $SCORING_RUN_DIR for an explicit resume." >&2; exit 1; }
"$PY" "$HERE/merge_shards.py" --results_dir "$SCORING_RUN_DIR" --run-manifest "$SCORING_RUN_DIR/scoring_run.json" --out "$SUMMARY"
