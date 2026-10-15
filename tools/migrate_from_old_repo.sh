#!/usr/bin/env bash
# Copy the parts of the old repo (DanielChen1128/ITTS_diversity) that the paper uses
# into this repo's layout. Run from the root of this repo:
#   bash tools/migrate_from_old_repo.sh /path/to/ITTS_diversity
# Nothing in the old repo is modified. Missing files are listed at the end; check them by hand.
set -u
OLD="${1:?usage: bash tools/migrate_from_old_repo.sh /path/to/ITTS_diversity}"
SE="$OLD/speaker_embedding_eval"
missing=()

cp_one () {  # cp_one <src> <dst>
  if [ -e "$1" ]; then mkdir -p "$(dirname "$2")"; cp -r "$1" "$2"; echo "ok   $2"
  else missing+=("$1"); echo "MISS $1"; fi
}

# ---- generation/ (7 paper systems only) ----
cp_one "$OLD/infer.py"                              generation/infer.py
cp_one "$OLD/download_weights.py"                   generation/download_weights.py
cp_one "$OLD/model_backends/__init__.py"            generation/backends/__init__.py
cp_one "$OLD/model_backends/base.py"                generation/backends/base.py
for m in moss_voicegen parler qwen3_tts voxinstruct voxcpm2 capspeech; do
  cp_one "$OLD/model_backends/${m}_model.py"        "generation/backends/${m}_model.py"
done
cp_one "$OLD/patches/voxinstruct"                   generation/patches/voxinstruct
cp_one "$OLD/setup_envs.sh"                         envs/setup_envs.sh

# ---- data/ ----
for c in gender age accent emotion gender_x_age; do
  cp_one "$OLD/attribute category prompts/$c.json"  "data/prompts/raw/$c.json"
done
for i in 1 2 3; do
  cp_one "$OLD/emilia_en_samples/sample_$i.jsonl"   "data/emilia/overall_partition_$i.jsonl"
done
cp_one "$OLD/download_emilia_prompts.py"            data/emilia/download_reference_audio.py

# ---- experiments/: original run scripts, to be rewritten as run.sh ----
cp_one "$OLD/run_infer.sh"                          experiments/02_overall_diversity/old_run_infer.sh
cp_one "$OLD/run_param_sweep_b200_1.sh"             experiments/03_sampling_sweep/old_run_param_sweep_b200_1.sh
cp_one "$OLD/run_param_sweep_b200_2.sh"             experiments/03_sampling_sweep/old_run_param_sweep_b200_2.sh
cp_one "$OLD/run_attribute_infer.sh"                experiments/04_attribute_diversity/old_run_attribute_infer.sh

# ---- embeddings/ ----
cp_one "$SE/eval/extract_embeddings.py"             embeddings/extract_embeddings.py
cp_one "$SE/scripts/run_embedding_subset.sh"        embeddings/run_embedding_subset.sh
cp_one "$SE/scripts/run_embedding_param_sweep.sh"   embeddings/run_embedding_param_sweep.sh

# ---- metrics/ (to be split into mpcd/vendi/chamfer/mmd/half_sampling_ci) ----
cp_one "$SE/eval/compute_metrics.py"                metrics/compute_metrics.py

# ---- controllability/ ----
cp_one "$SE/eval/make_attr_manifests.py"            controllability/make_manifests.py
cp_one "$SE/eval/score_controllability.py"          controllability/score_controllability.py
cp_one "$SE/eval/merge_controllability_shards.py"   controllability/merge_shards.py
cp_one "$SE/scripts/run_controllability_attributes.sh" controllability/run_scoring.sh
cp_one "$SE/hf_weights.py"                          controllability/download_checkpoints.py
for f in train_accent.py train_accent.sbatch train_accent_whisper.py train_accent_whisper.sbatch eval_accent.py eval_accent_ensemble.py; do
  cp_one "$SE/classifier/$f"                        "controllability/accent_classifier/$f"
done
cp_one "$SE/splits/accent"                          controllability/accent_classifier/splits

# ---- results/ (filtered afterwards) ----
for f in mpcd_ci.csv mpcd_ci_by_encoder.csv; do
  cp_one "$SE/results/diversity/$f"                 "results/attributes/raw_$f"
done

# ---- docs/ ----
cp_one "$OLD/docs/within_value_diversity_ci.md"     docs/within_value_diversity_ci.md
cp_one "$OLD/docs/controllability_scorer.md"        docs/controllability_scorer.md
cp_one "$OLD/docs/classifier.md"                    docs/accent_classifier.md
cp_one "$OLD/docs/BATCH_INFERENCE.md"               docs/batch_inference.md

# ---- filter to what the paper uses ----
python3 tools/filter_results.py

echo
if [ ${#missing[@]} -gt 0 ]; then
  echo "Missing in old repo (${#missing[@]}):"; printf '  %s\n' "${missing[@]}"
else
  echo "All files found."
fi
