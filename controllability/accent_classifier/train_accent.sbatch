#!/bin/bash
#SBATCH -J accent_wavlm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH -o controllability/checkpoint/sbatch_accent_wavlm_%j.log

# Train the WavLM-Large + LoRA member of the released 6-class accent classifier on a 24 GB GPU
# (an RTX 3090 in our runs). Submit from the repo root. Auto-resumes from $ACC_CKPT/last.pt, so
# re-submit after a wall-time limit to continue. Override any ACC_* variable with sbatch --export.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACC_SPLITS="${ACC_SPLITS:-controllability/accent_classifier/splits}"
export ACC_CKPT="${ACC_CKPT:-controllability/checkpoint/trained/accent_wavlm}"
export ACC_LABELS="${ACC_LABELS:-North America,English,South Asia,Germanic,Oceania,South African}"
export ACC_WEIGHTED="${ACC_WEIGHTED:-1}"      # inverse-frequency class-weighted cross-entropy
PY="${PY:-envs/vox_profile/bin/python}"

echo "[sbatch] $(hostname) job $SLURM_JOB_ID splits=$ACC_SPLITS ckpt=$ACC_CKPT weighted=${ACC_WEIGHTED:-0} start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY controllability/accent_classifier/train_accent.py
echo "[sbatch] train_accent.py exit $? $(date)"
