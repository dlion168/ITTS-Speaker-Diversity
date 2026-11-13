# controllability/

Checks whether each generated utterance has the requested attribute, using one classifier per
attribute, and reports per-class recall and confusion (paper §3.5, §5, Appendices B to D).
Runs in the `vox_profile` environment (`envs/vox_profile.lock.txt`) on a GPU.

## Files

| File | What it does | Input → Output |
|---|---|---|
| `setup_vox_profile.sh` | Clones the Vox-Profile model code (`tiantiaf0627/vox-profile-release`) at commit `85100e6` into `controllability/vox-profile-release/` | → model wrappers imported by the scorer |
| `download_checkpoints.py` | Downloads our accent classifier from Hugging Face (`Snooow1029/itts-attribute-classifiers`) | → `checkpoint/accent_{wavlm,whisper}/best.pt` |
| `make_manifests.py` | Reads the requested value from each WAV file name and writes a manifest with `target_gender`, `target_age`, `target_accent`, `target_emotion` columns | `data/outputs/attributes/<system>/<category>/` → `controllability_manifest.csv` in each folder |
| `score_controllability.py` | Runs the classifiers on the WAVs listed in a manifest (`--accent ensemble --age voxprofile`, the paper setting and the defaults) | WAVs + manifest → `controllability.json`, `controllability_per_utt.csv` |
| `merge_shards.py` | Merges sharded scoring runs into one summary with recall and confusion per system and category (plus per-cell results for gender × age); refuses incomplete, mixed or duplicated shards | shard outputs → `summary.json` |
| `scoring_run.py` | Records the inputs and settings of a scoring run before it starts, and checks them when a run is resumed | → `scoring_run.json` in the run folder |
| `run_scoring.sh` | Batch driver: scores every system and category in 10 shards on one GPU, then merges | `data/outputs/attributes/` → `<run folder>/summary.json` |
| `accent_classifier/train_accent.py` (+ `.sbatch`) | Trains the WavLM-Large + LoRA member of the accent classifier | `accent_classifier/splits/` → checkpoint |
| `accent_classifier/train_accent_whisper.py` (+ `.sbatch`) | Trains the Whisper-Large-V3 + LoRA member | `accent_classifier/splits/` → checkpoint |
| `accent_classifier/eval_accent_ensemble.py` | Test-split accuracy of the ensemble (Appendix C) | splits + checkpoints → `results/accent_classifier/id/` |
| `accent_classifier/eval_accent_ood.py` | Recall on four out-of-domain accent corpora (Appendix C) | manifest of OOD CSVs + checkpoints → `results/accent_classifier/ood/` |
| `accent_classifier/data_build/` | Builds the training splits and the SAA out-of-domain set from the public corpora; checks a copy of the original split shards (`verify_original_shards.py`); `source_manifests/` lists the clips of the released model's splits and of the other sources | CommonVoice 17, AfriSpeech, VoxPopuli, SAA → `accent_classifier/splits/` |
| `accent_classifier/splits/` | Train / validation / test shards of the accent classifier (not committed: third-party audio) | |

Gender, age and emotion use released models without retraining; see `configs/classifiers.yaml`.
Training data, recipe and results of the accent classifier: `docs/accent_classifier.md`.
Input format and options of the scorer: `docs/controllability_scorer.md`.

## Usage

```bash
bash controllability/setup_vox_profile.sh
python controllability/download_checkpoints.py
python controllability/make_manifests.py --root data/outputs/attributes
PY=envs/vox_profile/bin/python bash controllability/run_scoring.sh      # all 5 categories
```

`experiments/05_controllability/run.sh` runs these steps.

Each scoring run writes to its own folder (`SCORING_RUN_DIR`, printed at the start; default a new
`run-<time>-<pid>/` under `SHARD_DIR`), and writes `summary.json` there only if every shard finished.
To rerun failed shards, keep all settings and point to the same folder:

```bash
SCORING_RUN_DIR=/path/to/that/run RESUME=1 SHARDS="1 2" bash controllability/run_scoring.sh
```

## Settings needed to reproduce the results exactly

1. **Shards and batch size.** Keep `NPROC=10` and `BATCH=8` (the defaults of `run_scoring.sh`), as
   used for `results/controllability/summary.json`.
2. **LoRA loading.** Our checkpoints store already-merged LoRA weights. After `load_state_dict`, mark
   every `loralib` layer as merged (`score_controllability.py` does this).
