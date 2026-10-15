# controllability/

Checks whether each generated utterance has the requested attribute, using one classifier per attribute, and reports per-class recall (paper §3.5).

## Files

| File | What it does | Input → Output | Status |
|---|---|---|---|
| `make_manifests.py` | Reads the requested value from each WAV file name and writes a manifest with `target_gender`, `target_age`, `target_accent`, `target_emotion` columns | `data/outputs/attributes/` → manifest CSVs | from old repo (`speaker_embedding_eval/eval/make_attr_manifests.py`) |
| `score_controllability.py` | Runs the classifiers on the WAVs listed in a manifest. Always pass `--age voxprofile` and `--accent ensemble`. | WAVs + manifest → `controllability.json`, `controllability_per_utt.csv` | from old repo (`speaker_embedding_eval/eval/score_controllability.py`) |
| `merge_shards.py` | Merges sharded scoring runs into one summary with recall and confusion matrices per system and category | shard outputs → `results/controllability/summary.json` | from old repo (`speaker_embedding_eval/eval/merge_controllability_shards.py`) |
| `run_scoring.sh` | Batch driver: scores every system and category, split into shards on one GPU | | from old repo (`speaker_embedding_eval/scripts/run_controllability_attributes.sh`) |
| `download_checkpoints.py` | Downloads our accent classifier from Hugging Face (`Snooow1029/itts-attribute-classifiers`) into `checkpoint/` | → `checkpoint/accent_{wavlm,whisper}/best.pt` | from old repo (`speaker_embedding_eval/hf_weights.py`) |
| `accent_classifier/train_accent.py` (+ `.sbatch`) | Trains the WavLM-Large + LoRA accent classifier | `splits/` → checkpoint | from old repo (`speaker_embedding_eval/classifier/`) |
| `accent_classifier/train_accent_whisper.py` (+ `.sbatch`) | Trains the Whisper-Large-V3 + LoRA accent classifier | `splits/` → checkpoint | from old repo |
| `accent_classifier/eval_accent.py` | Tests the WavLM classifier on the held-out split | | from old repo |
| `accent_classifier/eval_accent_ensemble.py` | Tests the WavLM + Whisper ensemble (the numbers in Appendix C) | | from old repo |
| `accent_classifier/splits/` | Train / validation / test lists for the accent classifier | | **to upload** (whoever trained the accent classifier) |

Gender, age, and emotion use released models without retraining; see `configs/classifiers.yaml`.

## Two things that silently change results

1. **Batch size.** The Vox-Profile wrappers normalize each padded batch, so predictions depend on batch size. Use one batch size for every shard of a run.
2. **LoRA loading.** Our checkpoints store already-merged LoRA weights. After `load_state_dict`, mark every `loralib` layer as merged, or the update is applied twice.
