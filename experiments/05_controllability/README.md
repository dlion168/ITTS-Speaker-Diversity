# 05_controllability (paper §3.5, §5, Appendices B to D)

Does each utterance have the requested attribute? Scores the attribute-experiment audio of all
7 systems with one classifier per attribute and reports per-class recall and confusion.

## Environment

The `vox_profile` conda environment (`envs/vox_profile.yml`, `envs/vox_profile.lock.txt`; see
`envs/README.md`), and for `--from audio` also `itts_embed` (the download step). Point to existing
ones with:

```bash
export VOX_ENV=/path/to/vox_profile EMBED_ENV=/path/to/itts_embed
export HF_TOKEN=hf_...    # --from audio only: the dataset is gated (do not write the token into a file)
```

The scorer imports the Vox-Profile model code. `run.sh` clones it at the commit we used into
`controllability/vox-profile-release/` (`controllability/setup_vox_profile.sh`) unless that folder
exists or `VOX_PROFILE_DIR` points to a checkout.

## Run

There is no `generate.sh` here: the audio is the one of experiment 04. There is no `--from embeddings`
either, since the classifiers need audio.

```bash
# from the released audio (attribute and gender_x_age configs of the Hugging Face dataset)
ITTS_DATA_ROOT=/big/disk bash experiments/05_controllability/run.sh --from audio

# from the audio of experiments/04_attribute_diversity/generate.sh
bash experiments/05_controllability/run.sh --from generated
```

Steps:

1. Download the released audio into `$ITTS_DATA_ROOT/outputs/attributes/<system>/<category>/`
   (`--from audio`), or check that `generate.sh` of experiment 04 wrote it there (`--from generated`).
2. Download our accent classifier (`controllability/download_checkpoints.py`, 2 files, the Whisper
   member is 6.2 GB; later runs take it from the Hugging Face cache).
3. Write a manifest per folder (`controllability/make_manifests.py`; the requested value comes from
   the file name).
4. Score every system and category in `NPROC` shards on one GPU and merge
   (`controllability/run_scoring.sh`) into `$RESULTS_DIR/controllability/summary.json`.

Keep `NPROC=10` and `BATCH=8` (the defaults) to reproduce `results/controllability/summary.json`
exactly. Accent scoring loads Whisper-Large-V3, about 12 GB per worker.

## Files

| File | What it does |
|---|---|
| `run.sh` | Steps 1 to 4, `--from generated` or `audio` |

## Inputs and outputs

- Input: `$ITTS_DATA_ROOT/outputs/attributes/`, `configs/classifiers.yaml`
- Output: `results/controllability/summary.json`; per-shard outputs in `$ITTS_DATA_ROOT/controllability/`

The accent classifier's own test results (Appendix C) are in `results/accent_classifier/`.
