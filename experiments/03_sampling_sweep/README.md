# 03_sampling_sweep (paper §4.2)

Tests whether the gap to real speech comes from default decoding settings, by sweeping temperature
(MOSS-VoiceGenerator) and classifier-free guidance (VoxInstruct, VoxCPM2).

## Run

```bash
bash experiments/03_sampling_sweep/generate.sh                   # optional: generate the audio
bash experiments/03_sampling_sweep/run.sh --from generated|audio|embeddings
```

`generate.sh` does step 1, `run.sh` steps 2 and 3 (see `experiments/README.md` for `--from`).

Settings (`experiments/common.sh`):

- `SYSTEMS`: run only some of the three sweep systems, e.g. `SYSTEMS="voxcpm2" bash
  experiments/03_sampling_sweep/run.sh --from embeddings`. Systems outside the sweep are ignored;
  the CSV then has rows for the selected systems only. Direct `summarize.py` calls also
  accept `--systems`, which takes precedence over `SYSTEMS`.
- `REAL_AUDIO`: real Emilia recordings, `<dir>/set_1/*.wav` (default
  `$ITTS_DATA_ROOT/real_world_data`, where `data/emilia/download_reference_audio.py` writes them).
  Only `--from generated` and `--from audio` need it, and only when
  `$ITTS_DATA_ROOT/embeddings_overall/real/set_1` does not exist yet (02 writes it too).
  `--from embeddings` downloads the released real-speech embeddings instead.

1. Generate 10,000 utterances for each of 5 values × 3 seeds (17, 42, 99) × 3 systems
   (`generate.sh`), or download the released audio (`run.sh --from audio`).
2. Embed with the 8 retained encoders (`embeddings/run_embedding_param_sweep.sh`), or download the
   released embeddings (`--from embeddings`).
3. Compute Vendi and MPCD per configuration on the same 3,000 of the 10,000 texts, and the
   real-speech baseline at the same size (`summarize.py`).

## Files

| File | What it does |
|---|---|
| `generate.sh` | Step 1: generates the audio (`generation/run_sampling_sweep.sh`) |
| `run.sh` | Steps 2 and 3, `--from generated`, `audio` or `embeddings` |
| `summarize.py` | Writes `results/sampling_sweep/param_sweep_seeded_diversity.csv` and `results/overall/real_vendi_n3000.csv` |

## Inputs and outputs

- Input: `data/emilia/overall_partition_1.jsonl` (the sweep reuses partition 1's texts),
  `configs/experiments/sampling_sweep.yaml`
- Output: `results/sampling_sweep/param_sweep_seeded_diversity.csv`, `results/overall/real_vendi_n3000.csv`
- Intermediate: audio in `$ITTS_DATA_ROOT/outputs/param_sweep/<system>/<temp|cfg>_<value>/seed_<s>/`,
  embeddings in `$ITTS_DATA_ROOT/embeddings_param_sweep/<system>/<temp|cfg>_<value>/seed_<s>/<encoder>/`.

`run.sh` stops before extraction if a selected system has no `seed_<s>/` folders or a seed folder has
no WAV files. `summarize.py` stops if a configuration is missing any of the 8 encoders, or if its
embeddings sit one `seed_<s>/` level too deep (`.../seed_17/seed_17/<encoder>/`).

## Subsets

Vendi grows with the number of utterances, so every configuration is scored on the same 3,000 of
its 10,000 texts, and the real-speech baseline on 3,000 utterances of real partition 1
(`summarize.py`).
