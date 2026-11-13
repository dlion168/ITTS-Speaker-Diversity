# 04_attribute_diversity (paper §5, Appendix E)

Within-value speaker diversity for 7 systems × 24 requested attribute values (gender 2, age 4,
accent 6, emotion 4, gender × age 8), with 95% half-sampling confidence intervals averaged over the
8 retained encoders.

## Environment

The same two conda environments as experiment 01: `itts_embed` and `itts_nemo` (TitaNet; not needed
for `--from embeddings`). Build them as described in `experiments/01_encoder_validation/README.md`,
or point to existing ones, and export a read token for the gated Hugging Face dataset (do not write
it into a file in the repo):

```bash
export EMBED_ENV=/path/to/itts_embed NEMO_ENV=/path/to/itts_nemo
export HF_TOKEN=hf_...
```

Every path needs a GPU: the confidence intervals draw 10,000 subsets per value.

## Run

`generate.sh` generates the audio, `run.sh` evaluates it (`--from`: see `experiments/README.md`).
Each `run.sh` writes `$RESULTS_DIR/attributes/{vendi,mpcd}_ci{,_by_encoder}.csv` (default: a new
`results/reproductions/run-<time>-<pid>/` folder).

### `--from embeddings`: metrics only

```bash
RESULTS_DIR=/tmp/check04 bash experiments/04_attribute_diversity/run.sh --from embeddings
```

Downloads the `attribute` and `gender_x_age` embeddings of the 8 encoders (all 7 systems are
released), then `metrics/half_sampling_ci.py`. `SYSTEMS="voxcpm2"` runs one system.

### `--from audio`: embed the released audio

```bash
ITTS_DATA_ROOT=/big/disk bash experiments/04_attribute_diversity/run.sh --from audio
```

Downloads the `attribute` and `gender_x_age` audio (96,000 utterances per system) into
`$ITTS_DATA_ROOT/outputs/attributes/<system>/<category>/`, embeds it with the 8 encoders
(`embeddings/run_embedding_subset.sh`) into `$ITTS_DATA_ROOT/embeddings_attributes/`, then computes
the metrics. The same audio is what `05_controllability/run.sh --from audio` scores.

### `generate.sh`, then `--from generated`

```bash
bash experiments/04_attribute_diversity/generate.sh --dry-run   # print the generation commands
bash experiments/04_attribute_diversity/generate.sh
bash experiments/04_attribute_diversity/run.sh --from generated
```

`generate.sh` calls `generation/run_attributes.sh` (environments and weights in
`generation/README.md`): 3,000 utterances per single-attribute value and 6,000 per gender × age
cell, into `$ITTS_DATA_ROOT/outputs/attributes/`. `run.sh --from generated` embeds and scores them.
The embedding step refuses a category with fewer files than that, so audio made with
`generate.sh --limit N` can be checked with `--dry-run` but not evaluated. Freshly generated audio
will not reproduce the tables exactly (sampling).

## Method

`metrics/half_sampling_ci.py`: for each value, B = 10,000 random subsets of N = 1,500 clips without
replacement, the same subsets for every encoder (seed 0); point estimate = mean over the subsets,
95% interval from the rescaled draws. Details: `docs/within_value_diversity_ci.md`.

## Results

`results/attributes/{vendi,mpcd}_ci.csv` (encoder mean, one row per system and value, plus one column
per encoder) and `{vendi,mpcd}_ci_by_encoder.csv` (one row per encoder, system and value). The draws
(`*.npz`) are written next to them but not committed.

## Files

| File | What it does |
|---|---|
| `generate.sh` | Generates the audio (`generation/run_attributes.sh`) |
| `run.sh` | Evaluates it: `--from generated`, `audio` or `embeddings` |

Inputs: `data/prompts/single_attribute.json`, `data/prompts/gender_x_age.json`,
`configs/experiments/attributes.yaml`.
