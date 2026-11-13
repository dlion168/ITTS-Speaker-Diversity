# 02_overall_diversity (paper §4.1)

Compares the speaker diversity of each system under a neutral instruction with real recordings of
the same texts.

## Environment

The same two conda environments as experiment 01: `itts_embed` (the 7 encoders other than TitaNet,
the metrics and the Python steps) and `itts_nemo` (TitaNet). Build them as described in
`experiments/01_encoder_validation/README.md` (Environment), or point to existing ones:

```bash
export EMBED_ENV=/path/to/itts_embed NEMO_ENV=/path/to/itts_nemo
```

The released data are on the gated Hugging Face dataset `dlion168/itts-speaker-diversity`: accept
its terms on the dataset page, then export a read token (do not write it into a file in the repo):

```bash
export HF_TOKEN=hf_...
```

## Run

Generation and evaluation are separate scripts: `generate.sh` generates the audio, `run.sh`
evaluates it. `run.sh` chooses its starting point with `--from`; each writes
`$RESULTS_DIR/overall/summary_diversity.csv` (default: a new `results/reproductions/run-<time>-<pid>/`
folder).

### `--from embeddings`: metrics only

```bash
RESULTS_DIR=/tmp/check bash experiments/02_overall_diversity/run.sh --from embeddings
```

`embeddings/download_released.py` fetches the overall embeddings of the seven systems and the
real-speech (`real_emilia`) embeddings of the 8 encoders, then `summarize.py` computes the table.

`HF_LOCAL_DIR=/path/to/snapshot` uses an already downloaded copy of the dataset instead.

### `--from audio`: embed the released audio

```bash
ITTS_DATA_ROOT=/big/disk python data/emilia/download_reference_audio.py   # once: -> /big/disk/real_world_data/
ITTS_DATA_ROOT=/big/disk bash experiments/02_overall_diversity/run.sh --from audio
```

1. `embeddings/download_released_audio.py` downloads the overall audio (`data/overall/*.parquet`,
   7 systems × 30,000 utterances, 60 GB; `HF_HOME` sets where the download cache goes) and writes it
   back as WAV files to `$ITTS_DATA_ROOT/outputs/overall/<system>/sample{1,2,3}/` (about 165 GB).
   A dropped connection is retried and resumes from the files already downloaded.
2. Embeds every partition with the 8 encoders (`embeddings/extract_embeddings.py`) into
   `$ITTS_DATA_ROOT/embeddings_overall/`: 22 partitions (7 systems × 3, plus real `set_1`).
3. `summarize.py`.

The real recordings come from the gated `dlion168/emilia-en-itts-prompts` (same `HF_TOKEN`); only
`set_1` is used.

### `generate.sh`, then `--from generated`: generate the audio yourself

```bash
bash experiments/02_overall_diversity/generate.sh --dry-run    # print the generation commands
bash experiments/02_overall_diversity/generate.sh
python data/emilia/download_reference_audio.py   # once: -> $ITTS_DATA_ROOT/real_world_data/
bash experiments/02_overall_diversity/run.sh --from generated
```

`generate.sh` calls `generation/run_overall.sh` (generation module; environments and weights in
`generation/README.md`) to produce 3 partitions × 10,000 utterances per system with the neutral
instruction in `$ITTS_DATA_ROOT/outputs/overall/`; its options (`--dry-run`, `--limit N`,
`--batch-size B`) go to that script. `run.sh --from generated` then runs steps 2 and 3 above on that
audio, and stops if a selected system has no audio there. The generation interpreters are set
with `GENERATION_PYTHON_MAP` or `GENERATION_ENV_ROOT` (`generation/NEW_RUNS.md`). Freshly generated
audio will not reproduce the table exactly (sampling).

## Method

- Texts: `data/emilia/overall_partition_{1,2,3}.jsonl` (10,000 Emilia English texts each); generation
  settings in `configs/experiments/overall.yaml`; neutral instruction "Speak like a person — natural,
  neutral tone, moderate pace, conversational and clear delivery."
- `summarize.py`: Vendi score and MPCD (cosine) on each full partition and encoder, then the mean and
  std over the three partitions. Real speech uses partition 1 only (`--real_sets`).

## Results

`results/overall/summary_diversity.csv`: set, encoder, n_partitions, vendi_mean/std, mpcd_mean/std
for the seven systems and real speech, 8 encoders each.

The three ModelScope encoders (CAM++, ERes2Net, ERes2NetV2) take 16 kHz input; `embeddings/compat.py`
resamples other rates with `torchaudio.transforms.Resample`, as for all results in this repository.

## Files

| File | What it does |
|---|---|
| `generate.sh` | Generates the audio (generation module) |
| `run.sh` | Evaluates it: `--from generated`, `audio` or `embeddings` |
| `summarize.py` | Writes `summary_diversity.csv` |
