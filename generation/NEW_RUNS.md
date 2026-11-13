# Generating the paper's audio

`run_overall.sh`, `run_sampling_sweep.sh` and `run_attributes.sh` generate the audio of §4.1, §4.2
and §5. The experiment scripts `experiments/0{2,3,4}_*/generate.sh` call them.

## Setup

- **Environments.** Each system runs in its own Python environment (`envs/README.md`). Tell the
  driver where they are with `GENERATION_ENV_ROOT` (a folder that contains `<env name>/bin/python`;
  for named conda environments, `$(conda info --base)/envs`), or map each system to its interpreter
  with `GENERATION_PYTHON_MAP` (a JSON object from system name to a Python path or an argv list).
  The driver itself needs PyYAML (`GENERATION_DRIVER_PYTHON`, default `python`).
- **Weights.** Download them once with `python generation/download_weights.py` (all systems) or
  `--model <name>`, into `WEIGHTS_DIR` (default `<repo>/weights/`). VoxInstruct then needs the
  changes in `generation/patches/voxinstruct/`.
- **Data.** `ITTS_DATA_ROOT` (default `data/`).

## Run

Print the plan first; it creates nothing:

```bash
bash generation/run_overall.sh --dry-run
bash generation/run_sampling_sweep.sh --dry-run
bash generation/run_attributes.sh --models capspeech --dry-run
```

Then run without `--dry-run`. The full plans have 21 overall groups (7 systems × 3 partitions),
45 sweep groups (3 systems × 5 values × 3 seeds) and 35 attribute groups (7 systems × 5 categories).

| Option | Meaning |
|---|---|
| `--models` | a subset of the systems |
| `--config PATH` | another experiment YAML instead of `configs/experiments/<kind>.yaml` |
| `--instruction TEXT` | another neutral instruction for overall and sweep runs |
| `--seed N` | the seed of an overall or attribute run (`GENERATION_SEED`, default from the YAML); sweep seeds come from the YAML |
| `--batch-size N`, `--limit N` | batch size; only the first N rows (a quick test) |

The sweep varies MOSS-VoiceGenerator's `audio_temperature`, VoxCPM2's `cfg_value` and VoxInstruct's
`cfg_st_on_text` over 0.5 to 2.5 with seeds 17, 42 and 99.

## Outputs

Audio goes to `$ITTS_DATA_ROOT/outputs/`:

- overall: `overall/<system>/sample<partition>/`
- sweep: `param_sweep/<system>/<temp|cfg>_<value>/seed_<seed>/`
- attributes: `attributes/<system>/<category>/`

Each folder also gets `generation_run.json` (instruction, input and config hashes, seed, command)
and, once all its WAVs are written and readable, `generation_complete.json`. A run resumes in a
folder only with the same settings; changed settings need a new `ITTS_DATA_ROOT`.

Then evaluate with `run.sh --from generated` of the same experiment (`experiments/README.md`).
