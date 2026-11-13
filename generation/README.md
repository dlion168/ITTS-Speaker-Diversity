# generation/

Generates speech with the 7 ITTS systems in the paper. The experiment scripts
(`experiments/*/generate.sh`) call the three run scripts below; `infer.py` can also be used on its own.

## Files

| File | What it does |
|---|---|
| `run_overall.sh`, `run_sampling_sweep.sh`, `run_attributes.sh` | Generate the audio of §4.1, §4.2 and §5. All three call `run_generation.py` with the matching experiment config |
| `run_generation.py` | Plans the runs from `configs/experiments/*.yaml`, checks inputs and environments, runs `run_seeded_infer.py` per output folder and checks the WAVs afterwards (`--dry-run` prints the plan only) |
| `run_seeded_infer.py` | Seeds Python, NumPy and PyTorch, then runs `infer.py` in the system's environment |
| `infer.py` | Single entry point for all systems: reads `id, text, style_prompt` rows (CSV, JSON or JSONL) and writes `<output_dir>/<id>.wav`. Per-system decoding flags: `--moss_audio_temperature`, `--voxcpm_cfg_value`, `--vox_cfg_st_on_text`, `--vox_cfg_at_on_text`, `--vox_cfg_at_on_st`, `--vox_nar_iter_steps` |
| `settings.py` | Loads and checks the experiment YAML files |
| `preflight.py` | Checks input counts and ids, prompt selection and the finished WAVs |
| `seed_hook/` | `sitecustomize.py`, which seeds subprocesses started inside a run (used by `run_seeded_infer.py`) |
| `backends/` | One backend per system (`base.py` defines the interface; `__init__.py` maps CLI names to backends) |
| `patches/voxinstruct/` | Our changes to the official VoxInstruct code, and how to apply them |
| `download_weights.py` | Downloads the model weights at the revisions in `configs/systems.yaml` into `<repo>/weights/` (or `WEIGHTS_DIR`, `--weights-dir`); clones VoxInstruct at its recorded commit; `--check` lists what is already there without downloading |
| `model_pins.py` | Resolves each model to its recorded revision; loading a model that is not in the weights folder downloads that revision, not the latest one |
| `examples/` | Example inputs for `infer.py` |
| `tests/` | Tests of the run plans and seeding |

Model revisions, upstream commits and the generation settings of each system are in
`configs/systems.yaml`. CapSpeech-NAR uses a different checkpoint per attribute category
(`AccCapTTS` for accent, `EmoCapTTS` for emotion, `CapTTS` otherwise).

## Usage

See `NEW_RUNS.md` for complete runs. A single file:

```bash
python generation/infer.py --list-models
python generation/infer.py --model qwen3-voicedesign --input generation/examples/sample.csv \
    --output_dir data/outputs/example --batch_size 16
```

Each system runs in its own environment (`envs/README.md`). VoxInstruct weights are on Google Drive;
`patches/voxinstruct/README.md` explains the setup.

## Output file names

The file name carries the requested value, and the evaluation code reads it, so do not rename outputs.

- single attribute: `<attr>-<value>-s<k>-<idx>-<take>.wav`, e.g. `age-child-s1-0001-T1.wav`
- gender × age: `genderage-<gender>_<age>-s<k>-<idx>-<frame>_<order>.wav`
