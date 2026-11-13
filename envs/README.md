# envs/

The tools need incompatible package versions, so each group runs in its own conda environment.

## Evaluation environments

Used by every `experiments/*/run.sh`. `experiments/common.sh` expects them at `envs/<name>/`
(override with `EMBED_ENV`, `NEMO_ENV`, `VOX_ENV`).

| Environment | File | Used for |
|---|---|---|
| `itts_embed` | `itts_embed.yml` | the 10 speaker encoders other than TitaNet, all metrics, the Python steps of `run.sh`, experiment 01 |
| `itts_nemo` | `itts_nemo.yml` | TitaNet |
| `vox_profile` | `vox_profile.yml` | controllability classifiers (experiment 05) |

```bash
conda env create -p envs/itts_embed  -f envs/itts_embed.yml
conda env create -p envs/itts_nemo   -f envs/itts_nemo.yml
conda env create -p envs/vox_profile -f envs/vox_profile.yml
```

Create them one after another: two `conda env create` runs that share a package cache fail with
`LockError`. Each `*.lock.txt` is the `pip freeze` of the same environment; its first lines give the
commands to use it instead of the YAML file.

## Generation environments

Needed only to generate new audio (`experiments/*/generate.sh`).

| Environment | File | Systems |
|---|---|---|
| `itts_moss` | `itts_moss.yml` | MOSS-VoiceGenerator |
| `itts_parler` | `itts_parler.yml` | Parler-TTS Mini v1 and Large v1 |
| `itts_qwen` | `itts_qwen.yml` | Qwen3-TTS Voice Design |
| `itts_vox` | `itts_vox.yml` | VoxInstruct |
| `itts_voxcpm2` | `itts_voxcpm2.yml` | VoxCPM2 |
| `itts_capspeech` | `itts_capspeech.yml` | CapSpeech-NAR |

```bash
bash envs/setup_generation_envs.sh            # all six, as named conda environments
bash envs/setup_generation_envs.sh moss       # or any of: moss, parler, qwen, vox, voxcpm2, capspeech
```

Three systems need a step beyond the YAML file:

- **VoxCPM2 and CapSpeech-NAR**: install the upstream source, checked out at the `upstream_commit`
  in `configs/systems.yaml`, without its dependencies:

  ```bash
  git clone https://github.com/OpenBMB/VoxCPM && git -C VoxCPM checkout 616d3d3e630a9c96c2853250eef91b0f39dcd5fa
  conda run -n itts_voxcpm2 python -m pip install --no-deps -e ./VoxCPM

  git clone https://github.com/WangHelin1997/CapSpeech && git -C CapSpeech checkout 0868f3578ba61e50ceddf4ac7b05c286c008652a
  conda run -n itts_capspeech python -m pip install --no-deps -e ./CapSpeech
  conda run -n itts_capspeech python -c "import nltk; [nltk.download(r) for r in ('averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger', 'cmudict')]"
  ```

- **VoxInstruct**: `generation/download_weights.py` clones the upstream code at the recorded commit;
  then apply the changes in `generation/patches/voxinstruct/` (see its README).

The generation driver looks for `<GENERATION_ENV_ROOT>/<name>/bin/python`. For named conda
environments, set `GENERATION_ENV_ROOT=$(conda info --base)/envs`, or map each system to its
interpreter with `GENERATION_PYTHON_MAP` (see `generation/NEW_RUNS.md`).
