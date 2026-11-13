# 01_encoder_validation (paper §3.2, Appendix A)

Checks which speaker encoders can measure diversity. LibriSpeech subsets with a known number of
speakers k are embedded with all 11 encoders; an encoder is kept if its Vendi score (VS) grows clearly
with k.

The encoders use the same checkpoints as the diversity experiments (`configs/encoders.yaml`).

## Environment

**Requirements.** Linux, one NVIDIA GPU with a driver for CUDA 12.8, conda (Miniconda or Miniforge), `git` and `curl`, and about 30 GB of disk: 2 × ~8.5 GB
for the environments, 6.3 GB for LibriSpeech, ~2 GB for the encoder checkpoints and 0.4 GB of
embeddings.

**Two conda environments** (see also `envs/README.md`), because NeMo and the other toolkits need
different `transformers` versions:

| Environment | File | Python | Used for |
|---|---|---|---|
| `itts_embed` | `envs/itts_embed.yml` | 3.11 | the 10 encoders other than TitaNet (WeSpeaker, SpeechBrain, ModelScope, Hugging Face SV models), all metrics, and the Python steps of `run.sh` |
| `itts_nemo` | `envs/itts_nemo.yml` | 3.10 | TitaNet-Large (NeMo 2.7.3) |

From the repo root:

```bash
conda env create -p envs/itts_embed -f envs/itts_embed.yml
conda env create -p envs/itts_nemo  -f envs/itts_nemo.yml
```

Each takes about an hour to build, mostly pip downloads. Create them one after the
other: two `conda env create` sharing one package cache at the same time fail with `LockError`. The YAML files pull PyTorch 2.9.1+cu128 from
`https://download.pytorch.org/whl/cu128` and WeSpeaker from GitHub at commit `dfa7419`.

If conda cannot solve the YAML, the `*.lock.txt` files (`pip freeze` of the same environments) are
an alternative; the first lines of each file give the command, e.g.

```bash
conda create -p envs/itts_embed python=3.11 -y
envs/itts_embed/bin/pip install -r envs/itts_embed.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

`run.sh` looks for the environments at `envs/itts_embed/` and `envs/itts_nemo/`; if they are
elsewhere, point to them:

```bash
export EMBED_ENV=/path/to/itts_embed NEMO_ENV=/path/to/itts_nemo
```

**Checkpoints.** No manual download: each encoder fetches its checkpoint on first use
(`configs/encoders.yaml` lists them): from Hugging Face for TitaNet, SpeechBrain (ECAPA, ResNet,
x-vector), WavLM and UniSpeech-SAT, and from ModelScope for ERes2Net, ERes2NetV2, CAM++ and the two
WeSpeaker models (`wespeaker.load_model` fetches `voxceleb_resnet221_LM` and `voxblink2_samresnet34`
from the `wenet/wespeaker_pretrained_models` dataset there). Set `HF_HOME` or `MODELSCOPE_CACHE` to
keep them off the home directory. `run.sh` sets `SSL_CERT_FILE` to certifi's
bundle, which some ModelScope downloads need.

## Run

```bash
bash experiments/01_encoder_validation/prepare_data.sh   # step 1, once
bash experiments/01_encoder_validation/run.sh            # steps 2 to 4
```

`prepare_data.sh` downloads LibriSpeech train-clean-100 (6.3 GB) if `LIBRISPEECH_DIR` does not
exist (`OPENSLR_MIRROR` selects a faster mirror), writes the subsets and links the audio; it needs
only `itts_embed` and takes a few minutes after the download. `run.sh` embeds 28,539 utterances with
11 encoders. Both write under `ITTS_DATA_ROOT` (default
`data/`); results go to a new `results/reproductions/run-<time>-<pid>/` folder unless `RESULTS_DIR` is set.

## Method (paper §3.2)

1. `build_subsets.py`: for each k ∈ {6, 8, 10, 12, 15, 20, 24, 30, 40, 48, 60, 80, 120, 240}, 100 subsets of
   N = 240 utterances from LibriSpeech train-clean-100: k speakers, 240/k utterances from each, without
   replacement (seed 0). One utterance of each of the 251 speakers is first set aside as the reference
   pool R for Chamfer and MMD and never enters a subset.
2. `embeddings/extract_embeddings.py`: every utterance, with each of the 11 encoders.
3. `compute_curves.py`: MPCD and VS within each subset (cosine); Chamfer and MMD² between the subset and
   R (`metrics/chamfer.py`, `metrics/mmd.py`: raw embeddings, Euclidean, median-heuristic Gaussian
   kernel, biased estimator, as in the paper's appendix). Mean, std and sem over the 100 subsets per k.
4. `rank_correlation.py`: per encoder and metric, Spearman ρ between the metric and k over all 1,400
   subsets, the k=240 / k=6 ratio, and the number of the 13 adjacent k steps whose change in the mean
   exceeds the larger of the two subset standard deviations. An encoder is kept when its VS separates at
   least 12 of the 13 steps.

## Results

`results/encoder_validation/`: `metric_values.csv` (every subset), `metric_curves.csv` (mean, std and
sem per k) and `rank_correlation.csv`. Re-running gives the same subsets; the metrics vary very
slightly with GPU floating-point noise in the embeddings.

VS at k = 6 and k = 240, their ratio and steps separated (of 13):

| Encoder | VS k=6 | VS k=240 | ratio | steps | kept |
|---|---|---|---|---|---|
| TIT | 12.01 | 80.50 | 6.70 | 13 | yes |
| WSVB | 11.03 | 71.07 | 6.44 | 13 | yes |
| WS221 | 13.09 | 81.76 | 6.24 | 13 | yes |
| SBRN | 15.81 | 91.44 | 5.78 | 13 | yes |
| SBEC | 15.40 | 83.26 | 5.41 | 13 | yes |
| ER2v2 | 7.65 | 22.24 | 2.91 | 13 | yes |
| ER2 | 7.51 | 21.80 | 2.90 | 13 | yes |
| CAM | 7.77 | 21.05 | 2.71 | 12 | yes |
| WLM | 2.67 | 3.61 | 1.35 | 0 | no |
| USAT | 2.49 | 3.20 | 1.28 | 0 | no |
| XV | 1.61 | 1.86 | 1.16 | 1 | no |

The eight retained encoders expand VS by 2.7 to 6.7 and separate at least 12 of 13 steps; XV, WLM and
USAT rise by at most 1.35, stay below VS 4 at k = 240 and separate at most 1 step.

## Files

| File | What it does |
|---|---|
| `prepare_data.sh` | Step 1: LibriSpeech download, `build_subsets.py`, and `.wav` links to the FLAC files in `$ITTS_DATA_ROOT/librispeech/train-clean-100_flat/` |
| `run.sh` | Steps 2 to 4 |
| `build_subsets.py` | Writes `subsets.jsonl`, `reference_pool.txt` and `utterances.txt` to `$ITTS_DATA_ROOT/librispeech_validation/` (deterministic for a given seed, so not committed) |
| `compute_curves.py` | Writes `metric_values.csv` and `metric_curves.csv` |
| `rank_correlation.py` | Writes `rank_correlation.csv` |
