# ITTS Speaker Diversity

[![HF Dataset](https://img.shields.io/badge/HF%20Dataset-itts--speaker--diversity-yellow)](https://huggingface.co/datasets/dlion168/itts-speaker-diversity)
[![HF Model](https://img.shields.io/badge/HF%20Model-itts--attribute--classifiers-yellow)](https://huggingface.co/Snooow1029/itts-attribute-classifiers)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue)](LICENSE)
[![Data: CC BY-NC 4.0](https://img.shields.io/badge/data-CC%20BY--NC%204.0-lightgrey)](DATA_LICENSE.md)

Code to reproduce the experiments of **"Beyond Controllability: Speaker Diversity in Instruction-Guided Text-to-Speech"**.

Instruction-guided text-to-speech (ITTS) systems create a voice from a written description, such as
"Speak in a male voice." If you ask a system for a male voice a thousand times, do you get many
different men, or the same few voices again and again? The paper measures this **speaker diversity**
for seven open-source ITTS systems, compares it with real human speech, and compares it across
requested genders, ages, accents and emotions. It also checks whether each generated voice has the
requested attribute (**controllability**).

Main findings:

- Every system produces less diverse voices than real speech.
- In most systems, female prompts give less diverse voices than male prompts.
- Accent and emotion prompts barely change diversity, but the generated voices lean strongly toward
  North American English and neutral speech.
- Diversity and controllability reveal different weaknesses, so ITTS should be evaluated on both.

This repository contains the code and results for every experiment: choosing speaker encoders that
can tell how many different speakers a set contains (§3.2), comparing each system with real speech
(§4.1) and across sampling settings (§4.2), comparing diversity across requested attributes (§5), and
measuring controllability (§3.5, §5).

## Contents

- [What is evaluated](#what-is-evaluated)
- [Paper to code](#paper-to-code)
- [Three starting points](#three-starting-points)
- [Quick start](#quick-start)
- [Released data](#released-data)
- [Environments](#environments)
- [Repository layout](#repository-layout)
- [Tests](#tests)
- [License](#license)

## What is evaluated

**Seven ITTS systems** (`configs/systems.yaml`). The folder name is used in local paths and result
files; the split name is used in the Hugging Face dataset.

| System | Folder name | HF split | Weights |
|---|---|---|---|
| MOSS-VoiceGenerator | `moss-voicegen` | `moss_voicegen` | `OpenMOSS-Team/MOSS-VoiceGenerator` |
| Parler-TTS Large v1 | `parler-large` | `parler_large` | `parler-tts/parler-tts-large-v1` |
| Parler-TTS Mini v1 | `parler-mini` | `parler_mini` | `parler-tts/parler-tts-mini-v1` |
| CapSpeech-NAR | `capspeech` | `capspeech_nar` | `OpenSound/CapSpeech-models` |
| Qwen3-TTS Voice Design 1.7B | `qwen3-voicedesign` | `qwen3_tts` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| VoxInstruct | `voxinstruct` | `voxinstruct` | Google Drive, see `generation/README.md` |
| VoxCPM2 | `voxcpm2` | `voxcpm2` | `openbmb/VoxCPM2` |

**Speaker encoders** (`configs/encoders.yaml`). All 11 are compared in the encoder validation
(experiment 01); the 8 retained there are used for every diversity result.

| Abbrev. in paper | Folder name | Checkpoint | Used for diversity |
|---|---|---|---|
| TIT | `nemo_titanet` | `nvidia/speakerverification_en_titanet_large` | yes |
| SBEC | `speechbrain_ecapa` | `speechbrain/spkrec-ecapa-voxceleb` | yes |
| SBRN | `speechbrain_resnet` | `speechbrain/spkrec-resnet-voxceleb` | yes |
| WS221 | `wespeaker` | WeSpeaker `voxceleb_resnet221_LM` | yes |
| WSVB | `wespeaker_voxblink2_resnet152_16k` | WeSpeaker `voxblink2_samresnet34` | yes |
| ER2 | `modelscope_eres2net` | `iic/speech_eres2net_sv_zh-cn_16k-common` | yes |
| ER2v2 | `modelscope_eres2netv2` | `iic/speech_eres2netv2_sv_zh-cn_16k-common` | yes |
| CAM | `modelscope_campplus` | `iic/speech_campplus_sv_zh-cn_16k-common` | yes |
| XV | `speechbrain_xvector` | `speechbrain/spkrec-xvect-voxceleb` | no |
| WLM | `wavlm_base_plus_sv` | `microsoft/wavlm-base-plus-sv` | no |
| USAT | `unispeech_sat_base_plus_sv` | `microsoft/unispeech-sat-base-plus-sv` | no |

**Metrics** (`metrics/`, computed on L2-normalized embeddings):

| Metric | Definition | Direction | Used in |
|---|---|---|---|
| MPCD | 1 − mean cosine similarity over all pairs i ≠ j | higher = more diverse | all diversity results |
| Vendi | exp(entropy of the eigenvalues of K/n), K the cosine-similarity matrix; grows with n, so only sets of equal size are compared | higher = more diverse | all diversity results |
| Chamfer distance | between a set and a reference pool, Euclidean, both directions | lower = closer to the pool | encoder validation only |
| MMD² | Gaussian kernel, median-heuristic bandwidth, biased estimator | lower = closer to the pool | encoder validation only |

The 95% confidence intervals in §5 draw 10,000 random subsets of 1,500 utterances per requested value,
average the 8 encoders draw by draw, and take the 2.5 and 97.5 percentiles, rescaled for the
6,000-utterance gender × age cells (`metrics/half_sampling_ci.py`, method in
`docs/within_value_diversity_ci.md`).

**Controllability classifiers** (`configs/classifiers.yaml`). Controllability is the per-class recall:
the fraction of utterances generated for a requested value that the classifier assigns to that value.

| Attribute | Classifier | Classes |
|---|---|---|
| Gender | Vox-Profile `tiantiaf/wavlm-large-age-sex`, sex head | female, male |
| Age | same model, age regressor binned at 20 / 40 / 60 years | child, young_adult, middle_aged, elderly |
| Emotion | `emotion2vec/emotion2vec_plus_large`, argmax restricted to the 4 classes | angry, happy, neutral, sad |
| Accent | our classifier (WavLM-Large and Whisper-Large-V3 with LoRA, softmax-averaged), weights at `Snooow1029/itts-attribute-classifiers` | north_american, british_isles, oceanian, south_asian, south_african, germanic |

## Paper to code

| Paper | Experiment folder | Writes |
|---|---|---|
| §3.2 encoder validation, Appendix A | `experiments/01_encoder_validation` | `results/encoder_validation/metric_curves.csv`, `rank_correlation.csv` |
| §4.1 comparison with real speech | `experiments/02_overall_diversity` | `results/overall/summary_diversity.csv` |
| §4.2 sampling parameters | `experiments/03_sampling_sweep` | `results/sampling_sweep/param_sweep_seeded_diversity.csv`, `results/overall/real_vendi_n3000.csv` |
| §5 diversity with 95% CI, Appendix E | `experiments/04_attribute_diversity` | `results/attributes/vendi_ci.csv`, `vendi_ci_by_encoder.csv` (and the MPCD versions) |
| §3.5 and §5 controllability, Appendices B to D | `experiments/05_controllability` | `results/controllability/summary.json` |
| Appendix C accent classifier | `controllability/accent_classifier` | `results/accent_classifier/` |

Each experiment folder has its own README with inputs, outputs and options.
Column definitions of every result file are in `results/README.md`.

## Three starting points

Generation and evaluation are separate scripts. Experiments 02, 03 and 04 have a `generate.sh`
that writes audio, and every experiment has a `run.sh` that evaluates. `run.sh` takes `--from`:

| `run.sh --from` | Starts from | Needs |
|---|---|---|
| `embeddings` | the released speaker embeddings; only computes the metrics | `HF_TOKEN`; a GPU for experiment 04 |
| `audio` | the released audio; extracts embeddings or scores controllability, then computes the metrics | `HF_TOKEN`, a GPU |
| `generated` | the audio you generated with `generate.sh` | the generation environments and model weights (`generation/NEW_RUNS.md`), a GPU |

Experiment 05 needs audio, so it has no `--from embeddings`. Experiment 01 starts from LibriSpeech:
run `prepare_data.sh`, then `run.sh` (see its README). Each run writes its results to a new folder,
`results/reproductions/run-<time>-<pid>/`, so the committed files in `results/` are never
overwritten; set `RESULTS_DIR` to choose the folder, for example to collect several experiments in one place.

## Quick start

Recompute the §4.1 table from the released embeddings (no GPU needed):

```bash
conda env create -p envs/itts_embed -f envs/itts_embed.yml   # evaluation environment
export HF_TOKEN=...                                           # read token, after accepting the dataset terms
RESULTS_DIR=/tmp/itts_check bash experiments/02_overall_diversity/run.sh --from embeddings
```

This writes `/tmp/itts_check/overall/summary_diversity.csv`, which you can compare with the committed
`results/overall/summary_diversity.csv`.

The other experiments follow the same pattern; see `experiments/README.md`.

## Released data

| What | Where | Access |
|---|---|---|
| Generated audio (configs `overall`, `attribute`, `gender_x_age`, `sampling_sweep`; one split per system) and speaker embeddings of all 8 encoders, including those of the real Emilia reference recordings (`embeddings/`) | Hugging Face dataset [`dlion168/itts-speaker-diversity`](https://huggingface.co/datasets/dlion168/itts-speaker-diversity) | gated: accept the terms on the dataset page, then create a read token and set `HF_TOKEN` |
| Real Emilia recordings used as the real-speech reference (experiments 02 and 03 with `--from audio` or `--from generated`) | `data/emilia/download_reference_audio.py` downloads them from [`dlion168/emilia-en-itts-prompts`](https://huggingface.co/datasets/dlion168/emilia-en-itts-prompts) | gated, manual approval; needs `HF_TOKEN` |
| Accent classifier weights | Hugging Face model [`Snooow1029/itts-attribute-classifiers`](https://huggingface.co/Snooow1029/itts-attribute-classifiers), fetched by `controllability/download_checkpoints.py` | public |
| Texts and instructions | `data/emilia/` (Emilia texts and utterance ids), `data/prompts/` (attribute instructions) | in this repository |

## Environments

Each group of tools needs different package versions, so each has its own conda environment
(details in `envs/README.md`).

| Environment | Used for | Create with |
|---|---|---|
| `itts_embed` | all encoders except TitaNet, all metrics, the Python steps of `run.sh` | `conda env create -p envs/itts_embed -f envs/itts_embed.yml` |
| `itts_nemo` | TitaNet | `conda env create -p envs/itts_nemo -f envs/itts_nemo.yml` |
| `vox_profile` | controllability classifiers (experiment 05) | `conda env create -p envs/vox_profile -f envs/vox_profile.yml` |
| `itts_moss`, `itts_parler`, `itts_qwen`, `itts_vox`, `itts_voxcpm2`, `itts_capspeech` | generating audio, one per system (`generate.sh` only) | `bash envs/setup_generation_envs.sh`, then the extra steps in `envs/README.md` |

`experiments/common.sh` expects the evaluation environments at `envs/<name>/`; override with
`EMBED_ENV`, `NEMO_ENV`, `VOX_ENV`. Create the environments one after another; two
`conda env create` runs sharing one package cache fail with `LockError`.

Other settings read by the scripts (all optional):

| Variable | Meaning | Default |
|---|---|---|
| `ITTS_DATA_ROOT` | large intermediate files: audio, embeddings, scoring shards | `data/` |
| `RESULTS_DIR` | where result files are written | a new `results/reproductions/run-<time>-<pid>/` |
| `SYSTEMS` | space-separated subset of the 7 folder names | all 7 |
| `HF_REVISION` | revision of the released dataset | the one in `configs/release_assets.json` |
| `HF_LOCAL_DIR` | an already downloaded copy of the dataset; skips the download | |
| `REAL_AUDIO` | real Emilia recordings, `<dir>/set_{1,2,3}/*.wav` | `$ITTS_DATA_ROOT/real_world_data` |

## Repository layout

```
configs/          systems, encoders, classifiers, per-experiment parameters
data/             Emilia texts and ids, attribute instructions (no audio)
generation/       ITTS generation for the 7 systems
embeddings/       speaker-embedding extraction and download of released embeddings or audio
metrics/          MPCD, Vendi, Chamfer distance, MMD², confidence intervals
controllability/  attribute classifiers, scoring, accent classifier training
experiments/      one folder per paper experiment (generate.sh, run.sh)
results/          the result files behind the paper's tables and figures
envs/             conda environment files
docs/             method notes (confidence intervals, accent classifier, scorer)
tests/            regression tests that need no models or downloads
```

## Tests

```bash
python -m unittest discover -s tests -v          # in the itts_embed environment
python metrics/tests/test_diversity_metrics.py   # analytic checks of the four metrics
python -m unittest discover -s generation/tests -v   # generation plans and seeds
```

## License

The code is under the MIT License (`LICENSE`). The data files (`data/`, `results/`,
`controllability/accent_classifier/data_build/source_manifests/`) and the released audio and
embeddings are under CC BY-NC 4.0; text from third-party corpora keeps its corpus license
(`DATA_LICENSE.md`). Third-party code and model weights: `NOTICE.md`.
