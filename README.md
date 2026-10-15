# ITTS Speaker Diversity

Code and results for **"Attribute-Conditional Speaker Diversity in Instruction-Guided Text-to-Speech"**.

> Work in progress. The repo is being reorganized from an earlier codebase.

We measure how many distinct voices instruction-guided TTS (ITTS) systems produce for the same voice description,
compare it with real speech, and compare it across requested genders, ages, accents, and emotions together with controllability.

## Reproduction tiers

| Tier | Start from | Reproduces | Cost |
|---|---|---|---|
| T0 | `results/` | every number in the paper's figures and tables | read the CSV/JSON files |
| T1 | released speaker embeddings | all diversity metrics and confidence intervals | hours, 1 GPU |
| T2 | released audio (Hugging Face) | embeddings and controllability | days, GPU |
| T3 | scratch | all generated audio | about 1,200 B200 GPU-hours |

## Layout

```
configs/          systems, encoders, classifiers, and per-experiment parameters
data/             prompt templates, Emilia text ids, LibriSpeech validation subsets (no audio)
generation/       ITTS generation (7 systems)
embeddings/       speaker-embedding extraction (11 encoders)
metrics/          MPCD, Vendi, Chamfer, MMD, half-sampling confidence intervals
controllability/  attribute classifiers and scoring
experiments/      one folder per paper experiment, each with run.sh
results/          the result files the paper reads
docs/             method notes
tools/            migration and filtering helpers
```

## Paper to results

| Paper | Experiment | Result file |
|---|---|---|
| §3.2 encoder validation, Appendix A | `experiments/01_encoder_validation` | `results/encoder_validation/metric_curves.csv` |
| §4.1 comparison with real speech | `experiments/02_overall_diversity` | `results/overall/summary_diversity.csv` |
| §4.2 sampling parameters | `experiments/03_sampling_sweep` | `results/sampling_sweep/param_sweep_seeded_diversity.csv`, `results/overall/real_vendi_n3000.csv` |
| §5 diversity with 95% CI, Appendix E | `experiments/04_attribute_diversity` | `results/attributes/vendi_ci.csv`, `vendi_ci_by_encoder.csv` |
| §5 controllability, Appendices B to D | `experiments/05_controllability` | `results/controllability/summary.json` |
| Ethical Considerations, compute | | `results/compute/generation_speed.csv` |

See `results/README.md` for column definitions.

## License

TODO
