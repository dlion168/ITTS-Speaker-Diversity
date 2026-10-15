# 01_encoder_validation (paper §3.2, Appendix A)

Checks which speaker encoders can measure diversity. LibriSpeech subsets with a known number of speakers k are embedded with all 11 encoders; an encoder is kept if its diversity score grows clearly with k.

## Steps in `run.sh`

1. Build 1,400 subsets: 14 values of k, 100 subsets each, 240 utterances per subset (`build_subsets.py`).
2. Embed every subset with the 11 encoders (`embeddings/extract_embeddings.py`).
3. Compute MPCD and Vendi per subset, and Chamfer and MMD against the reference pool (`compute_curves.py`).
4. Compute the Spearman correlation between each metric and k over all 1,400 subsets (`rank_correlation.py`).

## Files

| File | What it does | Status |
|---|---|---|
| `run.sh` | Runs steps 1 to 4 | to write |
| `build_subsets.py` | Samples the subsets and writes `data/librispeech_validation/subsets.jsonl` | **to upload** (whoever ran the LibriSpeech encoder validation) |
| `compute_curves.py` | Writes `results/encoder_validation/metric_curves.csv` (mean, std, sem per encoder, metric, k) | **to upload** (same person) |
| `rank_correlation.py` | Writes `results/encoder_validation/rank_correlation.csv` (Spearman ρ per encoder and metric) | **to upload** (same person) |

## Inputs and outputs

- Input: LibriSpeech (see `configs/experiments/encoder_validation.yaml`)
- Output: `results/encoder_validation/metric_curves.csv`, `rank_correlation.csv`
