# 03_sampling_sweep (paper §4.2)

Tests whether the gap to real speech comes from default decoding settings, by sweeping temperature (MOSS-VoiceGenerator) and classifier-free guidance (VoxInstruct, VoxCPM2).

## Steps in `run.sh`

1. Generate 10,000 utterances for each of 5 values × 3 seeds × 3 systems (`generation/infer.py` with the decoding flags in `configs/experiments/sampling_sweep.yaml`).
2. Embed with the 8 retained encoders (`embeddings/run_embedding_param_sweep.sh`).
3. Compute Vendi and MPCD per configuration (`metrics/`, `summarize.py`).
4. Compute real speech at the same sample size for the baseline line in the figure.

## Files

| File | What it does | Status |
|---|---|---|
| `run.sh` | Runs steps 1 to 4 | to write (start from the two `old_run_*.sh`) |
| `old_run_param_sweep_b200_1.sh` | Original sweep script for MOSS-VoiceGenerator and VoxCPM2 | from old repo; delete once `run.sh` works |
| `old_run_param_sweep_b200_2.sh` | Original sweep script for VoxInstruct | from old repo; delete once `run.sh` works |
| `summarize.py` | Writes `results/sampling_sweep/param_sweep_seeded_diversity.csv` and `results/overall/real_vendi_n3000.csv` | to write |

## Inputs and outputs

- Input: `data/emilia/`, `configs/experiments/sampling_sweep.yaml`
- Output: `results/sampling_sweep/param_sweep_seeded_diversity.csv`, `results/overall/real_vendi_n3000.csv`
