# 02_overall_diversity (paper §4.1)

Compares the speaker diversity of each system under a neutral instruction with real recordings of the same texts.

## Steps in `run.sh`

1. Generate 3 partitions × 10,000 utterances per system with the neutral instruction (`generation/infer.py`).
2. Download the matching real Emilia recordings (`data/emilia/download_reference_audio.py`).
3. Embed generated and real audio with the 8 retained encoders (`embeddings/`).
4. Compute Vendi and MPCD per partition and encoder, then the mean and std over partitions (`metrics/`, `summarize.py`).

## Files

| File | What it does | Status |
|---|---|---|
| `run.sh` | Runs steps 1 to 4 | to write (start from `old_run_infer.sh`) |
| `old_run_infer.sh` | The original generation script, kept as a reference for writing `run.sh` | from old repo (`run_infer.sh`); delete once `run.sh` works |
| `summarize.py` | Collects per-partition results into `results/overall/summary_diversity.csv` | to write |

## Inputs and outputs

- Input: `data/emilia/overall_partition_{1,2,3}.jsonl`, `configs/experiments/overall.yaml`
- Output: `results/overall/summary_diversity.csv`
