# metrics/

Diversity metrics computed from speaker embeddings, and the confidence intervals used in §5.

## Files

| File | What it does | Input → Output | Status |
|---|---|---|---|
| `mpcd.py` | Mean pairwise cosine distance, `1 - mean(cosine similarity)` over all pairs. Higher = more diverse. | `embeddings.npy` → one number | to write (split out of `compute_metrics.py`) |
| `vendi.py` | Vendi Score with a cosine kernel: `exp(entropy of the eigenvalues of K/N)` | `embeddings.npy` → one number | to write (split out) |
| `chamfer.py` | Chamfer distance between a set and a reference pool (encoder validation only) | two embedding sets → one number | to write (split out) |
| `mmd.py` | MMD² with a Gaussian kernel, median-heuristic bandwidth, biased estimator (encoder validation only) | two embedding sets → one number | to write (split out) |
| `half_sampling_ci.py` | Draws 10,000 random subsets of N = 1,500 from each attribute value, averages the 8 encoders within each draw, and returns the mean and the 2.5/97.5 percentiles (rescaled by `sqrt(N/(n-N))` for the 6,000-utterance gender × age cells) | embeddings of all attribute values → `vendi_ci.csv`, `mpcd_ci.csv`, per-encoder versions | to write (split out of the `ci` subcommand) |
| `compute_metrics.py` | Command-line wrapper around the five files above. Will be slimmed down once they exist; remove `erank` (not in the paper). | | from old repo (`speaker_embedding_eval/eval/compute_metrics.py`) |

## Usage

```bash
python metrics/compute_metrics.py ci --emb_root data/embeddings_attributes --out_dir results/attributes
```

The method is described in `docs/within_value_diversity_ci.md`.
