# metrics/

Diversity metrics computed from speaker embeddings, and the confidence intervals used in §5.
Only the four metrics in the paper are implemented.

## Files

| File | What it does | Input → Output |
|---|---|---|
| `mpcd.py` | Mean pairwise cosine distance, `1 - mean cosine similarity` over all pairs i ≠ j. Higher = more diverse. Also the batched form used by the CI code. | `(n, d)` array → one number |
| `vendi.py` | Vendi score with a cosine kernel, `exp(entropy of the eigenvalues of K/n)`. Higher = more diverse; grows with n, so compare equal-size sets only. | `(n, d)` array → one number |
| `chamfer.py` | Chamfer distance between a set and a reference pool, Euclidean on raw embeddings, both directions (encoder validation only) | two arrays → one number |
| `mmd.py` | MMD² with a Gaussian kernel, median-heuristic bandwidth, biased estimator (encoder validation only) | two arrays → one number |
| `half_sampling_ci.py` | Draws 10,000 random subsets of N = 1,500 from each attribute value, averages the 8 encoders draw by draw, and reports the mean and the 2.5/97.5 percentiles (rescaled by `sqrt(N/(n-N))` for the 6,000-utterance gender × age cells) | `data/embeddings_attributes/` → `vendi_ci.csv`, `mpcd_ci.csv`, per-encoder versions, draws |
| `compute_metrics.py` | Command-line entry point: `diversity` (MPCD, Vendi of embedding sets, optional fixed-seed subset), `cross` (Chamfer, MMD²), `ci` (= `half_sampling_ci.py`) | |

## Usage

```bash
python metrics/compute_metrics.py diversity data/embeddings_overall/voxcpm2/sample1
python metrics/compute_metrics.py diversity data/embeddings_param_sweep/voxcpm2/cfg_1.0/seed_17 --n 3000
python metrics/compute_metrics.py ci --emb_root data/embeddings_attributes --out_dir results/attributes
```

The CI method is described in `docs/within_value_diversity_ci.md`.
