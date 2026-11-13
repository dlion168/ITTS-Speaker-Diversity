# Within-value speaker diversity with 95% confidence intervals (Vendi, MPCD)

Paper §5 and Appendix E. For every system and requested attribute value (for example
qwen3-voicedesign with `age = child`), how diverse are the generated voices, and how large is the
sampling error of that estimate?

- Code: `metrics/half_sampling_ci.py` (also `metrics/compute_metrics.py ci`)
- Run: `experiments/04_attribute_diversity/run.sh`
- Output: `results/attributes/`

The reported number is the **mean over 8 speaker encoders**; each encoder's own numbers are kept
in the same files (§7).

## 1. Data

### 1.1 Audio

Each system generated speech from the prompts in `data/prompts/`, one folder per category:
`data/outputs/attributes/<system>/<category>/`. The `style_prompt` of each prompt asks for the
attribute value (e.g. *"Speak in a child's voice."*); `text` is an Emilia transcript. Within one
value the 3,000 texts are almost all distinct (each text appears at most twice), so utterances
are treated as independent samples and no grouping by prompt is needed.

| category | values | utterances per value |
| -------- | ------ | -------------------: |
| gender | female / male | 3,000 |
| age | child / young_adult / middle_aged / elderly | 3,000 |
| accent | british_isles / germanic / north_american / oceanian / south_african / south_asian | 3,000 |
| emotion | angry / happy / neutral / sad | 3,000 |
| gender_x_age | {female, male} × {child, young_adult, middle_aged, elderly} | **6,000** |

24 values per system, **168 values** for the 7 systems.

### 1.2 Speaker embeddings

The eight encoders kept after validation (paper §3.2; checkpoints in `configs/encoders.yaml`):
`speechbrain_ecapa` (192-d), `speechbrain_resnet` (256), `wespeaker` (256),
`wespeaker_voxblink2_resnet152_16k` (256), `nemo_titanet` (192), `modelscope_campplus` (192),
`modelscope_eres2net` (192), `modelscope_eres2netv2` (192).

```
data/embeddings_attributes/<system>/<category>/<encoder>/
    embeddings.npy    # (utterances, 192 or 256)
    filenames.txt     # one per row, e.g. age-child-s1-0001-T1
```

None of the files has NaN or Inf rows, every value has its full 3,000 (or 6,000) utterances, and
within one (system, category) all eight encoders list the files in the same order.

## 2. Metrics

Every embedding is first L2-normalised, so a dot product is a cosine similarity.

**Vendi score** (higher = more diverse), `metrics/vendi.py`:

$$
\mathrm{Vendi}(X) = \exp\Big(-\sum_i \lambda_i \log \lambda_i\Big),\qquad
\lambda = \mathrm{eig}(K/N),\ K_{ij} = \cos(x_i, x_j)
$$

Roughly "how many distinct speakers this set amounts to", between 1 and N. **Vendi grows with N**,
so every cell uses the same **N = 1,500**.

**MPCD**, mean pairwise cosine distance (higher = more diverse), `metrics/mpcd.py`:

$$
\mathrm{MPCD}(X) = \frac{1}{N(N-1)} \sum_{i \ne j} \big(1 - \cos(x_i, x_j)\big)
$$

The diagonal is excluded. MPCD does not change systematically with N.

## 3. Procedure

For one cell, e.g. qwen3-voicedesign / age / child with 3,000 utterances:

1. **Group.** Read the value from each file name (`age-child-s1-0001-T1` → `child`).
2. **Normalise** every embedding to unit length.
3. **Subsample** 1,500 of the 3,000 utterances **without replacement**.
4. **Score** Vendi and MPCD on those 1,500 (both metrics on the same subset).
5. **Repeat** steps 3–4 **B = 10,000** times.
6. **Point estimate:** the mean of the 10,000 values (`vendi` / `mpcd` columns).
7. **95% interval:** the 2.5th and 97.5th percentiles of the 10,000 values after the rescaling
   of §4.3 (`ci_lo` / `ci_hi`).

Steps 1–7 run once per encoder, with the same seed (`--seed 0`) for every cell, so results are
exactly reproducible.

**Averaging the encoders.** All encoders share the file order and the seed, so draw b selects
**the same 1,500 utterances** for all eight. The eight values of draw b are averaged, giving
10,000 draws of the encoder mean, and steps 6–7 are applied to those. Each encoder's own values
are in the per-encoder columns (§7).

## 4. Why this subsampling

### 4.1 Half-sampling rather than, say, 2,500 of 3,000

Subsets drawn without replacement overlap; the more they overlap, the closer their values and the
narrower the interval, which then only reflects the choice of utterances, not the error of
generating a new batch. For an approximately linear statistic, drawing N of n:

- variance between subsets ≈ (1 − N/n) · Var_N
- variance of the estimate itself ≈ (N/n) · Var_N

The two are equal at **N = n/2**, so the subset percentiles can be used as the interval directly.

### 4.2 Why not the ordinary bootstrap

Resampling 3,000 with replacement repeats about 37% of the utterances. Repeated voices lower
Vendi systematically, so the whole interval shifts down and may not even contain the full-set value.

### 4.3 Rescaling for gender_x_age

The gender_x_age cells have n = 6,000, so N = 1,500 is not half and the subsets spread too much.
Each draw's deviation from the mean is scaled by

$$
c = \sqrt{\frac{N}{n - N}}
$$

before taking percentiles: `adj = mean + c · (draw − mean)`. c = 1 for n = 3,000 and 0.577 for
n = 6,000; the `c` column records it for every row.

## 5. Computation

Because the rows are unit-norm, K = XXᵀ (N × N) and XᵀX (d × d, d = 192 or 256) have the same
non-zero eigenvalues, so Vendi uses the d × d matrix. MPCD only needs the sum s of the subset:

$$
\mathrm{MPCD} = 1 - \frac{\lVert s \rVert^2 - N}{N(N-1)}
$$

Subsets are processed 500 at a time on the GPU.

## 6. Running

```bash
python metrics/half_sampling_ci.py --emb_root data/embeddings_attributes --out_dir results/attributes
```

| option | default | meaning |
| ------ | ------- | ------- |
| `--encoders` | the 8 paper encoders | encoders to average |
| `--systems` | every folder in `--emb_root` | systems to score |
| `--categories` | the 5 categories | categories to score |
| `--N` | 1500 | utterances per subset |
| `--B` | 10000 | number of subsets |
| `--chunk` | 500 | subsets per GPU batch |
| `--seed` | 0 | random seed |

## 7. Output files

`results/attributes/`

| file | content |
| ---- | ------- |
| `vendi_ci.csv` | encoder mean, one row per value (168 rows), plus one point-estimate column per encoder |
| `mpcd_ci.csv` | the same for MPCD |
| `vendi_ci_by_encoder.csv` | every encoder separately (168 × 8 = 1,344 rows), with an `encoder` column |
| `mpcd_ci_by_encoder.csv` | the same for MPCD |
| `vendi_draws.npz`, `mpcd_draws.npz` | the 10,000 encoder-mean draws, key `<system>/<category>/<value>` (written by the script; not committed) |
| `*_draws_by_encoder.npz` | the 10,000 draws per encoder, key `<encoder>/<system>/<category>/<value>` (not committed) |

Columns of `vendi_ci.csv` / `mpcd_ci.csv`:

| column | meaning |
| ------ | ------- |
| `system`, `category`, `value` | the cell |
| `n` | utterances in the cell (3,000 or 6,000) |
| `N`, `B` | subset size, number of subsets |
| `c` | interval rescaling factor (§4.3) |
| `n_encoders` | encoders averaged (8) |
| `vendi` / `mpcd` | point estimate of the encoder mean |
| `ci_lo`, `ci_hi` | 95% interval of the encoder mean |
| `se` | standard error, c × standard deviation of the draws |
| `vendi__<encoder>` / `mpcd__<encoder>` | that encoder's point estimate (8 columns) |

`*_by_encoder.csv` has `encoder` plus the same columns except `n_encoders` and `__<encoder>`.
The draws allow paired comparisons: compute, say, child − middle_aged for every draw and check
whether the 2.5–97.5 percentiles of the difference include 0.

## 8. Example: age (mean of 8 encoders)

**Vendi [95% CI]** (higher = more diverse)

| system | child | young_adult | middle_aged | elderly |
| --- | --- | --- | --- | --- |
| moss-voicegen | 25.1 [24.5, 25.6] | 55.5 [54.9, 56.1] | 57.5 [56.9, 58.2] | 57.8 [57.1, 58.4] |
| parler-large | 46.7 [46.2, 47.3] | 46.3 [45.7, 46.9] | 48.2 [47.6, 48.8] | 49.0 [48.5, 49.5] |
| parler-mini | 39.5 [39.0, 40.0] | 41.1 [40.6, 41.6] | 41.2 [40.7, 41.7] | 40.2 [39.7, 40.8] |
| capspeech | 23.9 [23.4, 24.4] | 17.0 [16.5, 17.4] | 27.4 [26.8, 27.9] | 26.9 [26.4, 27.4] |
| qwen3-voicedesign | **5.5** [5.4, 5.6] | 15.3 [14.8, 15.7] | 31.4 [30.8, 32.0] | 33.6 [32.9, 34.3] |
| voxinstruct | 19.7 [19.2, 20.2] | 22.8 [22.3, 23.3] | 25.5 [25.0, 26.0] | 28.4 [27.8, 28.9] |
| voxcpm2 | 23.7 [23.1, 24.3] | 15.6 [15.1, 16.0] | 12.8 [12.5, 13.2] | 16.3 [15.9, 16.7] |

**MPCD [95% CI]** (higher = more diverse)

| system | child | young_adult | middle_aged | elderly |
| --- | --- | --- | --- | --- |
| moss-voicegen | 0.611 [0.606, 0.616] | 0.741 [0.739, 0.743] | 0.741 [0.739, 0.744] | 0.750 [0.747, 0.752] |
| parler-large | 0.727 [0.725, 0.730] | 0.725 [0.722, 0.728] | 0.736 [0.733, 0.738] | 0.734 [0.731, 0.736] |
| parler-mini | 0.696 [0.692, 0.699] | 0.705 [0.702, 0.707] | 0.703 [0.700, 0.706] | 0.699 [0.695, 0.702] |
| capspeech | 0.577 [0.573, 0.581] | 0.509 [0.504, 0.515] | 0.596 [0.592, 0.601] | 0.597 [0.592, 0.601] |
| qwen3-voicedesign | **0.287** [0.283, 0.291] | 0.502 [0.495, 0.509] | 0.633 [0.629, 0.637] | 0.662 [0.659, 0.666] |
| voxinstruct | 0.531 [0.524, 0.537] | 0.571 [0.566, 0.575] | 0.591 [0.586, 0.595] | 0.620 [0.616, 0.623] |
| voxcpm2 | 0.638 [0.634, 0.642] | 0.501 [0.495, 0.507] | 0.459 [0.453, 0.465] | 0.511 [0.506, 0.516] |

- Intervals are narrow: the Vendi half-width is 1.5% of the point estimate at the median, 3.0% at most.
- Large gaps are reliable, e.g. qwen3-voicedesign's child voices (Vendi 5.5, MPCD 0.287) are far
  less diverse than its middle-aged voices (31.4, 0.633).
- Small gaps need non-overlapping intervals before they are read as differences.

### 8.1 Agreement between encoders

qwen3-voicedesign / age, Vendi per encoder:

| encoder | child | young_adult | middle_aged | elderly |
| ------- | ----: | ----------: | ----------: | ------: |
| speechbrain_ecapa | 7.8 | 21.3 | 54.0 | 54.5 |
| speechbrain_resnet | 6.1 | 21.2 | 46.6 | 49.6 |
| wespeaker | 5.2 | 18.3 | 41.5 | 45.2 |
| nemo_titanet | 7.9 | 18.2 | 41.7 | 43.7 |
| wespeaker_voxblink2_resnet152_16k | 4.3 | 15.0 | 28.2 | 29.6 |
| modelscope_eres2netv2 | 4.3 | 10.2 | 13.4 | 15.6 |
| modelscope_campplus | 4.4 | 9.1 | 13.8 | 17.1 |
| modelscope_eres2net | 4.2 | 8.8 | 11.8 | 13.7 |

Spearman ρ with `speechbrain_ecapa` over the 168 values:

| encoder | Vendi | MPCD |
| ------- | ----: | ---: |
| speechbrain_resnet | 0.992 | 0.974 |
| wespeaker | 0.988 | 0.979 |
| nemo_titanet | 0.988 | 0.974 |
| wespeaker_voxblink2_resnet152_16k | 0.973 | 0.955 |
| modelscope_campplus | 0.946 | 0.916 |
| modelscope_eres2netv2 | 0.942 | 0.917 |
| modelscope_eres2net | 0.934 | 0.908 |

