# results/

The files the paper's figures and tables are read from. They contain only the 7 paper systems
(plus real speech where it applies), the metrics MPCD, Vendi, Chamfer and MMD², and the paper's
attribute values. `experiments/*/run.sh` writes each of them.

Systems: `moss-voicegen`, `parler-large`, `parler-mini`, `capspeech`, `qwen3-voicedesign`,
`voxinstruct`, `voxcpm2`. Encoders (code names, see `configs/encoders.yaml`): `speechbrain_ecapa`,
`speechbrain_resnet`, `wespeaker`, `wespeaker_voxblink2_resnet152_16k` (WSVB, VoxBlink2 SimAM-ResNet34), `nemo_titanet`,
`modelscope_campplus`, `modelscope_eres2net`, `modelscope_eres2netv2`; the encoder validation adds
`speechbrain_xvector`, `wavlm_base_plus_sv`, `unispeech_sat_base_plus_sv`.
MPCD = 1 − mean pairwise cosine similarity; for both MPCD and Vendi, higher = more diverse.

## encoder_validation/ (§3.2, Appendix A)

`metric_curves.csv`: one row per (encoder, metric, k), the same 11 encoders as `configs/encoders.yaml`.

| column | meaning |
|---|---|
| `encoder`, `metric` | 11 encoders; `mpcd`, `vendi`, `chamfer`, `mmd` |
| `k` | number of LibriSpeech speakers in the subset (14 values, 6 to 240) |
| `n_set`, `utts_per_speaker` | 240 utterances per subset = k × utts_per_speaker |
| `value_mean`, `value_std`, `sem` | over the `n_draws` = 100 subsets |
| `faithful_direction` | `up` if a faithful metric should grow with k (MPCD, Vendi), `down` if it should shrink (Chamfer, MMD² to the reference pool) |

Written by `experiments/01_encoder_validation/run.sh`, with
`metric_values.csv` (one row per encoder, metric and subset) and `rank_correlation.csv`:

| column | meaning |
|---|---|
| `spearman_rho` | Spearman ρ between the metric and k over all 1,400 subsets |
| `ratio_k240_k6` | mean at k = 240 / mean at k = 6 |
| `steps_separated`, `n_steps` | adjacent k steps (of 13) whose change in the mean, in the faithful direction, exceeds the larger subset std |
| `monotonic` | the mean moves in the faithful direction at every step |
| `retained` | the encoder's Vendi separates ≥ 12 of 13 steps (the paper's criterion; same for all its metrics) |

## overall/ (§4.1, §4.2)

`summary_diversity.csv`: one row per (set, encoder); `set` is a system or `real`.

| column | meaning |
|---|---|
| `n_partitions` | partitions of 10,000 utterances averaged (3; real speech: 1) |
| `vendi_mean`, `vendi_std` | Vendi on each full partition: mean and population std over partitions |
| `mpcd_mean`, `mpcd_std` | the same for MPCD |

`real_vendi_n3000.csv`: real speech (Emilia partition 1) on 3,000 utterances, the baseline line of
the sampling-sweep figure. Columns `set, encoder, n, seed, vendi, mpcd`.

## sampling_sweep/ (§4.2)

`param_sweep_seeded_diversity.csv`: one row per (system, parameter value, seed, encoder), 360 rows.

| column | meaning |
|---|---|
| `param_kind`, `param` | `temp` (moss-voicegen) or `cfg` (voxinstruct, voxcpm2); values 0.5 to 2.5 |
| `seed` | generation seed: 17, 42, 99 |
| `n` | 3,000 utterances, the same texts in every row |
| `vendi`, `mpcd` | metric values |

## attributes/ (§5, Appendix E)

`vendi_ci.csv`, `mpcd_ci.csv`: one row per (system, category, value), 168 rows; the metric is the
mean over 8 encoders. `*_by_encoder.csv`: the same per encoder (1,344 rows, extra `encoder` column).
Column definitions: `docs/within_value_diversity_ci.md` §7.

## controllability/ (§3.5, §5, Appendices B to D)

`summary.json`: keyed by `<system>/<category>` (35 entries). For each: `n_rows` and, per scored
attribute, `controllability_acc` (accuracy), `uar` (mean per-class recall), `n`,
`per_class_recall` and `confusion` (`{requested: {predicted: count}}`). `gender_x_age` entries also
have `per_cell` (`n`, `age_acc`, `gender_acc`, `both_acc` per requested cell). Emotion is the argmax
over the four requested emotions. Labels are the prompt names (`north_american`, ..., `elderly`), as
in `attributes/`.

## accent_classifier/ (Appendix C)

Test results of our accent classifier (`docs/accent_classifier.md`): `id/` on the held-out
speaker-independent test split (members and ensemble), `ood/` on four out-of-domain corpora, with the
released Vox-Profile accent head as reference.
