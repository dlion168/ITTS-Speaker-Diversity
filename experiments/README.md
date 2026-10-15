# experiments/

One folder per experiment in the paper. Each folder has a `run.sh` that calls the modules (`generation/`, `embeddings/`, `metrics/`, `controllability/`) in order and writes its results to `results/`.

`run.sh` takes `--from scratch | audio | embeddings`, so a reader can start from generated audio or from released embeddings instead of regenerating everything.

| Folder | Paper | Result written to |
|---|---|---|
| `01_encoder_validation/` | §3.2, Appendix A | `results/encoder_validation/` |
| `02_overall_diversity/` | §4.1 | `results/overall/` |
| `03_sampling_sweep/` | §4.2 | `results/sampling_sweep/` |
| `04_attribute_diversity/` | §5, Appendix E | `results/attributes/` |
| `05_controllability/` | §3.5, §5, Appendices B to D | `results/controllability/` |
