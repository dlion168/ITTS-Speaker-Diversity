# experiments/

One folder per experiment in the paper. Shared settings (data root, environments, Hugging Face
dataset) are in `common.sh`. Experiments 02 to 04 have two scripts:

- `generate.sh` generates the audio with `generation/` into `$ITTS_DATA_ROOT/outputs/` (generation
  environments and model weights, many GPU-hours). Options such as `--dry-run` or `--limit N` go to
  `generation/run_<experiment>.sh`.
- `run.sh` evaluates (`embeddings/`, `metrics/`, `controllability/`) and writes its results to
  `results/`. It takes `--from generated | audio | embeddings`, so a reader can start from the
  released audio or the released embeddings instead of generating anything:

| `--from` | uses | needs |
|---|---|---|
| `generated` | the audio `generate.sh` wrote; embeds / scores it, computes the metrics | GPU |
| `audio` | downloads the released audio; embeds / scores it, computes the metrics | `HF_TOKEN` (gated dataset), GPU |
| `embeddings` | downloads the released speaker embeddings, computes the metrics | `HF_TOKEN`; a GPU for `04` |

| Folder | Paper | Result written to | Scripts |
|---|---|---|---|
| `01_encoder_validation/` | §3.2, Appendix A | `results/encoder_validation/` | `prepare_data.sh` downloads LibriSpeech and builds the subsets, then `run.sh` (no `--from`) |
| `02_overall_diversity/` | §4.1 | `results/overall/summary_diversity.csv` | `generate.sh`; `run.sh` all three |
| `03_sampling_sweep/` | §4.2 | `results/sampling_sweep/`, `results/overall/real_vendi_n3000.csv` | `generate.sh`; `run.sh` all three |
| `04_attribute_diversity/` | §5, Appendix E | `results/attributes/` | `generate.sh`; `run.sh` all three |
| `05_controllability/` | §3.5, §5, Appendices B to D | `results/controllability/summary.json` | `run.sh` `generated` (the audio of 04's `generate.sh`) or `audio` |

Each run writes to a new folder, `results/reproductions/run-<time>-<pid>/`, so the committed files in
`results/` stay as they are and can be compared with the rerun. Set `RESULTS_DIR` to collect several
experiments in one folder.
