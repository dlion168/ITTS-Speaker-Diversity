# tests/

Tests that need no models, downloads or GPU. Run them in the `itts_embed` environment
(PyYAML, PyArrow, soundfile and huggingface_hub are required):

```bash
python -m unittest discover -s tests -v
python metrics/tests/test_diversity_metrics.py          # analytic checks of the four metrics
python -m unittest discover -s generation/tests -v      # generation plans and seeding (needs PyTorch)
```

| File | Checks |
|---|---|
| `test_generation_configuration.py` | YAML and command-line precedence, instructions passed to inference, sweep seeds and flags, prompt selection, output checks of the generation driver |
| `test_accent_source_validation.py` | Source coverage and schema checks of the accent-classifier split builder, speaker-disjoint splits, no overwriting of existing outputs |
| `test_controllability_labels.py` | `merge_shards.py` and `results/controllability/summary.json` use the prompt names (`north_american`, `elderly`, ...) |
| `test_experiment_entrypoints.py` | `run.sh --from scratch` is rejected; folders without WAVs are not taken as generated audio |
| `test_sampling_sweep_contract.py` | Experiment 03: system selection, folder layout of audio and embeddings, metric values on toy arrays, worker exit codes |
| `test_configs_and_envs.py` | `configs/systems.yaml` format, the real-audio download default, `envs/setup_generation_envs.sh` |
| `test_weight_download_contract.py` | `generation/download_weights.py` and `model_pins.py`: one weights folder, recorded revisions, incomplete downloads are not reused, existing files are not overwritten (Hugging Face calls are mocked) |
| `test_entrypoint_checks.py` | Experiments 04 and 05 stop when a selected category has no audio; fresh result folders |
| `test_scoring_coverage.py` | Controllability scoring: all shards present, no duplicate clips, resume only with identical inputs and settings, no silent fallback from the accent ensemble |
| `test_release_assets.py` | Default dataset and classifier revisions, released embeddings and audio written back without overwriting different files, the CI workflow |
| `test_original_shard_validation.py` | `verify_original_shards.py` on small synthetic shards |
| `test_cv17_sidecar.py` | `source_manifests/cv17_utt_map.csv` covers all 115,443 CV17 clips of the released splits |
