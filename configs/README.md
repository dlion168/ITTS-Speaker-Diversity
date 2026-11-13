# configs/

Settings of the paper's systems, encoders, classifiers and experiments.

`generation/run_generation.py` reads `experiments/{overall,sampling_sweep,attributes}.yaml`
(through `generation/settings.py`); `--config PATH` selects another file. `generation/download_weights.py`
and the generation backends download and load each model at the `revision` in `systems.yaml`
(through `generation/model_pins.py`). The other files document the setup; the code that uses each
value says so in its own README.

| File | Content |
|---|---|
| `systems.yaml` | The 7 ITTS systems: Hugging Face id and revision, upstream repository and commit, conda environment, generation settings (CapSpeech-NAR: checkpoint per attribute category); revisions of the supporting checkpoints (MOSS audio tokenizer, CapSpeech vocoder and caption encoder) |
| `manual_checkpoints.json` | Size and SHA-256 of the two VoxInstruct checkpoints (Google Drive), checked after download |
| `release_assets.json` | Revisions of the Hugging Face dataset and of the accent-classifier weights that the download scripts fetch by default |
| `encoders.yaml` | The 11 speaker encoders: folder name, paper abbreviation, checkpoint, embedding size, environment, weights license, and whether it is used for the diversity results |
| `classifiers.yaml` | The 4 controllability classifiers: checkpoint, how outputs map to the requested values, chance level |
| `experiments/encoder_validation.yaml` | §3.2: LibriSpeech split, values of k, subset size, subsets per k, reference pool, seed |
| `experiments/overall.yaml` | §4.1: instruction text, 3 partitions × 10,000 texts, seed |
| `experiments/sampling_sweep.yaml` | §4.2: parameter values, seeds, the parameter swept per system |
| `experiments/attributes.yaml` | §5: attribute values, utterances per value, confidence-interval settings, seed |

For overall and sweep runs, `--instruction` overrides the YAML instruction. Seeds of new overall and
attribute runs resolve as `--seed` > `GENERATION_SEED` > YAML `generation_seed`; sweep seeds come
from the YAML `seeds` list. How the generation driver finds each system's environment is in
`generation/NEW_RUNS.md`.
