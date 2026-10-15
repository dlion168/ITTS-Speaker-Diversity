# configs/

All settings in one place. Scripts read these files instead of hard-coding values. Fields marked `TODO` still need the real value.

| File | Content | Status |
|---|---|---|
| `systems.yaml` | The 7 ITTS systems: Hugging Face id, revision, upstream repo commit, environment, default decoding parameters | in repo; revisions, commits, and decoding parameters **to fill** (whoever ran generation) |
| `encoders.yaml` | The 11 speaker encoders: code name, paper abbreviation, the checkpoint actually loaded, embedding size, environment, weights license, and whether it was kept after validation | in repo; x-vector checkpoint and ModelScope licenses **to fill** |
| `classifiers.yaml` | The 4 controllability classifiers: checkpoint, how outputs map to our labels, chance level | in repo |
| `experiments/encoder_validation.yaml` | §3.2: LibriSpeech split, values of k, subset size, subsets per k, reference pool, seed | in repo; corpus, reference pool, seed **to fill** (whoever ran the LibriSpeech encoder validation) |
| `experiments/overall.yaml` | §4.1: instruction text, 3 partitions × 10,000 texts | in repo; **verify** the instruction actually used |
| `experiments/sampling_sweep.yaml` | §4.2: parameter values, seeds, which flag was swept per system | in repo; seeds and the VoxInstruct flag **to fill** |
| `experiments/attributes.yaml` | §5: attribute values, utterances per value, confidence-interval settings | in repo |
