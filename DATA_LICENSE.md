# Data license

The code in this repository is under the MIT License (`LICENSE`). The data files listed below are
under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0,
https://creativecommons.org/licenses/by-nc/4.0/legalcode):

- `data/` (Emilia texts and utterance ids, attribute instructions)
- `results/` (the result files behind the paper's tables and figures)
- `controllability/accent_classifier/data_build/source_manifests/` (clip lists of the accent classifier)

Text and identifiers taken from third-party corpora keep the license of their corpus:

| Corpus | Where it appears | License |
|---|---|---|
| Emilia | `data/emilia/`, `data/prompts/` | CC BY-NC 4.0 |
| CommonVoice 17 | `source_manifests/cv17_utt_map.csv`, `source_manifests/splits_6cls/` | CC0 1.0 |
| VoxPopuli | `source_manifests/germanic_voxpopuli.csv`, `voxpopuli_utt_map.csv`, `splits_6cls/` | CC0 |
| L2-ARCTIC | `source_manifests/southasia_l2arctic*.csv`, `splits_6cls/` | CC BY-NC 4.0 |
| AfriSpeech-200 | `source_manifests/splits_6cls/` | CC BY-NC-SA 4.0 |
| EdAcc | `source_manifests/ood_edacc*.csv` | CC BY-SA 4.0 |
| NCHLT English | `source_manifests/ood_nchlt*.csv` | CC BY 3.0 |

The generated audio and speaker embeddings released on Hugging Face
(`dlion168/itts-speaker-diversity`) are also CC BY-NC 4.0.
