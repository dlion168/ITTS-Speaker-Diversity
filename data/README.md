# data/

Small text files only. Audio is released on Hugging Face; generated WAVs, embeddings and other
large intermediate files go to `$ITTS_DATA_ROOT` (default: this folder) and are not committed.

| File | Content |
|---|---|
| `prompts/single_attribute.json` | Instructions for gender (2 values), age (4), accent (6) and emotion (4): 3,000 entries per value from 5 instruction templates, entries `{id, text, style_prompt, language}` |
| `prompts/gender_x_age.json` | Instructions for the 8 gender × age cells: 6,000 entries per cell from 10 templates |
| `emilia/overall_partition_{1,2,3}.jsonl` | The 3 × 10,000 Emilia texts for §4.1, with Emilia utterance ids |
| `emilia/manifest.json` | How the texts were sampled from Emilia-EN (source tars, seeds, filters) |
| `emilia/download_reference_audio.py` | Downloads the real Emilia recordings used as the real-speech reference (gated dataset, needs `HF_TOKEN`); writes `<--out_dir>/set_{1,2,3}/`, default `$ITTS_DATA_ROOT/real_world_data/` |

Experiment 01 writes its LibriSpeech subsets under `$ITTS_DATA_ROOT/librispeech_validation/`
(`subsets.jsonl`: the 1,400 subsets with their 240 utterance ids each; `reference_pool.txt`: one
utterance of each of the 251 speakers), with `experiments/01_encoder_validation/build_subsets.py`
(seed 0).
