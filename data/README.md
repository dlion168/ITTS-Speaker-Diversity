# data/

Small text files only. Audio is released on Hugging Face, and large intermediate files (generated WAVs, embeddings) live outside the repo.

| File | Content | Status |
|---|---|---|
| `prompts/single_attribute.json` | Instructions for gender (2 values), age (4), accent (6), emotion (4): 5 templates per value, entries `{id, text, style_prompt, language}` | from old repo (`attribute category prompts/{gender,age,accent,emotion}.json`), filtered by `tools/filter_results.py` to drop `non_binary` |
| `prompts/gender_x_age.json` | Instructions for the 8 gender × age cells: 10 templates per cell | from old repo (`attribute category prompts/gender_x_age.json`) |
| `prompts/raw/` | Unfiltered copies of the old prompt files, kept for checking the filter | from old repo |
| `emilia/overall_partition_{1,2,3}.jsonl` | The 3 × 10,000 Emilia texts for §4.1, with Emilia utterance ids | from old repo (`emilia_en_samples/sample_{1,2,3}.jsonl`) |
| `emilia/download_reference_audio.py` | Downloads the real Emilia recordings used as the real-speech reference (gated dataset, needs `HF_TOKEN`) | from old repo (`download_emilia_prompts.py`) |
| `librispeech_validation/subsets.jsonl` | The 1,400 encoder-validation subsets: `k`, `subset_id`, and 240 LibriSpeech utterance ids each | **to upload** (whoever ran the LibriSpeech encoder validation) |
| `librispeech_validation/reference_pool.txt` | LibriSpeech utterance ids of the reference pool used for Chamfer and MMD | **to upload** (same person) |
