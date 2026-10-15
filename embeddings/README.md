# embeddings/

Turns a folder of WAVs into speaker embeddings with any of the 11 speaker encoders. All diversity metrics are computed from these embeddings.

## Files

| File | What it does | Input → Output | Status |
|---|---|---|---|
| `extract_embeddings.py` | Main script. Loads the chosen encoders and embeds every WAV in a folder. `--models all` runs every encoder available in the current environment. | `--wav_dir <folder>` → `<output_dir>/<tts_model>/<folder_name>/<encoder>/embeddings.npy` and `filenames.txt` (row *i* of the `.npy` is line *i* of the `.txt`; same order for every encoder) | from old repo (`speaker_embedding_eval/eval/extract_embeddings.py`) |
| `encoders/speechbrain.py` | Loaders for SpeechBrain ECAPA, ResNet, and x-vector | | to write (split out of `extract_embeddings.py`; x-vector loader **to upload**) |
| `encoders/wespeaker.py` | Loaders for WeSpeaker ResNet221-LM and VoxBlink2 SimAM-ResNet34 | | to write (split out of `extract_embeddings.py`) |
| `encoders/nemo_titanet.py` | Loader for TitaNet-Large (needs the separate `itts_nemo` environment) | | to write (split out) |
| `encoders/modelscope.py` | Loaders for the three Chinese ModelScope encoders: ERes2Net, ERes2NetV2, CAM++ | | to write (split out) |
| `encoders/hf_sv.py` | Loaders for WavLM-base-plus-sv and UniSpeech-SAT-base-plus-sv (only used in encoder validation) | | **to upload** (whoever ran the LibriSpeech encoder validation) |
| `run_embedding_subset.sh` | Batch driver: extracts embeddings for many `<system>/<category>` folders, resumable | | from old repo (`speaker_embedding_eval/scripts/run_embedding_subset.sh`) |
| `run_embedding_param_sweep.sh` | Batch driver for the sampling-sweep folders | | from old repo (`speaker_embedding_eval/scripts/run_embedding_param_sweep.sh`) |

The exact checkpoint each encoder loads is listed in `configs/encoders.yaml`.

## Usage

```bash
python embeddings/extract_embeddings.py --wav_dir data/outputs/attributes/qwen3-voicedesign/age \
    --tts_model qwen3-voicedesign --output_dir data/embeddings_attributes --models all
```

TitaNet must be run from the `itts_nemo` environment; the other seven paper encoders run in `itts_embed`.
