# embeddings/

Turns a folder of WAVs into speaker embeddings with any of the 11 speaker encoders, or fetches the
released embeddings. All diversity metrics are computed from these embeddings.

The download scripts fetch the dataset revision in `configs/release_assets.json` unless
`HF_REVISION` (or `--revision`) is set. They do not overwrite existing files that differ from the
download; use a new `--out_root` in that case.

## Files

| File | What it does | Input → Output |
|---|---|---|
| `extract_embeddings.py` | Main script. Embeds every WAV in a folder with the chosen encoders; re-running resumes. `--models all` = the 8 paper encoders, `--models validation` = the 3 validation-only encoders. | `--wav_dir <folder>` → `<output_dir>/<tts_model>/<folder name>/<encoder>/embeddings.npy` and `filenames.txt` (row *i* of the `.npy` is line *i* of the `.txt`; rows sorted by file name, same order for every encoder) |
| `encoders/__init__.py` | Encoder registry: name → loader module, imported lazily so each environment only needs its own toolkit | |
| `encoders/speechbrain.py` | SpeechBrain ECAPA, ResNet (paper) and x-vector (validation) | |
| `encoders/wespeaker.py` | WeSpeaker ResNet221-LM (`wespeaker`) and VoxBlink2 SimAM-ResNet34 (folder name `wespeaker_voxblink2_resnet152_16k`) | |
| `encoders/nemo_titanet.py` | TitaNet-Large (needs the separate `itts_nemo` environment) | |
| `encoders/modelscope.py` | The three Mandarin ModelScope encoders: ERes2Net, ERes2NetV2, CAM++ | |
| `encoders/hf_sv.py` | WavLM-base-plus-sv and UniSpeech-SAT-base-plus-sv (encoder validation only) | |
| `audio.py`, `compat.py` | Audio loading and padding; shims for newer `huggingface_hub` / `torchaudio` that NeMo and WeSpeaker still expect | |
| `run_embedding_subset.sh` | Batch driver for the attribute experiment: `--task <system>/<category>` (or `<system>/all`), all 8 encoders, TitaNet in its own env. Several workers can share one output root (lock per task); finished outputs are verified and skipped. | `data/outputs/attributes/` → `data/embeddings_attributes/` |
| `run_embedding_param_sweep.sh` | Batch driver for the sampling sweep: `--task <system>/<param>/<seed>` | `data/outputs/param_sweep/` → `data/embeddings_param_sweep/<system>/<param>/<seed>/<encoder>/` |
| `download_released.py` | Downloads the released embeddings from Hugging Face and writes them in the layout above (for `--from embeddings`) | HF parquet → `data/embeddings_{attributes,overall,param_sweep}/` |
| `download_released_audio.py` | Downloads the released audio and writes WAVs in the generation layout (for `--from audio`; gated, needs `HF_TOKEN`) | HF parquet (FLAC) → `data/outputs/{attributes,overall,param_sweep}/` |

The exact checkpoint each encoder loads is listed in `configs/encoders.yaml`; the encoder validation
uses the same checkpoints.

## Usage

```bash
python embeddings/extract_embeddings.py --wav_dir data/outputs/attributes/qwen3-voicedesign/age \
    --tts_model qwen3-voicedesign --output_dir data/embeddings_attributes --models all
```

TitaNet runs in `itts_nemo` (`envs/itts_nemo.lock.txt`); the other seven paper encoders and all
metrics run in `itts_embed` (`envs/itts_embed.lock.txt`).

## Notes

- The released embeddings used batch 16 for both SpeechBrain encoders and 256 for ModelScope, over
  the whole folder in file-name order; the batch drivers do the same.
- ModelScope and WeSpeaker embeddings can differ slightly between runs because of GPU
  nondeterminism.
