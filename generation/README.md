# generation/

Generates speech with the 7 ITTS systems in the paper. Every experiment calls `infer.py`; nothing else in the repo generates audio.

**Status legend**: *in repo* = already here; *from old repo* = copied by `tools/migrate_from_old_repo.sh` (old path given); *to write* = new code needed; *to upload* = someone has it locally and must add it.

## Files

| File | What it does | Input → Output | Status |
|---|---|---|---|
| `infer.py` | Single entry point for all systems. Loads one backend, reads a list of texts and instructions, writes one WAV per row. Flags: `--model`, `--input`, `--output_dir`, `--batch_size`, plus per-system decoding flags (`--moss_audio_temperature`, `--voxcpm_cfg_value`, VoxInstruct cfg flags). | CSV/JSON/JSONL with columns `id, text, style_prompt` → `<output_dir>/<model>/<id>.wav` | from old repo (`infer.py`) |
| `backends/__init__.py` | `MODEL_REGISTRY`: maps the CLI name (e.g. `qwen3-voicedesign`) to its backend class and weights. **Keep only the 7 paper systems.** | | from old repo (`model_backends/__init__.py`), then edit |
| `backends/base.py` | Abstract class `ITTSBackend` with `load()` and `synthesize(text, style_prompt) -> (waveform, sample_rate)`. Every backend implements it. | | from old repo (`model_backends/base.py`) |
| `backends/moss_voicegen_model.py` | MOSS-VoiceGenerator backend | | from old repo |
| `backends/parler_model.py` | Parler-TTS Mini v1 and Large v1 backend (same code, different weights) | | from old repo |
| `backends/qwen3_tts_model.py` | Qwen3-TTS Voice Design 1.7B backend | | from old repo |
| `backends/voxinstruct_model.py` | VoxInstruct backend | | from old repo |
| `backends/voxcpm2_model.py` | VoxCPM2 backend | | from old repo |
| `backends/capspeech_model.py` | CapSpeech-NAR backend | | from old repo |
| `patches/voxinstruct/` | Our changes to the official VoxInstruct code, with a README on how to apply them | | from old repo (`patches/voxinstruct/`) |
| `download_weights.py` | Downloads all Hugging Face weights into `weights/`; `--check` lists what is already downloaded | → `weights/` | from old repo |
| `upstream_commits.txt` | One line per cloned upstream repo (CapSpeech, MOSS-TTS, VoxCPM, VoxInstruct): repo URL and the exact commit used. Get it with `git -C <clone> rev-parse HEAD` on the machine that ran generation. | | **to upload** (whoever ran generation) |

## Usage

```bash
python generation/infer.py --list-models
python generation/infer.py --model qwen3-voicedesign --input data/emilia/overall_partition_1.jsonl \
    --output_dir data/outputs/overall --batch_size 16
```

Each system needs its own environment (see `envs/`). VoxInstruct weights are on Google Drive and must be downloaded by hand; `patches/voxinstruct/README.md` explains the setup.

## Output file names

The file name carries the label, and the evaluation code parses it, so do not rename outputs.
- single attribute: `<attr>-<value>-s<k>-<idx>-<take>.wav`, e.g. `age-child-s1-0001-T1.wav`
- gender × age: `genderage-<gender>_<age>-s<k>-<idx>-<frame>_<order>.wav`
