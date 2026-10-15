# envs/

The systems and tools need incompatible package versions, so each group has its own environment. Every file here should be **exported from the machine that actually ran the experiment** (`conda env export` or `pip freeze`), not written by hand.

| File | Used for | Status |
|---|---|---|
| `setup_envs.sh` | Creates all generation environments | from old repo (`setup_envs.sh`) |
| `itts_parler.yml` | Parler-TTS Mini v1 and Large v1 | **to upload** (whoever ran generation) |
| `itts_moss.yml` | MOSS-VoiceGenerator | **to upload** |
| `itts_qwen.yml` | Qwen3-TTS | **to upload** |
| `itts_voxcpm2.yml` | VoxCPM2 | **to upload** |
| `itts_capspeech.yml` | CapSpeech-NAR | **to upload** |
| `itts_vox.yml` | VoxInstruct | **to upload** |
| `itts_embed.lock.txt` | 7 speaker encoders and all metrics (Python 3.11, wespeaker at commit `dfa7419`) | **to upload** (whoever ran embedding extraction) |
| `itts_nemo.lock.txt` | TitaNet | **to upload** |
| `itts_validation.lock.txt` | x-vector, WavLM-SV, UniSpeech-SAT-SV for encoder validation | **to upload** (whoever ran the LibriSpeech encoder validation) |
| `vox_profile.lock.txt` | Controllability classifiers (Python 3.10, torch 2.8.0+cu128, transformers 4.46.3, speechbrain 1.0.3) | **to upload** (whoever ran controllability scoring) |
