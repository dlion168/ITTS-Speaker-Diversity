# Controllability scorer

`controllability/score_controllability.py` measures how well an instruction-guided TTS system
produces the speaker attributes its prompt asked for (paper §3.5). Point it at **one folder per
system and category** (generated `.wav` files plus a CSV with each clip's *requested* attribute);
it runs one classifier per attribute and reports, per attribute, how often the **perceived** value
matches the **requested** one, with per-class recall and the confusion matrix.

- Environment: `vox_profile` (`envs/vox_profile.lock.txt`; Python 3.10, torch 2.8.0+cu128).
- Model code: a checkout of `tiantiaf0627/vox-profile-release` at commit `85100e6`
  (`bash controllability/setup_vox_profile.sh`), at `controllability/vox-profile-release` or `$VOX_PROFILE_DIR`.
- Accent checkpoints: `python controllability/download_checkpoints.py`.
- GPU required: the Vox-Profile wrappers call `.cuda()` internally.

## 1. Classifiers

| attribute | model | labels | chance |
| --------- | ----- | ------ | -----: |
| accent | our WavLM-Large + LoRA and Whisper-Large-V3 + LoRA, softmax-averaged (`--accent ensemble`) | North America / English / South Asia / Germanic / Oceania / South African | 1/6 |
| age | Vox-Profile `tiantiaf/wavlm-large-age-sex`, regression head, binned <20 / 20–39 / 40–59 / ≥60 | child / young_adult / middle_aged / older | 1/4 |
| gender | Vox-Profile `tiantiaf/wavlm-large-age-sex`, sex head | female / male | 1/2 |
| emotion | `emotion2vec/emotion2vec_plus_large` (via `funasr`), argmax restricted to the four requested emotions | angry / happy / neutral / sad | 1/4 |

Accent classifier, held-out speakers (Appendix C): ensemble accuracy 0.873 (WavLM 0.849,
Whisper 0.847); pooled UAR 0.750 on four out-of-domain corpora (`docs/accent_classifier.md`).
Gender, age and emotion use the released models unchanged; their
checkpoints are fetched from the Hugging Face hub on first use (`configs/classifiers.yaml`).

### 1.1 Loading our accent checkpoints elsewhere

`best.pt` stores LoRA weights that are **already merged** into `.weight`, but loralib's `merged`
flag is not part of the state dict. Mark every LoRA layer as merged after loading, so that
`.eval()` does not add the update again:

```python
import sys, torch, loralib as lora
sys.path.insert(0, "controllability/vox-profile-release/src/model/accent")
from wavlm_accent import WavLMWrapper

LABELS = ["North America", "English", "South Asia", "Germanic", "Oceania", "South African"]
m = WavLMWrapper(pretrain_model="wavlm_large", finetune_method="lora", lora_rank=16,
                 output_class_num=6, apply_gradient_reversal=False).cuda()
m.load_state_dict(torch.load("controllability/checkpoint/accent_wavlm/best.pt")["state_dict"])
for mod in m.modules():                 # required: the update is already merged
    if isinstance(mod, lora.LoRALayer):
        mod.merged = True
m.eval()
```

The Whisper member is the same with `from whisper_accent import WhisperWrapper` and
`pretrain_model="whisper_large"`.

## 2. Input

One folder per system and category: the `.wav` files (also found in a `wav/` or `audio/`
subfolder) and one CSV. Any sample rate or channel count works; clips are resampled to 16 kHz mono
and truncated to `--max_s` seconds (default 20).

`controllability/make_manifests.py --root data/outputs/attributes` writes
`controllability_manifest.csv` into every `<system>/<category>/` folder, reading the requested
value from the file name (`age-child-s1-0001-T1.wav` → `target_age = child`).

### 2.1 CSV columns

Column names are detected case-insensitively; any one alias works.

| field | accepted names | required |
| ----- | -------------- | -------- |
| file | `file` / `filename` / `wav` / `path` / `audio` / `id` | yes |
| accent | `target_accent` / `accent` | no |
| age | `target_age` / `age` | no |
| gender | `target_gender` / `gender` / `sex` | no |
| emotion | `target_emotion` / `emotion` | no |

Only attributes with a column are scored. A blank or unrecognised cell skips that attribute for
that clip (counted in `n_unrecognised_target`).

### 2.2 Accepted values

| attribute | canonical | also accepted |
| --------- | --------- | ------------- |
| accent | `North America` `English` `South Asia` `Germanic` `Oceania` `South African` | american, north_american → North America; british, british_isles → English; indian, south_asian → South Asia; german → Germanic; australian, oceanian → Oceania; south_african → South African |
| age | `child` `young_adult` `middle_aged` `older` | kid → child; young → young_adult; adult, middle → middle_aged; old, elderly, senior → older |
| gender | `female` `male` | woman, f → female; man, m → male |
| emotion | `angry` `happy` `neutral` `sad` | anger, mad → angry; joy → happy; sadness → sad |

The scorer's own outputs (`controllability.json`, the per-utterance CSVs) use these canonical labels.
`merge_shards.py` writes `results/controllability/summary.json` with the prompt names instead
(`north_american`, `british_isles`, `south_asian`, `germanic`, `oceanian`, `south_african`; `elderly`),
the names used in `data/prompts/`, `configs/` and `results/attributes/`.

## 3. Running

```bash
python controllability/score_controllability.py data/outputs/attributes/voxcpm2/accent \
    --csv controllability_manifest.csv --attrs accent --accent ensemble --age voxprofile
```

For the paper, `controllability/run_scoring.sh` scores all 7 systems × 5 categories in 10 shards
per category on one GPU and merges them into `results/controllability/summary.json`
(`experiments/05_controllability/run.sh` does all steps).

| option | default | meaning |
| ------ | ------- | ------- |
| `folders` | | one or more folders (positional) |
| `--csv` | auto | CSV name inside each folder (tries `prompts.csv`, `metadata.csv`, `meta.csv`, `manifest.csv`, then the first `*.csv`) |
| `--accent` | `ensemble` | `ensemble` (paper), `wavlm` or `whisper` |
| `--age` | `voxprofile` | the Vox-Profile age regressor (only option) |
| `--attrs` | all present | subset of `accent,age,gender,emotion`; also limits which classifiers are loaded |
| `--shard` | `0/1` | `I/N`: score rows I, I+N, I+2N, …; merge with `merge_shards.py` |
| `--out_dir` | the folder | write results to one directory, prefixed `<system>__<category>` |
| `--resume` | off | skip (folder, shard) pairs that already have a result |
| `--batch` | `8` | batch size |
| `--workers` | `8` | audio-decoding threads |
| `--max_s` | `20` | truncate clips to this many seconds |

**Settings for exact reproduction.** The paper used 10 shards and `--batch 8` (the defaults of
`run_scoring.sh`); keep both to reproduce `results/controllability/summary.json` exactly.

## 4. Output

Per folder (or per shard): `controllability.json` with `controllability_acc` (accuracy), `uar`
(mean per-class recall), `n`, `per_class_recall` and `confusion` per attribute, and
`controllability_per_utt.csv` with `target_*` and `pred_*` per clip (plus `pred_age_years`, the
regressed age before binning).

`merge_shards.py` recombines per-utterance CSVs into the same statistics as an unsharded run; for
gender × age folders it also reports each requested cell (`per_cell`: age, gender and both correct).

```json
"<system>/gender": {
  "n_rows": 6000,
  "attributes": {
    "gender": {
      "controllability_acc": 0.95, "uar": 0.95, "n": 6000,
      "per_class_recall": {"female": 0.92, "male": 0.98},
      "confusion": {"female": {"female": 2760, "male": 240}, "male": {"male": 2940, "female": 60}}
    }
  }
}
```
(illustrative numbers)
