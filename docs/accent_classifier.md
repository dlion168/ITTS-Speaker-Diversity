# Accent classifier

The accent controllability scores (paper §3.5, §5, Appendix C) use our own 6-class accent
classifier: a softmax-averaged ensemble of **WavLM-Large + LoRA** and **Whisper-Large-V3 + LoRA**,
following the Vox-Profile recipe and trained **speaker-independently** (no speaker appears in more
than one of train / validation / test).

- Classes: North America, English (British Isles), South Asia, Germanic, Oceania, South African.
- Released checkpoints: `Snooow1029/itts-attribute-classifiers` (`accent_wavlm.pt`,
  `accent_whisper.pt`); `controllability/download_checkpoints.py` fetches them.
- Code: `controllability/accent_classifier/`. Reports behind every number below:
  `results/accent_classifier/`.

Gender, age and emotion use released models without retraining (`configs/classifiers.yaml`).

## 1. Data

| region | sources | train clips / speakers / hours | val clips / speakers | test clips / speakers |
| --- | --- | --- | --- | --- |
| North America | CommonVoice 17 | 40,000 / 4,350 / 58.6 h | 2,000 / 672 | 2,000 / 595 |
| English | CommonVoice 17 | 20,000 / 1,410 / 29.0 h | 2,000 / 291 | 2,000 / 298 |
| South Asia | CommonVoice 17 (+ L2-ARCTIC Hindi, 76 clips / 2 speakers) | 16,000 / 1,221 / 24.8 h | 2,000 / 256 | 2,000 / 315 |
| Oceania | CommonVoice 17 | 14,000 / 548 / 20.6 h | 2,000 / 157 | 2,000 / 153 |
| South African | AfriSpeech-200 ZA (2,793 / 119) + CommonVoice 17 (5,702 / 194) | 8,495 / 313 / 15.4 h | 2,000 / 101 | 2,000 / 99 |
| Germanic | CommonVoice 17 (800 / 34) + VoxPopuli `en_de`, `en_nl` (999 / 75) | 1,799 / 109 / 3.5 h | 600 / 36 | 551 / 38 |

- CommonVoice 17 is `fixie-ai/common_voice_17_0` (validated split); its free-text accent field is
  mapped to the six regions.
- Germanic is small on purpose: CommonVoice's German tag is dominated by one speaker, so VoxPopuli
  supplies speaker diversity (109 speakers) and the model learns accent rather than identity.
- The Speech Accent Archive (SAA) and VCTK are **excluded from training** and used as out-of-domain tests.

**Split.** Every source is split 60/20/20 by speaker (`client_id`); train/test speaker overlap is 0
for every class. Each speaker contributes at most 150 clips before the class caps (train:
North America 40k, English 20k, South Asia 16k, Oceania 14k, Germanic 16k, South African uncapped;
validation and test 2,000 per class).

**Audio.** Decoded to 16 kHz mono and filtered to 3–15 s. Split shards are parquet files with
columns `audio_bytes, label, cls, client_id, dur, source`.

The split shards contain the corpora's audio, which is not redistributed.
`controllability/accent_classifier/data_build/source_manifests/` lists every clip of the splits and
links it to its source utterance; `data_build/` has the build scripts.

## 2. Model

`vox-profile-release/src/model/accent/{wavlm_accent.py, whisper_accent.py}` (commit `85100e6`):

- frozen WavLM-Large or Whisper-Large-V3 encoder with LoRA rank 16 on the upper-half FFN layers;
- WavLM member: learnable softmax-weighted sum over all hidden layers; Whisper member: the last
  encoder layer, as in the Vox-Profile wrappers;
- head: 3 point-wise Conv1d (hidden size→256→256→256), mean pooling, Linear → ReLU → Linear(256→6).

The ensemble averages the two members' softmax outputs.

## 3. Training

| setting | value |
| --- | --- |
| augmentation (train only, `audiomentations`, in order) | AddGaussianSNR 3–30 dB (p 1.0); TimeMask 10–15% (p 1.0); TimeStretch 0.9–1.1, length-preserving (p 1.0); PolarityInversion (p 0.5). No pitch shift: it would distort the accent cue. |
| optimiser | Adam, learning rate 5e-4 constant |
| batch | effective 16 (WavLM 8 × 2, Whisper 4 × 4 gradient accumulation, on a 24 GB GPU) |
| epochs | 15; best checkpoint by validation macro-F1 |
| loss | inverse-frequency class-weighted cross-entropy (`ACC_WEIGHTED=1`; Germanic ≈ 22× North America) |
| precision, seed | AMP fp16, 1234 |

```bash
sbatch controllability/accent_classifier/train_accent.sbatch           # WavLM member
sbatch controllability/accent_classifier/train_accent_whisper.sbatch   # Whisper member
```

Both read `ACC_SPLITS`, `ACC_CKPT`, `ACC_LABELS` and `ACC_WEIGHTED` and resume from
`$ACC_CKPT/last.pt`. At the end each script scores its best checkpoint on the test split.

**Loading the checkpoints.** `best.pt` holds LoRA weights already merged into `.weight`, but not
loralib's `merged` flag. Mark every `loralib.LoRALayer` as merged after `load_state_dict`, or
`.eval()` adds the LoRA update a second time (see `docs/controllability_scorer.md` §1.1).

## 4. Results

### 4.1 In-distribution (held-out speakers)

`controllability/accent_classifier/eval_accent_ensemble.py` →
`results/accent_classifier/id/accent_6cls_ensemble.json` (members: `accent_6cls_{wavlm,whisper}.json`).

| model | accuracy | macro-F1 | UAR |
| --- | ---: | ---: | ---: |
| WavLM | 0.849 | 0.848 | 0.850 |
| Whisper | 0.847 | 0.856 | 0.848 |
| **ensemble** | **0.873** | **0.876** | **0.873** |

Per-class F1 of the ensemble: North America 0.881, English 0.825, South Asia 0.918,
Germanic 0.887, Oceania 0.873, South African 0.871 (2,000 test clips per class, 551 for Germanic).

### 4.2 Out-of-domain corpora

`controllability/accent_classifier/eval_accent_ood.py` → `results/accent_classifier/ood/`.
Four corpora not used in training: SAA (read paragraph, near-native, about one clip per speaker),
NCHLT (South African read speech, 210 speakers), EdAcc (spontaneous conversation) and VCTK
(studio read; English and North America only, 300 clips each).

Ensemble recall per corpus and region:

| corpus | N. America | English | Oceania | S. Asia | Germanic | S. African | acc | UAR |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| SAA | 0.99 | 0.89 | 0.95 | 0.69 | 0.60 | 1.00 | 0.907 | 0.855 |
| NCHLT | — | — | — | — | — | 0.81 | 0.807 | 0.807 |
| EdAcc | 0.38 | 0.49 | — | 0.91 | 0.32 | 0.85 | 0.515 | 0.590 |
| VCTK | 0.95 | 0.84 | — | — | — | — | 0.892 | 0.892 |

Pooled over the four corpora, beside the released Vox-Profile accent head
(`tiantiaf/wavlm-large-narrow-accent`, 16 classes mapped to our 6):

| region | WavLM | Whisper | **ensemble** | Vox-Profile |
| --- | --: | --: | --: | --: |
| North America | 0.719 | 0.723 | **0.729** | 0.337 |
| English | 0.502 | 0.714 | 0.609 | 0.573 |
| Oceania | 0.975 | 0.900 | 0.950 | 0.500 |
| South Asia | 0.867 | 0.795 | 0.862 | 0.886 |
| Germanic | 0.638 | 0.238 | 0.543 | 0.219 |
| South African | 0.690 | 0.825 | 0.808 | 0.399 |
| **pooled UAR** | 0.732 | 0.699 | **0.750** | 0.543 |

- The two members are complementary (WavLM is better on Germanic, Oceania and South Asia; Whisper on
  English and South African), so the ensemble has the best pooled UAR.

The OOD corpora have their own licences and are not redistributed; build the manifest CSVs
(`file, target_accent`) locally to rerun `eval_accent_ood.py`.
