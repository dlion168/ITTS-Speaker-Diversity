# accent_classifier/data_build/

Scripts that build the training splits of the accent classifier and its Speech Accent Archive
out-of-domain test set (`docs/accent_classifier.md` §1, §4.2) from public corpora. The corpora's audio is not
redistributed, so the splits are built locally.

All scripts write under `$ACC_DATA_ROOT` (default `controllability/accent_classifier/data/`, not
committed); `make_splits.py` writes the shards to `$ACC_SPLITS` (default
`controllability/accent_classifier/splits/`), where `train_accent*.py` and `eval_accent_ensemble.py`
read them.

| Order | Script | Builds | Output under `$ACC_DATA_ROOT` |
|---|---|---|---|
| 1 | `scan_cv17_accents.py` | per-accent hours of CommonVoice 17 (`fixie-ai/common_voice_17_0`, validated), streamed shard by shard | `cv17_accent_scan/raw_tally.json` |
| 2 | `map_cv17_to_voxprofile.py` | maps CV17's free-text accent field to regions (`classify()`, imported by step 3); run alone it reports hours per class | (report only) |
| 3 | `build_accent_dataset.py` | the six-class CV17 mirror, clips filtered to 3–15 s | `cv17_6class/<class>/` |
| 4 | `build_extra_sa.py afrispeech` | AfriSpeech-200 ZA clips for South African | `extra_sa/afrispeech/` |
| 4 | `build_voxpopuli_germanic.py` | VoxPopuli `en_de` accented English for Germanic | `extra_germanic/voxpopuli/` |
| 5 | `make_splits.py` | the speaker-independent 60/20/20 split with the class caps of `docs/accent_classifier.md` §1 | `$ACC_SPLITS/{train,val,test}/shard-*.parquet` |
| OOD | `build_saa_ood.py` | Speech Accent Archive test set (`HamdanXI/speech-accent-archive-v2`) | `saa_ood/saa_accent.csv` + audio |

`source_validation.py` holds the checks used by `make_splits.py` and the training scripts:
every required source present with the expected schema, and no speaker in more than one split.
`verify_original_shards.py` checks a copy of the original split shards instead (below).

For Germanic, `make_splits.py` reads the VoxPopuli `en_de` and `en_nl` clips from
`extra_germanic/voxpopuli_de_nl/`, and for South Asia the L2-ARCTIC Hindi clips from
`extra_southasia/l2arctic/`; `source_manifests/germanic_voxpopuli.csv` and
`southasia_l2arctic.csv` list those clips.

```bash
P=envs/vox_profile/bin/python; D=controllability/accent_classifier/data_build
$P $D/scan_cv17_accents.py && $P $D/build_accent_dataset.py
$P $D/build_extra_sa.py afrispeech && $P $D/build_voxpopuli_germanic.py
$P $D/make_splits.py --check-sources      # check the sources without writing anything
$P $D/make_splits.py --output /path/to/new/splits
```

## Clips of the released model

`source_manifests/splits_6cls/{train,val,test}.jsonl.gz` list the 121,445 clips of the released
model's splits (100,294 / 10,600 / 10,551): split, class, source corpus, speaker, duration and the
SHA-256 of each clip's audio. The `*_utt_map.csv` files link every clip to its source utterance
(`source_manifests/README.md`).

To train on a copy of the original split shards, check it against these lists first:

```bash
$P $D/verify_original_shards.py --splits /path/to/original/shards
ACC_SPLITS=/path/to/original/shards ACC_ORIGINAL_SPLIT_MANIFESTS=$D/source_manifests/splits_6cls \
    $P controllability/accent_classifier/train_accent.py
```
