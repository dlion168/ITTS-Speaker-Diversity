# source_manifests/

Clip lists for the accent classifier's data.

| File | Rows | Columns |
|---|---:|---|
| `splits_6cls/{train,val,test}.jsonl.gz` | 100,294 / 10,600 / 10,551 | the released model's splits: `split, class, source, label, client_id, dur, final_shard, final_shard_row, audio_sha256, source_identifier` (`audio_sha256` identifies each clip; `source_identifier` is empty, the files below link the clips to their source utterances) |
| `cv17_utt_map.csv` | 115,443 | every CommonVoice 17 clip of those splits, by `audio_sha256`: `client_id, dur, cls, cv_sentence` |
| `germanic_voxpopuli.csv` | 1,726 | VoxPopuli `en_de` (807) and `en_nl` (919) clips of the Germanic source: `client_id, dur, raw_accent, shard` |
| `voxpopuli_utt_map.csv` | 1,726 | the same clips with their VoxPopuli utterance: `voxpopuli_audio_id, text` |
| `southasia_l2arctic.csv` | 273 | L2-ARCTIC Hindi clips of the South Asia source: `client_id, dur, raw_accent, shard` |
| `southasia_l2arctic_utt_map.csv` | 273 | the same clips with their L2-ARCTIC utterance: `l2arctic_speaker, l2arctic_file_id, transcript` |
| `ood_nchlt.csv` | 3,053 | NCHLT out-of-domain test clips: `filename, target_accent` |
| `ood_nchlt_utt_map.csv` | 3,053 | the same clips with their NCHLT speaker and utterance: `orig_speaker, utterance_id, text, dur` |
| `ood_edacc.csv` | 3,798 | EdAcc out-of-domain test clips: `filename, target_accent` (the evaluation drops `Other`, leaving 1,525) |
| `ood_edacc_utt_map.csv` | 3,798 | the same clips with their EdAcc speaker and text: `orig_speaker, text, dur` |

The source datasets on Hugging Face: `fixie-ai/common_voice_17_0`, `facebook/voxpopuli` (`en_accented`), `NathanRoll/l2-arctic-dataset`,
`danielshaps/nchlt_speech_eng`, `edinburghcstr/edacc`.
