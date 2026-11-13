#!/usr/bin/env bash
# Paper §3.2, Appendix A: which speaker encoders can measure diversity?
#   bash experiments/01_encoder_validation/prepare_data.sh   # once: LibriSpeech, subsets, wav links
#   bash experiments/01_encoder_validation/run.sh
# Steps:
#   1. embeddings/extract_embeddings.py   every utterance, with each of the 11 encoders
#   2. compute_curves.py    MPCD and Vendi per subset, Chamfer and MMD^2 against R
#   3. rank_correlation.py  Spearman rho, k=240/k=6 ratio and separated steps per encoder
# Writes $RESULTS_DIR/encoder_validation/. Settings: see experiments/common.sh.
source "$(dirname "$0")/../common.sh"
need_python "$EMBED_ENV"
need_python "$NEMO_ENV"
SUB="$DATA/librispeech_validation"
FLAT="$DATA/librispeech/train-clean-100_flat"
EMB="$DATA/embeddings_validation"
OUT="$RESULTS_DIR/encoder_validation"

if [ ! -f "$SUB/subsets.jsonl" ] || [ ! -d "$FLAT" ]; then
    echo "missing $SUB/subsets.jsonl or $FLAT: run experiments/01_encoder_validation/prepare_data.sh first" >&2
    exit 1
fi

# The nine encoders trained for speaker verification plus the two SSL models (paper §3.2), with the
# same checkpoints as the diversity experiments (configs/encoders.yaml).
SET=(nemo_titanet speechbrain_ecapa speechbrain_resnet wespeaker wespeaker_voxblink2_resnet152_16k
     modelscope_eres2net modelscope_eres2netv2 modelscope_campplus
     speechbrain_xvector wavlm_base_plus_sv unispeech_sat_base_plus_sv)

export SSL_CERT_FILE="${SSL_CERT_FILE:-$("$PY" -c 'import certifi; print(certifi.where())')}"
for enc in "${SET[@]}"; do
    py="$EMBED_ENV/bin/python"; batch=256
    [ "$enc" = nemo_titanet ] && py="$NEMO_ENV/bin/python"
    case "$enc" in speechbrain_*) batch=16 ;; esac
    "$py" -u embeddings/extract_embeddings.py --wav_dir "$FLAT" --tts_model librispeech \
        --output_dir "$EMB" --models "$enc" --batch_size "$batch"
done

"$PY" experiments/01_encoder_validation/compute_curves.py --emb_root "$EMB/librispeech/$(basename "$FLAT")" \
    --data_dir "$SUB" --encoders "${SET[@]}" --out_dir "$OUT"
"$PY" experiments/01_encoder_validation/rank_correlation.py --dir "$OUT"
