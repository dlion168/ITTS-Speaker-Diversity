#!/usr/bin/env bash
# Paper §3.2, Appendix A: data for the encoder validation. Run once before run.sh.
#   bash experiments/01_encoder_validation/prepare_data.sh
#   1. downloads LibriSpeech train-clean-100 (6.3 GB) from OpenSLR if LIBRISPEECH_DIR does not exist
#   2. build_subsets.py     1,400 subsets: 14 values of k x 100 subsets x 240 utterances
#                           (240/k per speaker), plus the reference pool R
#   3. links every FLAC file under a .wav name into one folder for embeddings/extract_embeddings.py
# Writes $ITTS_DATA_ROOT/librispeech_validation/ and $ITTS_DATA_ROOT/librispeech/train-clean-100_flat/.
#
# Extra settings (see also experiments/common.sh):
#   LIBRISPEECH_DIR  .../LibriSpeech/train-clean-100; downloaded from OpenSLR if missing
#                    [$ITTS_DATA_ROOT/librispeech/LibriSpeech/train-clean-100]
#   OPENSLR_MIRROR   [https://www.openslr.org/resources/12]
source "$(dirname "$0")/../common.sh"
need_python "$EMBED_ENV"
LIBRI="${LIBRISPEECH_DIR:-$DATA/librispeech/LibriSpeech/train-clean-100}"
MIRROR="${OPENSLR_MIRROR:-https://www.openslr.org/resources/12}"
SUB="$DATA/librispeech_validation"
FLAT="$DATA/librispeech/train-clean-100_flat"

if [ ! -d "$LIBRI" ]; then
    d="$(dirname "$(dirname "$LIBRI")")"; mkdir -p "$d"
    echo "downloading LibriSpeech train-clean-100 (6.3 GB) to $d"
    curl -fL -o "$d/train-clean-100.tar.gz" "$MIRROR/train-clean-100.tar.gz"
    (cd "$d" && curl -fsL https://www.openslr.org/resources/12/md5sum.txt | grep " train-clean-100.tar.gz" | md5sum -c -)
    tar xzf "$d/train-clean-100.tar.gz" -C "$d"
fi

"$PY" experiments/01_encoder_validation/build_subsets.py --librispeech_dir "$LIBRI" --out_dir "$SUB"

# extract_embeddings.py embeds every *.wav of one folder: link the FLAC files there under .wav names
# (soundfile reads them by content).
mkdir -p "$FLAT"
find "$LIBRI" -name '*.flac' | while read -r f; do
    b="$(basename "$f" .flac)"; [ -e "$FLAT/$b.wav" ] || ln -s "$(realpath "$f")" "$FLAT/$b.wav"
done
echo "-> $SUB, $FLAT ($(find "$FLAT" -name '*.wav' | wc -l) files)"
