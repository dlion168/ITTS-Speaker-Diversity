#!/usr/bin/env python3
"""
Download dlion168/emilia-en-itts-prompts and save each set as a folder of WAV files.

Output structure (--out_dir, default $ITTS_DATA_ROOT/real_world_data, i.e. data/real_world_data,
the default REAL_AUDIO of experiments/common.sh):
    real_world_data/
    ├── set_1/       ← corresponds to sample1
    │   ├── EN_B00079_S04984_W000000.wav
    │   └── ...
    ├── set_2/       ← corresponds to sample2
    └── set_3/       ← corresponds to sample3

Requires: HF_TOKEN environment variable (dataset is gated).
"""

import argparse
import io
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

REPO_ID = "dlion168/emilia-en-itts-prompts"
TARGET_SR = 16000

SETS = {
    "set_1": ["data/set_1-00000-of-00002.parquet", "data/set_1-00001-of-00002.parquet"],
    "set_2": ["data/set_2-00000-of-00002.parquet", "data/set_2-00001-of-00002.parquet"],
    "set_3": ["data/set_3-00000-of-00002.parquet", "data/set_3-00001-of-00002.parquet"],
}


def convert_mp3_bytes_to_wav(mp3_bytes: bytes, wav_path: str, target_sr: int = TARGET_SR):
    audio, sr = sf.read(io.BytesIO(mp3_bytes))
    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(target_sr, sr)
        audio = resample_poly(audio, target_sr // g, sr // g)
        sr = target_sr
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    sf.write(wav_path, audio, sr)


def process_set(set_name: str, parquet_files: list, output_dir: Path):
    set_dir = output_dir / set_name
    set_dir.mkdir(parents=True, exist_ok=True)

    existing = set(f.stem for f in set_dir.glob("*.wav"))
    if existing:
        print(f"  [{set_name}] {len(existing)} WAVs already exist, will skip them")

    total = 0
    saved = 0
    skipped = 0

    for pq_name in parquet_files:
        print(f"  [{set_name}] Downloading {pq_name} ...")
        local_path = hf_hub_download(REPO_ID, pq_name, repo_type="dataset")
        table = pq.read_table(local_path, columns=["id", "audio"])
        rows = table.to_pydict()

        for i in range(len(rows["id"])):
            file_id = rows["id"][i]
            total += 1

            if file_id in existing:
                skipped += 1
                continue

            mp3_bytes = rows["audio"][i]["bytes"]
            wav_path = str(set_dir / f"{file_id}.wav")

            try:
                convert_mp3_bytes_to_wav(mp3_bytes, wav_path)
                saved += 1
            except Exception as e:
                print(f"    FAILED {file_id}: {e}")

            if (saved + skipped) % 1000 == 0:
                print(f"    [{set_name}] progress: {saved} saved, {skipped} skipped / {total} total")

        del table, rows

    print(f"  [{set_name}] Done: {saved} saved, {skipped} skipped, {total} total")


def main():
    data_root = Path(os.environ.get("ITTS_DATA_ROOT") or Path(__file__).resolve().parents[1])
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path, default=data_root / "real_world_data")
    args = ap.parse_args()
    if not os.environ.get("HF_TOKEN"):
        print("Error: set HF_TOKEN environment variable first.")
        return 1

    output_dir = args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    for set_name, pq_files in SETS.items():
        print(f"\n{'='*60}")
        print(f"Processing {set_name} ...")
        process_set(set_name, pq_files, output_dir)

    print(f"\n{'='*60}")
    print("All done.")
    for set_name in SETS:
        set_dir = output_dir / set_name
        n = len(list(set_dir.glob("*.wav")))
        print(f"  {set_name}: {n} WAV files")

    return 0


if __name__ == "__main__":
    sys.exit(main())
