#!/usr/bin/env python3
"""Fetch the released generated audio from Hugging Face and write it back as WAV files in the
layout generation/ produces, so experiments/*/run.sh --from audio can embed and score it.

Hugging Face dataset (default dlion168/itts-speaker-diversity, gated: needs HF_TOKEN):
    data/<config>/<split>-XXXXX-of-YYYYY.parquet   with columns original_filename, partition,
                                                   sweep_param, sweep_value, seed, attribute, audio (FLAC)
Local layout under --out_root:
    outputs/attributes/<system>/<category>/<original_filename>                (attribute, gender_x_age)
    outputs/overall/<system>/sample<partition>/<original_filename>            (overall)
    outputs/param_sweep/<system>/<temp|cfg>_<value>/seed_<s>/<original_filename>   (sampling_sweep)

FLAC -> WAV keeps the samples and sample rate unchanged (both 16-bit PCM).

Usage:
    python embeddings/download_released_audio.py --configs attribute --systems voxcpm2 --out_root data
"""
import argparse
import io
import math
import time
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
import numpy as np
if __package__:
    from .release_assets import revision_for
else:
    from release_assets import revision_for

SYSTEMS = {"moss_voicegen": "moss-voicegen", "parler_large": "parler-large",
           "parler_mini": "parler-mini", "capspeech_nar": "capspeech",
           "qwen3_tts": "qwen3-voicedesign", "voxinstruct": "voxinstruct", "voxcpm2": "voxcpm2"}
SWEEP_PARAM = {"temperature": "temp", "cfg": "cfg"}
CONFIGS = ("attribute", "gender_x_age", "overall", "sampling_sweep")
COLUMNS = ["original_filename", "attribute", "partition", "sweep_param", "sweep_value", "seed", "audio"]


def target_dir(config, system, r):
    if config == "attribute":
        if r['attribute'] not in {'gender', 'age', 'accent', 'emotion'}:
            raise ValueError('Unknown released attribute category')
        return Path("outputs/attributes") / system / r["attribute"]
    if config == "gender_x_age":
        return Path("outputs/attributes") / system / "gender_x_age"
    if config == "overall":
        if isinstance(r['partition'], bool) or not isinstance(r['partition'], int) or r['partition'] not in (1, 2, 3):
            raise ValueError('Unknown released overall partition')
        return Path("outputs/overall") / system / f"sample{r['partition']}"
    if r['sweep_param'] not in SWEEP_PARAM or not math.isfinite(float(r['sweep_value'])) or float(r['sweep_value']) <= 0:
        raise ValueError('Invalid released sweep setting')
    if isinstance(r['seed'], bool) or not isinstance(r['seed'], int) or not 0 <= r['seed'] < 2**32:
        raise ValueError('Invalid released sweep seed')
    return (Path("outputs/param_sweep") / system
            / f"{SWEEP_PARAM[r['sweep_param']]}_{r['sweep_value']}" / f"seed_{r['seed']}")


def write_wav(path, flac_bytes):
    data, sr = sf.read(io.BytesIO(flac_bytes), dtype="int16")
    if path.exists():
        previous, previous_rate = sf.read(path, dtype='int16')
        if previous_rate != sr or not np.array_equal(previous, data):
            raise FileExistsError(f'Existing audio differs from selected release at {path}; use a fresh out_root')
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.wav")
    sf.write(tmp, data, sr, subtype="PCM_16")
    tmp.replace(path)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="dlion168/itts-speaker-diversity")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--local_dir", type=Path, default=None,
                    help="use an existing snapshot of the dataset instead of downloading")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS),
                    help="Hugging Face split names")
    ap.add_argument("--out_root", type=Path, default=Path("data"))
    args = ap.parse_args()
    args.revision = revision_for(args.repo, args.revision)

    root = args.local_dir
    if root is None:
        from huggingface_hub import snapshot_download
        patterns = [f"data/{c}/{s}-*.parquet" for c in args.configs for s in args.systems]
        # Tens of GB: a dropped connection is likely; each retry resumes from the files already done.
        for attempt in range(1, 6):
            try:
                root = Path(snapshot_download(args.repo, repo_type="dataset", revision=args.revision,
                                              allow_patterns=patterns))
                break
            except Exception as e:
                if attempt == 5:
                    raise
                print(f"download failed ({e}); retry {attempt}/4 in 30 s", flush=True)
                time.sleep(30)

    for config in args.configs:
        for split in args.systems:
            shards = sorted((root / "data" / config).glob(f"{split}-*.parquet"))
            if not shards:
                if config == 'sampling_sweep' and split not in {'moss_voicegen', 'voxinstruct', 'voxcpm2'}:
                    continue
                raise FileNotFoundError(f'Missing selected released audio: {config}/{split}')
            n = skipped = 0
            for shard in shards:
                pf = pq.ParquetFile(shard)
                for g in range(pf.num_row_groups):
                    for r in pf.read_row_group(g, columns=COLUMNS).to_pylist():
                        out = args.out_root / target_dir(config, SYSTEMS[split], r) / r["original_filename"]
                        filename = r['original_filename']
                        if not isinstance(filename, str) or Path(filename).name != filename or filename in {'.', '..'} or '\\' in filename:
                            raise ValueError('Unsafe original filename in released audio')
                        if not write_wav(out, r['audio']['bytes']):
                            skipped += 1
                            continue
                        n += 1
            print(f"{config}/{split}: wrote {n} WAVs ({skipped} already present)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
