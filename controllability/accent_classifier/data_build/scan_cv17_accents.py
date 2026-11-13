#!/usr/bin/env python3
"""Scan Common Voice 17 (fixie-ai reupload) `validated` split for per-accent HOURS.

Streaming download->tally->delete so peak disk ~= one shard (~0.5 GB), not 67 GB.
For each shard we read only [accent, audio]; duration is taken from the mp3 header
(mutagen, no full decode) and only for rows that actually carry an accent label
(~15% of clips), so decode cost is bounded to the labelled pool.

We persist the RAW accent-string -> seconds/clip tallies (full fidelity), so the
Vox-Profile-class mapping can be recomputed later without re-scanning. Checkpoint
after every shard -> resumable (already-done shards are skipped).

Out: $ACC_DATA_ROOT/cv17_accent_scan/{raw_tally.json, progress.json}
"""
import io
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from mutagen.mp3 import MP3

DATA_ROOT = Path(os.environ.get("ACC_DATA_ROOT", Path(__file__).resolve().parent.parent / "data"))

REPO = "fixie-ai/common_voice_17_0"
SPLIT = "validated"
OUT = DATA_ROOT / "cv17_accent_scan"
TMP = OUT / "_shard_tmp"


def load_ckpt():
    p = OUT / "raw_tally.json"
    if p.exists():
        d = json.loads(p.read_text())
        return (defaultdict(float, d["sec"]), defaultdict(int, d["n"]),
                set(d["done_shards"]), d.get("no_accent_clips", 0),
                d.get("no_accent_sec", 0.0))
    return defaultdict(float), defaultdict(int), set(), 0, 0.0


def save_ckpt(sec, n, done, na_clips, na_sec, files):
    (OUT / "raw_tally.json").write_text(json.dumps({
        "sec": dict(sec), "n": dict(n), "done_shards": sorted(done),
        "no_accent_clips": na_clips, "no_accent_sec": na_sec,
        "n_shards_total": len(files),
    }))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    files = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset")
                   if re.fullmatch(rf"en/{SPLIT}-\d+-of-\d+\.parquet", f))
    print(f"[cv17] {SPLIT}: {len(files)} shards", flush=True)

    sec, n, done, na_clips, na_sec = load_ckpt()
    t0 = time.time()
    for i, fn in enumerate(files):
        if fn in done:
            continue
        local = hf_hub_download(REPO, fn, repo_type="dataset",
                                local_dir=TMP, local_dir_use_symlinks=False)
        tbl = pq.ParquetFile(local).read(columns=["accent", "audio"])
        accs = tbl.column("accent").to_pylist()
        auds = tbl.column("audio").to_pylist()
        for a, au in zip(accs, auds):
            a = (a or "").strip()
            try:
                dur = MP3(io.BytesIO(au["bytes"])).info.length
            except Exception:
                dur = 0.0
            if not a:
                na_clips += 1
                na_sec += dur
                continue
            sec[a] += dur
            n[a] += 1
        done.add(fn)
        try:
            os.remove(local)
        except OSError:
            pass
        save_ckpt(sec, n, done, na_clips, na_sec, files)
        lab_h = sum(sec.values()) / 3600
        rate = (i + 1) / max(1e-9, time.time() - t0)
        eta = (len(files) - (i + 1)) / max(1e-9, rate) / 60
        print(f"[{len(done)}/{len(files)}] {fn.split('/')[-1]}  "
              f"labelled={sum(n.values()):,} clips / {lab_h:.1f} h  "
              f"raw_accents={len(sec)}  ETA {eta:.0f} min", flush=True)

    print(f"\n[done] labelled {sum(n.values()):,} clips / "
          f"{sum(sec.values())/3600:.1f} h across {len(sec)} raw accent strings")
    print(f"       unlabelled {na_clips:,} clips / {na_sec/3600:.1f} h")
    # tidy tmp
    for p in TMP.glob("**/*"):
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass
    os._exit(0)


if __name__ == "__main__":
    main()
