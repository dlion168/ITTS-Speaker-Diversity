#!/usr/bin/env python3
"""Mirror the 6 target accent classes' audio from CV17 (fixie-ai reupload) locally,
so we can build Vox-Profile-style speaker-independent splits offline.

Streaming: download one `validated` shard -> keep only target-class clips filtered
to [3,15] s (Vox-Profile filter) -> write per (input-shard, class) parquet with the
mp3 bytes + client_id + duration -> delete the shard. Peak disk ~= 1 input shard.
Resumable: done input shards are skipped. Duration from the mp3 header (mutagen,
no full decode). Reuses the validated 16-class mapping from map_cv17_to_voxprofile.

Out: $ACC_DATA_ROOT/cv17_6class/<class_slug>/from-<NNN>.parquet
     $ACC_DATA_ROOT/cv17_6class/_build_ckpt.json
"""
import io
import json
import os
import re
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from mutagen.mp3 import MP3

from map_cv17_to_voxprofile import classify  # validated primary-token substring rules

DATA_ROOT = Path(os.environ.get("ACC_DATA_ROOT", Path(__file__).resolve().parent.parent / "data"))

REPO = "fixie-ai/common_voice_17_0"
SPLIT = "validated"
TARGETS = {"North America", "English", "South Asia", "Germanic", "Oceania", "South African"}
SLUG = {c: c.lower().replace(" ", "_") for c in TARGETS}
MIN_S, MAX_S = 3.0, 15.0

OUT = DATA_ROOT / "cv17_6class"
TMP = OUT / "_shard_tmp"
CKPT = OUT / "_build_ckpt.json"

SCHEMA = pa.schema([
    ("client_id", pa.string()),
    ("dur", pa.float32()),
    ("raw_accent", pa.string()),
    ("sentence", pa.string()),
    ("audio_bytes", pa.large_binary()),
])


def load_ckpt():
    if CKPT.exists():
        d = json.loads(CKPT.read_text())
        return set(d["done"]), d.get("kept", {})
    return set(), {}


def save_ckpt(done, kept):
    CKPT.write_text(json.dumps({"done": sorted(done), "kept": kept}))


def main():
    for c in TARGETS:
        (OUT / SLUG[c]).mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset")
                   if re.fullmatch(rf"en/{SPLIT}-\d+-of-\d+\.parquet", f))
    print(f"[build] {SPLIT}: {len(files)} shards -> classes {sorted(TARGETS)}", flush=True)

    done, kept = load_ckpt()
    t0 = time.time()
    for i, fn in enumerate(files):
        if fn in done:
            continue
        local = hf_hub_download(REPO, fn, repo_type="dataset",
                                local_dir=TMP, local_dir_use_symlinks=False)
        tbl = pq.ParquetFile(local).read(columns=["client_id", "accent", "audio", "sentence"])
        cid = tbl.column("client_id").to_pylist()
        acc = tbl.column("accent").to_pylist()
        aud = tbl.column("audio").to_pylist()
        snt = tbl.column("sentence").to_pylist()

        buf = {c: {"client_id": [], "dur": [], "raw_accent": [], "sentence": [], "audio_bytes": []}
               for c in TARGETS}
        for ci, a, au, s in zip(cid, acc, aud, snt):
            a = (a or "").strip()
            if not a:
                continue
            cls = classify(a)
            if cls not in TARGETS:
                continue
            b = au["bytes"]
            try:
                dur = MP3(io.BytesIO(b)).info.length
            except Exception:
                continue
            if not (MIN_S <= dur <= MAX_S):
                continue
            d = buf[cls]
            d["client_id"].append(ci)
            d["dur"].append(dur)
            d["raw_accent"].append(a)
            d["sentence"].append(s or "")
            d["audio_bytes"].append(b)

        for c in TARGETS:
            d = buf[c]
            if not d["client_id"]:
                continue
            t = pa.table({k: d[k] for k in SCHEMA.names}, schema=SCHEMA)
            pq.write_table(t, OUT / SLUG[c] / f"from-{i:03d}.parquet", compression="zstd")
            kept[c] = kept.get(c, 0) + t.num_rows

        done.add(fn)
        try:
            os.remove(local)
        except OSError:
            pass
        save_ckpt(done, kept)

        rate = (len(done)) / max(1e-9, time.time() - t0)
        eta = (len(files) - len(done)) / max(1e-9, rate) / 60
        tot = sum(kept.values())
        print(f"[{len(done)}/{len(files)}] {fn.split('/')[-1]}  kept_total={tot:,}  "
              f"({', '.join(f'{SLUG[c]}={kept.get(c,0):,}' for c in sorted(TARGETS))})  "
              f"ETA {eta:.0f} min", flush=True)

    print("\n[done] kept per class:", {c: kept.get(c, 0) for c in sorted(TARGETS)}, flush=True)
    for p in TMP.glob("**/*"):
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass
    os._exit(0)


if __name__ == "__main__":
    main()
