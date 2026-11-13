#!/usr/bin/env python3
"""Build extra South African English audio into the same schema as the CV17 mirror,
so make_splits can merge it into the 'South African' class (speaker-independent).

Source:
  afrispeech -> AfriSpeech-200 ZA subset: 11 SA-L1 accents, country==ZA, 3-15 s

Schema: [client_id, dur, raw_accent, sentence, audio_bytes] (audio_bytes = raw file
bytes, any format soundfile can decode). Out: $ACC_DATA_ROOT/extra_sa/<src>/
"""
import csv
import io
import os
import sys
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

DATA_ROOT = Path(os.environ.get("ACC_DATA_ROOT", Path(__file__).resolve().parent.parent / "data"))

BASE = DATA_ROOT
OUT = BASE / "extra_sa"
MIN_S, MAX_S = 3.0, 15.0
SCHEMA = pa.schema([
    ("client_id", pa.string()), ("dur", pa.float32()), ("raw_accent", pa.string()),
    ("sentence", pa.string()), ("audio_bytes", pa.large_binary()),
])
AFRI = "tobiolatunji/afrispeech-200"
SA_ACCENTS = ["afrikaans", "isixhosa", "isizulu", "sepedi", "sesotho",
              "setswana", "sotho", "tswana", "venda", "xhosa", "zulu"]


def _dur(b):
    try:
        info = sf.info(io.BytesIO(b))
        return info.frames / info.samplerate
    except Exception:
        return None


def write_shard(rows, dst, tag):
    if not rows["client_id"]:
        return 0
    t = pa.table({k: rows[k] for k in SCHEMA.names}, schema=SCHEMA)
    pq.write_table(t, dst, compression="zstd")
    return t.num_rows


def build_afrispeech():
    from huggingface_hub import hf_hub_download, HfApi
    d = OUT / "afrispeech"
    d.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "_afri_tmp"
    tmp.mkdir(exist_ok=True)
    api = HfApi()
    repo_files = set(api.list_repo_files(AFRI, repo_type="dataset"))
    total = 0
    for accent in SA_ACCENTS:
        # metadata: id/stem -> (duration, country, user_id, transcript)
        meta = {}
        for split in ("train", "dev", "test"):
            csvf = f"transcripts/{accent}/{split}.csv"
            if csvf not in repo_files:
                continue
            p = hf_hub_download(AFRI, csvf, repo_type="dataset")
            for r in csv.DictReader(open(p)):
                if (r.get("country", "") or "").strip().upper() not in ("ZA", "SOUTH AFRICA"):
                    continue
                stem = Path((r.get("audio_paths") or "").strip()).stem
                key = stem or (r.get("audio_ids") or "").strip()
                if not key:
                    continue
                try:
                    du = float(r.get("duration") or 0)
                except ValueError:
                    du = 0.0
                meta[key] = (du, r.get("user_ids", ""), r.get("transcript", ""), accent)
                meta[(r.get("audio_ids") or "").strip()] = meta[key]  # alt key
        kept_acc = 0
        for split in ("train", "dev", "test"):
            tar_list = sorted(f for f in repo_files
                              if f.startswith(f"audio/{accent}/{split}/") and f.endswith(".tar.gz"))
            for ti, tf in enumerate(tar_list):
                try:
                    tarp = hf_hub_download(AFRI, tf, repo_type="dataset", local_dir=tmp)
                except Exception as e:
                    print(f"  [{accent}] dl fail {tf}: {e}", flush=True)
                    continue
                rows = {k: [] for k in SCHEMA.names}
                with tarfile.open(tarp) as tar:
                    for m in tar.getmembers():
                        if not m.isfile():
                            continue
                        stem = Path(m.name).stem
                        info = meta.get(stem)
                        if info is None:
                            continue
                        b = tar.extractfile(m).read()
                        du = _dur(b)
                        if du is None or not (MIN_S <= du <= MAX_S):
                            continue
                        uid, tx, acc = info[1], info[2], info[3]
                        rows["client_id"].append(f"afri_{uid or stem}")
                        rows["dur"].append(du)
                        rows["raw_accent"].append(f"afrispeech_{acc}")
                        rows["sentence"].append(tx or "")
                        rows["audio_bytes"].append(b)
                kept_acc += write_shard(rows, d / f"{accent}_{split}_{ti}.parquet", accent)
                os.remove(tarp)
        total += kept_acc
        print(f"[afrispeech] {accent}: kept {kept_acc}  (running total {total})", flush=True)
    for p in tmp.glob("**/*"):
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass
    print(f"[afrispeech] TOTAL kept {total}", flush=True)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "afrispeech"
    if src == "afrispeech":
        build_afrispeech()
    else:
        print("usage: build_extra_sa.py [afrispeech]")
