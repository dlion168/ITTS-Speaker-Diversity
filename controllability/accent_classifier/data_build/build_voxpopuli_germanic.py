#!/usr/bin/env python3
"""Supplement the Germanic class with VoxPopuli German-accented English (en_de).

CV17's Germanic tag is dominated by a single speaker; VoxPopuli en_accented en_de adds
European-Parliament speakers, enabling a speaker-independent Germanic val/test. Same output schema as the
CV mirror so make_splits merges it (speaker-independent, keyed on speaker_id).

Out: $ACC_DATA_ROOT/extra_germanic/voxpopuli/*.parquet
"""
import os
import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

DATA_ROOT = Path(os.environ.get("ACC_DATA_ROOT", Path(__file__).resolve().parent.parent / "data"))

REPO = "facebook/voxpopuli"
SHARDS = ["en_accented/test-00000-of-00002.parquet", "en_accented/test-00001-of-00002.parquet"]
ACCENT = "en_de"
MIN_S, MAX_S = 3.0, 15.0
OUT = DATA_ROOT / "extra_germanic" / "voxpopuli"
SCHEMA = pa.schema([
    ("client_id", pa.string()), ("dur", pa.float32()), ("raw_accent", pa.string()),
    ("sentence", pa.string()), ("audio_bytes", pa.large_binary()),
])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kept = 0
    for si, sh in enumerate(SHARDS):
        f = hf_hub_download(REPO, sh, repo_type="dataset")
        pf = pq.ParquetFile(f)
        rows = {c: [] for c in SCHEMA.names}
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["audio", "accent", "speaker_id", "normalized_text"])
            audio = t.column("audio").combine_chunks().to_pylist()  # avoid chunked-struct error
            acc = t.column("accent").to_pylist()
            spk = t.column("speaker_id").to_pylist()
            txt = t.column("normalized_text").to_pylist()
            for au, a, s, tx in zip(audio, acc, spk, txt):
                if a != ACCENT or not au:
                    continue
                b = au["bytes"]
                try:
                    info = sf.info(io.BytesIO(b))
                    dur = info.frames / info.samplerate
                except Exception:
                    continue
                if not (MIN_S <= dur <= MAX_S):
                    continue
                rows["client_id"].append(f"vp_{s}")
                rows["dur"].append(dur)
                rows["raw_accent"].append(a)
                rows["sentence"].append(tx or "")
                rows["audio_bytes"].append(b)
        if rows["client_id"]:
            pq.write_table(pa.table(rows, schema=SCHEMA), OUT / f"from-{si}.parquet",
                           compression="zstd")
            kept += len(rows["client_id"])
    print(f"[voxpopuli en_de] kept {kept} clips -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
