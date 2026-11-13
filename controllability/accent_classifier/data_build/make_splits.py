#!/usr/bin/env python3
"""Speaker-independent accent splits. Every source (CV17 included) is split 60/20/20 by
client_id, so no speaker crosses splits.

Besides CV17, it uses VoxPopuli en_de+en_nl (Germanic), AfriSpeech-ZA (South African) and
L2-ARCTIC Hindi (South Asia); VCTK is kept out as an out-of-domain test set.

Usage: make_splits.py [--check-sources] [--output DIR]
Out: $ACC_SPLITS (default accent_classifier/splits)/{train,val,test}/shard-*.parquet
"""
import os
import argparse
import math
import hashlib
import json, random
from collections import defaultdict
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
if __package__:
    from .source_validation import required_sources, validate_sources, assert_speaker_disjoint
else:
    from source_validation import required_sources, validate_sources, assert_speaker_disjoint

DATA_ROOT = Path(os.environ.get("ACC_DATA_ROOT", Path(__file__).resolve().parent.parent / "data"))

BASE = DATA_ROOT
MIRROR = BASE / "cv17_6class"
# where train_accent*.py read the splits (ACC_SPLITS)
OUT = Path(os.environ.get("ACC_SPLITS", Path(__file__).resolve().parent.parent / "splits"))

REGIONS = ["North America", "English", "South Asia", "Germanic", "Oceania", "South African"]
LABELS = list(REGIONS)
LAB_IDX = {c: i for i, c in enumerate(LABELS)}
SLUG = {c: c.lower().replace(" ", "_") for c in LABELS}
TRAIN_CAP = {"North America": 40000, "English": 20000, "South Asia": 16000,
             "Oceania": 14000, "Germanic": 16000, "South African": None}
EVAL_CAP = 2000
MAX_PER_SPK = 150   # cap clips per speaker (before total cap) so no single speaker dominates
                    # -- important for Germanic, where one CV17 speaker dominates.
SEED = 1234
SHARD_ROWS = 2000

# per-class extra (non-CV) sources; NO VCTK. Schema: client_id,dur,raw_accent,audio_bytes
EXTRA = {
    "South African": [("afrispeech", BASE / "extra_sa" / "afrispeech")],
    "Germanic": [("voxpopuli", BASE / "extra_germanic" / "voxpopuli_de_nl")],
    "South Asia": [("l2arctic", BASE / "extra_southasia" / "l2arctic")],
}


def read_refs(d, source, has_cv_mirror):
    """-> list of (path,row,client_id,dur,source). cv mirror & extras share client_id/dur cols."""
    rows = []
    for f in sorted(Path(d).glob("*.parquet")):
        pf = pq.ParquetFile(f)
        cid = pf.read(columns=["client_id"]).column("client_id").to_pylist()
        dur = pf.read(columns=["dur"]).column("dur").to_pylist()
        for i, (c, du) in enumerate(zip(cid, dur)):
            if c is None or not str(c).strip() or du is None or not math.isfinite(float(du)) or float(du) <= 0:
                raise ValueError(f"Invalid speaker/duration metadata: {f.name}, row {i}")
            rows.append((str(f), i, str(c), float(du), source))
    return rows


def gather(cls):
    rows = []
    rows += read_refs(MIRROR / SLUG[cls], "cv17", True)
    for name, ed in EXTRA.get(cls, []):
        rows += read_refs(ed, name, False)
    return rows


def speaker_split(rows, rng):
    by = defaultdict(list)
    for r in rows:
        by[r[2]].append(r)
    spk = sorted(by); rng.shuffle(spk)
    n = len(spk); ntr, nva = int(n * 0.6), int(n * 0.2)
    parts = {"train": spk[:ntr], "val": spk[ntr:ntr + nva], "test": spk[ntr + nva:]}
    out = {}
    for s, sl in parts.items():
        acc = []
        for k in sl:
            acc.extend(by[k])
        rng.shuffle(acc); out[s] = acc
    return out


def main(argv=None):
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Fresh split destination; existing output shards are never overwritten')
    parser.add_argument('--source-manifest', type=Path, help='Optional expected counts and source-file SHA-256 values')
    parser.add_argument('--check-sources', action='store_true', help='Validate coverage/schema only, without writing any output')
    args = parser.parse_args(argv)
    OUT = args.output or OUT
    inventory = validate_sources(required_sources(MIRROR, REGIONS, EXTRA), args.source_manifest)
    if args.check_sources:
        print(json.dumps({'status': 'source_coverage_checked', 'sources': inventory}, indent=2))
        return 0
    if OUT.exists() and (any(OUT.rglob('*.parquet')) or (OUT / 'manifest_summary.json').exists()):
        raise FileExistsError('Split output already contains artifacts; choose a fresh --output directory')
    rng = random.Random(SEED)
    summary = {}
    pooled = {"train": [], "val": [], "test": []}
    def cap_per_speaker(rows):
        seen = defaultdict(int); out = []
        for r in rows:
            if seen[r[2]] < MAX_PER_SPK:
                seen[r[2]] += 1; out.append(r)
        return out

    for cls in LABELS:
        rows = gather(cls)
        sp = speaker_split(rows, rng)
        for s in ("train", "val", "test"):   # per-speaker cap first (rows already shuffled)
            sp[s] = cap_per_speaker(sp[s])
        cap = TRAIN_CAP.get(cls)
        if cap and len(sp["train"]) > cap:
            sp["train"] = sp["train"][:cap]
        for s in ("val", "test"):
            if len(sp[s]) > EVAL_CAP:
                sp[s] = sp[s][:EVAL_CAP]
        if any(not sp[s] for s in ('train', 'val', 'test')):
            raise ValueError(f'Class {cls} cannot produce nonempty train/val/test partitions; no shards were written')
        for s in ("train", "val", "test"):
            for r in sp[s]:
                pooled[s].append((cls, *r))
        nspk = {s: len({r[2] for r in sp[s]}) for s in ("train", "val", "test")}
        summary[cls] = {s: len(sp[s]) for s in ("train", "val", "test")} | {f"{s}_spk": nspk[s] for s in nspk}
        print(f"{cls:<15} train={len(sp['train']):>6} ({nspk['train']}spk) "
               f"val={len(sp['val']):>5} ({nspk['val']}spk) test={len(sp['test']):>5} ({nspk['test']}spk)", flush=True)

    assert_speaker_disjoint(pooled)

    schema = pa.schema([("audio_bytes", pa.large_binary()), ("label", pa.int32()),
                        ("cls", pa.string()), ("client_id", pa.string()),
                        ("dur", pa.float32()), ("source", pa.string())])
    FLUSH = 256
    (OUT / 'manifests').mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        recs = pooled[split]; rng.shuffle(recs)
        (OUT / split).mkdir(parents=True, exist_ok=True)
        nshards = max(1, (len(recs) + SHARD_ROWS - 1) // SHARD_ROWS)
        by_path = defaultdict(list)
        for k, (cls, path, ri, cid, du, src) in enumerate(recs):
            by_path[path].append((ri, k, cls, cid, du, src))
        writers = [None] * nshards
        sinks = [None] * nshards
        bufs = [{c: [] for c in schema.names} for _ in range(nshards)]

        def flush(si):
            b = bufs[si]
            if not b["label"]:
                return
            if writers[si] is None:
                sinks[si] = (OUT / split / f"shard-{si:04d}.parquet").open('xb')
                writers[si] = pq.ParquetWriter(sinks[si], schema, compression="zstd")
            writers[si].write_table(pa.table(b, schema=schema))
            bufs[si] = {c: [] for c in schema.names}

        written = 0
        manifest_path = OUT / 'manifests' / f'{split}.jsonl'
        with manifest_path.open('x') as manifest:
            for path, items in by_path.items():
                parquet = pq.ParquetFile(path)
                identifier_field = next((name for name in ('utterance_id', 'id', 'filename', 'path') if name in parquet.schema_arrow.names), None)
                table = parquet.read(columns=['audio_bytes'] + ([identifier_field] if identifier_field else []))
                tb = table.column('audio_bytes')
                for ri, k, cls, cid, du, src in items:
                    audio = tb[ri].as_py()
                    if not isinstance(audio, bytes) or not audio:
                        raise ValueError(f'Empty/nonbinary source audio: {Path(path).name}, row {ri}')
                    si = k % nshards
                    b = bufs[si]
                    b["audio_bytes"].append(audio); b["label"].append(LAB_IDX[cls])
                    b["cls"].append(cls); b["client_id"].append(cid); b["dur"].append(du); b["source"].append(src)
                    source_identifier = table.column(identifier_field)[ri].as_py() if identifier_field else None
                    manifest.write(json.dumps({'split': split, 'class': cls, 'source': src,
                                               'client_id': cid, 'source_parquet': os.path.relpath(path, BASE),
                                               'source_row': ri, 'audio_sha256': hashlib.sha256(audio).hexdigest(),
                                               'source_identifier_field': identifier_field,
                                               'source_identifier': str(source_identifier) if source_identifier is not None else None}) + '\n')
                    written += 1
                    if len(b["label"]) >= FLUSH:
                        flush(si)
        for si in range(nshards):
            flush(si)
            if writers[si] is not None:
                writers[si].close()
                sinks[si].close()
        print(f"[{split}] wrote {written:,} clips in {nshards} shards", flush=True)
        summary[f"_{split}_total"] = written

    summary["_labels"] = LABELS
    summary['_source_inventory'] = inventory
    summary['_speaker_disjoint'] = True
    summary['_split_seed'] = SEED
    summary['_expected_source_manifest_sha256'] = hashlib.sha256(args.source_manifest.read_bytes()).hexdigest() if args.source_manifest else None
    (OUT / "manifest_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nlabels:", LABELS)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
