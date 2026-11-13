#!/usr/bin/env python3
"""Fetch the released speaker embeddings from Hugging Face and lay them out like
extract_embeddings.py output, so metrics/ and experiments/*/run.sh --from embeddings can use them.

Hugging Face dataset (default dlion168/itts-speaker-diversity; gated: accept its terms, then set HF_TOKEN):
    embeddings/<config>/<split>/<encoder>.parquet          columns id, embedding
    metadata/emilia_ids_overall_partition_{1,2,3}.txt      Emilia ids of each overall partition

Local layout written under --out_root (rows sorted by file name, as extract_embeddings.py writes them):
    embeddings_attributes/<system>/<category>/<encoder>/{embeddings.npy,filenames.txt}
    embeddings_overall/<system>/sample{1,2,3}/<encoder>/...          (overall, §4.1)
    embeddings_overall/real/set_{1,2,3}/<encoder>/...                (real_emilia, if released)
    embeddings_param_sweep/<system>/<temp|cfg>_<v>/seed_<s>/<encoder>/...   (sampling_sweep, §4.2)

Usage:
    python embeddings/download_released.py --configs attribute gender_x_age --out_root data
    python embeddings/download_released.py --local_dir /path/to/snapshot --out_root data   # no download
"""
import argparse
import sys
import os
import math
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
if __package__:
    from .release_assets import revision_for
else:
    from release_assets import revision_for

# Hugging Face split name -> system name used in this repo's folders and result files
SYSTEMS = {"moss_voicegen": "moss-voicegen", "parler_large": "parler-large",
           "parler_mini": "parler-mini", "capspeech_nar": "capspeech",
           "qwen3_tts": "qwen3-voicedesign", "voxinstruct": "voxinstruct", "voxcpm2": "voxcpm2"}
SWEEP_PARAM = {"temperature": "temp", "cfg": "cfg"}
CONFIGS = ("attribute", "gender_x_age", "overall", "real_emilia", "sampling_sweep")


def category_of(stem):
    # gender-female-s1-0001-T1 -> gender ; genderage-male_child-s2-0001-F1_... -> gender_x_age
    head = stem.split("-", 1)[0]
    return "gender_x_age" if head == "genderage" else head


def write_set(folder, names, X):
    if not names or len(set(names)) != len(names) or any(not isinstance(name, str) or not name or '\n' in name or '\r' in name for name in names):
        raise ValueError('Released embeddings require unique nonempty file IDs')
    if X.ndim != 2 or len(X) != len(names) or not np.isfinite(X).all():
        raise ValueError('Invalid/nonfinite released embedding matrix')
    order = np.argsort(np.asarray(names), kind="stable")
    sorted_names = ''.join(f'{names[i]}\n' for i in order)
    sorted_X = np.ascontiguousarray(X[order])
    emb_path, names_path = folder / 'embeddings.npy', folder / 'filenames.txt'
    if emb_path.exists() or names_path.exists():
        if emb_path.is_file() and names_path.is_file() and names_path.read_text() == sorted_names and np.array_equal(np.load(emb_path, allow_pickle=False), sorted_X):
            return
        raise FileExistsError(f'Refusing to overwrite different/incomplete embeddings at {folder}; use a fresh out_root')
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=folder, suffix='.npy', delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, sorted_X)
    os.replace(temporary, emb_path)
    names_path.write_text(sorted_names)


def convert(path, config, split, encoder, partitions, out_root):
    t = pq.read_table(path, columns=["id", "embedding"])
    ids = t["id"].to_pylist()
    X = np.stack(t["embedding"].to_numpy(zero_copy_only=False)).astype(np.float32)
    groups = defaultdict(list)                        # local folder -> row indices
    for i, id_ in enumerate(ids):
        parts = id_.split("/")
        stem = parts[-1]
        if config in ("attribute", "gender_x_age"):
            category = category_of(stem)
            if category not in {'gender', 'age', 'accent', 'emotion', 'gender_x_age'}:
                raise ValueError('Unknown released embedding category')
            d = Path("embeddings_attributes") / SYSTEMS[split] / category
        elif config == "overall":
            d = Path("embeddings_overall") / SYSTEMS[split] / f"sample{partitions[stem]}"
        elif config == "real_emilia":
            d = Path("embeddings_overall") / "real" / split.replace("partition_", "set_")
        else:                                         # sampling_sweep/<split>/<param>_<v>/seed_<s>/<stem>
            param, value = parts[2].rsplit("_", 1)
            seed = parts[3]
            if param not in SWEEP_PARAM or not math.isfinite(float(value)) or float(value) <= 0 or not re.fullmatch(r'seed_\d+', seed):
                raise ValueError('Invalid released sweep embedding ID')
            d = (Path("embeddings_param_sweep") / SYSTEMS[split] / f"{SWEEP_PARAM[param]}_{value}"
                 / seed)
        groups[d].append(i)
    for d, rows in groups.items():
        write_set(out_root / d / encoder, [ids[i].split("/")[-1] for i in rows], X[rows])
    return len(groups)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="dlion168/itts-speaker-diversity")
    ap.add_argument("--revision", default=None, help="branch, tag or commit (default: configs/release_assets.json)")
    ap.add_argument("--local_dir", type=Path, default=None,
                    help="use an existing snapshot of the dataset instead of downloading")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    ap.add_argument("--encoders", nargs="+", default=None, help="default: all released")
    ap.add_argument("--out_root", type=Path, default=Path("data"))
    args = ap.parse_args()
    args.revision = revision_for(args.repo, args.revision)

    root = args.local_dir
    if root is None:
        from huggingface_hub import snapshot_download
        enc = args.encoders or ["*"]
        patterns = [f"embeddings/{c}/*/{e}.parquet" for c in args.configs for e in enc]
        patterns.append("metadata/emilia_ids_overall_partition_*.txt")
        root = Path(snapshot_download(args.repo, repo_type="dataset", revision=args.revision,
                                      allow_patterns=patterns))

    partitions = {}
    for k in (1, 2, 3):
        f = root / "metadata" / f"emilia_ids_overall_partition_{k}.txt"
        if f.exists():
            partitions.update({i: k for i in f.read_text().split()})

    missing = [c for c in args.configs if not any((root / "embeddings" / c).glob("*/*.parquet"))]
    if missing:
        where = args.local_dir or f"{args.repo} at revision {args.revision or 'main'}"
        print(f"no released embeddings for {', '.join(missing)} in {where}.", file=sys.stderr)
        return 1

    for config in args.configs:
        files = sorted((root / "embeddings" / config).glob("*/*.parquet"))
        for f in files:
            if args.encoders and f.stem not in args.encoders:
                continue
            n = convert(f, config, f.parent.name, f.stem, partitions, args.out_root)
            print(f"{config}/{f.parent.name}/{f.stem}: {n} folders", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
