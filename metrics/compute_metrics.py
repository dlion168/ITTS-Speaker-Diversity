#!/usr/bin/env python3
"""Command-line entry point for the diversity metrics.

    diversity   MPCD and Vendi of one embedding set per encoder (paper §4.1, §4.2)
    cross       Chamfer and MMD^2 between two embedding sets (encoder validation, §3.2)
    ci          within-value Vendi and MPCD with half-sampling CIs (§5); see half_sampling_ci.py

An embedding set is a folder with embeddings.npy and filenames.txt, as written by
embeddings/extract_embeddings.py. `diversity` also accepts a parent folder and then
evaluates every <encoder>/ subfolder in it.

Usage:
    python metrics/compute_metrics.py diversity data/embeddings_overall/voxcpm2/sample1 --csv out.csv
    python metrics/compute_metrics.py diversity data/embeddings_param_sweep/voxcpm2/cfg_1.0/seed_17 --n 3000
    python metrics/compute_metrics.py cross data/emb/a/speechbrain_ecapa data/emb/b/speechbrain_ecapa
    python metrics/compute_metrics.py ci --emb_root data/embeddings_attributes --out_dir results/attributes
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import half_sampling_ci
from chamfer import chamfer
from mmd import mmd2
from mpcd import mpcd
from vendi import vendi


def load_set(folder: Path) -> np.ndarray:
    """embeddings.npy of one set, without non-finite rows."""
    X = np.load(folder / "embeddings.npy")
    return X[np.isfinite(X).all(axis=1)]


def subsample_rows(X: np.ndarray, n: int, seed: int) -> np.ndarray:
    """n rows chosen with np.random.default_rng(seed), kept in their original order.

    The same seed picks the same row positions in every set of the same size; since rows are
    sorted by file name and every system shares the same text ids, that is the same texts."""
    if n is None or n >= len(X):
        return X
    idx = np.sort(np.random.default_rng(seed).choice(len(X), size=n, replace=False))
    return X[idx]


def set_folders(path: Path):
    if (path / "embeddings.npy").exists():
        return [path]
    return sorted(d for d in path.iterdir() if (d / "embeddings.npy").exists())


def diversity_main(argv):
    ap = argparse.ArgumentParser(prog="compute_metrics.py diversity",
                                 description="MPCD and Vendi of each embedding set.")
    ap.add_argument("paths", nargs="+", type=Path,
                    help="embedding-set folders, or parents of <encoder>/ folders")
    ap.add_argument("--n", type=int, default=None, help="evaluate a fixed random subset of n rows")
    ap.add_argument("--seed", type=int, default=42, help="seed of the --n subset (default 42)")
    ap.add_argument("--csv", type=Path, default=None, help="append rows to this CSV")
    args = ap.parse_args(argv)

    rows = []
    for path in args.paths:
        for folder in set_folders(path):
            X = subsample_rows(load_set(folder), args.n, args.seed)
            row = {"path": str(folder), "encoder": folder.name, "n": len(X),
                   "mpcd": round(mpcd(X), 6), "vendi": round(vendi(X), 4)}
            print(f"{row['path']}  n={row['n']}  MPCD={row['mpcd']:.6f}  Vendi={row['vendi']:.4f}",
                  flush=True)
            rows.append(row)
    if args.csv and rows:
        new = not args.csv.exists()
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
    return 0


def cross_main(argv):
    ap = argparse.ArgumentParser(prog="compute_metrics.py cross",
                                 description="Chamfer distance and MMD^2 between two embedding sets.")
    ap.add_argument("set_a", type=Path)
    ap.add_argument("set_b", type=Path)
    args = ap.parse_args(argv)
    X, Y = load_set(args.set_a), load_set(args.set_b)
    print(f"chamfer={chamfer(X, Y):.6f}  mmd2={mmd2(X, Y):.6f}  (n_a={len(X)}, n_b={len(Y)})")
    return 0


COMMANDS = {"diversity": diversity_main, "cross": cross_main, "ci": half_sampling_ci.main}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        return 2
    return COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
