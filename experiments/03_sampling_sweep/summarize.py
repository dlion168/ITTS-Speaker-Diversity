#!/usr/bin/env python3
"""Paper §4.2: Vendi and MPCD across sampling temperature / CFG scale, and the real-speech baseline.

Input:  <sweep_root>/<system>/<temp|cfg>_<value>/seed_<s>/<encoder>/embeddings.npy
        <real_root>/set_1/<encoder>/embeddings.npy
Output: results/sampling_sweep/param_sweep_seeded_diversity.csv
            system, param_kind, param, encoder, seed, n, vendi, mpcd
        results/overall/real_vendi_n3000.csv   (the dashed real-speech line in the figure)
            set, encoder, n, seed, vendi, mpcd

Vendi grows with set size, so every configuration is scored on the same n = 3,000 of its 10,000
utterances, and the real-speech baseline on 3,000 utterances of real partition 1.

Usage:
    python experiments/03_sampling_sweep/summarize.py --sweep_root data/embeddings_param_sweep \\
        --real_root data/embeddings_overall/real
"""
import argparse
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "metrics"))
from compute_metrics import load_set, subsample_rows  # noqa: E402
from mpcd import mpcd  # noqa: E402
from vendi import vendi  # noqa: E402

SYSTEMS = ["moss-voicegen", "voxcpm2", "voxinstruct"]
PAPER_SYSTEMS = set(SYSTEMS) | {"parler-mini", "parler-large", "capspeech", "qwen3-voicedesign"}
ENCODERS = ["modelscope_campplus", "modelscope_eres2net", "modelscope_eres2netv2", "nemo_titanet",
            "speechbrain_ecapa", "speechbrain_resnet", "wespeaker",
            "wespeaker_voxblink2_resnet152_16k"]
N, SEED = 3000, 42


def selected_systems(explicit=None):
    """CLI wins over SYSTEMS; known non-sweep paper systems are ignored."""
    names = explicit if explicit is not None else os.environ.get("SYSTEMS", " ".join(SYSTEMS)).split()
    if set(names) - PAPER_SYSTEMS:
        raise ValueError("SYSTEMS contains an unknown paper system")
    selected = list(dict.fromkeys(name for name in names if name in SYSTEMS))
    if not selected:
        raise ValueError("SYSTEMS has no selected sampling-sweep systems")
    return selected


def discover_leaves(root, systems):
    """Validate all selected groups without reading arrays or creating outputs."""
    result = []
    for system in systems:
        directory = root / system
        if not directory.is_dir():
            raise ValueError(f"no embeddings for selected system {system} under {root}")
        kind = "temp" if system == "moss-voicegen" else "cfg"
        params = []
        for path in directory.iterdir():
            if not path.is_dir() or not path.name.startswith(kind + "_"):
                continue
            try:
                value = float(path.name[len(kind) + 1:])
            except ValueError as error:
                raise ValueError(f"invalid sweep parameter directory: {path}") from error
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"sweep parameter must be finite and positive: {path}")
            params.append((value, path))
        if not params:
            raise ValueError(f"no {kind} parameter directories for selected system {system}")
        for value, param in sorted(params, key=lambda item: item[0]):
            seeds = sorted((path for path in param.iterdir() if path.is_dir() and re.fullmatch(r"seed_\d+", path.name)),
                           key=lambda path: int(path.name[5:]))
            if not seeds:
                raise ValueError(f"no seed directories under {param}")
            for leaf in seeds:
                for encoder in ENCODERS:
                    if not (leaf / encoder / "embeddings.npy").is_file():
                        legacy = leaf / leaf.name / encoder / "embeddings.npy"
                        if legacy.is_file():
                            raise ValueError(f"embeddings one seed level too deep at {legacy}; expected {leaf / encoder / 'embeddings.npy'}")
                        raise ValueError(f"missing selected embedding array: {leaf / encoder / 'embeddings.npy'}")
                result.append((system, kind, value, int(leaf.name[5:]), leaf))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep_root", type=Path, required=True)
    ap.add_argument("--real_root", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=Path("results"))
    ap.add_argument("--real_subset", choices=["permutation", "choice"], default="permutation")
    ap.add_argument("--systems", nargs="+", choices=SYSTEMS, default=None,
                    help="Selected sweep systems; otherwise use SYSTEMS or all three")
    args = ap.parse_args()
    try:
        systems = selected_systems(args.systems)
        leaves = discover_leaves(args.sweep_root, systems)
    except ValueError as error:
        ap.error(str(error))

    rows = []
    for system, kind, value, seed, leaf in leaves:
        for enc in ENCODERS:
            X = subsample_rows(load_set(leaf / enc), N, SEED)
            rows.append({"system": system, "param_kind": kind, "param": value,
                         "encoder": enc, "seed": seed, "n": len(X),
                         "vendi": vendi(X), "mpcd": mpcd(X)})
            r = rows[-1]
            print(f"{system:14s} {leaf.parent.name:9s} {leaf.name:8s} {enc:34s} "
                  f"Vendi {r['vendi']:8.4f}  MPCD {r['mpcd']:.6f}", flush=True)
    out = args.out_dir / "sampling_sweep" / "param_sweep_seeded_diversity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out} ({len(rows)} rows)")

    if args.real_root:
        real = []
        for enc in ENCODERS:
            X = load_set(args.real_root / "set_1" / enc)
            if args.real_subset == "permutation":
                X = X[np.random.default_rng(SEED).permutation(len(X))[:N]]
            else:
                X = subsample_rows(X, N, SEED)
            real.append({"set": "real", "encoder": enc, "n": len(X), "seed": SEED,
                         "vendi": round(vendi(X), 4), "mpcd": round(mpcd(X), 6)})
            print(f"real set_1 {enc:34s} Vendi {real[-1]['vendi']:8.4f}  MPCD {real[-1]['mpcd']:.6f}")
        out = args.out_dir / "overall" / "real_vendi_n3000.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(real).to_csv(out, index=False)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
