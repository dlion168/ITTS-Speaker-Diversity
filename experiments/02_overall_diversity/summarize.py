#!/usr/bin/env python3
"""Paper §4.1: Vendi and MPCD of each system's 3 x 10,000 neutral-prompt utterances and of real speech.

Input:  <emb_root>/<system>/sample{1,2,3}/<encoder>/embeddings.npy   (generated speech)
        <emb_root>/real/set_{1,2,3}/<encoder>/embeddings.npy          (real Emilia recordings)
Output: results/overall/summary_diversity.csv with one row per (set, encoder):
        set, encoder, n_partitions, vendi_mean, vendi_std, mpcd_mean, mpcd_std
        (mean and population std over partitions; each partition uses all 10,000 utterances).

The paper's real-speech row uses partition 1 only (--real_sets set_1, the default).

Usage:
    python experiments/02_overall_diversity/summarize.py --emb_root data/embeddings_overall
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "metrics"))
from compute_metrics import load_set  # noqa: E402
from mpcd import mpcd  # noqa: E402
from vendi import vendi  # noqa: E402

SYSTEMS = ["moss-voicegen", "parler-large", "parler-mini", "capspeech",
           "qwen3-voicedesign", "voxinstruct", "voxcpm2"]
ENCODERS = ["modelscope_campplus", "modelscope_eres2net", "modelscope_eres2netv2", "nemo_titanet",
            "speechbrain_ecapa", "speechbrain_resnet", "wespeaker",
            "wespeaker_voxblink2_resnet152_16k"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/overall/summary_diversity.csv"))
    ap.add_argument("--systems", nargs="+", default=SYSTEMS)
    ap.add_argument("--real_sets", nargs="*", default=["set_1"], help="empty to skip real speech")
    args = ap.parse_args()

    sets = [("real", args.real_sets)] if args.real_sets else []
    sets += [(s, ["sample1", "sample2", "sample3"]) for s in args.systems]
    rows = []
    for name, parts in sets:
        for enc in ENCODERS:
            folders = [args.emb_root / name / p / enc for p in parts]
            missing = [f for f in folders if not (f / "embeddings.npy").exists()]
            if missing:
                print(f"[skip] {name}/{enc}: missing {missing[0]}")
                continue
            v, m = [], []
            for f in folders:
                X = load_set(f)
                v.append(vendi(X))
                m.append(mpcd(X))
            rows.append({"set": name, "encoder": enc, "n_partitions": len(parts),
                         "vendi_mean": round(np.mean(v), 4), "vendi_std": round(np.std(v), 4),
                         "mpcd_mean": round(np.mean(m), 6), "mpcd_std": round(np.std(m), 6)})
            print(f"{name:18s} {enc:34s} Vendi {rows[-1]['vendi_mean']:9.4f}  "
                  f"MPCD {rows[-1]['mpcd_mean']:.6f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"-> {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
