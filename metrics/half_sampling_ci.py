#!/usr/bin/env python3
"""Within-value Vendi and MPCD with a 95% half-sampling confidence interval (paper §5, Appendix E).

For each (system, category, value), e.g. capspeech / age / child with n = 3,000 clips, draw B random
subsets of N clips *without replacement* and compute Vendi and MPCD on the same subsets:

    point estimate = mean over the B subsets (Vendi grows with N, so N is the same for every value)
    95% interval   = 2.5 / 97.5 percentiles of  mean + c * (draw - mean),   c = sqrt(N / (n - N))

The spread of the subset values matches the sampling uncertainty when N = n/2 (half-sampling);
for other N it is rescaled by c (c = 1 for n = 3,000; 0.577 for the 6,000-clip gender x age cells).
Every encoder uses the same seed and clip order, so draw b is the same N clips for all encoders;
averaging the encoders draw by draw gives B draws of the encoder-averaged metric, whose interval
is computed the same way. Method: docs/within_value_diversity_ci.md.

Input layout:  <emb_root>/<system>/<category>/<encoder>/{embeddings.npy, filenames.txt}
File names carry the requested value: age-child-s1-0001-T1 -> child,
genderage-male_child-s2-0001-F1_speakas-agefirst -> male_child.

Outputs in --out_dir, for metric in {vendi, mpcd}:
    <metric>_ci.csv                 encoder mean: one row per value, plus a <metric>__<encoder> column per encoder
    <metric>_ci_by_encoder.csv      one row per (encoder, value)
    <metric>_draws.npz              the B encoder-averaged draws per value (key system/category/value)
    <metric>_draws_by_encoder.npz   the B draws per encoder and value

Usage:
    python metrics/half_sampling_ci.py --emb_root data/embeddings_attributes --out_dir results/attributes
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

from mpcd import mpcd_from_sum
from vendi import vendi_from_gram

ENCODERS = ("modelscope_campplus", "modelscope_eres2net", "modelscope_eres2netv2", "nemo_titanet",
            "speechbrain_ecapa", "speechbrain_resnet", "wespeaker",
            "wespeaker_voxblink2_resnet152_16k")
CATEGORIES = ("gender", "age", "accent", "emotion", "gender_x_age")
KEY = ["system", "category", "value", "n", "N", "B", "c"]


def value_of(name):
    prefix = re.sub(r"-s\d+-\d{4}-.*$", "", name)
    return prefix.split("-", 1)[1]


def subsample(X, N, B, gen, chunk):
    """(Vendi, MPCD) of B random N-subsets of the unit rows of X, each as a (B,) array."""
    import torch
    n = X.shape[0]
    vendi, mpcd = [], []
    for start in range(0, B, chunk):
        b = min(chunk, B - start)
        idx = torch.rand(b, n, generator=gen, device=X.device).argsort(dim=1)[:, :N]
        Xs = X[idx]                                   # (b, N, d)
        vendi.append(vendi_from_gram(Xs.transpose(1, 2) @ Xs, N))
        mpcd.append(mpcd_from_sum(Xs.sum(1), N))
    return torch.cat(vendi).cpu().numpy(), torch.cat(mpcd).cpu().numpy()


def summarize(name, v, c):
    """Point estimate, c-rescaled 95% percentile interval and standard error of B draws."""
    mean = float(v.mean())
    adj = mean + c * (v - mean)
    return {
        name: round(mean, 6),
        "ci_lo": round(float(np.percentile(adj, 2.5)), 6),
        "ci_hi": round(float(np.percentile(adj, 97.5)), 6),
        "se": round(c * float(v.std(ddof=1)), 6),
    }


def one_encoder(emb_root, encoder, systems, categories, N, B, chunk, seed, dev):
    """{(system, category, value): {"key": ..., "c": c, "draws": {"vendi": (B,), "mpcd": (B,)}}}."""
    import torch
    out, t0 = {}, time.time()
    for system in systems:
        for cat in categories:
            d = emb_root / system / cat / encoder
            if not (d / "embeddings.npy").exists():
                print(f"[skip] {d} has no embeddings.npy")
                continue
            E = np.load(d / "embeddings.npy").astype(np.float64)
            names = (d / "filenames.txt").read_text().split()
            assert len(names) == len(E), f"{d}: {len(names)} names vs {len(E)} rows"
            groups = {}
            for i, nm in enumerate(names):
                if np.isfinite(E[i]).all():
                    groups.setdefault(value_of(nm), []).append(i)
            for value, ids in sorted(groups.items()):
                n = len(ids)
                if n <= N:
                    print(f"[skip] {system}/{cat}/{value}: n={n} <= N={N}")
                    continue
                X = torch.tensor(E[ids], device=dev)
                X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-8)
                gen = torch.Generator(device=dev).manual_seed(seed)
                v, p = subsample(X, N, B, gen, chunk)
                c = (N / (n - N)) ** 0.5
                out[(system, cat, value)] = {
                    "key": dict(zip(KEY, (system, cat, value, n, N, B, round(c, 4)))),
                    "c": c, "draws": {"vendi": v, "mpcd": p}}
                print(f"{encoder:34s} {system:18s} {cat:14s} {value:20s} n={n:5d}  "
                      f"Vendi {v.mean():7.2f}  MPCD {p.mean():.4f}  ({time.time() - t0:.0f}s)",
                      flush=True)
    return out


def main(argv=None):
    import pandas as pd
    import torch
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--encoders", nargs="*", default=list(ENCODERS))
    ap.add_argument("--systems", nargs="*", default=None, help="default: every folder in --emb_root")
    ap.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    ap.add_argument("--N", type=int, default=1500, help="subset size")
    ap.add_argument("--B", type=int, default=10000, help="number of subsets")
    ap.add_argument("--chunk", type=int, default=500, help="subsets per GPU batch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    systems = args.systems or sorted(p.name for p in args.emb_root.iterdir() if p.is_dir())
    encoders = sorted(args.encoders)
    per_enc = {e: one_encoder(args.emb_root, e, systems, args.categories,
                              args.N, args.B, args.chunk, args.seed, dev) for e in encoders}
    keys = list(per_enc[encoders[0]])
    for e in encoders:
        assert list(per_enc[e]) == keys, f"{e}: values differ from {encoders[0]}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for m in ("vendi", "mpcd"):
        rows, long, mean_draws, enc_draws = [], [], {}, {}
        for k in keys:
            key, name = per_enc[encoders[0]][k]["key"], "/".join(k)
            enc_rows = {}
            for e in encoders:
                r = per_enc[e][k]
                enc_rows[e] = {**r["key"], **summarize(m, r["draws"][m], r["c"])}
                long.append({"encoder": e, **enc_rows[e]})
                enc_draws[f"{e}/{name}"] = r["draws"][m].astype(np.float32)
            # average the float32 draws as saved, so the npz files reproduce the table exactly
            v = np.stack([enc_draws[f"{e}/{name}"] for e in encoders]).astype(np.float64).mean(0)
            mean_draws[name] = v.astype(np.float32)
            row = {**key, "n_encoders": len(encoders), **summarize(m, v, key["c"])}
            row.update({f"{m}__{e}": enc_rows[e][m] for e in encoders})
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.out_dir / f"{m}_ci.csv", index=False)
        pd.DataFrame(long).to_csv(args.out_dir / f"{m}_ci_by_encoder.csv", index=False)
        np.savez_compressed(args.out_dir / f"{m}_draws.npz", **mean_draws)
        np.savez_compressed(args.out_dir / f"{m}_draws_by_encoder.npz", **enc_draws)
        print(f"-> {args.out_dir / f'{m}_ci.csv'}: {len(rows)} values x {len(encoders)} encoders")
    return 0


if __name__ == "__main__":
    sys.exit(main())
