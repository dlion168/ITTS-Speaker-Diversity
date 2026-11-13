"""Step 3 of the encoder validation (paper §3.2, Appendix A): metric values per subset and per k.

For every encoder and every subset S of data/librispeech_validation/subsets.jsonl:
  MPCD and Vendi (cosine, within S), Chamfer and MMD^2 between S and the reference pool R
  (raw embeddings, Euclidean; MMD^2 with a Gaussian kernel, median-heuristic bandwidth, biased
  estimator; metrics/chamfer.py, metrics/mmd.py).

    python experiments/01_encoder_validation/compute_curves.py --emb_root <dir with <encoder>/embeddings.npy>

Writes <out_dir>/metric_values.csv (one row per encoder, metric, subset) and
<out_dir>/metric_curves.csv (mean, std and sem over the 100 subsets of each k).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metrics"))
from chamfer import chamfer  # noqa: E402
from mmd import mmd2  # noqa: E402
from mpcd import mpcd  # noqa: E402
from vendi import vendi  # noqa: E402

# A faithful metric moves this way as the number of speakers k grows.
DIRECTION = {"mpcd": "up", "vendi": "up", "chamfer": "down", "mmd": "down"}


def load_encoder(d: Path):
    X = np.load(d / "embeddings.npy").astype(np.float64)
    names = [l.strip().rsplit(".", 1)[0] for l in open(d / "filenames.txt")]
    return X, {n: i for i, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb_root", required=True, type=Path)
    ap.add_argument("--data_dir", type=Path, default=ROOT / "data/librispeech_validation")
    ap.add_argument("--encoders", nargs="+", help="default: every folder under --emb_root")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "results/encoder_validation")
    a = ap.parse_args()

    subsets = [json.loads(l) for l in open(a.data_dir / "subsets.jsonl")]
    reference = (a.data_dir / "reference_pool.txt").read_text().split()
    encoders = a.encoders or sorted(p.name for p in a.emb_root.iterdir() if (p / "embeddings.npy").exists())

    rows = []
    for enc in encoders:
        X, idx = load_encoder(a.emb_root / enc)
        R = X[[idx[u] for u in reference]]
        for s in subsets:
            A = X[[idx[u] for u in s["utts"]]]
            for metric, value in (("mpcd", mpcd(A)), ("vendi", vendi(A)),
                                  ("chamfer", chamfer(A, R)), ("mmd", mmd2(A, R))):
                rows.append(dict(encoder=enc, metric=metric, k=s["k"], subset_id=s["subset_id"],
                                 utts_per_speaker=s["utts_per_speaker"], value=value))
        print(f"{enc}: {len(subsets)} subsets", flush=True)

    values = pd.DataFrame(rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    values.to_csv(a.out_dir / "metric_values.csv", index=False)
    g = values.groupby(["encoder", "metric", "k", "utts_per_speaker"]).value
    curves = pd.concat([g.mean().rename("value_mean"), g.std().rename("value_std"), g.size().rename("n_draws")], axis=1).reset_index()
    curves["n_set"] = len(subsets[0]["utts"])
    curves["sem"] = curves.value_std / np.sqrt(curves.n_draws)
    curves["faithful_direction"] = curves.metric.map(DIRECTION)
    curves = curves[["encoder", "metric", "k", "n_set", "utts_per_speaker", "value_mean", "value_std", "sem",
                     "n_draws", "faithful_direction"]].sort_values(["encoder", "metric", "k"])
    curves.to_csv(a.out_dir / "metric_curves.csv", index=False)
    print(f"-> {a.out_dir / 'metric_values.csv'} ({len(values)} rows), {a.out_dir / 'metric_curves.csv'} ({len(curves)} rows)")


if __name__ == "__main__":
    main()
