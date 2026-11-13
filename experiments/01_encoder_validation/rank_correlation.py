"""Step 4 of the encoder validation (paper §3.2, Appendix A): which encoders track k?

From metric_values.csv / metric_curves.csv (compute_curves.py), per encoder and metric:
  spearman_rho       Spearman rho between the metric and k over all subsets (1,400)
  ratio_k240_k6      mean at the largest k / mean at the smallest k
  steps_separated    adjacent k steps (13) whose change in the mean, in the faithful direction,
                     exceeds the larger of the two subset standard deviations
  monotonic          the mean moves in the faithful direction at every step
The paper keeps an encoder when its Vendi score separates at least 12 of the 13 steps.

    python experiments/01_encoder_validation/rank_correlation.py

Writes <dir>/rank_correlation.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=ROOT / "results/encoder_validation")
    ap.add_argument("--min_steps", type=int, default=12, help="Vendi steps needed to keep an encoder")
    a = ap.parse_args()

    values = pd.read_csv(a.dir / "metric_values.csv")
    curves = pd.read_csv(a.dir / "metric_curves.csv")
    rows = []
    for (enc, metric), c in curves.sort_values("k").groupby(["encoder", "metric"]):
        v = values[(values.encoder == enc) & (values.metric == metric)]
        sign = 1 if c.faithful_direction.iloc[0] == "up" else -1
        m, s = c.value_mean.to_numpy(), c.value_std.to_numpy()
        step = sign * np.diff(m)
        rows.append(dict(encoder=enc, metric=metric, n_subsets=len(v),
                         spearman_rho=spearmanr(v.k, v.value).statistic,
                         ratio_k240_k6=m[-1] / m[0] if m[0] else np.nan,
                         steps_separated=int((step > np.maximum(s[:-1], s[1:])).sum()),
                         n_steps=len(step), monotonic=bool((step > 0).all())))
    out = pd.DataFrame(rows)
    vs = out[out.metric == "vendi"].set_index("encoder")
    out["retained"] = out.encoder.map(vs.steps_separated >= a.min_steps)
    out.to_csv(a.dir / "rank_correlation.csv", index=False)
    print(vs.join(out[out.metric == "vendi"].set_index("encoder").retained)[
        ["spearman_rho", "ratio_k240_k6", "steps_separated", "monotonic", "retained"]].round(3).to_string())
    print(f"-> {a.dir / 'rank_correlation.csv'}")


if __name__ == "__main__":
    main()
