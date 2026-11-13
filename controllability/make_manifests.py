#!/usr/bin/env python3
"""Build the controllability scorer's manifest CSV for every (model, attribute) folder.

The attribute-conditioned outputs are laid out as

    <root>/<model>/<attribute>/<attr>-<value>-<sX>-<idx>-<take>.wav

so the *requested* attribute value is recoverable straight from the filename. This
writes one `controllability_manifest.csv` per folder with the columns
`score_controllability.py` expects (`file` + `target_<attr>`), which is all the scorer
needs on top of the WAVs themselves.

Categories: accent (6 values), age (4), gender (female|male), emotion (angry|happy|neutral|sad),
and gender_x_age (8 cells, both a gender and an age target).

Usage:
  python controllability/make_manifests.py --root data/outputs/attributes [--models m1 m2]
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

# folder name -> the target_* columns its filenames encode
CATEGORIES = {
    "accent": ("accent",),
    "age": ("age",),
    "gender": ("gender",),
    "gender_x_age": ("gender", "age"),
    "emotion": ("emotion",),
}

GENDERS = ("female", "male")


def split_value(category: str, value: str) -> dict:
    """filename value field -> {attr: raw target value} (verbatim prompt vocab)."""
    if category == "gender_x_age":
        # "female_child", "male_middle_aged" -> gender is the first underscore field
        gender, _, age = value.partition("_")
        if gender not in GENDERS or not age:
            raise ValueError(f"cannot split gender_x_age value {value!r}")
        return {"gender": gender, "age": age}
    return {CATEGORIES[category][0]: value}


def build(folder: Path, category: str) -> tuple[int, Counter]:
    cols = CATEGORIES[category]
    rows, seen = [], Counter()
    for wav in sorted(folder.glob("*.wav")):
        parts = wav.stem.split("-")
        if len(parts) < 3:
            raise ValueError(f"unexpected filename: {wav.name}")
        targets = split_value(category, parts[1])
        rows.append({"file": wav.name, **{f"target_{a}": targets[a] for a in cols}})
        seen[parts[1]] += 1
    out = folder / "controllability_manifest.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["file"] + [f"target_{a}" for a in cols])
        wr.writeheader()
        wr.writerows(rows)
    return len(rows), seen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="outputs/attributes root")
    ap.add_argument("--models", nargs="+", default=None, help="default: every sub-dir of --root")
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=list(CATEGORIES))
    args = ap.parse_args()

    models = args.models or sorted(d.name for d in args.root.iterdir() if d.is_dir())
    total = 0
    for model in models:
        for category in args.categories:
            folder = args.root / model / category
            if not folder.is_dir():
                print(f"[skip] {model}/{category}: no such folder", flush=True)
                continue
            n, seen = build(folder, category)
            total += n
            values = " ".join(f"{v}={c}" for v, c in sorted(seen.items()))
            print(f"[ok] {model}/{category}: {n} rows  ({values})", flush=True)
    print(f"[done] {total} rows across {len(models)} models", flush=True)


if __name__ == "__main__":
    main()
