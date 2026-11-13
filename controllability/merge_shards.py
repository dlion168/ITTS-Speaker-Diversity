#!/usr/bin/env python3
"""Combine the per-utterance CSVs written by sharded `score_controllability.py` runs.

`--shard I/N` lets N processes split one folder across a single GPU, each writing its own
`controllability.<model>__<attr>.shardIofN_per_utt.csv`. This re-reads those per-utterance
predictions and applies the same aggregation formulas as the unsharded scorer
(controllability accuracy, UAR, per-class recall, confusion).

Rows that request both a gender and an age (`gender_x_age`) are also scored per requested
combination (`per_cell`, e.g. `male_child`): age accuracy, gender accuracy, and both correct.

Labels are written with the prompt names used in data/prompts/, configs/ and results/attributes/
(`north_american`, `elderly`, ...), not the classifier's own label names (`North America`,
`older`, ...) that the per-utterance CSVs hold.

Usage:
  python controllability/merge_shards.py --results_dir DIR [--out results/controllability/summary.json]
"""
import argparse
import csv
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
if __package__:
    from .scoring_run import validated_files
else:
    from scoring_run import validated_files

ATTRS = ("accent", "age", "gender", "emotion")
# classifier label (score_controllability.py) -> prompt name; other labels are already the same
PROMPT_NAME = {"North America": "north_american", "English": "british_isles",
               "South Asia": "south_asian", "Germanic": "germanic", "Oceania": "oceanian",
               "South African": "south_african", "older": "elderly"}


def prompt_name(label):
    return PROMPT_NAME.get(label, label)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default: <results_dir>/summary.json")
    ap.add_argument("--run-manifest", type=Path, help="scoring_run.json of the run: check that every shard and input is present before merging")
    ap.add_argument("--models", nargs='+', help="Optional model filter when merging without --run-manifest")
    ap.add_argument("--groups", nargs='+', help="Optional category filter when merging without --run-manifest")
    args = ap.parse_args()
    if args.run_manifest and (args.models or args.groups):
        ap.error('A run manifest defines the exact model/group scope; --models/--groups cannot narrow it')

    # label -> attr -> gt -> Counter[pred]
    conf = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    n_rows = Counter()
    # label -> "<gender>_<age>" -> Counter(n, age, gender, both)
    cells = defaultdict(lambda: defaultdict(Counter))
    files = validated_files(args.run_manifest) if args.run_manifest else sorted(args.results_dir.glob("controllability.*_per_utt.csv"))
    parsed, shard_sets = [], defaultdict(list)
    for path in files:
        match = re.fullmatch(r'controllability\.(.+?)(?:\.shard(\d+)of(\d+))?_per_utt\.csv', path.name)
        if not match or '__' not in match[1]:
            raise ValueError(f'Unsupported controllability result filename: {path.name}')
        label = match[1].replace('__', '/')
        model, group = label.split('/', 1)
        if args.models and model not in args.models or args.groups and group not in args.groups:
            continue
        shard_sets[label].append((int(match[2]), int(match[3])) if match[2] is not None else None)
        parsed.append((path, label))
    files = parsed
    if not files:
        raise SystemExit(f"no controllability.*_per_utt.csv under {args.results_dir}")
    for label, shards in shard_sets.items():
        if None in shards:
            if len(shards) != 1:
                raise ValueError(f'Mixed complete/sharded results for {label}; choose one verified run')
        else:
            totals = {total for _, total in shards}
            if len(totals) != 1:
                raise ValueError(f'Mixed shard counts for {label}')
            total = next(iter(totals))
            if total < 1 or {index for index, _ in shards} != set(range(total)) or len(shards) != total:
                raise ValueError(f'Incomplete/invalid shard coverage for {label}')

    seen = set()
    for f, label in files:
        with open(f, newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or 'file' not in reader.fieldnames:
                raise ValueError(f'Missing file column in {f}')
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f'Malformed CSV record in {f}')
                identifier = row.get('file')
                if not identifier or (label, identifier) in seen:
                    raise ValueError(f'Empty/duplicate sample key in {label}; refusing to double-count')
                seen.add((label, identifier))
                n_rows[label] += 1
                row = {k: prompt_name(v) for k, v in row.items()}
                for a in ATTRS:
                    gt, pred = row.get(f"target_{a}", ""), row.get(f"pred_{a}", "")
                    if gt and pred:
                        conf[label][a][gt][pred] += 1
                if row.get("target_gender") and row.get("target_age"):
                    age_ok = row["pred_age"] == row["target_age"]
                    gender_ok = row["pred_gender"] == row["target_gender"]
                    c = cells[label][f"{row['target_gender']}_{row['target_age']}"]
                    c["n"] += 1
                    c["age"] += age_ok
                    c["gender"] += gender_ok
                    c["both"] += age_ok and gender_ok

    summary = {}
    if not conf:
        raise ValueError('No valid target/prediction pairs; no summary was written')
    for label in sorted(conf):
        entry = {"n_rows": n_rows[label], "attributes": {}}
        for a, table in conf[label].items():
            total = sum(sum(c.values()) for c in table.values())
            hits = sum(c[g] for g, c in table.items())
            recalls = {g: (c[g] / sum(c.values()) if sum(c.values()) else 0.0)
                       for g, c in sorted(table.items())}
            entry["attributes"][a] = {
                "controllability_acc": round(hits / total, 4) if total else 0.0,
                "uar": round(sum(recalls.values()) / len(recalls), 4) if recalls else 0.0,
                "n": total,
                "per_class_recall": {g: round(v, 4) for g, v in recalls.items()},
                "confusion": {g: dict(c) for g, c in sorted(table.items())},
            }
        if cells[label]:
            entry["per_cell"] = {
                k: {"n": c["n"],
                    "age_acc": round(c["age"] / c["n"], 4),
                    "gender_acc": round(c["gender"] / c["n"], 4),
                    "both_acc": round(c["both"] / c["n"], 4)}
                for k, c in sorted(cells[label].items())
            }
        summary[label] = entry

    out = args.out or args.results_dir / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Validate/aggregate fully before replacing the output. Invalid input must
    # never turn a previous valid summary into an apparently successful subset.
    with tempfile.NamedTemporaryFile('w', dir=out.parent, prefix=out.name+'.', suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(summary, indent=2) + '\n')
    os.replace(temporary, out)

    print(f"\n=== controllability (acc | UAR) — {len(summary)} folders ===")
    print("model/attribute".ljust(34) + "".join(a.ljust(20) for a in ATTRS))
    for label, entry in summary.items():
        line = label[:33].ljust(34)
        for a in ATTRS:
            d = entry["attributes"].get(a)
            line += (f"{d['controllability_acc']:.3f}|{d['uar']:.3f} (n={d['n']})".ljust(20)
                     if d else "-".ljust(20))
        print(line)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
