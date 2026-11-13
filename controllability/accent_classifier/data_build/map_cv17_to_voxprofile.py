#!/usr/bin/env python3
"""Map the raw CV17 accent-string tallies (from scan_cv17_accents.py) onto the
Vox-Profile 16-class narrow taxonomy and report HOURS per class.

CV accent labels are free text and often compound ("United States English,New
York English"). We take the PRIMARY token (before the first comma) and apply
substring rules. Unmapped primaries are reported under __unmapped__ so nothing
is silently dropped. Reads $ACC_DATA_ROOT/cv17_accent_scan/raw_tally.json.
"""
import os
import json
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path(os.environ.get("ACC_DATA_ROOT", Path(__file__).resolve().parent.parent / "data"))

RAW = DATA_ROOT / "cv17_accent_scan" / "raw_tally.json"

# substring -> Vox-Profile narrow class. First match wins; order matters
# (specific before generic, e.g. 'northern irish' before 'irish').
RULES = [
    ("united states", "North America"), ("american", "North America"),
    ("canadian", "North America"), ("canada", "North America"),
    ("northern irish", "Northern Irish"),
    ("england english", "English"), ("english english", "English"),
    ("scottish", "Scottish"), ("welsh", "Welsh"),
    ("ireland", "Irish"), ("irish", "Irish"),
    ("australian", "Oceania"), ("new zealand", "Oceania"),
    ("india and south asia", "South Asia"), ("indian", "South Asia"),
    ("india", "South Asia"), ("pakistan", "South Asia"), ("sri lanka", "South Asia"),
    ("nepal", "South Asia"), ("banglad", "South Asia"), ("south asia", "South Asia"),
    ("hong kong", "East Asia"), ("china", "East Asia"), ("chinese", "East Asia"),
    ("east asia", "East Asia"),
    ("malaysia", "Southeast Asia"), ("singapore", "Southeast Asia"),
    ("filipino", "Southeast Asia"), ("philippine", "Southeast Asia"),
    ("vietnam", "Southeast Asia"), ("indonesia", "Southeast Asia"),
    ("thai", "Southeast Asia"), ("southeast asia", "Southeast Asia"),
    # South African must precede the generic 'african' -> Other fallback
    ("south african", "South African"), ("southern african", "South African"),
    ("south africa", "South African"),
    ("german", "Germanic"), ("dutch", "Germanic"), ("austrian", "Germanic"),
    ("spanish", "Romance"), ("hispanic", "Romance"), ("latin", "Romance"),
    ("italian", "Romance"), ("french", "Romance"), ("portug", "Romance"),
    ("romanian", "Romance"),
    ("russian", "Slavic"), ("polish", "Slavic"), ("slav", "Slavic"),
    ("ukrainian", "Slavic"), ("eastern european", "Slavic"), ("bulgarian", "Slavic"),
    ("arabic", "Semitic"), ("hebrew", "Semitic"), ("israeli", "Semitic"),
    ("northumbrian", "English"), ("british", "English"),
    # non-South-African Africa, Caribbean, South Atlantic -> Other
    ("south atlantic", "Other"), ("west indies", "Other"), ("caribbean", "Other"),
    ("bermuda", "Other"), ("jamaica", "Other"), ("trinidad", "Other"),
    ("nigeria", "Other"), ("kenya", "Other"), ("ghana", "Other"), ("african", "Other"),
]
NARROW = ["North America", "English", "Scottish", "Welsh", "Irish", "Northern Irish",
          "South Asia", "East Asia", "Southeast Asia", "Oceania", "South African",
          "Germanic", "Romance", "Slavic", "Semitic", "Other"]


def classify(raw):
    primary = raw.split(",")[0].strip().lower()
    for needle, cls in RULES:
        if needle in primary:
            return cls
    # fall back to full string if primary didn't hit
    full = raw.lower()
    for needle, cls in RULES:
        if needle in full:
            return cls
    return "__unmapped__"


def main():
    d = json.loads(RAW.read_text())
    sec, n = d["sec"], d["n"]
    done = len(d.get("done_shards", []))
    tot = d.get("n_shards_total", "?")
    cls_sec, cls_n = defaultdict(float), defaultdict(int)
    unmapped = defaultdict(float)
    for raw, s in sec.items():
        c = classify(raw)
        cls_sec[c] += s
        cls_n[c] += n.get(raw, 0)
        if c == "__unmapped__":
            unmapped[raw] += s

    print(f"# CV17 validated accent scan  ({done}/{tot} shards processed)\n")
    print(f"{'Vox-Profile class':<16} {'hours':>8} {'clips':>10}")
    print("-" * 38)
    th = tc = 0.0
    for c in NARROW:
        h = cls_sec[c] / 3600
        print(f"{c:<16} {h:>8.1f} {cls_n[c]:>10,}")
        th += h
        tc += cls_n[c]
    um_h = cls_sec['__unmapped__'] / 3600
    print("-" * 38)
    print(f"{'(mapped total)':<16} {th:>8.1f} {int(tc):>10,}")
    print(f"{'__unmapped__':<16} {um_h:>8.1f} {cls_n['__unmapped__']:>10,}")
    if unmapped:
        top = sorted(unmapped.items(), key=lambda x: -x[1])[:15]
        print("\nTop unmapped raw accents (hours):")
        for raw, s in top:
            print(f"  {s/3600:6.1f}  {raw[:70]}")


if __name__ == "__main__":
    main()
