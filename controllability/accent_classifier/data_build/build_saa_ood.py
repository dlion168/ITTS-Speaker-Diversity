#!/usr/bin/env python3
"""Build a balanced SAA out-of-domain set from HamdanXI/speech-accent-archive-v2.

Accent test set, out-of-domain for our classifier (SAA is not in training): 5 regions
(no South African -- SAA has only 9 such speakers): North America / English / Oceania /
South Asia / Germanic, balanced to the smallest region.

Mapping (speaker metadata -> our label space):
  accent: native_language german/dutch -> Germanic;
          native english + country{usa,canada} -> North America;
          native english + country{uk,england,scotland,wales,ireland,n.ireland} -> English;
          native english + country{australia,new zealand} -> Oceania;
          country{india,pakistan,bangladesh,sri lanka,nepal} (any L1, = Indian English) -> South Asia.

Writes $ACC_DATA_ROOT/saa_ood/{audio/*.wav, saa_accent.csv}.
CSV columns: file(abs wav path), target_accent. Score with eval_accent_ood.py.
Run: python controllability/accent_classifier/data_build/build_saa_ood.py
"""
import os
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

DATA_ROOT = Path(os.environ.get("ACC_DATA_ROOT", Path(__file__).resolve().parent.parent / "data"))

REPO = "HamdanXI/speech-accent-archive-v2"
HERE = Path(__file__).resolve().parent.parent
OUT = DATA_ROOT / "saa_ood"
AUD = OUT / "audio"
SEED = 42

NA_C = {"usa", "canada"}
EN_C = {"uk", "england", "scotland", "wales", "ireland", "northern ireland", "united kingdom"}
OC_C = {"australia", "new zealand"}
SA_C = {"india", "pakistan", "bangladesh", "sri lanka", "nepal"}
ACCENT_ORDER = ["North America", "English", "Oceania", "South Asia", "Germanic"]


def accent_region(nat, ctry):
    nat = (nat or "").lower().strip(); ctry = (ctry or "").lower().strip()
    if nat in ("german", "dutch"):
        return "Germanic"
    if ctry in SA_C:
        return "South Asia"
    if nat == "english":
        if ctry in NA_C: return "North America"
        if ctry in EN_C: return "English"
        if ctry in OC_C: return "Oceania"
    return None


def main():
    AUD.mkdir(parents=True, exist_ok=True)
    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet")]
    print(f"[meta] {len(files)} parquet files")

    # ---- pass 1: metadata -> per-label pools of (filename) ----
    meta = {}   # filename -> dict(accent, parquet)
    acc_pool = {k: [] for k in ACCENT_ORDER}
    local = {}
    for f in files:
        lp = hf_hub_download(REPO, f, repo_type="dataset")
        local[f] = lp
        t = pq.read_table(lp, columns=["filename", "native_language", "country"])
        d = t.to_pydict()
        for fn, nat, ctry in zip(d["filename"], d["native_language"], d["country"]):
            reg = accent_region(nat, ctry)
            meta[fn] = {"accent": reg, "pq": f}
            if reg: acc_pool[reg].append(fn)
    print("[accent pools]", {k: len(v) for k, v in acc_pool.items()})

    rng = random.Random(SEED)
    n_acc = min(len(v) for v in acc_pool.values())
    print(f"[balance] accent -> {n_acc}/region ({n_acc*len(ACCENT_ORDER)} total)")

    sel_acc = {}
    for k in ACCENT_ORDER:
        for fn in rng.sample(acc_pool[k], n_acc): sel_acc[fn] = k
    wanted = set(sel_acc)
    print(f"[select] {len(wanted)} unique clips to extract")

    # ---- pass 2: read audio only for selected filenames, write wavs ----
    by_pq = {}
    for fn in wanted:
        by_pq.setdefault(meta[fn]["pq"], set()).add(fn)
    written = set()
    for f, fns in by_pq.items():
        t = pq.read_table(local[f], columns=["filename", "audio"])
        names = t.column("filename").to_pylist()
        audio = t.column("audio").to_pylist()
        for nm, au in zip(names, audio):
            if nm in fns:
                arr = np.asarray(au["array"], dtype=np.float32)
                sr = int(au["sampling_rate"])
                wav = AUD / f"{nm}.wav"
                sf.write(str(wav), arr, sr)
                written.add(nm)
    print(f"[audio] wrote {len(written)} wavs")

    def dump(csvname, col, sel):
        with open(OUT / csvname, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["file", col])
            n = 0
            for fn, lab in sel.items():
                wav = AUD / f"{fn}.wav"
                if wav.is_file():
                    w.writerow([str(wav), lab]); n += 1
        print(f"[csv] {csvname}: {n} rows")

    dump("saa_accent.csv", "target_accent", sel_acc)
    print("[done]")


if __name__ == "__main__":
    main()
