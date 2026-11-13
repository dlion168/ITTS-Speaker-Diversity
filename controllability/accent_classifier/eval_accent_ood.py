#!/usr/bin/env python3
"""Cross-corpus out-of-domain test of the accent classifier (paper Appendix C).

Scores the checkpoint(s) (WavLM and/or Whisper; both -> softmax-avg ensemble) on every
OOD corpus listed in a manifest: {"<corpus>": {"csv": path, "channel": ...}}, each CSV with
columns file, target_accent. The paper used SAA (Speech Accent Archive), NCHLT, EdAcc and VCTK;
those corpora have their own licences and are not redistributed, so build the CSVs locally.
Labels are read from the checkpoint. Reports,
per corpus and pooled: overall acc, UAR (mean over regions present), and per-region recall +
where errors go. `Other`-GT rows are skipped for a 6-class model (reported as n_unscorable).

Usage (vox_profile env):
  python controllability/accent_classifier/eval_accent_ood.py \
      --wavlm controllability/checkpoint/accent_wavlm/best.pt \
      --whisper controllability/checkpoint/accent_whisper/best.pt \
      --tag 6cls_ensemble --manifest ood_manifest.json --out results/accent_classifier/ood
"""
import argparse, csv, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

HERE = Path(__file__).resolve().parent.parent  # controllability/
REPO = Path(os.environ.get("VOX_PROFILE_DIR", HERE / "vox-profile-release"))
sys.path.insert(0, str(REPO / "src" / "model" / "accent"))
SR = 16000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mark_merged(m):
    import loralib as lora
    for mod in m.modules():
        if isinstance(mod, lora.LoRALayer):
            mod.merged = True


def load_member(ckpt, kind):
    ck = torch.load(ckpt, map_location=device)
    labels = ck["labels"]
    if kind == "wavlm":
        from wavlm_accent import WavLMWrapper as W
        pre = "wavlm_large"
    else:
        from whisper_accent import WhisperWrapper as W
        pre = "whisper_large"
    m = W(pretrain_model=pre, finetune_method="lora", lora_rank=16,
          output_class_num=len(labels), apply_gradient_reversal=False).to(device)
    m.load_state_dict(ck["state_dict"]); _mark_merged(m); m.eval()
    return m, labels


def load_wav(p):
    w, s = sf.read(p, dtype="float32")
    if w.ndim > 1:
        w = w.mean(-1)
    w = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
    if s != SR:
        w = AF.resample(w, s, SR)
    return w[: 15 * SR]


@torch.no_grad()
def predict(members, wavs):
    x = torch.zeros(len(wavs), max(w.numel() for w in wavs))
    lengths = torch.tensor([w.numel() for w in wavs])
    for i, w in enumerate(wavs):
        x[i, : w.numel()] = w
    x, lengths = x.to(device), lengths.to(device)
    probs = None
    for m in members:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = m(x, length=lengths)
        logits = out[0] if isinstance(out, tuple) else out
        p = F.softmax(logits.float(), -1)
        probs = p if probs is None else probs + p
    return (probs / len(members)).argmax(-1).cpu().numpy()


def score_csv(members, labels, csv_path, batch=8, label_col="target_accent"):
    conf = defaultdict(Counter); n_ok = Counter(); n_tot = Counter(); n_unscorable = 0
    labset = set(labels)
    buf, meta = [], []
    rows = list(csv.DictReader(open(csv_path)))

    def flush():
        nonlocal buf, meta
        if not buf:
            return
        preds = predict(members, buf)
        for gt, pi in zip(meta, preds):
            pred = labels[pi]
            conf[gt][pred] += 1; n_tot[gt] += 1; n_ok[gt] += int(pred == gt)
        buf, meta = [], []

    for r in rows:
        gt = r[label_col]
        if gt not in labset:          # e.g. 'Other' GT under a 6-class model
            n_unscorable += 1
            continue
        try:
            buf.append(load_wav(r["file"])); meta.append(gt)
        except Exception:
            continue
        if len(buf) == batch:
            flush()
    flush()
    regions = sorted(n_tot)
    recall = {g: n_ok[g] / n_tot[g] for g in regions}
    acc = sum(n_ok.values()) / max(1, sum(n_tot.values()))
    uar = float(np.mean(list(recall.values()))) if recall else 0.0
    return {"acc": round(acc, 4), "uar": round(uar, 4), "n": sum(n_tot.values()),
            "n_unscorable": n_unscorable,
            "per_region_recall": {g: round(recall[g], 4) for g in regions},
            "per_region_n": {g: n_tot[g] for g in regions},
            "confusion": {g: dict(conf[g]) for g in regions}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavlm"); ap.add_argument("--whisper")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=str(HERE.parent / "results" / "accent_classifier" / "ood"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--label_col", default="target_accent")
    a = ap.parse_args()

    members, labels = [], None
    for ck, kind in ((a.wavlm, "wavlm"), (a.whisper, "whisper")):
        if ck:
            m, lb = load_member(ck, kind)
            members.append(m); labels = lb
            print(f"[load] {kind} {ck} labels={lb}", flush=True)
    if not members:
        sys.exit("need --wavlm and/or --whisper")

    manifest = json.loads(Path(a.manifest).read_text())
    report = {"tag": a.tag, "labels": labels, "corpora": {}}
    pooled_ok = Counter(); pooled_tot = Counter()
    for corp, m in manifest.items():
        res = score_csv(members, labels, m["csv"], a.batch, a.label_col)
        report["corpora"][corp] = {"channel": m.get("channel"), **res}
        for g, n in res["per_region_n"].items():
            pooled_tot[g] += n; pooled_ok[g] += round(res["per_region_recall"][g] * n)
        print(f"\n=== {corp} ({m.get('channel')}) ===  acc={res['acc']} UAR={res['uar']} n={res['n']}"
              f"{'  unscorable(Other)='+str(res['n_unscorable']) if res['n_unscorable'] else ''}", flush=True)
        for g in res["per_region_recall"]:
            print(f"    {g:15s} recall {res['per_region_recall'][g]:.3f} (n={res['per_region_n'][g]})", flush=True)
    pooled = {g: round(pooled_ok[g] / pooled_tot[g], 4) for g in sorted(pooled_tot)}
    report["pooled_per_region_recall"] = pooled
    report["pooled_uar"] = round(float(np.mean(list(pooled.values()))), 4) if pooled else 0.0
    print("\n=== POOLED per-region recall (all corpora) ===", flush=True)
    for g in pooled:
        print(f"    {g:15s} recall {pooled[g]:.3f} (n={pooled_tot[g]})", flush=True)
    print(f"    pooled UAR {report['pooled_uar']}", flush=True)
    outp = Path(a.out) / f"ood_report_{a.tag}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {outp}", flush=True)


if __name__ == "__main__":
    main()
