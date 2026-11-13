#!/usr/bin/env python3
"""In-distribution test score (paper Appendix C) of the released 6-class accent classifier:
the WavLM + Whisper softmax-averaged ensemble on the speaker-independent held-out test split.
(Each member's own test score is written by its training script at the end of training.)

Reads <splits>/test/*.parquet (columns audio_bytes, cls) and the two downloaded checkpoints;
writes results/accent_classifier/id/accent_6cls_ensemble.json.

Run (vox_profile env):
    python controllability/accent_classifier/eval_accent_ensemble.py [--splits DIR] [--out FILE]
"""
import argparse, io, json, os, sys, glob
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, soundfile as sf, torch
import torch.nn.functional as F
import torchaudio.functional as AF
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent.parent  # controllability/
REPO = Path(os.environ.get("VOX_PROFILE_DIR", HERE / "vox-profile-release"))
sys.path.insert(0, str(REPO / "src" / "model" / "accent"))
SR = 16000
WAVLM = HERE / "checkpoint" / "accent_wavlm" / "best.pt"
WHISPER = HERE / "checkpoint" / "accent_whisper" / "best.pt"
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


def decode(b):
    w, s = sf.read(io.BytesIO(b), dtype="float32")
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=Path,
                    default=Path(os.environ.get("ACC_SPLITS", Path(__file__).resolve().parent / "splits")))
    ap.add_argument("--out", type=Path,
                    default=HERE.parent / "results" / "accent_classifier" / "id" / "accent_6cls_ensemble.json")
    args = ap.parse_args()
    TEST, OUT = args.splits / "test", args.out
    members, labels = [], None
    for ck, kind in ((WAVLM, "wavlm"), (WHISPER, "whisper")):
        m, lb = load_member(str(ck), kind); members.append(m); labels = lb
        print(f"[load] {kind}", flush=True)
    lab2i = {l: i for i, l in enumerate(labels)}

    conf = defaultdict(Counter); n_ok = Counter(); n_tot = Counter()
    buf, gts = [], []

    def flush():
        nonlocal buf, gts
        if not buf:
            return
        preds = predict(members, buf)
        for gt, pi in zip(gts, preds):
            pred = labels[pi]
            conf[gt][pred] += 1; n_tot[gt] += 1; n_ok[gt] += int(pred == gt)
        buf, gts = [], []

    for shard in sorted(glob.glob(str(TEST / "*.parquet"))):
        t = pq.read_table(shard, columns=["audio_bytes", "cls"]).to_pydict()
        for b, lab in zip(t["audio_bytes"], t["cls"]):
            if lab not in lab2i:
                continue
            try:
                buf.append(decode(b)); gts.append(lab)
            except Exception:
                continue
            if len(buf) == 8:
                flush()
        print(f"[shard] {Path(shard).name} cum_n={sum(n_tot.values())}", flush=True)
    flush()

    regions = labels
    recall = {g: (n_ok[g] / n_tot[g] if n_tot[g] else 0.0) for g in regions}
    # macro-F1 from confusion
    f1 = {}
    for g in regions:
        tp = conf[g][g]
        fp = sum(conf[o][g] for o in regions if o != g)
        fn = sum(conf[g][o] for o in regions if o != g)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1[g] = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    acc = sum(n_ok.values()) / max(1, sum(n_tot.values()))
    uar = float(np.mean([recall[g] for g in regions]))
    macro_f1 = float(np.mean([f1[g] for g in regions]))
    res = {"tag": "6cls_ensemble_ID", "labels": labels,
           "acc": round(acc, 4), "macro_f1": round(macro_f1, 4), "uar": round(uar, 4),
           "per_class_f1": {g: round(f1[g], 4) for g in regions},
           "per_class_recall": {g: round(recall[g], 4) for g in regions},
           "per_class_n": {g: n_tot[g] for g in regions}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
