#!/usr/bin/env python3
"""Train the 6-class WavLM-Large accent classifier, following Vox-Profile's recipe.

Model: src/model/accent/wavlm_accent.py WavLMWrapper (LoRA r16 on upper-half FFN,
learnable layer-weighted-sum, pointwise-conv head, mean-pool). Backbone frozen
except LoRA. Loss: cross-entropy, inverse-frequency class-weighted with ACC_WEIGHTED=1 (as released).
Optim: Adam lr 5e-4, batch 16, 15 epochs. Audio: mp3->16 kHz mono, already
filtered to [3,15] s upstream. AMP (fp16).

Run in the vox_profile environment (envs/vox_profile.lock.txt):
  python controllability/accent_classifier/train_accent.py   (vox_profile env; see train_accent*.sbatch)
"""
import io
import json
import os
import random
import sys
import time
from pathlib import Path

import audiomentations as A
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.functional as AF
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

# Vox-Profile training augmentations (applied to train waveforms only, paper order):
# Gaussian noise p=1.0 SNR 3-30 dB; time mask p=1.0 ratio 10-15%; time stretch p=1.0
# rate 0.9-1.1; polarity inversion p=0.5.
AUG = A.Compose([
    A.AddGaussianSNR(min_snr_db=3.0, max_snr_db=30.0, p=1.0),
    A.TimeMask(min_band_part=0.10, max_band_part=0.15, p=1.0),
    A.TimeStretch(min_rate=0.9, max_rate=1.1, leave_length_unchanged=True, p=1.0),
    A.PolarityInversion(p=0.5),
])

HERE = Path(__file__).resolve().parent.parent  # controllability/
VOX_PROFILE = Path(os.environ.get("VOX_PROFILE_DIR", HERE / "vox-profile-release"))
sys.path.append(str(VOX_PROFILE / "src" / "model" / "accent"))
if __package__:
    from .data_build.source_validation import validate_training_splits
else:
    from data_build.source_validation import validate_training_splits

# Settings of the released 6-class model; override with env vars.
SPLITS = Path(os.environ.get("ACC_SPLITS", Path(__file__).resolve().parent / "splits"))
# Never the download location checkpoint/accent_wavlm/: that best.pt links into the HF cache.
CKPT = Path(os.environ.get("ACC_CKPT", HERE / "checkpoint" / "trained" / "accent_wavlm"))
LABELS = os.environ.get("ACC_LABELS",
                        "North America,English,South Asia,Germanic,Oceania,South African").split(",")
NUM_CLASS = len(LABELS)
SR = 16000
# Paper uses effective batch 16 (with LoRA). A 24 GB 3090 can't fit 16 x 15 s through
# WavLM-Large, so use physical batch 8 x grad-accum 2 = effective 16 (identical update).
BATCH = 8
ACCUM = 2
EPOCHS = 15
LR = 5e-4
SEED = 1234
device = "cuda"


def _decode(b):
    wav, srr = sf.read(io.BytesIO(b), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(-1)
    if srr != SR:
        wav = AF.resample(torch.from_numpy(wav), srr, SR).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32)


class ShardDataset(IterableDataset):
    """Reads each shuffled parquet shard once per epoch (shard order + within-shard +
    buffer shuffle). make_splits already globally shuffled rows into shards, so this
    keeps randomness while reading each file exactly once. Shards partitioned across
    workers. Set `train=False` for deterministic single-pass eval."""
    def __init__(self, split, train=True, buffer=4096):
        self.files = sorted((SPLITS / split).glob("*.parquet"))
        self.train = train
        self.buffer = buffer
        self.epoch = 0
        n = 0
        labels = []
        for f in self.files:
            lab = pq.ParquetFile(f).read(columns=["label"]).column("label").to_pylist()
            n += len(lab)
            labels.extend(int(x) for x in lab)
        self.n = n
        self.labels = labels

    def __len__(self):
        return self.n

    def set_epoch(self, e):
        self.epoch = e

    def _shards_for_worker(self):
        wi = get_worker_info()
        files = list(self.files)
        if self.train:
            random.Random(SEED + self.epoch).shuffle(files)
        if wi is not None:
            files = files[wi.id::wi.num_workers]
        return files

    def __iter__(self):
        buf = []
        rng = random.Random(SEED + 7919 * (self.epoch + 1))
        # deterministic-but-epoch-varying seed for audiomentations (numpy global RNG)
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        np.random.seed((SEED + 100003 * self.epoch + wid) % (2 ** 31 - 1))
        for f in self._shards_for_worker():
            t = pq.ParquetFile(f).read(columns=["audio_bytes", "label"])
            ab = t.column("audio_bytes")
            lb = t.column("label")
            order = list(range(len(lb)))
            if self.train:
                rng.shuffle(order)
            for r in order:
                wav = _decode(ab[r].as_py())
                if self.train:
                    wav = AUG(samples=wav, sample_rate=SR)
                item = (torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)),
                        int(lb[r].as_py()))
                if not self.train:
                    yield item
                    continue
                buf.append(item)
                if len(buf) >= self.buffer:
                    j = rng.randrange(len(buf))
                    buf[j], buf[-1] = buf[-1], buf[j]
                    yield buf.pop()
        rng.shuffle(buf)
        for item in buf:
            yield item


def collate(batch):
    wavs, labs = zip(*batch)
    lengths = torch.tensor([w.numel() for w in wavs], dtype=torch.long)
    m = int(lengths.max())
    x = torch.zeros(len(wavs), m, dtype=torch.float32)
    for i, w in enumerate(wavs):
        x[i, : w.numel()] = w
    return x, lengths, torch.tensor(labs, dtype=torch.long)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, gts = [], []
    for x, lengths, y in loader:
        x, lengths = x.to(device), lengths.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x, length=lengths)
        preds.append(logits.argmax(-1).cpu())
        gts.append(y)
    preds = torch.cat(preds).numpy()
    gts = torch.cat(gts).numpy()
    # per-class F1 + macro-F1 + UAR, no sklearn dependency
    f1s, recalls = [], []
    for c in range(NUM_CLASS):
        tp = int(((preds == c) & (gts == c)).sum())
        fp = int(((preds == c) & (gts != c)).sum())
        fn = int(((preds != c) & (gts == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        recalls.append(rec)
    acc = float((preds == gts).mean())
    cm = np.zeros((NUM_CLASS, NUM_CLASS), int)
    for p, g in zip(preds, gts):
        cm[g, p] += 1
    return {"acc": acc, "macro_f1": float(np.mean(f1s)), "uar": float(np.mean(recalls)),
            "per_class_f1": {LABELS[c]: round(f1s[c], 4) for c in range(NUM_CLASS)},
            "confusion": cm.tolist()}


def main():
    validate_training_splits(SPLITS, LABELS, os.environ.get('ACC_ORIGINAL_SPLIT_MANIFESTS'))
    from wavlm_accent import WavLMWrapper
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    CKPT.mkdir(parents=True, exist_ok=True)
    # Small worker count and shuffle buffer keep host memory low.
    tr = ShardDataset("train", train=True, buffer=512)
    va = ShardDataset("val", train=False)
    te = ShardDataset("test", train=False)
    steps_per_epoch = len(tr) // BATCH
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,} | {steps_per_epoch} steps/ep", flush=True)
    from collections import Counter
    print("train label dist:", {LABELS[k]: v for k, v in sorted(Counter(tr.labels).items())}, flush=True)

    tl = DataLoader(tr, batch_size=BATCH, num_workers=3, collate_fn=collate,
                    pin_memory=False, drop_last=True, prefetch_factor=2)
    vl = DataLoader(va, batch_size=BATCH, num_workers=2, collate_fn=collate)
    el = DataLoader(te, batch_size=BATCH, num_workers=2, collate_fn=collate)

    model = WavLMWrapper(pretrain_model="wavlm_large", finetune_method="lora", lora_rank=16,
                         output_class_num=NUM_CLASS, apply_gradient_reversal=False).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params):,}", flush=True)
    opt = torch.optim.Adam(params, lr=LR)
    scaler = torch.amp.GradScaler("cuda")
    if os.environ.get("ACC_WEIGHTED", "0") not in ("", "0"):
        cnt = Counter(tr.labels)
        w = torch.tensor([len(tr.labels) / (NUM_CLASS * cnt[i]) for i in range(NUM_CLASS)],
                         dtype=torch.float32, device=device)
        print("class weights:", {LABELS[i]: round(float(w[i]), 3) for i in range(NUM_CLASS)}, flush=True)
        crit = nn.CrossEntropyLoss(weight=w)
    else:
        crit = nn.CrossEntropyLoss()

    best = -1.0
    hist = []
    start_ep = 1
    # Resume: last.pt holds the full state (model/opt/scaler/epoch), so a re-submitted
    # job continues from it.
    last_ck = CKPT / "last.pt"
    if last_ck.exists():
        ck = torch.load(last_ck, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        best = ck["best"]
        hist = ck["hist"]
        start_ep = ck["epoch"] + 1
        print(f"[resume] from epoch {ck['epoch']} (best macroF1 {best:.4f}) -> start ep {start_ep}", flush=True)

    for ep in range(start_ep, EPOCHS + 1):
        model.train()
        tr.set_epoch(ep)
        t0 = time.time()
        run = 0.0
        opt.zero_grad(set_to_none=True)
        for it, (x, lengths, y) in enumerate(tl):
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x, length=lengths)
                loss = crit(logits, y) / ACCUM
            scaler.scale(loss).backward()
            run += loss.item() * ACCUM
            if (it + 1) % ACCUM == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            if (it + 1) % 200 == 0:
                print(f"  ep{ep} it{it+1}/{steps_per_epoch} loss {run/(it+1):.4f} "
                      f"({(it+1)*BATCH/(time.time()-t0):.1f} clip/s)", flush=True)
        val = evaluate(model, vl)
        dt = (time.time() - t0) / 60
        print(f"[ep {ep}] train_loss {run/steps_per_epoch:.4f} | val acc {val['acc']:.4f} "
              f"macroF1 {val['macro_f1']:.4f} UAR {val['uar']:.4f} | {dt:.1f} min", flush=True)
        print("   per-class F1:", val["per_class_f1"], flush=True)
        hist.append({"epoch": ep, "train_loss": run / steps_per_epoch, "val": val})
        (CKPT / "history.json").write_text(json.dumps(hist, indent=2))
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "epoch": ep, "val": val,
                        "labels": LABELS}, CKPT / "best.pt")
            print(f"   * saved best (macroF1 {best:.4f})", flush=True)
        # full-state checkpoint for resume (atomic write)
        tmp = CKPT / "last.pt.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best,
                    "hist": hist, "labels": LABELS}, tmp)
        tmp.replace(CKPT / "last.pt")

    print("\n=== TEST (best val checkpoint) ===", flush=True)
    ck = torch.load(CKPT / "best.pt", map_location=device)
    model.load_state_dict(ck["state_dict"])
    test = evaluate(model, el)
    print(json.dumps({k: v for k, v in test.items() if k != "confusion"}, indent=2), flush=True)
    print("confusion (rows=true, cols=pred):", flush=True)
    print("labels:", LABELS, flush=True)
    for row in test["confusion"]:
        print("  ", row, flush=True)
    (CKPT / "test_result.json").write_text(json.dumps(
        {"labels": LABELS, "best_epoch": ck["epoch"], **test}, indent=2))


if __name__ == "__main__":
    main()
