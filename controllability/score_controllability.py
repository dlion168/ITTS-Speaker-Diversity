#!/usr/bin/env python3
"""Controllability scorer for ITTS model outputs.

Given ONE folder per ITTS model, each containing generated .wav files plus a CSV
that records the *target* attribute of every clip (the attribute the prompt asked
for), this runs our attribute classifiers and reports, per attribute, how often the
perceived attribute matches the requested one -> the controllability score.

Classifiers (all 16 kHz; checkpoints in configs/classifiers.yaml):
  accent  : our 6-way WavLM-Large + LoRA (checkpoint/accent_wavlm/best.pt), softmax-averaged
            with our Whisper-Large-V3 + LoRA member (checkpoint/accent_whisper/best.pt) for
            --accent ensemble (the paper setting). Fetch both with download_checkpoints.py.
            labels: North America / English / South Asia / Germanic / Oceania / South African
  age     : Vox-Profile tiantiaf/wavlm-large-age-sex, regression head, binned
            <20 / 20-39 / 40-59 / 60+ into child / young_adult / middle_aged / older
  gender  : Vox-Profile tiantiaf/wavlm-large-age-sex, sex head
            labels: female / male
  emotion : emotion2vec/emotion2vec_plus_large via funasr. Its label space is 9-way; the
            prompts ask for four, so pred_emotion is the argmax restricted to
            {angry, happy, neutral, sad} (chance 0.25).

CSV contract (per folder). One row per clip. Column names are auto-detected:
  file    : one of  file/filename/wav/path/audio/id     (resolved under the folder)
  accent  : one of  target_accent/accent
  age     : one of  target_age/age
  gender  : one of  target_gender/gender/sex
  emotion : one of  target_emotion/emotion
Only attributes whose column is present (and whose value is non-empty & recognised)
are scored for a given clip; missing/blank cells are simply skipped for that attr.

Usage (in the vox_profile environment, envs/vox_profile.lock.txt):
  python controllability/score_controllability.py FOLDER [FOLDER ...] \
      [--csv NAME] [--accent {wavlm,whisper,ensemble}] [--age voxprofile] \
      [--attrs accent,age,gender,emotion] [--shard I/N] [--out_dir DIR] [--resume] [--max_s 20]

NOTE: to reproduce results/controllability/summary.json exactly, keep --batch 8 and 10 shards
(the defaults of run_scoring.sh).

Per folder it writes  <folder>/controllability.json  and  controllability_per_utt.csv,
and prints a summary. With >1 folder it also prints a cross-model table.

The Vox-Profile model wrappers are imported from a checkout of
https://github.com/tiantiaf0627/vox-profile-release at commit 85100e6 (setup_vox_profile.sh),
found at controllability/vox-profile-release or $VOX_PROFILE_DIR.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

HERE = Path(__file__).resolve().parent         # controllability/
REPO = Path(os.environ.get("VOX_PROFILE_DIR", HERE / "vox-profile-release"))
for p in ("src/model/accent", "src/model/age_sex"):
    sys.path.insert(0, str(REPO / p))

SR = 16000
ACCENT_LABELS = ["North America", "English", "South Asia", "Germanic", "Oceania", "South African"]
AGE_LABELS = ["child", "young_adult", "middle_aged", "older"]
GENDER_LABELS = ["female", "male"]
# --age voxprofile: regressed years -> AGE_LABELS via <20 / 20-39 / 40-59 / >=60.
VOXPROFILE_AGE_BINS = [20.0, 40.0, 60.0]
# The four emotions the prompts request; emotion2vec+ predicts a 9-way space that contains all four.
EMOTION_LABELS = ["angry", "happy", "neutral", "sad"]
EMO2VEC_REPO = "emotion2vec/emotion2vec_plus_large"

CHECKPOINT_ROOT = Path(os.environ.get('ACCENT_CHECKPOINT_ROOT', HERE / 'checkpoint'))
ACCENT_CKPT = CHECKPOINT_ROOT / "accent_wavlm" / "best.pt"
WHISPER_CKPT = CHECKPOINT_ROOT / "accent_whisper" / "best.pt"

# --- target-value normalisation (prompt vocab -> classifier label space) ---------
# The prompts use the 6 accent labels verbatim, but we stay case/space tolerant and
# fold a few common synonyms so a stray "American"/"elderly"/"woman" still scores.
FILE_COLS = ["file", "filename", "wav", "path", "audio", "id", "audio_path", "wav_path"]
ATTR_COLS = {
    "accent": ["target_accent", "accent"],
    "age": ["target_age", "age"],
    "gender": ["target_gender", "gender", "sex"],
    "emotion": ["target_emotion", "emotion"],
}


def _norm_accent(v):
    s = v.strip().lower()
    alias = {
        "north america": "North America", "american": "North America", "america": "North America",
        "us": "North America", "usa": "North America", "north_american": "North America",
        "english": "English", "british": "English", "england": "English", "uk": "English",
        "british_isles": "English", "british isles": "English",
        "south asia": "South Asia", "indian": "South Asia", "india": "South Asia",
        "south_asian": "South Asia", "south asian": "South Asia",
        "germanic": "Germanic", "german": "Germanic",
        "oceania": "Oceania", "australian": "Oceania", "australia": "Oceania",
        "new zealand": "Oceania", "oceanian": "Oceania",
        "south african": "South African", "south africa": "South African",
        "south_african": "South African",
    }
    return alias.get(s)


def _norm_age(v):
    s = v.strip().lower().replace("-", "_").replace(" ", "_")
    alias = {
        "child": "child", "kid": "child", "children": "child",
        "young_adult": "young_adult", "young": "young_adult", "youth": "young_adult",
        "twenties": "young_adult", "young_adults": "young_adult",
        "middle_aged": "middle_aged", "middle": "middle_aged", "adult": "middle_aged",
        "middleaged": "middle_aged",
        "older": "older", "old": "older", "elderly": "older", "senior": "older",
        "old_adult": "older",
    }
    return alias.get(s)


def _norm_gender(v):
    s = v.strip().lower()
    alias = {"female": "female", "woman": "female", "f": "female", "women": "female",
             "male": "male", "man": "male", "m": "male", "men": "male"}
    return alias.get(s)


def _norm_emotion(v):
    s = v.strip().lower()
    alias = {
        "angry": "angry", "anger": "angry", "mad": "angry",
        "happy": "happy", "happiness": "happy", "joy": "happy", "joyful": "happy",
        "neutral": "neutral", "none": "neutral",
        "sad": "sad", "sadness": "sad", "sorrow": "sad",
    }
    return alias.get(s)


NORMALISERS = {"accent": _norm_accent, "age": _norm_age, "gender": _norm_gender,
               "emotion": _norm_emotion}


def _emo2vec_scores(rec):
    """funasr result -> {plain label: score}. Its labels look like '生气/angry'; '<unk>' has no '/'."""
    return {lab.split("/")[-1]: sc for lab, sc in zip(rec["labels"], rec["scores"])}


# --------------------------------------------------------------------------------
def load_audio(path, max_s):
    w, srr = sf.read(str(path), dtype="float32")
    if w.ndim > 1:
        w = w.mean(-1)
    w = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
    if srr != SR:
        w = AF.resample(w, srr, SR)
    if max_s and w.numel() > int(max_s * SR):
        w = w[: int(max_s * SR)]
    if w.numel() == 0:
        w = torch.zeros(SR // 10)
    return w


def collate(wavs):
    lengths = torch.tensor([w.numel() for w in wavs])
    m = int(lengths.max())
    x = torch.zeros(len(wavs), m)
    for i, w in enumerate(wavs):
        x[i, : w.numel()] = w
    return x, lengths


def detect_col(header, candidates):
    low = {h.lower(): h for h in header}
    for c in candidates:
        if c in low:
            return low[c]
    return None


def read_manifest(folder, csv_name, only_attrs=None):
    if csv_name:
        csv_path = folder / csv_name
    else:
        cands = [folder / n for n in ("prompts.csv", "metadata.csv", "meta.csv", "manifest.csv")]
        csv_path = next((c for c in cands if c.is_file()), None)
        if csv_path is None:
            found = sorted(folder.glob("*.csv"))
            csv_path = found[0] if found else None
    if csv_path is None or not csv_path.is_file():
        raise FileNotFoundError(f"no CSV found in {folder} (pass --csv NAME)")

    with open(csv_path, newline="") as fh:
        rd = csv.DictReader(fh)
        header = rd.fieldnames or []
        file_col = detect_col(header, FILE_COLS)
        if file_col is None:
            raise ValueError(f"{csv_path}: no filename column (looked for {FILE_COLS})")
        attr_col = {a: detect_col(header, cols) for a, cols in ATTR_COLS.items()}
        attr_col = {a: c for a, c in attr_col.items() if c}
        if only_attrs is not None:
            attr_col = {a: c for a, c in attr_col.items() if a in only_attrs}
        rows = list(rd)
    return csv_path, file_col, attr_col, rows


def resolve_wav(folder, name):
    p = Path(name)
    for cand in (folder / p, folder / p.name, folder / "wav" / p.name,
                 folder / "audio" / p.name):
        if cand.is_file():
            return cand
    # last resort: add .wav
    for cand in (folder / f"{name}.wav", folder / "wav" / f"{name}.wav"):
        if cand.is_file():
            return cand
    return None


# --------------------------------------------------------------------------------
def _mark_lora_merged(model):
    """Our accent best.pt files store the LoRA update already merged into `.weight`
    (weight = W + BA), but loralib's `merged` flag is not part of the state_dict. Mark every
    LoRA layer as merged after loading, so that .eval() does not add the update again."""
    import loralib as lora
    n = 0
    for m in model.modules():
        if isinstance(m, lora.LoRALayer):
            m.merged = True
            n += 1
    return n


def build_models(accent_mode, device, need=("accent", "age", "gender", "emotion"),
                 age_mode="voxprofile"):
    models = {}
    need = set(need)
    if 'accent' in need:
        if __package__:
            from .scoring_run import require_accent_checkpoints
        else:
            from scoring_run import require_accent_checkpoints
        require_accent_checkpoints(CHECKPOINT_ROOT, accent_mode)
    if need - {"emotion"}:
        # Only import the Vox-Profile wrappers when a Vox-Profile classifier is actually
        # wanted — they pull in speechbrain and hard-code .cuda() at import time.
        from wavlm_accent import WavLMWrapper
        from wavlm_demographics import WavLMWrapper as AgeSexWavLM

    if "accent" not in need:
        print("[load] skipping accent classifier (not in --attrs)", flush=True)
    else:
        if accent_mode in ('wavlm', 'ensemble'):
            print("[load] accent WavLM ...", flush=True)
            acc = WavLMWrapper(pretrain_model="wavlm_large", finetune_method="lora", lora_rank=16,
                               output_class_num=len(ACCENT_LABELS), apply_gradient_reversal=False).to(device)
            acc.load_state_dict(torch.load(ACCENT_CKPT, map_location=device)["state_dict"])
            print(f"       [lora-fix] marked {_mark_lora_merged(acc)} LoRA layers as already-merged", flush=True)
            acc.eval()
            models["accent_wavlm"] = acc

        if accent_mode in ("whisper", "ensemble"):
            if not WHISPER_CKPT.is_file():
                raise FileNotFoundError(f'{WHISPER_CKPT} missing; requested ensemble cannot silently become wavlm-only')
            else:
                from whisper_accent import WhisperWrapper
                print("[load] accent Whisper-Large-V3 ...", flush=True)
                wh = WhisperWrapper(pretrain_model="whisper_large", finetune_method="lora", lora_rank=16,
                                    output_class_num=len(ACCENT_LABELS), apply_gradient_reversal=False).to(device)
                wh.load_state_dict(torch.load(WHISPER_CKPT, map_location=device)["state_dict"])
                print(f"       [lora-fix] marked {_mark_lora_merged(wh)} LoRA layers as already-merged", flush=True)
                wh.eval()
                models["accent_whisper"] = wh

    if "age" in need:
        # Vox-Profile's pretrained age regressor (the age head of the same model the gender
        # track uses) -- no local checkpoint; predict_batch bins its age into AGE_LABELS.
        print("[load] age Vox-Profile age_sex (regression head) ...", flush=True)
        models["age_sex"] = AgeSexWavLM.from_pretrained("tiantiaf/wavlm-large-age-sex").to(device).eval()

    if "gender" in need:
        if "age_sex" in models:                  # same weights; don't load them twice
            models["gender"] = models["age_sex"]
        else:
            print("[load] gender Vox-Profile age_sex ...", flush=True)
            models["gender"] = AgeSexWavLM.from_pretrained("tiantiaf/wavlm-large-age-sex").to(device).eval()

    if "emotion" in need:
        # emotion2vec+ ships no checkpoint in our HF repo; funasr loads the hub snapshot.
        # snapshot_download resolves straight from the cache under HF_HUB_OFFLINE=1.
        from huggingface_hub import snapshot_download
        from funasr import AutoModel
        print("[load] emotion emotion2vec_plus_large ...", flush=True)
        models["emotion"] = AutoModel(model=snapshot_download(EMO2VEC_REPO),
                                      device="cuda:0" if device.type == "cuda" else "cpu",
                                      disable_update=True, disable_pbar=True, disable_log=True)

    return models, accent_mode, age_mode


@torch.no_grad()
def predict_batch(models, accent_mode, wavs, device, need=("accent", "age", "gender", "emotion"),
                  age_mode="voxprofile"):
    """Return dict attr -> list[str] predictions for a batch of raw waveforms.

    With "age" requested, the extra key "age_years" carries the regressed age before binning.
    """
    need = set(need)
    x = lengths = None
    if need - {"emotion"}:                      # emotion2vec does its own batching
        x, lengths = collate(wavs)
        x, lengths = x.to(device), lengths.to(device)
    use_amp = device.type == "cuda"

    def run(m):
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = m(x, length=lengths)
        logits = out[0] if isinstance(out, tuple) else out
        return F.softmax(logits.float(), dim=-1)

    out = {}

    if "accent" in need:
        # accent (optionally ensembled)
        if accent_mode == "whisper":
            pa = run(models["accent_whisper"])
        else:
            pa = run(models["accent_wavlm"])
            if accent_mode == "ensemble":
                pa = (pa + run(models["accent_whisper"])) / 2.0
        out["accent"] = [ACCENT_LABELS[i] for i in pa.argmax(-1).cpu().numpy()]

    sex_logits = None
    if "age" in need:
        # Scored one clip at a time.
        # The age head is a sigmoid regressor: age_out * 100 = years (Vox-Profile README).
        years, sex_logits = [], []
        for w in wavs:
            age_out, sex_out = models["age_sex"](w.unsqueeze(0).to(device))
            years.append(float(age_out.reshape(-1)[0]) * 100.0)
            sex_logits.append(sex_out)
        out["age_years"] = years
        out["age"] = [AGE_LABELS[int(np.digitize(y, VOXPROFILE_AGE_BINS))] for y in years]

    if "gender" in need:
        # Vox-Profile wrapper takes one clip [1,T] and returns (age_out, sex_out).
        if sex_logits is None:
            sex_logits = [models["gender"](w.unsqueeze(0).to(device))[1] for w in wavs]
        gender_pred = [GENDER_LABELS[int(s.softmax(-1).argmax())] for s in sex_logits]
        out["gender"] = gender_pred

    if "emotion" in need:
        res = models["emotion"].generate([w.numpy() for w in wavs], fs=SR,
                                         granularity="utterance", extract_embedding=False,
                                         batch_size=len(wavs), disable_pbar=True)
        out["emotion"] = []
        for rec in res:
            sc = _emo2vec_scores(rec)
            out["emotion"].append(max(EMOTION_LABELS, key=lambda e: sc.get(e, -1.0)))

    return out


def score_folder(folder, models, accent_mode, csv_name, max_s, batch, device,
                 only_attrs=None, shard=(0, 1), out_dir=None, workers=8, label=None,
                 age_mode="voxprofile", resume=False):
    csv_path, file_col, attr_col, rows = read_manifest(folder, csv_name, only_attrs)
    shard_i, shard_n = shard
    if shard_n > 1:
        rows = rows[shard_i::shard_n]
    label = label or f"{folder.parent.name}/{folder.name}"
    tag = f"  shard {shard_i+1}/{shard_n}" if shard_n > 1 else ""
    stem = "controllability"
    if out_dir:
        stem += "." + label.replace("/", "__")
    if shard_n > 1:
        stem += f".shard{shard_i}of{shard_n}"
    dest = Path(out_dir) if out_dir else folder
    if resume and (dest / f"{stem}.json").is_file():
        if __package__:
            from .scoring_run import sha256 as input_sha256, validate_pair
        else:
            from scoring_run import sha256 as input_sha256, validate_pair
        expected_rows = [{'file': row[file_col], **{'target_' + attr: NORMALISERS[attr]((row.get(column) or '').strip()) or ''
                         for attr, column in attr_col.items()}} for row in rows]
        validate_pair(dest / f'{stem}_per_utt.csv', dest / f'{stem}.json', expected_rows, list(attr_col),
                      input_hash=input_sha256(csv_path), batch=batch, label=label, shard=f'{shard_i}of{shard_n}', max_s=max_s)
        previous = json.loads((dest / f'{stem}.json').read_text())
        provenance = previous.get('input_provenance', {})
        if provenance.get('max_s') != max_s or previous.get('accent_mode') != accent_mode or previous.get('age_mode') != age_mode:
            raise ValueError('Resume scoring settings differ; choose a fresh run directory')
        print(f"\n=== {label} ==={tag}  [resume] already scored, skipping", flush=True)
        return previous
    print(f"\n=== {label} ===  csv={csv_path.name}  n={len(rows)}  "
          f"attrs={list(attr_col)}{tag}", flush=True)

    per_utt = []
    conf = {a: defaultdict(Counter) for a in attr_col}   # attr -> gt -> Counter[pred]
    n_ok = Counter(); n_tot = Counter(); n_skip = Counter()
    missing_wav = 0

    buf_wavs, buf_meta = [], []
    need = set(attr_col)
    pool = ThreadPoolExecutor(max_workers=workers)

    def flush():
        nonlocal buf_wavs, buf_meta
        if not buf_wavs:
            return
        preds = predict_batch(models, accent_mode, [w.result() for w in buf_wavs], device, need,
                              age_mode)
        for j, meta in enumerate(buf_meta):
            row = {"file": meta["name"]}
            if "age_years" in preds:
                row["pred_age_years"] = round(preds["age_years"][j], 2)
            for a in attr_col:
                pred = preds[a][j]
                row[f"pred_{a}"] = pred
                gt = meta["gt"].get(a)
                row[f"target_{a}"] = gt if gt is not None else ""
                if gt is None:
                    continue
                ok = int(pred == gt)
                n_ok[a] += ok; n_tot[a] += 1
                conf[a][gt][pred] += 1
            per_utt.append(row)
        buf_wavs, buf_meta = [], []

    for r in rows:
        name = r[file_col]
        wav_path = resolve_wav(folder, name)
        gt = {}
        for a, col in attr_col.items():
            raw = (r.get(col) or "").strip()
            if not raw:
                continue
            norm = NORMALISERS[a](raw)
            if norm is None:
                n_skip[a] += 1
            else:
                gt[a] = norm
        if wav_path is None:
            missing_wav += 1
            continue
        if not gt:
            continue
        buf_wavs.append(pool.submit(load_audio, wav_path, max_s))
        buf_meta.append({"name": name, "gt": gt})
        if len(buf_wavs) == batch:
            flush()
    flush()
    pool.shutdown()

    # --- metrics ---
    summary = {"folder": folder.name, "label": label, "model": folder.parent.name,
               "shard": f"{shard_i}of{shard_n}", "n_rows": len(rows), "accent_mode": accent_mode,
               "age_mode": age_mode,
                "missing_wav": missing_wav, "attributes": {}}
    if __package__:
        from .scoring_run import sha256 as input_sha256
    else:
        from scoring_run import sha256 as input_sha256
    summary['input_provenance'] = {'csv_sha256': input_sha256(csv_path), 'batch': batch, 'max_s': max_s,
                                   'scorer_sha256': input_sha256(Path(__file__))}
    for a in attr_col:
        if not n_tot[a]:
            continue
        acc = n_ok[a] / n_tot[a]
        classes = sorted(conf[a])
        recalls = {}
        for g in classes:
            tot = sum(conf[a][g].values())
            recalls[g] = conf[a][g][g] / tot if tot else 0.0
        uar = float(np.mean(list(recalls.values()))) if recalls else 0.0
        summary["attributes"][a] = {
            "controllability_acc": round(acc, 4), "uar": round(uar, 4),
            "n": n_tot[a], "n_unrecognised_target": n_skip[a],
            "per_class_recall": {g: round(v, 4) for g, v in recalls.items()},
            "confusion": {g: dict(conf[a][g]) for g in classes},
        }

    # --- write outputs ---
    all_cols = ["file"]
    for a in attr_col:
        all_cols += [f"target_{a}", f"pred_{a}"]
    if "age" in attr_col:
        all_cols.append("pred_age_years")    # Vox-Profile regressed age before binning
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / f"{stem}_per_utt.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=all_cols)
        wr.writeheader()
        for row in per_utt:
            wr.writerow({k: row.get(k, "") for k in all_cols})
    (dest / f"{stem}.json").write_text(json.dumps(summary, indent=2))

    # --- print ---
    if missing_wav:
        print(f"  [warn] {missing_wav} rows had no matching wav (skipped)", flush=True)
    for a, s in summary["attributes"].items():
        extra = f"  ({s['n_unrecognised_target']} target vals unrecognised)" if s["n_unrecognised_target"] else ""
        print(f"  {a:7s}: acc={s['controllability_acc']:.3f}  UAR={s['uar']:.3f}  "
              f"n={s['n']}{extra}", flush=True)
    print(f"  -> {dest}/{stem}.json + {stem}_per_utt.csv", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path, help="one folder per ITTS model")
    ap.add_argument("--csv", default=None, help="CSV filename inside each folder (default: auto-detect)")
    ap.add_argument("--accent", choices=["wavlm", "whisper", "ensemble"], default="ensemble",
                    help="accent classifier (default and paper setting: ensemble, the softmax "
                         "average of the WavLM and Whisper members).")
    ap.add_argument("--age", choices=["voxprofile"], default="voxprofile",
                    help="age classifier: the pretrained Vox-Profile age regressor "
                         "(tiantiaf/wavlm-large-age-sex), binned <20/20-39/40-59/60+.")
    ap.add_argument("--resume", action="store_true",
                    help="skip any (folder, shard) whose result JSON already exists — use it to "
                         "re-run only the shards that died without redoing finished work.")
    ap.add_argument("--max_s", type=float, default=20.0, help="truncate clips to this many seconds")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--attrs", default=None,
                    help="comma-separated subset of accent,age,gender,emotion. Restricts BOTH which "
                         "classifiers are loaded and which columns are scored, so e.g. an "
                         "age-only folder never pays for Whisper-Large. Default: all present.")
    ap.add_argument("--shard", default="0/1", metavar="I/N",
                    help="process only rows I, I+N, I+2N ... of each folder's CSV. Run N processes "
                         "with I=0..N-1 to fan a folder out across one GPU, then combine with "
                         "merge_shards.py. Default 0/1 (no sharding).")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="write results here instead of into each audio folder; filenames are "
                         "prefixed with <model>__<attribute> so many folders can share one dir.")
    ap.add_argument("--workers", type=int, default=8,
                    help="threads used to decode/resample WAVs ahead of the GPU (default 8)")
    args = ap.parse_args()

    only_attrs = None
    if args.attrs:
        only_attrs = [a.strip() for a in args.attrs.split(",") if a.strip()]
        bad = set(only_attrs) - set(ATTR_COLS)
        if bad:
            sys.exit(f"--attrs: unknown attribute(s) {sorted(bad)}; pick from {list(ATTR_COLS)}")
    try:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    except ValueError:
        sys.exit(f"--shard must look like I/N, got {args.shard!r}")
    if not (shard_n >= 1 and 0 <= shard_i < shard_n):
        sys.exit(f"--shard out of range: {args.shard}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dev] {device}  accent_mode={args.accent}  age_mode={args.age}", flush=True)
    models, accent_mode, age_mode = build_models(args.accent, device,
                                                 only_attrs or list(ATTR_COLS), args.age)

    all_summaries = []
    for folder in args.folders:
        folder = folder.resolve()
        if not folder.is_dir():
            print(f"[skip] not a dir: {folder}", flush=True)
            continue
        try:
            all_summaries.append(score_folder(folder, models, accent_mode,
                                              args.csv, args.max_s, args.batch, device,
                                              only_attrs=only_attrs, shard=(shard_i, shard_n),
                                              out_dir=args.out_dir, workers=args.workers,
                                              age_mode=age_mode, resume=args.resume))
        except (FileNotFoundError, ValueError) as e:
            print(f"[skip] {folder.name}: {e}", flush=True)

    if len(all_summaries) > 1:
        attrs = ["accent", "age", "gender", "emotion"]
        print("\n=== controllability summary (acc | UAR) ===", flush=True)
        head = "model".ljust(22) + "".join(a.ljust(18) for a in attrs)
        print(head, flush=True)
        for s in all_summaries:
            line = s.get("label", s["folder"])[-21:].ljust(22)
            for a in attrs:
                if a in s["attributes"]:
                    d = s["attributes"][a]
                    line += f"{d['controllability_acc']:.3f}|{d['uar']:.3f}".ljust(18)
                else:
                    line += "-".ljust(18)
            print(line, flush=True)


if __name__ == "__main__":
    main()
