#!/usr/bin/env python3
"""Download our accent classifier from Hugging Face into controllability/checkpoint/.

    checkpoint/accent_wavlm/best.pt    <- Snooow1029/itts-attribute-classifiers/accent_wavlm.pt   (WavLM-Large + LoRA r16)
    checkpoint/accent_whisper/best.pt  <- Snooow1029/itts-attribute-classifiers/accent_whisper.pt (Whisper-Large-V3 + LoRA r16)

Gender, age and emotion use released models (configs/classifiers.yaml) that are fetched from the
Hugging Face hub on first use. The repo is public; no token is needed.

Usage:
    python controllability/download_checkpoints.py [--repo REPO] [--revision REV]
"""
import argparse
import filecmp
import os
import shutil
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embeddings.release_assets import revision_for

HERE = Path(__file__).resolve().parent
CHECKPOINT_ROOT = Path(os.environ.get('ACCENT_CHECKPOINT_ROOT', HERE / 'checkpoint'))

CHECKPOINTS = {
    "accent_wavlm": CHECKPOINT_ROOT / "accent_wavlm" / "best.pt",
    "accent_whisper": CHECKPOINT_ROOT / "accent_whisper" / "best.pt",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="Snooow1029/itts-attribute-classifiers")
    ap.add_argument("--revision", default=None, help="commit or branch (default: main)")
    ap.add_argument("--retries", type=int, default=40,
                    help="attempts per file; each resumes the partial download (default 40)")
    args = ap.parse_args()
    args.revision = revision_for(args.repo, args.revision)

    # The 6.2 GB Whisper member often stalls mid-transfer; the xet backend makes it worse and
    # hf_hub_download resumes from the .incomplete blob, so disable xet, bound the socket
    # timeout, and retry.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    from huggingface_hub import hf_hub_download

    for name, dst in CHECKPOINTS.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {args.repo}/{name}.pt -> {dst}", flush=True)
        for attempt in range(1, args.retries + 1):
            try:
                path = hf_hub_download(repo_id=args.repo, filename=f"{name}.pt",
                                       repo_type="model", revision=args.revision)
                break
            except Exception as e:  # noqa: BLE001 - any transport error is worth retrying
                print(f"  [retry {attempt}/{args.retries}] {type(e).__name__}: {e}", flush=True)
                time.sleep(5)
        else:
            sys.exit(f"{name}: giving up after {args.retries} attempts")
        # hf_hub_download returns a cache path; link (or copy) it into the expected location
        if dst.exists() or dst.is_symlink():
            if dst.is_file() and (dst.resolve() == Path(path).resolve() or filecmp.cmp(dst, path, shallow=False)):
                print(f'[skip] matching existing checkpoint preserved: {dst}')
                continue
            raise FileExistsError(f'Existing checkpoint differs at {dst}; preserve it and choose a fresh ACCENT_CHECKPOINT_ROOT')
        try:
            dst.symlink_to(Path(path).resolve())
        except OSError:
            shutil.copy2(path, dst)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
