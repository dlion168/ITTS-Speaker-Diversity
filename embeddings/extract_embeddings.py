#!/usr/bin/env python3
"""
Extract speaker embeddings from a folder of WAV files with one or more speaker encoders.

Output, per encoder:
    <output_dir>/<tts_model>/<wav_dir name>/<encoder>/embeddings.npy   (n, dim) float32
    <output_dir>/<tts_model>/<wav_dir name>/<encoder>/filenames.txt    row i of the .npy = line i
Rows are sorted by file name, so every encoder has the same row order.

Usage:
    python embeddings/extract_embeddings.py --wav_dir data/outputs/attributes/voxcpm2/age \
        --tts_model voxcpm2 --output_dir data/embeddings_attributes --models all

`--models all` runs the eight paper encoders; TitaNet only loads in the itts_nemo environment,
the other seven in itts_embed (an encoder that fails to load is reported and skipped).
`--models validation` runs the three encoders used only in the encoder validation.

Re-running resumes: files already in embeddings.npy are skipped.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import compat

compat.apply()

import numpy as np
import torch

import encoders
from audio import load_audio_tensor


# ── Resume helpers ──────────────────────────────────────────────────────────

def load_existing(model_output_dir: Path):
    """Load previously saved embeddings and filenames. Returns (dict, set)."""
    emb_path = model_output_dir / "embeddings.npy"
    names_path = model_output_dir / "filenames.txt"
    if emb_path.exists() and names_path.exists():
        matrix = np.load(emb_path)
        names = names_path.read_text().strip().split("\n")
        if len(names) == matrix.shape[0]:
            return {n: matrix[i] for i, n in enumerate(names)}, set(names)
    return {}, set()


def submit_audio_batch(pool, wav_paths, target_sr=16000):
    """Submit audio reads so the next batch can load during GPU inference."""
    def _load(path):
        try:
            return load_audio_tensor(str(path), target_sr)
        except Exception:
            return None

    return [pool.submit(_load, path) for path in wav_paths]


# ── Batch extraction for supported models ───────────────────────────────────

SAVE_INTERVAL = 5000


def save_checkpoint(model_output_dir, embeddings_dict):
    """Atomically replace the two resume files with current progress."""
    if not embeddings_dict:
        return
    all_names = sorted(embeddings_dict.keys())
    emb_matrix = np.stack([embeddings_dict[n] for n in all_names])
    emb_path = model_output_dir / "embeddings.npy"
    names_path = model_output_dir / "filenames.txt"
    emb_tmp = model_output_dir / "embeddings.tmp.npy"
    names_tmp = model_output_dir / "filenames.tmp.txt"
    np.save(emb_tmp, emb_matrix)
    with open(names_tmp, "w") as f:
        for name in all_names:
            f.write(name + "\n")
    os.replace(emb_tmp, emb_path)
    os.replace(names_tmp, names_path)


def run_batched_signals(model, batch_fn, wav_files, batch_size, num_workers,
                        model_output_dir, existing_embs):
    """Batched inference on decoded waveforms, with threaded I/O prefetch and checkpointing.

    On CUDA OOM the batch is split in half, then retried one file at a time."""
    embeddings = dict(existing_embs)
    failed = 0
    total = len(wav_files)
    new_count = 0

    starts = list(range(0, total, batch_size))
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        pending = submit_audio_batch(pool, wav_files[:batch_size])
        for start in starts:
            end = min(start + batch_size, total)
            batch_paths = wav_files[start:end]
            signals = [future.result() for future in pending]

            next_start = end
            next_end = min(next_start + batch_size, total)
            pending = submit_audio_batch(pool, wav_files[next_start:next_end])

            valid_signals = []
            valid_names = []
            for j, sig in enumerate(signals):
                if sig is not None:
                    valid_signals.append(sig)
                    valid_names.append(batch_paths[j].stem)
                else:
                    failed += 1

            if valid_signals:
                cur_signals = valid_signals
                cur_names = valid_names
                while cur_signals:
                    try:
                        with torch.no_grad():
                            emb_matrix = batch_fn(model, cur_signals)
                        for k, name in enumerate(cur_names):
                            embeddings[name] = emb_matrix[k]
                            new_count += 1
                        break
                    except Exception as exc:
                        # Catch all CUDA OOM variants and 32-bit index errors
                        exc_str = str(exc).lower()
                        is_oom = "out of memory" in exc_str or "cuda" in exc_str
                        is_index = "32-bit index" in str(exc)
                        if not (is_oom or is_index):
                            raise
                        torch.cuda.empty_cache()
                        half = max(1, len(cur_signals) // 2)
                        if len(cur_signals) == 1:
                            print(f"  OOM on single file {cur_names[0]}, skipping")
                            failed += 1
                            break
                        print(f"  OOM at batch={len(cur_signals)}, retrying with {half}")
                        for sub_start in range(0, len(cur_signals), half):
                            sub_sigs = cur_signals[sub_start:sub_start+half]
                            sub_names = cur_names[sub_start:sub_start+half]
                            try:
                                with torch.no_grad():
                                    sub_emb = batch_fn(model, sub_sigs)
                                for k, name in enumerate(sub_names):
                                    embeddings[name] = sub_emb[k]
                                    new_count += 1
                            except Exception as exc2:
                                exc2_str = str(exc2).lower()
                                is_oom2 = "out of memory" in exc2_str or "cuda" in exc2_str
                                is_index2 = "32-bit index" in str(exc2)
                                if not (is_oom2 or is_index2):
                                    raise
                                torch.cuda.empty_cache()
                                for si, sn in zip(sub_sigs, sub_names):
                                    try:
                                        with torch.no_grad():
                                            single_emb = batch_fn(model, [si])
                                        embeddings[sn] = single_emb[0]
                                        new_count += 1
                                    except Exception:
                                        failed += 1
                        break

            if new_count >= SAVE_INTERVAL:
                save_checkpoint(model_output_dir, embeddings)
                new_count = 0

            done = len(embeddings)
            if end % 500 == 0 or end == total:
                print(f"  [{done}/{total}] processed")

    return embeddings, failed


def run_batched_paths(model, batch_fn, extract_fn, wav_files, batch_size,
                      model_output_dir, existing_embs):
    """Batched inference for encoders that take file paths (ModelScope pipelines).

    A failed batch is retried one file at a time."""
    embeddings = dict(existing_embs)
    failed = 0
    total = len(wav_files)
    new_count = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_paths = [str(p) for p in wav_files[start:end]]

        try:
            with torch.no_grad():
                emb_matrix = batch_fn(model, batch_paths)
            for k, name in enumerate(wav_files[start:end]):
                embeddings[name.stem] = emb_matrix[k]
                new_count += 1
        except Exception as e:
            print(f"  [{start}] batch failed ({len(batch_paths)} files): {e}")
            # Fallback: one by one
            for wav_path in wav_files[start:end]:
                try:
                    emb = extract_fn(model, str(wav_path))
                    embeddings[wav_path.stem] = emb
                    new_count += 1
                except Exception as e2:
                    print(f"  FAILED {wav_path.name}: {e2}")
                    failed += 1

        if new_count >= SAVE_INTERVAL:
            save_checkpoint(model_output_dir, embeddings)
            new_count = 0

        done = len(embeddings)
        if end % 500 == 0 or end == total:
            print(f"  [{done}/{total}] processed")

    return embeddings, failed


def run_sequential_prefetch(model, extractor, wav_files, num_workers,
                            model_output_dir, existing_embs):
    """Sequential GPU inference with threaded I/O prefetch and checkpointing."""
    embeddings = dict(existing_embs)
    failed = 0
    total = len(wav_files)
    new_count = 0
    chunk = num_workers * 2

    def _prefetch(paths):
        for path in paths:
            try:
                with open(path, "rb") as audio_file:
                    audio_file.read()
            except Exception:
                pass

    with ThreadPoolExecutor(max_workers=1) as pool:
        prefetch_future = None
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            batch_paths = wav_files[start:end]
            if prefetch_future is not None:
                prefetch_future.result()
            next_paths = wav_files[end:min(end + chunk, total)]
            prefetch_future = pool.submit(_prefetch, next_paths) if next_paths else None

            for j, wav_path in enumerate(batch_paths):
                try:
                    with torch.no_grad():
                        emb = extractor(model, str(wav_path))
                    embeddings[wav_path.stem] = emb
                    new_count += 1
                except Exception as e:
                    print(f"  [{start+j}/{total}] FAILED {wav_path.name}: {e}")
                    failed += 1

            if new_count >= SAVE_INTERVAL:
                save_checkpoint(model_output_dir, embeddings)
                new_count = 0

            done = len(embeddings)
            if end % 500 == 0 or end == total:
                print(f"  [{done}/{total}] processed")

    return embeddings, failed


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract speaker embeddings from WAV files.")
    parser.add_argument("--wav_dir", required=True, help="Directory containing WAV files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save embeddings (.npy).")
    parser.add_argument("--tts_model", required=True, help="Name of the TTS model that generated the WAVs (e.g. parler_mini).")
    parser.add_argument(
        "--models", nargs="+", default=["all"],
        help=f"Encoders to use: all (the 8 paper encoders), validation (the 3 validation-only "
             f"encoders), or names from {', '.join(encoders.SUPPORTED)}",
    )
    parser.add_argument("--device", default=None, help="Device (cuda / cpu). Auto-detect if omitted.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for batch-capable encoders (SpeechBrain, ModelScope). Default: 16")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of I/O prefetch threads. Default: 8")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N WAV files (for quick testing, e.g. --limit 100). Default: all")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if "cuda" in device:
        _mem_frac = float(os.environ.get("EMB_MEM_FRACTION", "0.45"))
        torch.cuda.set_per_process_memory_fraction(_mem_frac)
    print(f"[extract] Device: {device}")

    wav_dir = Path(args.wav_dir)
    wav_files = sorted(list(wav_dir.glob("*.wav")) + list(wav_dir.glob("*.WAV")))
    if not wav_files:
        print(f"[extract] No WAV files found in {wav_dir}")
        return 1
    print(f"[extract] Found {len(wav_files)} WAV files in total.")

    if args.limit is not None:
        wav_files = wav_files[:args.limit]
        print(f"[extract] --limit set: only processing first {len(wav_files)} files.")

    models_to_run = []
    for m in args.models:
        group = {"all": encoders.PAPER_ENCODERS, "validation": encoders.VALIDATION_ONLY_ENCODERS}
        models_to_run += group.get(m, [m])
    for m in models_to_run:
        if m not in encoders.SUPPORTED:
            print(f"[extract] Unknown model: {m}. Supported: {', '.join(encoders.SUPPORTED)}")
            return 1

    output_dir = Path(args.output_dir)
    dataset_name = wav_dir.name

    for model_name in models_to_run:
        model_output_dir = output_dir / args.tts_model / dataset_name / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # ── Resume check ──
        existing_embs, done_names = load_existing(model_output_dir)
        all_wav_names = {p.stem for p in wav_files}

        if done_names >= all_wav_names:
            print(f"\n{'='*60}")
            print(f"[extract] SKIP {model_name}: all {len(done_names)} files already done at {model_output_dir}")
            continue

        remaining = [p for p in wav_files if p.stem not in done_names]

        print(f"\n{'='*60}")
        if existing_embs:
            print(f"[extract] RESUME {model_name}: {len(done_names)} done, {len(remaining)} remaining")
        else:
            print(f"[extract] START  {model_name}: {len(remaining)} files")

        enc = encoders.get(model_name)
        try:
            model = enc.load(device)
        except Exception as e:
            print(f"[extract] FAILED to load {model_name}: {e}")
            continue

        t0 = time.time()

        if enc.batch_mode == "signals":
            print(f"[extract] Using BATCH mode (batch_size={args.batch_size})")
            embeddings, failed = run_batched_signals(
                model, enc.batch, remaining, args.batch_size,
                args.num_workers, model_output_dir, existing_embs,
            )
        elif enc.batch_mode == "paths":
            print(f"[extract] Using path BATCH mode (batch_size={args.batch_size})")
            embeddings, failed = run_batched_paths(
                model, enc.batch, enc.extract, remaining, args.batch_size,
                model_output_dir, existing_embs,
            )
        else:
            print("[extract] Using sequential mode with I/O prefetch")
            embeddings, failed = run_sequential_prefetch(
                model, enc.extract, remaining, args.num_workers,
                model_output_dir, existing_embs,
            )

        elapsed = time.time() - t0
        new_done = len(embeddings) - len(existing_embs)

        # final save
        save_checkpoint(model_output_dir, embeddings)

        speed = new_done / elapsed if elapsed > 0 else 0
        if embeddings:
            dim = next(iter(embeddings.values())).shape[-1]
            print(f"[extract] Saved {model_name}: ({len(embeddings)}, {dim}) -> {model_output_dir}")
        else:
            print(f"[extract] No embeddings extracted for {model_name}.")
        print(f"[extract] {model_name}: {new_done} new, {failed} failed, {elapsed:.1f}s ({speed:.0f} files/s)")

        del model
        if "cuda" in device:
            torch.cuda.empty_cache()

    print("\n[extract] All done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
