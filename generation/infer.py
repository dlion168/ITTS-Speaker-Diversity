#!/usr/bin/env python3
"""
ITTS Diversity — Unified Instruction-to-Speech Inference Pipeline
=================================================================

Usage
-----
    python infer.py --model <MODEL_NAME> --input <FILE> [OPTIONS]

Supported models
----------------
    parler-mini         Parler-TTS Mini v1  (parler-tts/parler-tts-mini-v1)
    parler-large        Parler-TTS Large v1 (parler-tts/parler-tts-large-v1)
    moss-voicegen       MOSS-VoiceGenerator (OpenMOSS-Team/MOSS-VoiceGenerator)
    qwen3-voicedesign   Qwen3-TTS VoiceDesign (Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
    voxinstruct         VoxInstruct (requires manual setup + Google Drive ckpts)
    voxcpm2             VoxCPM2 (openbmb/VoxCPM2)
    capspeech           CapSpeech-NAR (OpenSound/CapSpeech-models)

Input file format (CSV)
-----------------------
Required columns:   text
Optional columns:   id, style_prompt, ref_audio, ref_text, language

    id,text,style_prompt,ref_audio,ref_text,language
    001,"Hello world!","A warm female voice",,,"en"
    002,"今天天氣很好","充滿活力的年輕聲音",,,"zh"

Input file format (JSON)
------------------------
A JSON array of objects, or a JSONL file (one JSON object per line):

    [
      {
        "id": "001",
        "text": "Hello world!",
        "style_prompt": "A warm female voice",
        "language": "en"
      }
    ]

Output
------
    WAV files saved directly to --output_dir (default: ./outputs/)
    Named as: <id>.wav  (or <index>.wav if no id column)

Weights
-------
    Model weights are resolved under --weights_dir (default: WEIGHTS_DIR or <repo>/weights).
    Missing known HF checkpoints use recorded revisions in the standard HF cache.
    HuggingFace models are auto-downloaded; others need manual placement.
    See individual model backend files in backends/ for setup instructions.

Examples
--------
    # Run Parler-TTS Mini on a CSV file
    python infer.py --model parler-mini --input examples/sample.csv

    # Run MOSS-VoiceGenerator on a JSON file, save to custom dir
    python infer.py --model moss-voicegen --input examples/sample.json \\
        --output_dir /tmp/tts_out --device cuda

    # Run Qwen3 VoiceDesign with pre-downloaded weights
    python infer.py --model qwen3-voicedesign --input examples/sample.csv \\
        --weights_dir ../tts_weights

    # List supported models
    python infer.py --list-models
"""

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional progress bar — silently degraded if tqdm not installed.
try:
    from tqdm import tqdm as _tqdm_cls
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

import numpy as np

# ---------------------------------------------------------------------------
# Ensure script can find backends/ when invoked from any cwd.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from backends import MODEL_REGISTRY  # noqa: E402

# Neutral "speak like a person" style prompt — applied when the input row
# has no style_prompt field.  Override via --default_style_prompt.
DEFAULT_STYLE_PROMPT = (
    "Speak like a person — natural, neutral tone, moderate pace, "
    "conversational and clear delivery."
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_input(path: str) -> List[Dict[str, Any]]:
    """
    Load CSV or JSON/JSONL input file into a list of dicts.

    Supported formats:
      - .csv  : standard CSV with header row
      - .json : JSON array  [{"text": ...}, ...]
      - .jsonl: one JSON object per line
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = p.suffix.lower()

    if suffix == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader]
        return rows

    if suffix == ".json":
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError(
            f"Unsupported JSON structure in {path}; expected array or object.")

    if suffix == ".jsonl":
        rows = []
        with open(p, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON at line {lineno} of {path}: {e}") from e
        return rows

    raise ValueError(
        f"Unsupported file extension '{suffix}'. Use .csv, .json, or .jsonl."
    )


def save_wav(audio: np.ndarray, sr: int, path: Path) -> None:
    """Validate and atomically save a float32 mono numpy array as WAV."""
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    if audio.ndim != 1 or audio.size < max(1, int(sr * 0.1)):
        raise ValueError(
            f"Invalid audio shape/length: shape={audio.shape}, sr={sr}")
    if sr <= 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio values or sample rate: sr={sr}")
    if float(np.max(np.abs(audio))) <= 1e-7:
        raise ValueError("Generated audio is silent")

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".wav", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        try:
            import soundfile as sf
            sf.write(str(tmp_path), audio, sr)
        except ImportError:
            try:
                from scipy.io import wavfile
                wavfile.write(str(tmp_path), sr, audio)
            except ImportError:
                _write_wav_manual(audio, sr, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _write_wav_manual(audio: np.ndarray, sr: int, path: Path) -> None:
    """Minimal WAV writer (PCM 16-bit mono) — no external dependencies."""
    import struct
    audio_i16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    data_bytes = audio_i16.tobytes()
    n_bytes = len(data_bytes)
    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n_bytes))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", n_bytes))
        f.write(data_bytes)


# ---------------------------------------------------------------------------
# Row normalisation
# ---------------------------------------------------------------------------

def normalise_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise column names to a canonical set.

    Accepted aliases:
        text            : text, content, transcript, sentence
        style_prompt    : style_prompt, description, prompt, style, instruction, voice_prompt
        ref_audio       : ref_audio, reference_audio, audio_ref, speaker_audio
        ref_text        : ref_text, reference_text, ref_transcript
        language        : language, lang
        id              : id, name, sample_id, utt_id
    """
    alias_map = {
        "text":         ["text", "content", "transcript", "sentence"],
        "style_prompt": ["style_prompt", "description", "prompt", "style",
                         "instruction", "voice_prompt"],
        "ref_audio":    ["ref_audio", "reference_audio", "audio_ref", "speaker_audio"],
        "ref_text":     ["ref_text", "reference_text", "ref_transcript"],
        "language":     ["language", "lang"],
        "id":           ["id", "name", "sample_id", "utt_id"],
    }
    normalised = {}
    lower_row = {
        k.lower().strip(): v for k, v in row.items() if isinstance(k, str)
    }
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            if alias in lower_row:
                normalised[canonical] = lower_row[alias]
                break
    return normalised


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infer.py",
        description="Unified ITTS Inference Pipeline — run any supported TTS model from a CSV/JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        metavar="MODEL",
        help=(
            "Model to use. Choices: "
            + ", ".join(MODEL_REGISTRY.keys())
        ),
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        metavar="FILE",
        help="Path to input CSV, JSON, or JSONL file.",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./outputs",
        metavar="DIR",
        help="Directory for the generated WAV files, written directly into it (no per-model sub-folder). (default: ./outputs)",
    )
    parser.add_argument(
        "--weights_dir", "-w",
        type=str,
        default=os.environ.get('WEIGHTS_DIR', str(_HERE.parent / 'weights')),
        metavar="DIR",
        help="Local weight root (WEIGHTS_DIR or <repo>/weights); missing recorded HF checkpoints use the pinned HF cache.",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        metavar="DEVICE",
        help="Compute device: 'cuda', 'cuda:0', 'cpu', etc. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print all supported model names and exit.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip rows that fail to synthesize instead of aborting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output WAV files. Default: skip if file exists.",
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Samples per GPU batch. Values > 1 use synthesize_batch() for faster "
            "throughput on large datasets. (default: 1)"
        ),
    )
    parser.add_argument(
        "--default_style_prompt",
        type=str,
        default=DEFAULT_STYLE_PROMPT,
        metavar="PROMPT",
        help=(
            "Style prompt applied to rows that have no style_prompt field. "
            "(default: neutral human voice)"
        ),
    )
    parser.add_argument(
        "--num_save_workers",
        type=int,
        default=4,
        metavar="N",
        help="Thread workers for concurrent WAV file saving. (default: 4)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N input rows (useful for smoke tests).",
    )
    parser.add_argument(
        "--attribute_category",
        choices=[
            "age", "gender", "gender_x_age", "l2_accent", "native_accent",
            "accent", "emotion",
        ],
        default=None,
        help="Attribute category. Required by CapSpeech for checkpoint routing.",
    )
    parser.add_argument(
        "--moss_audio_temperature",
        type=float,
        default=1.5,
        metavar="F",
        help="MOSS-VoiceGen audio sampling temperature. (default: 1.5)",
    )
    parser.add_argument(
        "--voxcpm_cfg_value",
        type=float,
        default=2.0,
        metavar="F",
        help="VoxCPM2 classifier-free guidance strength. (default: 2)",
    )
    parser.add_argument(
        "--vox_cfg_st_on_text",
        type=float,
        default=1.0,
        metavar="F",
        help="VoxInstruct semantic/text CFG. Set above 1 for stronger guidance. (default: 1)",
    )
    parser.add_argument(
        "--vox_cfg_at_on_text",
        type=float,
        default=1.0,
        metavar="F",
        help="VoxInstruct acoustic/text CFG. Set above 1 for stronger guidance. (default: 1)",
    )
    parser.add_argument(
        "--vox_cfg_at_on_st",
        type=float,
        default=1.0,
        metavar="F",
        help="VoxInstruct acoustic/semantic CFG. Set above 1 for stronger guidance. (default: 1)",
    )
    parser.add_argument(
        "--vox_nar_iter_steps",
        type=int,
        default=1,
        metavar="N",
        help="VoxInstruct NAR refinement iterations. Higher may improve quality. (default: 1)",
    )
    return parser


# ---------------------------------------------------------------------------
# Device auto-detection
# ---------------------------------------------------------------------------

def auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def synthesize_with_retries(backend, batch, **kwargs):
    """Return aligned (result, error) outcomes, isolating failed samples."""
    try:
        results = backend.synthesize_batch(batch, **kwargs)
        if len(results) != len(batch):
            raise RuntimeError(
                f"Backend returned {len(results)} result(s) for "
                f"a batch of {len(batch)} sample(s).")
        return [(result, None) for result in results]
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        if len(batch) == 1:
            return [(None, exc)]
        midpoint = len(batch) // 2
        print(
            f"[infer] Batch of {len(batch)} failed; retrying as "
            f"{midpoint} + {len(batch) - midpoint} samples. "
            f"Cause: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        try:
            import torch
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        except ImportError:
            pass
        left = synthesize_with_retries(
            backend, batch[:midpoint], **kwargs)
        right = synthesize_with_retries(
            backend, batch[midpoint:], **kwargs)
        return left + right


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # --list-models
    if args.list_models:
        print("\nSupported models:")
        max_len = max(len(k) for k in MODEL_REGISTRY)
        for name, (cls, hf_id) in MODEL_REGISTRY.items():
            hf_str = f"  [{hf_id}]" if hf_id else "  [manual setup required]"
            print(f"  {name:<{max_len}}  {cls.__name__}{hf_str}")
        print()
        return 0

    # Validate required args
    if not args.model:
        parser.error("--model is required. Use --list-models to see options.")
    if not args.input:
        parser.error("--input is required.")
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1.")
    if args.num_save_workers < 1:
        parser.error("--num_save_workers must be >= 1.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1.")
    if args.vox_nar_iter_steps < 1:
        parser.error("--vox_nar_iter_steps must be >= 1.")

    # Resolve device.
    device = args.device or auto_device()
    print(f"[infer] Device: {device}")

    # Resolve dirs.
    weights_dir = Path(args.weights_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[infer] Weights dir : {weights_dir}")
    print(f"[infer] Output dir  : {output_dir}")

    # Load input file.
    print(f"[infer] Loading input: {args.input}")
    rows = load_input(args.input)
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        print("[infer] No rows found in input file. Exiting.")
        return 0
    print(f"[infer] {len(rows)} row(s) to process.")

    # -----------------------------------------------------------------------
    # Pre-filter: validate rows and build the pending work list.
    # -----------------------------------------------------------------------
    success = 0
    skipped = 0
    failed = 0

    pending: List[Dict[str, Any]] = []
    seen_outputs: Dict[Path, int] = {}
    for idx, raw_row in enumerate(rows):
        row = normalise_row(raw_row)

        text_value = row.get("text", "")
        text = text_value.strip() if isinstance(text_value, str) else str(
            text_value or "").strip()
        if not text:
            print(f"  [row {idx}] SKIP — 'text' field is empty.")
            skipped += 1
            continue

        id_value = row.get("id", "")
        sample_id = (
            id_value.strip() if isinstance(id_value, str)
            else str(id_value or "").strip()
        ) or str(idx).zfill(4)
        style_value = row.get("style_prompt", "")
        style_prompt = (
            style_value.strip() if isinstance(style_value, str)
            else str(style_value or "").strip()
        ) or args.default_style_prompt
        ref_audio = row.get("ref_audio") or None
        ref_text_val = row.get("ref_text") or None
        language_value = row.get("language", "auto")
        language = (
            language_value.strip() if isinstance(language_value, str)
            else str(language_value or "auto").strip()
        )

        # Output filename = <id>.wav  (no model suffix, as required).
        out_name = (sample_id + ".wav").replace("/", "_").replace("\\", "_")
        out_path = output_dir / out_name

        if out_path in seen_outputs:
            print(
                f"[ERROR] Rows {seen_outputs[out_path]} and {idx} map to the "
                f"same output file: {out_path.name}", file=sys.stderr)
            return 1
        seen_outputs[out_path] = idx

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        pending.append({
            "text":         text,
            "style_prompt": style_prompt,
            "ref_audio":    ref_audio,
            "ref_text":     ref_text_val,
            "language":     language,
            "sample_id":    sample_id,
            "out_path":     out_path,
        })

    print(
        f"[infer] {len(pending)} sample(s) queued for synthesis "
        f"({skipped} already exist / empty, batch_size={args.batch_size})."
    )

    if not pending:
        print("[infer] Nothing to do.")
        return 0

    if args.model == "moss-voicegen" and args.batch_size > 1:
        # AR generation runs to the longest sample in a batch. Grouping similar
        # text lengths reduces padded decode steps without changing outputs.
        pending.sort(key=lambda item: len(item["text"]))
        print("[MOSS-VoiceGen] Grouped pending samples by text length for batching.")

    # Instantiate and load backend only after pre-filtering. This avoids an
    # expensive model load when all requested WAV files already exist.
    BackendClass, default_hf_id = MODEL_REGISTRY[args.model]
    backend = BackendClass(
        model_id=default_hf_id or args.model,
        weights_dir=weights_dir,
        device=device,
    )
    backend.attribute_category = args.attribute_category
    native_batch = getattr(backend, "supports_native_batch", False)
    print(f"[infer] Native backend batch: {'yes' if native_batch else 'no'}")
    if args.batch_size > 1 and not native_batch:
        print(
            "[infer] NOTE: this backend has no native batch implementation; "
            "batch_size will group scheduler work, but synthesis is sequential."
        )

    print(f"[infer] Loading model '{args.model}'...")
    try:
        backend.ensure_loaded()
    except Exception as e:
        print(
            f"\n[ERROR] Failed to load model '{args.model}':\n  {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

    synthesis_kwargs: Dict[str, Any] = {}
    if args.model == "moss-voicegen":
        synthesis_kwargs = {
            "audio_temperature": args.moss_audio_temperature,
        }
        print(
            f"[MOSS-VoiceGen] Audio temperature="
            f"{args.moss_audio_temperature:g}"
        )
    elif args.model == "voxcpm2":
        synthesis_kwargs = {
            "cfg_value": args.voxcpm_cfg_value,
        }
        print(f"[VoxCPM2] CFG value={args.voxcpm_cfg_value:g}")
    elif args.model == "voxinstruct":
        synthesis_kwargs = {
            "cfg_st_on_text": args.vox_cfg_st_on_text,
            "cfg_at_on_text": args.vox_cfg_at_on_text,
            "cfg_at_on_st": args.vox_cfg_at_on_st,
            "nar_iter_steps": args.vox_nar_iter_steps,
        }
        print(
            "[VoxInstruct] CFG(st/text, at/text, at/st)="
            f"{args.vox_cfg_st_on_text:g}, {args.vox_cfg_at_on_text:g}, "
            f"{args.vox_cfg_at_on_st:g}; NAR iterations={args.vox_nar_iter_steps}"
        )

    # -----------------------------------------------------------------------
    # Batched synthesis + concurrent WAV saving.
    # -----------------------------------------------------------------------
    pbar = (
        _tqdm_cls(total=len(pending), unit="sample", dynamic_ncols=True)
        if _HAS_TQDM else None
    )
    save_errors: List[str] = []
    run_started = time.perf_counter()
    synthesized_audio_seconds = 0.0

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.num_save_workers
    ) as save_pool:
        save_futs: Dict[concurrent.futures.Future, str] = {}
        max_pending_saves = max(2, args.num_save_workers * 2)
        interrupted = False
        aborted = False
        batch_start = 0

        def drain_saves(wait_all: bool = False) -> None:
            nonlocal success, failed
            if not save_futs:
                return
            return_when = (
                concurrent.futures.ALL_COMPLETED if wait_all
                else concurrent.futures.FIRST_COMPLETED
            )
            done, _ = concurrent.futures.wait(
                save_futs, return_when=return_when)
            for fut in done:
                sid = save_futs.pop(fut)
                try:
                    fut.result()
                    success += 1
                except Exception as exc:
                    save_errors.append(f"  [{sid}] SAVE ERROR: {exc}")
                    failed += 1
                if pbar:
                    pbar.update(1)

        while batch_start < len(pending) and not interrupted:
            batch = pending[batch_start: batch_start + args.batch_size]
            batch_start += len(batch)
            t0 = time.perf_counter()

            try:
                import torch
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
            except (ImportError, RuntimeError):
                torch = None

            try:
                outcomes = synthesize_with_retries(
                    backend, batch, **synthesis_kwargs)
                if torch is not None and device.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                per_sample = elapsed / len(batch)
                batch_audio_seconds = 0.0

                batch_failed = False
                for item, (result, error) in zip(batch, outcomes):
                    if error is not None:
                        failed += 1
                        batch_failed = True
                        msg = f"  [{item['sample_id']}] FAILED: {error}"
                        if pbar:
                            pbar.write(msg)
                            pbar.update(1)
                        else:
                            print(msg, file=sys.stderr)
                        continue
                    audio_arr, sr = result
                    batch_audio_seconds += len(audio_arr) / sr
                    while len(save_futs) >= max_pending_saves:
                        drain_saves()
                    fut = save_pool.submit(
                        save_wav, audio_arr, sr, item["out_path"])
                    save_futs[fut] = item["sample_id"]

                synthesized_audio_seconds += batch_audio_seconds
                peak_gib = None
                if torch is not None and device.startswith("cuda"):
                    peak_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)
                rtf = elapsed / batch_audio_seconds if batch_audio_seconds else 0
                metric = (
                    f"[infer] batch={len(batch)} synth={elapsed:.2f}s "
                    f"samples/s={len(batch) / elapsed:.3f} "
                    f"audio_s/s={batch_audio_seconds / elapsed:.2f} rtf={rtf:.3f}"
                )
                if peak_gib is not None:
                    metric += f" peak_reserved={peak_gib:.2f}GiB"
                if pbar:
                    pbar.write(metric)
                    pbar.set_postfix(t=f"{per_sample:.2f}s", batch=len(batch))
                else:
                    print(metric)

                if batch_failed and not args.skip_errors:
                    print(
                        "\n[infer] Aborting after isolated sample failure. "
                        "Use --skip-errors to continue.", file=sys.stderr)
                    aborted = True
                    break

            except KeyboardInterrupt:
                interrupted = True
                if pbar:
                    pbar.close()
                print("\n[infer] Interrupted by user.")

            except Exception as e:
                elapsed = time.perf_counter() - t0
                failed += len(batch)
                ids = ", ".join(it["sample_id"] for it in batch)
                msg = f"  [BATCH ({ids})] FAILED ({elapsed:.1f}s): {e}"
                if pbar:
                    pbar.write(msg)
                else:
                    print(msg, file=sys.stderr)
                if not args.skip_errors:
                    if pbar:
                        pbar.close()
                    traceback.print_exc()
                    print(
                        "\n[infer] Aborting. Use --skip-errors to continue on failures.")
                    aborted = True
                    break
                traceback.print_exc()

        # Drain the save thread pool — surface any disk-write errors.
        drain_saves(wait_all=True)

    if pbar:
        pbar.close()
    for err in save_errors:
        print(err, file=sys.stderr)

    # Summary.
    unprocessed = max(0, len(pending) - batch_start)
    total = success + skipped + failed + unprocessed
    wall_elapsed = time.perf_counter() - run_started
    wall_audio_rate = (
        synthesized_audio_seconds / wall_elapsed if wall_elapsed else 0.0)
    print(
        f"\n[infer] Done. {total} row(s) processed: "
        f"{success} succeeded, {skipped} skipped, {failed} failed, "
        f"{unprocessed} unprocessed."
    )
    print(
        f"[infer] End-to-end: {wall_elapsed:.2f}s, "
        f"audio_s/s={wall_audio_rate:.2f}")
    if interrupted:
        return 130
    return 0 if failed == 0 and not aborted and unprocessed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
