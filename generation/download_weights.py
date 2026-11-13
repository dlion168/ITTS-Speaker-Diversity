#!/usr/bin/env python3
"""
ITTS Diversity — Weight Downloader
====================================
Downloads the model weights, at the revisions in configs/systems.yaml, into <repo>/weights/
(or WEIGHTS_DIR / --weights-dir). VoxInstruct's weights are on Google Drive and are fetched
with gdown; its upstream code is cloned at the recorded commit.

Usage:
    python download_weights.py                    # download all HF models
    python download_weights.py --model parler-mini
    python download_weights.py --model moss-voicegen
    python download_weights.py --list
    python download_weights.py --check            # check which are already downloaded
"""

import argparse
import json
import os
import sys
from pathlib import Path
if __package__:
    from .model_pins import load_pins, verify_voxinstruct_files
else:
    from model_pins import load_pins, verify_voxinstruct_files

# ---------------------------------------------------------------------------
# Model download table
# ---------------------------------------------------------------------------
# key -> (hf_repo_id, local_dir_name, requires_pip_note)
HF_MODELS = {
    "parler-mini": (
        "parler-tts/parler-tts-mini-v1",
        "parler-tts--parler-tts-mini-v1",
        None,
    ),
    "parler-large": (
        "parler-tts/parler-tts-large-v1",
        "parler-tts--parler-tts-large-v1",
        None,
    ),
    "moss-voicegen": (
        "OpenMOSS-Team/MOSS-VoiceGenerator",
        "OpenMOSS-Team--MOSS-VoiceGenerator",
        (
            "MOSS-VoiceGenerator needs transformers>=5.0.0.\n"
            "  Install: git clone https://github.com/OpenMOSS/MOSS-TTS.git && "
            "cd MOSS-TTS && pip install --extra-index-url "
            "https://download.pytorch.org/whl/cu128 -e .\n"
            "  NOTE: moss-audio-tokenizer will also be downloaded automatically."
        ),
    ),
    "moss-audio-tokenizer": (
        "OpenMOSS-Team/MOSS-Audio-Tokenizer",
        "OpenMOSS-Team--MOSS-Audio-Tokenizer",
        None,
    ),
    "qwen3-voicedesign": (
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        (
            "Qwen3-TTS needs qwen-tts package.\n"
            "  Install: pip install -U qwen-tts"
        ),
    ),
    "voxcpm2": (
        "openbmb/VoxCPM2",
        "openbmb--VoxCPM2",
        "The official PyTorch backend is the release default; Nano-vLLM requires separate runtime validation.",
    ),
    "capspeech": (
        "OpenSound/CapSpeech-models",
        "OpenSound--CapSpeech-models",
        "CC BY-NC 4.0: non-commercial use only.",
    ),
    "capspeech-vocoder": (
        "nvidia/bigvgan_v2_24khz_100band_256x",
        "nvidia--bigvgan_v2_24khz_100band_256x",
        None,
    ),
    "capspeech-caption-encoder": (
        "google/flan-t5-large",
        "google--flan-t5-large",
        None,
    ),
}

# Explicit per-invocation overrides only; defaults come from immutable YAML pins.
HF_REVISIONS = {}

# Models that need manual setup (no HF download).
MANUAL_MODELS = {}

# Google Drive models (auto-download via gdown + optional git clone)
# key -> (gdrive_folder_id, local_dir_name, repo_url_or_None, pip_note_or_None)
GDRIVE_MODELS = {
    "voxinstruct": (
        "1VW5PDWC94bGH-VOFica0RdSXqoco9aBu",   # Google Drive folder ID
        "voxinstruct",
        "https://github.com/thuhcsi/VoxInstruct.git",
        (
            "Environment: envs/itts_vox.yml (Python 3.12, torch 2.9.1+cu128).\n"
            "  Then apply generation/patches/voxinstruct/ (see its README)."
        ),
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent.resolve()
ROOT = _HERE.parent
WEIGHTS_DIR = Path(os.environ.get('WEIGHTS_DIR', ROOT / 'weights')).resolve()
SYSTEMS_CONFIG = ROOT / 'configs/systems.yaml'


def download_gdrive(
    model_key: str,
    folder_id: str,
    local_name: str,
    repo_url: str | None,
    pip_note: str | None,
) -> bool:
    """Clone optional git repo and download a Google Drive folder via gdown."""
    local_dir = WEIGHTS_DIR / local_name

    # --- Clone repo if provided ---
    if repo_url:
        repo_dir = local_dir / "repo"
        commit = load_pins(SYSTEMS_CONFIG)[1].get(model_key)
        if not commit:
            raise ValueError(f'Missing recorded upstream commit for {model_key}')
        if repo_dir.exists():
            import subprocess
            existing = subprocess.check_output(['git', '-C', str(repo_dir), 'rev-parse', 'HEAD'], text=True).strip()
            if existing != commit:
                raise ValueError(f'Existing {model_key} checkout differs from the recorded commit; use a fresh weights directory. It was not changed.')
            print(f"  [{model_key}] [SKIP] matching checkout at {repo_dir}")
        else:
            print(f"  [{model_key}] Cloning {repo_url} ...")
            import subprocess
            result = subprocess.run(
                ["git", "clone", "--no-checkout", repo_url, str(repo_dir)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(
                    f"  [{model_key}] git clone failed:\n{result.stderr}",
                    file=sys.stderr,
                )
                return False
            subprocess.run(['git', '-C', str(repo_dir), 'checkout', '--detach', commit], check=True)
            print(f"  [{model_key}] Repo cloned.")

    # --- Download checkpoints from Google Drive ---
    ckpt_dir = local_dir / "pretrained"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    expected = [ckpt_dir / 'voxinstruct-sft-checkpoint' / name for name in ['ar_1800k.pyt', 'nar_1800k.pyt']]
    existing_pyt = list(ckpt_dir.rglob('*.pyt'))
    if all(path.is_file() and path.stat().st_size for path in expected):
        verify_voxinstruct_files(ckpt_dir / 'voxinstruct-sft-checkpoint')
        print(f"  [SKIP] {model_key}: checkpoints already at {ckpt_dir}")
        return True
    if existing_pyt:
        raise ValueError('Partial/unrecognised VoxInstruct checkpoints exist; preserve them and choose a fresh weights directory')
    try:
        import gdown
    except ImportError:
        print(f'[{model_key}] gdown is required for a new Drive download', file=sys.stderr)
        return False

    print(f"  [{model_key}] Downloading checkpoints from Google Drive (id={folder_id}) ...")
    try:
        result = gdown.download_folder(
            id=folder_id,
            output=str(ckpt_dir),
            quiet=False,
            use_cookies=False,
        )
        if result is None:
            print(
                f"  [{model_key}] gdown returned None — download may have failed.",
                file=sys.stderr,
            )
            return False
        if not all(path.is_file() and path.stat().st_size for path in expected):
            raise ValueError('Download did not produce both required AR/NAR checkpoints')
        verify_voxinstruct_files(ckpt_dir / 'voxinstruct-sft-checkpoint')
        print(f"  [{model_key}] Checkpoints downloaded to: {ckpt_dir}")
        if pip_note:
            print(f"\n  NOTE — runtime dependency for {model_key}:\n  {pip_note}\n")
        return True
    except Exception as e:
        print(f"  [{model_key}] FAILED: {e}", file=sys.stderr)
        return False


def download_hf(model_key: str, repo_id: str, local_name: str, pip_note: str | None):
    """Download a single HF model via snapshot_download."""
    # Use the hub's supported default transfer backend; do not require an
    # optional hf-transfer installation just to download recorded assets.
    from huggingface_hub import snapshot_download

    local_dir = WEIGHTS_DIR / local_name
    revision = HF_REVISIONS.get(model_key) or load_pins(SYSTEMS_CONFIG)[0].get(repo_id)
    if not revision:
        raise ValueError(f'No immutable checkpoint pin recorded for {model_key}')
    receipt = local_dir / '.itts_download.json'
    identity = {'repo_id': repo_id, 'revision': revision}
    if receipt.is_file():
        previous = json.loads(receipt.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError('Existing download has a different revision; choose a fresh weights directory')
        if previous.get('complete'):
            inventory = previous.get('files', {})
            if not inventory or any(Path(name).is_absolute() or '..' in Path(name).parts or not (local_dir / name).is_file() or (local_dir / name).stat().st_size != size for name, size in inventory.items()):
                raise ValueError('Recorded weights are missing/changed; preserve the directory and use a fresh weight root')
            print(f'  [SKIP] {model_key}: recorded successful download at revision {revision}')
            return True
    elif local_dir.exists() and any(local_dir.iterdir()):
        raise ValueError('Existing weights have no matching download receipt; they were not overwritten. Use a fresh weights directory.')
    revision_note = f" (revision={revision})" if revision else ""
    print(f"  [{model_key}] Downloading {repo_id}{revision_note}  ->  {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({**identity, 'complete': False}, indent=2) + '\n')

    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(local_dir),
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*",
                             "rust_model*", "coreml*", "onnx*"],
        )
        inventory = {str(path.relative_to(local_dir)): path.stat().st_size for path in local_dir.rglob('*')
                     if path.is_file() and '.cache' not in path.relative_to(local_dir).parts and path != receipt}
        if not any(Path(name).suffix in {'.bin', '.pt', '.pth', '.pyt', '.safetensors'} and size > 0 for name, size in inventory.items()):
            raise ValueError('Snapshot returned no nonempty checkpoint tensors')
        receipt.write_text(json.dumps({**identity, 'complete': True, 'files': inventory,
                                      'scope': 'successful pinned snapshot_download'}, indent=2) + '\n')
        print(f"  [{model_key}] Done.")
        if pip_note:
            print(f"\n  NOTE — runtime dependency for {model_key}:")
            print(f"  {pip_note}\n")
        return True
    except Exception as e:
        print(f"  [{model_key}] FAILED: {e}", file=sys.stderr)
        return False


def check_status():
    """Print download status of all models."""
    print("\n=== Weight Download Status ===\n")
    pins, _ = load_pins(SYSTEMS_CONFIG)
    for key, (repo_id, local_name, _) in HF_MODELS.items():
        local_dir = WEIGHTS_DIR / local_name
        if local_dir.exists() and any(local_dir.iterdir()):
            size = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
            size_gb = size / 1e9
            receipt = local_dir / '.itts_download.json'
            previous = json.loads(receipt.read_text()) if receipt.is_file() else {}
            recorded = previous.get('repo_id') == repo_id and previous.get('revision') == pins.get(repo_id) and previous.get('complete', False) and bool(previous.get('files')) and all(
                not Path(name).is_absolute() and '..' not in Path(name).parts and (local_dir / name).is_file()
                and (local_dir / name).stat().st_size == size for name, size in previous.get('files', {}).items())
            print(f"  {'✓' if recorded else '?'} {key:<20} {local_dir.name}  ({size_gb:.2f} GB; {'recorded download' if recorded else 'no download receipt'})")
        else:
            print(f"  ✗ {key:<20} not downloaded")

    print()
    for key, (folder_id, local_name, repo_url, _) in GDRIVE_MODELS.items():
        local_dir = WEIGHTS_DIR / local_name
        has_repo = (local_dir / "repo").is_dir() if repo_url else True
        expected = json.loads((ROOT / 'configs/manual_checkpoints.json').read_text())[key]
        ckpt_root = local_dir / 'pretrained/voxinstruct-sft-checkpoint'
        has_ckpts = all((ckpt_root / name).is_file() and (ckpt_root / name).stat().st_size == record['bytes']
                        for name, record in expected.items())
        marker = "?" if (has_repo and has_ckpts) else "✗"
        print(f"  {marker} {key:<20} gdrive:{folder_id}  (repo={'present' if has_repo else 'missing'}, AR/NAR sizes={'match' if has_ckpts else 'missing/different'}; content hash verified only on explicit download/load)")

    print()
    for key, (name, instructions) in MANUAL_MODELS.items():
        marker = "✓" if _check_manual(key) else "✗"
        print(f"  {marker} {key:<20} {name}  (manual setup)")
    print()


def _check_manual(key: str) -> bool:
    return False


def print_manual_instructions():
    print("\n=== Models Requiring Manual Setup ===\n")
    for key, (name, instructions) in MANUAL_MODELS.items():
        status = "✓ DONE" if _check_manual(key) else "✗ NOT SET UP"
        print(f"[{key}] {name}  ({status})")
        print(instructions)
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Download ITTS model weights to ./weights/ subfolder."
    )
    all_keys = list(HF_MODELS.keys()) + list(GDRIVE_MODELS.keys()) + list(MANUAL_MODELS.keys())
    p.add_argument(
        "--model", "-m",
        nargs="+",
        choices=all_keys + ["all"],
        default=["all"],
        metavar="MODEL",
        help=(
            "Model(s) to download. 'all' downloads all HF + GDrive models. "
            f"Choices: {', '.join(all_keys + ['all'])}"
        ),
    )
    p.add_argument("--list",   action="store_true", help="List all models and exit.")
    p.add_argument("--check",  action="store_true", help="Show download status and exit.")
    p.add_argument("--manual", action="store_true", help="Show manual-setup instructions and exit.")
    p.add_argument('--weights-dir', '--weights_dir', type=Path, default=WEIGHTS_DIR,
                   help='Weight root; defaults to WEIGHTS_DIR or <repo>/weights')
    p.add_argument('--systems-config', type=Path, default=SYSTEMS_CONFIG, help='Recorded immutable checkpoint pins')
    return p


def main():
    global WEIGHTS_DIR, SYSTEMS_CONFIG
    args = build_parser().parse_args()
    WEIGHTS_DIR = args.weights_dir.resolve()
    SYSTEMS_CONFIG = args.systems_config.resolve()

    if args.list:
        print("\nHuggingFace models (auto-download):")
        for k, (r, l, _) in HF_MODELS.items():
            print(f"  {k:<22} {r}")
        print("\nGoogle Drive models (auto-download via gdown):")
        for k, (fid, l, repo, _) in GDRIVE_MODELS.items():
            print(f"  {k:<22} gdrive:{fid}")
        print("\nManual-setup models:")
        for k, (n, _) in MANUAL_MODELS.items():
            print(f"  {k:<22} {n}")
        print()
        return

    if args.check:
        check_status()
        return

    if args.manual:
        print_manual_instructions()
        return

    # Determine which to download.
    targets = args.model
    if "all" in targets:
        hf_targets     = list(HF_MODELS.keys())
        gdrive_targets = list(GDRIVE_MODELS.keys())
        manual_targets = list(MANUAL_MODELS.keys())
    else:
        hf_targets     = [m for m in targets if m in HF_MODELS]
        gdrive_targets = [m for m in targets if m in GDRIVE_MODELS]
        manual_targets = [m for m in targets if m in MANUAL_MODELS]

    # Auto-include moss-audio-tokenizer when moss-voicegen is selected.
    if "moss-voicegen" in hf_targets and "moss-audio-tokenizer" not in hf_targets:
        hf_targets.append("moss-audio-tokenizer")
    if "capspeech" in hf_targets:
        for dependency in ("capspeech-vocoder", "capspeech-caption-encoder"):
            if dependency not in hf_targets:
                hf_targets.append(dependency)

    # Download HF models.
    if hf_targets:
        print(f"\n=== Downloading {len(hf_targets)} HuggingFace model(s) ===\n")
        ok, fail = 0, 0
        for key in hf_targets:
            repo_id, local_name, pip_note = HF_MODELS[key]
            if download_hf(key, repo_id, local_name, pip_note):
                ok += 1
            else:
                fail += 1
        print(f"\n  HF downloads: {ok} OK, {fail} failed.")

    # Download Google Drive models.
    if gdrive_targets:
        print(f"\n=== Downloading {len(gdrive_targets)} Google Drive model(s) ===\n")
        ok, fail = 0, 0
        for key in gdrive_targets:
            folder_id, local_name, repo_url, pip_note = GDRIVE_MODELS[key]
            if download_gdrive(key, folder_id, local_name, repo_url, pip_note):
                ok += 1
            else:
                fail += 1
        print(f"\n  GDrive downloads: {ok} OK, {fail} failed.")

    # Print manual instructions for requested manual models.
    if manual_targets:
        print()
        for key in manual_targets:
            name, instructions = MANUAL_MODELS[key]
            status = "✓ DONE" if _check_manual(key) else "setup needed"
            print(f"[{key}] {name}  ({status})")
            print(instructions)
            print()

    # Final status.
    print()
    check_status()


if __name__ == "__main__":
    main()
