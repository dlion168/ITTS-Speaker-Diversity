"""
VoxInstruct backend.

Paper: https://arxiv.org/abs/2408.15676 (ACM MM 2024)
Repo:  https://github.com/thuhcsi/VoxInstruct

VoxInstruct uses an AR + NAR codec language model for expressive
multilingual TTS from natural-language instructions.

Setup notes
-----------
1. Clone the VoxInstruct repo:
   git clone https://github.com/thuhcsi/VoxInstruct.git

2. Download checkpoints (via gdown — run automatically by download_weights.py):
   python download_weights.py --model voxinstruct
   Google Drive folder: https://drive.google.com/drive/folders/1VW5PDWC94bGH-VOFica0RdSXqoco9aBu

3. Place the repo + checkpoints under:
   <weights_dir>/voxinstruct/
   Expected structure:
     voxinstruct/
       repo/          <- git clone of VoxInstruct placed here
         configs/
         model/
         inference.py
         requirements.txt
         ...
       pretrained/
         voxinstruct-sft-checkpoint/
           ar_1800k.pyt
           nar_1800k.pyt

4. Install VoxInstruct dependencies:
   cd <weights_dir>/voxinstruct/repo
   conda create -n voxinstruct python=3.9.18
   pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \\
       --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt

5. Set VOXINSTRUCT_REPO to the repo path:
   export VOXINSTRUCT_REPO=/path/to/voxinstruct/repo

Input format for VoxInstruct:
    Instruction lines: <name>|<lang_id>|<instruction_text>|<optional_ref_audio>
    lang_id: 0=English, 1=Chinese

CSV/JSON fields used:
    text          - instruction text (required); VoxInstruct interprets this as
                    the full instruction (e.g., 'Speak in an angry tone: Hello!')
    style_prompt  - if provided, prepended to text as: '<style>: <text>'
    language      - 'en' -> lang_id=0, 'zh'/'cn' -> lang_id=1 (default: 'en')
    ref_audio     - optional path to reference speaker audio
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .base import ITTSBackend


_SETUP_GUIDE = """
VoxInstruct weights/repo not found. Please follow these steps:

1. Clone VoxInstruct:
   git clone https://github.com/thuhcsi/VoxInstruct.git {weights_dir}/voxinstruct/repo

2. Download checkpoints via gdown (handled by download_weights.py):
   python download_weights.py --model voxinstruct
   Google Drive: https://drive.google.com/drive/folders/1VW5PDWC94bGH-VOFica0RdSXqoco9aBu

3. Expected pretrained structure under {weights_dir}/voxinstruct/pretrained/:
   encodec-checkpoint/, google-mt5-base-checkpoint/, hubert-base-checkpoint/
   vocos-encodec-24khz/, voxinstruct-sft-checkpoint/

4. Install dependencies:
   cd {weights_dir}/voxinstruct/repo
   pip install torch==2.3.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt

5. Set VOXINSTRUCT_REPO:
   export VOXINSTRUCT_REPO={weights_dir}/voxinstruct/repo
"""


class VoxInstructBackend(ITTSBackend):
    """
    VoxInstruct wrapper. Calls VoxInstruct's inference.py as a subprocess.

    Acceleration
    ------------
    VoxInstruct's inference.py loads the AR+NAR models once and then iterates
    over every line in the synth_file. The batched path below writes a whole
    batch into a single synth_file and runs ONE subprocess, so the (very
    expensive) model load happens once per batch instead of once per sample.
    Use a large --batch_size (e.g. 256-1024) for best throughput.
    """

    supports_native_batch = True

    _LANG_MAP = {"zh": 1, "cn": 1, "chinese": 1, "en": 0, "english": 0}

    def __init__(self, model_id: str, weights_dir: Path, device: str = "cuda"):
        super().__init__(model_id, weights_dir, device)
        self._repo_path: Optional[Path] = None
        self._ar_ckpt: Optional[Path] = None
        self._nar_ckpt: Optional[Path] = None

    def _find_paths(self):
        """Resolve repo path and checkpoint paths."""
        env_repo = os.environ.get("VOXINSTRUCT_REPO")
        if env_repo and Path(env_repo).is_dir():
            repo = Path(env_repo)
        else:
            repo = self.weights_dir / "voxinstruct" / "repo"

        pretrained = self.weights_dir / "voxinstruct" / "pretrained"
        ckpt_dir = pretrained / "voxinstruct-sft-checkpoint"

        ar_ckpt = ckpt_dir / "ar_1800k.pyt"
        nar_ckpt = ckpt_dir / "nar_1800k.pyt"

        return repo, ar_ckpt, nar_ckpt

    def load(self) -> None:
        repo, ar_ckpt, nar_ckpt = self._find_paths()

        missing = []
        if not repo.is_dir():
            missing.append(f"repo dir: {repo}")
        if not ar_ckpt.is_file():
            missing.append(f"AR checkpoint: {ar_ckpt}")
        if not nar_ckpt.is_file():
            missing.append(f"NAR checkpoint: {nar_ckpt}")

        if missing:
            raise RuntimeError(
                _SETUP_GUIDE.format(weights_dir=self.weights_dir)
                + "\nMissing:\n  " + "\n  ".join(missing)
            )

        if __package__.startswith('generation.'):
            from ..model_pins import verify_voxinstruct_files
        else:
            from model_pins import verify_voxinstruct_files
        verify_voxinstruct_files(ar_ckpt.parent)
        self._repo_path = repo
        self._ar_ckpt = ar_ckpt
        self._nar_ckpt = nar_ckpt
        print(f"[VoxInstruct] Repo: {repo}")
        print(f"[VoxInstruct] AR ckpt: {ar_ckpt}")
        print("[VoxInstruct] Ready.")

    def synthesize(
        self,
        text: str,
        style_prompt: str = "",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        language: str = "auto",
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        import soundfile as sf

        self.ensure_loaded()

        # Build instruction text.
        if style_prompt.strip():
            instruction = f"{style_prompt.strip()}, \"{text}\""
        else:
            instruction = f"\"{text}\""

        # Map language to lang_id.
        lang_map = {"zh": 1, "cn": 1, "chinese": 1, "en": 0, "english": 0}
        lang_id = lang_map.get(language.lower(), 0)

        # Reference audio suffix.
        ref_suffix = f"|{ref_audio}" if ref_audio else "|"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            synth_file = tmpdir / "input.txt"
            out_dir = tmpdir / "output"
            out_dir.mkdir()

            # VoxInstruct input format: name|lang_id|instruction|[ref_audio]
            line = f"sample|{lang_id}|{instruction}{ref_suffix}\n"
            synth_file.write_text(line, encoding="utf-8")

            cmd = [
                sys.executable,
                str(self._repo_path / "inference.py"),
                "--ar_config",    str(self._repo_path /
                                      "configs" / "train_ar.yaml"),
                "--nar_config",   str(self._repo_path /
                                      "configs" / "train_nar.yaml"),
                "--ar_ckpt",      str(self._ar_ckpt),
                "--nar_ckpt",     str(self._nar_ckpt),
                "--synth_file",   str(synth_file),
                "--out_dir",      str(out_dir),
                "--device",       "cuda" if self.device.startswith(
                    "cuda") else "cpu",
                "--vocoder",      "vocos",
                "--cfg_st_on_text", str(kwargs.get("cfg_st_on_text", 1.0)),
                "--cfg_at_on_text", str(kwargs.get("cfg_at_on_text", 1.0)),
                "--cfg_at_on_st",   str(kwargs.get("cfg_at_on_st", 1.0)),
                "--nar_iter_steps", str(kwargs.get("nar_iter_steps", 1)),
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = str(self._repo_path) + \
                os.pathsep + env.get("PYTHONPATH", "")

            result = subprocess.run(
                cmd, cwd=str(self._repo_path), env=env,
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"VoxInstruct inference failed:\n{result.stderr}"
                )

            # Find generated wav.
            wav_files = list(out_dir.rglob("*.wav"))
            if not wav_files:
                raise RuntimeError(
                    f"VoxInstruct produced no WAV files.\nstdout:\n{result.stdout}"
                )

            audio_arr, sr = sf.read(str(wav_files[0]), dtype="float32")
            if audio_arr.ndim > 1:
                audio_arr = audio_arr.mean(axis=1)

        return audio_arr.astype(np.float32), sr

    # ------------------------------------------------------------------
    # Batched path: one subprocess (one model load) for the whole batch.
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_field(text: str) -> str:
        """Strip characters that would break the pipe-delimited synth_file."""
        return (text or "").replace("|", "/").replace("\n", " ").replace("\r", " ").strip()

    def _build_line(
        self, item: Dict[str, Any], basename: str
    ) -> Tuple[str, str]:
        """Return (basename, synth_file_line) for a pending item."""
        text = item["text"]
        style_prompt = (item.get("style_prompt") or "").strip()
        if style_prompt:
            instruction = f'{style_prompt}, "{text}"'
        else:
            instruction = f'"{text}"'
        instruction = self._sanitize_field(instruction)

        lang_id = self._LANG_MAP.get((item.get("language") or "en").lower(), 0)
        ref_audio = self._sanitize_field(item.get("ref_audio") or "")
        line = f"{basename}|{lang_id}|{instruction}|{ref_audio}"
        return basename, line

    def synthesize_batch(
        self,
        items: List[Dict[str, Any]],
        **kwargs,
    ) -> List[Tuple[np.ndarray, int]]:
        import soundfile as sf

        self.ensure_loaded()

        basenames: List[str] = []
        lines: List[str] = []
        for index, item in enumerate(items):
            # Internal names are independent of user IDs, so sanitized ID
            # collisions cannot overwrite another sample in the subprocess.
            internal_name = f"sample_{index:08d}"
            basename, line = self._build_line(item, internal_name)
            basenames.append(basename)
            lines.append(line)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            synth_file = tmpdir / "input.txt"
            out_dir = tmpdir / "output"
            out_dir.mkdir()
            synth_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

            # Internal cross-sample AR batch size for inference.py (no ref audio).
            # Tunable via env; lower it if the GPU runs out of memory.
            ar_batch_size = int(os.environ.get("VOX_AR_BATCH_SIZE", "16"))

            cmd = [
                sys.executable,
                str(self._repo_path / "inference.py"),
                "--ar_config",    str(self._repo_path /
                                      "configs" / "train_ar.yaml"),
                "--nar_config",   str(self._repo_path /
                                      "configs" / "train_nar.yaml"),
                "--ar_ckpt",      str(self._ar_ckpt),
                "--nar_ckpt",     str(self._nar_ckpt),
                "--synth_file",   str(synth_file),
                "--out_dir",      str(out_dir),
                "--device",       "cuda" if self.device.startswith(
                    "cuda") else "cpu",
                "--vocoder",      "vocos",
                "--cfg_st_on_text", str(kwargs.get("cfg_st_on_text", 1.0)),
                "--cfg_at_on_text", str(kwargs.get("cfg_at_on_text", 1.0)),
                "--cfg_at_on_st",   str(kwargs.get("cfg_at_on_st", 1.0)),
                "--nar_iter_steps", str(kwargs.get("nar_iter_steps", 1)),
                "--batch_size",     str(ar_batch_size),
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = str(self._repo_path) + \
                os.pathsep + env.get("PYTHONPATH", "")

            result = subprocess.run(
                cmd, cwd=str(self._repo_path), env=env,
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"VoxInstruct inference failed:\n{result.stderr}"
                )

            results: List[Tuple[np.ndarray, int]] = []
            for basename in basenames:
                wav_path = out_dir / f"{basename}.wav"
                if not wav_path.is_file():
                    raise RuntimeError(
                        f"VoxInstruct produced no WAV for {basename}")
                audio_arr, sr = sf.read(str(wav_path), dtype="float32")
                if audio_arr.ndim > 1:
                    audio_arr = audio_arr.mean(axis=1)
                results.append((audio_arr.astype(np.float32), int(sr)))

        return results
