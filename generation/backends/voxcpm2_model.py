"""VoxCPM2 Voice Design backend."""

import concurrent.futures
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import ITTSBackend


class VoxCPM2Backend(ITTSBackend):
    """Voice Design through the official PyTorch or Nano-vLLM runtime."""

    _DEFAULT_INSTRUCTION = (
        "A clear, natural voice with neutral emotion and moderate pace."
    )

    @property
    def supports_native_batch(self) -> bool:
        return os.environ.get("VOXCPM_ENGINE", "official").lower() == "nano"

    def __init__(self, model_id: str, weights_dir: Path, device: str = "cuda"):
        super().__init__(model_id, weights_dir, device)
        self._model = None
        self._engine = os.environ.get("VOXCPM_ENGINE", "official").lower()
        self._sample_rate = 48000

    def load(self) -> None:
        path = self.effective_path()
        if self._engine == "nano":
            try:
                from nanovllm_voxcpm import VoxCPM
            except ImportError as exc:
                raise ImportError(
                    "VOXCPM_ENGINE=nano requires: pip install nano-vllm-voxcpm"
                ) from exc
            if not self.device.startswith("cuda"):
                raise ValueError("Nano-vLLM VoxCPM requires a CUDA device")
            index = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
            print(f"[VoxCPM2] Loading Nano-vLLM engine from: {path}")
            self._model = VoxCPM.from_pretrained(model=path, devices=[index])
            return

        if self._engine != "official":
            raise ValueError("VOXCPM_ENGINE must be 'official' or 'nano'")
        try:
            from voxcpm import VoxCPM
        except ImportError as exc:
            raise ImportError(
                "voxcpm is not installed. Run: pip install -e ./VoxCPM"
            ) from exc
        optimize = os.environ.get("VOXCPM_COMPILE", "1") != "0"
        print(f"[VoxCPM2] Loading official engine from: {path}")
        self._model = VoxCPM.from_pretrained(
            path,
            load_denoiser=False,
            optimize=optimize,
            device=self.device,
        )
        self._sample_rate = int(self._model.tts_model.sample_rate)

    @classmethod
    def _design_text(cls, text: str, style_prompt: str) -> str:
        instruction = (style_prompt or "").strip() or cls._DEFAULT_INSTRUCTION
        return f"({instruction}){text}"

    @staticmethod
    def _mono(audio) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        return arr.reshape(-1)

    def synthesize(
        self,
        text: str,
        style_prompt: str = "",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        language: str = "auto",
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        self.ensure_loaded()
        designed_text = self._design_text(text, style_prompt)
        if self._engine == "nano":
            chunks = self._model.generate(target_text=designed_text)
            return self._mono(np.concatenate(list(chunks))), self._sample_rate

        audio = self._model.generate(
            text=designed_text,
            cfg_value=float(kwargs.get("cfg_value", 2.0)),
            inference_timesteps=int(kwargs.get("inference_timesteps", 10)),
            seed=kwargs.get("seed"),
        )
        return self._mono(audio), self._sample_rate

    def synthesize_batch(
        self, items: List[Dict[str, Any]], **kwargs
    ) -> List[Tuple[np.ndarray, int]]:
        if self._engine != "nano":
            return super().synthesize_batch(items, **kwargs)
        self.ensure_loaded()
        workers = min(len(items), int(os.environ.get("VOXCPM_CONCURRENCY", "16")))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    self.synthesize,
                    item["text"],
                    item.get("style_prompt", ""),
                    item.get("ref_audio"),
                    item.get("ref_text"),
                    item.get("language", "auto"),
                    **kwargs,
                )
                for item in items
            ]
            return [future.result() for future in futures]
