"""
Parler-TTS backend (mini-v1 / large-v1 / mini-v1.1).

Install:
    pip install git+https://github.com/huggingface/parler-tts.git soundfile

HF models:
    parler-tts/parler-tts-mini-v1
    parler-tts/parler-tts-large-v1
    parler-tts/parler-tts-mini-v1.1
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import numpy as np

from .base import ITTSBackend


class ParlerTTSBackend(ITTSBackend):
    """
    Wrapper around HuggingFace Parler-TTS models.

    CSV/JSON fields used:
        text          - text to synthesize  (required)
        style_prompt  - voice/style description (optional, falls back to default)
    """

    supports_native_batch = True

    _DEFAULT_DESCRIPTION = (
        "A female speaker delivers a slightly expressive and animated speech "
        "with a moderate speed and pitch. The recording is of very high quality, "
        "with the speaker's voice sounding clear and very close up."
    )

    def __init__(self, model_id: str, weights_dir: Path, device: str = "cuda"):
        super().__init__(model_id, weights_dir, device)
        self._model = None
        self._tokenizer = None

    @staticmethod
    def _clean_audio(audio: np.ndarray) -> np.ndarray:
        """Sanitize model output to avoid NaN/Inf and hard-clipping noise."""
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        if peak > 1.0:
            arr = arr / peak
        return np.clip(arr, -1.0, 1.0).astype(np.float32)

    def load(self) -> None:
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "parler_tts not installed. Run:\n"
                "  pip install git+https://github.com/huggingface/parler-tts.git"
            ) from e

        import torch

        path = self.effective_path()
        print(f"[ParlerTTS] Loading model from: {path}")
        # Parler-TTS DAC decoder requires float32; use float16 only for the
        # autoregressive decoder stack (not the vocoder).
        # Safest cross-component dtype is float32 on CUDA.
        self._model = (
            ParlerTTSForConditionalGeneration.from_pretrained(path)
            .to(self.device)
        )
        # torch.compile can speed up inference, but some setups may produce
        # unstable audio artifacts. Keep it opt-in for quality-first runs.
        enable_compile = os.environ.get("PARLER_ENABLE_COMPILE", "0") == "1"
        if self.device.startswith("cuda") and enable_compile:
            try:
                self._model = torch.compile(self._model, mode="default")
                print("[ParlerTTS] torch.compile enabled (PARLER_ENABLE_COMPILE=1).")
            except Exception as e:
                print(f"[ParlerTTS] torch.compile skipped: {e}")
        else:
            print("[ParlerTTS] torch.compile disabled by default for audio stability.")
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        print("[ParlerTTS] Model loaded.")

    def synthesize(
        self,
        text: str,
        style_prompt: str = "",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        language: str = "auto",
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        import torch

        self.ensure_loaded()
        description = style_prompt.strip() if style_prompt.strip() else self._DEFAULT_DESCRIPTION

        enc_desc = self._tokenizer(
            description, return_tensors="pt").to(self.device)
        enc_text = self._tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generation = self._model.generate(
                input_ids=enc_desc.input_ids,
                attention_mask=enc_desc.attention_mask,
                prompt_input_ids=enc_text.input_ids,
                prompt_attention_mask=enc_text.attention_mask,
            )

        audio_arr = self._clean_audio(generation.cpu().numpy().squeeze())
        sr = self._model.config.sampling_rate
        return audio_arr, sr

    def synthesize_batch(
        self,
        items: List[Dict[str, Any]],
        **kwargs,
    ) -> List[Tuple[np.ndarray, int]]:
        """Batched generation — pads tokenizer inputs and runs a single GPU forward pass."""
        import torch
        self.ensure_loaded()

        descriptions = [
            (item.get("style_prompt") or "").strip() or self._DEFAULT_DESCRIPTION
            for item in items
        ]
        texts = [item["text"] for item in items]

        # Parler-TTS feeds BOTH the description and the prompt through a T5
        # encoder, which is bidirectional and expects RIGHT padding (its default).
        # Forcing left padding shifts the real tokens and corrupts the encoder's
        # relative position bias, producing garbled / noisy audio for the whole
        # batch. We therefore keep the tokenizer's native right padding and rely
        # on the attention masks to ignore the trailing pad tokens.
        enc_desc = self._tokenizer(
            descriptions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        enc_text = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            generation = self._model.generate(
                input_ids=enc_desc.input_ids,
                attention_mask=enc_desc.attention_mask,
                prompt_input_ids=enc_text.input_ids,
                prompt_attention_mask=enc_text.attention_mask,
            )

        sr = self._model.config.sampling_rate
        results: List[Tuple[np.ndarray, int]] = []
        for audio in generation:
            arr = self._clean_audio(audio.cpu().numpy())
            # Parler right-pads each batched waveform with zeros up to the
            # longest sample; strip that trailing silence.
            nonzero = np.nonzero(np.abs(arr) > 1e-4)[0]
            if nonzero.size:
                arr = arr[: nonzero[-1] + 1]
            results.append((arr, sr))
        return results
