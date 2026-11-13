"""
Qwen3-TTS VoiceDesign backend.

Repo:     https://github.com/QwenLM/Qwen3-TTS
HF Model: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign

Install:
    pip install -U qwen-tts
    # Optional for better speed:
    pip install -U flash-attn --no-build-isolation

Qwen3-TTS VoiceDesign generates speech from a natural-language description of
the desired voice. No reference audio needed.

CSV/JSON fields used:
    text          - text to synthesize  (required)
    style_prompt  - voice design description  (required)
    language      - target language (optional, default 'auto')
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .base import ITTSBackend


class Qwen3TTSVoiceDesignBackend(ITTSBackend):
    """
    Wrapper around Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign.
    """

    supports_native_batch = True

    _DEFAULT_INSTRUCTION = (
        "A clear, natural, and expressive voice with moderate speed and neutral tone."
    )

    _LANG_MAP = {
        "en": "english", "zh": "chinese", "ja": "japanese", "ko": "korean",
        "fr": "french",  "de": "german",  "es": "spanish", "it": "italian",
        "pt": "portuguese", "ru": "russian",
    }

    @classmethod
    def _norm_lang(cls, language: str) -> str:
        lang = (language or "auto").lower().strip()
        return cls._LANG_MAP.get(lang, lang)

    @staticmethod
    def _to_mono_f32(audio) -> np.ndarray:
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        arr = np.array(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        return arr.astype(np.float32)

    def __init__(self, model_id: str, weights_dir: Path, device: str = "cuda"):
        super().__init__(model_id, weights_dir, device)
        self._model = None

    def load(self) -> None:
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as e:
            raise ImportError(
                "qwen-tts not installed. Run:\n"
                "  pip install -U qwen-tts\n"
                "Optional: pip install -U flash-attn --no-build-isolation"
            ) from e

        import torch

        path = self.effective_path()
        print(f"[Qwen3-VoiceDesign] Loading model from: {path}")

        dtype = torch.bfloat16 if self.device.startswith(
            "cuda") else torch.float32

        # Detect flash-attention availability.
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = None

        load_kwargs = dict(
            device_map=self.device,
            dtype=dtype,
        )
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        self._model = Qwen3TTSModel.from_pretrained(path, **load_kwargs)
        print("[Qwen3-VoiceDesign] Model loaded.")

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

        instruct = style_prompt.strip() if style_prompt.strip() else self._DEFAULT_INSTRUCTION
        lang = self._norm_lang(language)

        wavs, sr = self._model.generate_voice_design(
            text=text,
            language=lang,
            instruct=instruct,
        )

        return self._to_mono_f32(wavs[0]), int(sr)

    def synthesize_batch(
        self,
        items: List[Dict[str, Any]],
        **kwargs,
    ) -> List[Tuple[np.ndarray, int]]:
        """Native batched generation: a single generate_voice_design() call."""
        self.ensure_loaded()

        texts = [item["text"] for item in items]
        instructs = [
            (item.get("style_prompt") or "").strip() or self._DEFAULT_INSTRUCTION
            for item in items
        ]
        langs = [self._norm_lang(item.get("language", "auto"))
                 for item in items]

        wavs, sr = self._model.generate_voice_design(
            text=texts,
            language=langs,
            instruct=instructs,
        )

        return [(self._to_mono_f32(w), int(sr)) for w in wavs]
