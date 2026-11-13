"""
MOSS-VoiceGenerator backend.

Paper/Repo: https://github.com/OpenMOSS/MOSS-TTS
HF Model:   https://huggingface.co/OpenMOSS-Team/MOSS-VoiceGenerator

Install:
    git clone https://github.com/OpenMOSS/MOSS-TTS.git
    cd MOSS-TTS
    pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e .

Or install directly:
    pip install transformers>=5.0.0 torchaudio

MOSS-VoiceGenerator creates voices from free-form text descriptions without
any reference audio. It takes (text, instruction) and synthesises speech.

CSV/JSON fields used:
    text          - text to synthesize  (required)
    style_prompt  - voice style / character description  (required)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import importlib.util
import os
import numpy as np

from .base import ITTSBackend


class MOSSVoiceGenBackend(ITTSBackend):
    """
    Wrapper around OpenMOSS-Team/MOSS-VoiceGenerator.

    Requires MOSS-TTS package (provides the custom modeling code via
    trust_remote_code=True in AutoModel).
    """

    supports_native_batch = True

    _DEFAULT_INSTRUCTION = (
        "A clear, natural-sounding voice with neutral emotion, "
        "moderate speed, and good audio quality."
    )

    def __init__(self, model_id: str, weights_dir: Path, device: str = "cuda"):
        super().__init__(model_id, weights_dir, device)
        self._model = None
        self._processor = None

    def load(self) -> None:
        try:
            import torch
            import torchaudio
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "transformers / torchaudio not installed. Run:\n"
                "  pip install transformers torchaudio\n"
                "For full MOSS-TTS support also run:\n"
                "  git clone https://github.com/OpenMOSS/MOSS-TTS.git && "
                "cd MOSS-TTS && pip install -e ."
            ) from e

        import torch

        path = self.effective_path()
        print(f"[MOSS-VoiceGen] Loading model from: {path}")

        # Resolve attention implementation.
        dtype = torch.bfloat16 if self.device.startswith(
            "cuda") else torch.float32
        attn_impl = self._resolve_attn(dtype)

        # Check for local MOSS-Audio-Tokenizer weights
        codec_local = self.weights_dir / "OpenMOSS-Team--MOSS-Audio-Tokenizer"
        if __package__.startswith('generation.'):
            from ..model_pins import checkpoint_path
        else:
            from model_pins import checkpoint_path
        codec_kwargs = {'codec_path': checkpoint_path('OpenMOSS-Team/MOSS-Audio-Tokenizer', codec_local)}
        print(f"[MOSS-VoiceGen] Using recorded codec: {codec_kwargs['codec_path']}")

        self._processor = AutoProcessor.from_pretrained(
            path, trust_remote_code=True, **codec_kwargs
        )
        self._processor.audio_tokenizer = (
            self._processor.audio_tokenizer.to(self.device)
        )

        # Full BF16 by default; 4-bit is opt-in for smaller GPUs via
        # MOSS_QUANTIZATION=4bit.
        quantization_config = None
        if os.environ.get("MOSS_QUANTIZATION", "none").lower() == "4bit":
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print("[MOSS-VoiceGen] Using opt-in 4-bit NF4 quantization.")
            except ImportError as exc:
                raise ImportError(
                    "MOSS_QUANTIZATION=4bit requires bitsandbytes") from exc
        else:
            print("[MOSS-VoiceGen] Using BF16 weights for maximum throughput.")

        self._model = AutoModel.from_pretrained(
            path,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            torch_dtype=dtype,
            device_map={"": self.device},
            quantization_config=quantization_config,
        )
        self._model.eval()
        self._input_device = next(self._model.parameters()).device
        self._dtype = dtype
        print(f"[MOSS-VoiceGen] Model loaded (attn={attn_impl}).")

    def _resolve_attn(self, dtype) -> str:
        import torch
        if (
            self.device.startswith("cuda")
            and importlib.util.find_spec("flash_attn") is not None
            and dtype in {torch.float16, torch.bfloat16}
        ):
            try:
                major, _ = torch.cuda.get_device_capability()
                if major >= 8:
                    return "flash_attention_2"
            except Exception:
                pass
        if self.device.startswith("cuda"):
            return "sdpa"
        return "eager"

    def synthesize(
        self,
        text: str,
        style_prompt: str = "",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        language: str = "auto",
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        results = self.synthesize_batch(
            [{"text": text, "style_prompt": style_prompt}], **kwargs
        )
        return results[0]

    def synthesize_batch(
        self,
        items: List[Dict[str, Any]],
        **kwargs,
    ) -> List[Tuple[np.ndarray, int]]:
        import torch

        self.ensure_loaded()

        conversations = [
            [self._processor.build_user_message(
                text=item["text"],
                instruction=(item.get("style_prompt") or "").strip()
                or self._DEFAULT_INSTRUCTION,
            )]
            for item in items
        ]

        batch = self._processor(conversations, mode="generation")
        # Move all tensor values to device
        batch = {k: v.to(self._input_device) if hasattr(v, "to") else v
                 for k, v in batch.items()}

        with torch.inference_mode():
            outputs = self._model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                audio_temperature=kwargs.get("audio_temperature", 1.5),
                audio_top_p=kwargs.get("audio_top_p", 0.6),
                audio_top_k=kwargs.get("audio_top_k", 50),
                audio_repetition_penalty=kwargs.get(
                    "audio_repetition_penalty", 1.1),
            )

        sr = self._processor.model_config.sampling_rate
        results: List[Tuple[np.ndarray, int]] = []
        for message in self._processor.decode(outputs):
            audio = message.audio_codes_list[0]
            audio_np = audio.detach().cpu().to(torch.float32).numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=0)
            results.append((audio_np.astype(np.float32), int(sr)))

        return results
