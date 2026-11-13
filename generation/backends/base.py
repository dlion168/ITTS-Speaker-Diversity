"""
Abstract base class for all ITTS model backends.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ITTSBackend(ABC):
    """
    Every model backend must implement load() and synthesize().

    synthesize() returns (audio_array: np.ndarray[float32], sample_rate: int)
    The audio_array should be mono, values in [-1, 1].
    """

    supports_native_batch = False

    def __init__(self, model_id: str, weights_dir: Path, device: str = "cuda"):
        self.model_id = model_id          # HF model ID or local path
        self.weights_dir = Path(weights_dir)
        self.device = device
        self._loaded = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _local_weights(self) -> Optional[Path]:
        """Return local weight path if it exists, else None."""
        candidate = self.weights_dir / self._safe_name()
        if candidate.exists():
            return candidate
        return None

    def _safe_name(self) -> str:
        """Convert model_id to a filesystem-safe directory name."""
        return self.model_id.replace("/", "--") if self.model_id else self.__class__.__name__

    def effective_path(self) -> str:
        """Return local path (if weights pre-downloaded) or HF model ID."""
        if __package__.startswith('generation.'):
            from ..model_pins import checkpoint_path
        else:
            from model_pins import checkpoint_path
        # Only reached during an explicit model load, never plan construction.
        return checkpoint_path(self.model_id, self.weights_dir / self._safe_name())

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        style_prompt: str = "",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        language: str = "auto",
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech.

        Parameters
        ----------
        text         : text to synthesize
        style_prompt : natural-language style/voice description
        ref_audio    : path to reference audio file (for voice-clone models)
        ref_text     : transcript of reference audio
        language     : target language code ('auto', 'en', 'zh', ...)

        Returns
        -------
        (audio_array, sample_rate)
        """

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True

    def synthesize_batch(
        self,
        items: List[Dict[str, Any]],
        **kwargs,
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Synthesise a batch of items.
        Default implementation calls synthesize() sequentially.
        Override in subclasses to exploit model-level batching for speed.
        """
        return [
            self.synthesize(
                text=item["text"],
                style_prompt=item.get("style_prompt", ""),
                ref_audio=item.get("ref_audio"),
                ref_text=item.get("ref_text"),
                language=item.get("language", "auto"),
                **kwargs,
            )
            for item in items
        ]
