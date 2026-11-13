"""CapSpeech category-routed, native-batch backend."""

import contextlib
import io
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import ITTSBackend


class CapSpeechBackend(ITTSBackend):
    """Use CapTTS, AccCapTTS, or EmoCapTTS according to attribute category."""

    supports_native_batch = True
    CATEGORY_TASK = {
        "age": "CapTTS",
        "gender": "CapTTS",
        "gender_x_age": "CapTTS",
        "accent": "AccCapTTS",
        "emotion": "EmoCapTTS",
    }

    def __init__(self, model_id: str, weights_dir: Path, device: str = "cuda"):
        super().__init__(model_id, weights_dir, device)
        self.attribute_category = None
        self._components = None
        self._sample_rate = 24000

    def load(self) -> None:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            T5EncoderModel,
        )
        from g2p_en import G2p
        import laion_clap
        from capspeech.nar import bigvgan
        from capspeech.nar.model.modules import MelSpec
        from capspeech.nar.network.crossdit import CrossDiT
        from capspeech.nar.utils import load_yaml_with_includes
        if __package__.startswith('generation.'):
            from ..model_pins import checkpoint_path
        else:
            from model_pins import checkpoint_path

        category = self.attribute_category
        if category is None:
            # Diversity runs (sample_1/2/3.jsonl) carry no attribute category.
            # Route to the general caption-TTS checkpoint; override with
            # CAPSPEECH_DIVERSITY_TASK if a different task is desired.
            task = os.environ.get("CAPSPEECH_DIVERSITY_TASK", "CapTTS")
            category = "diversity"
        elif category in self.CATEGORY_TASK:
            task = self.CATEGORY_TASK[category]
        else:
            raise ValueError(
                "CapSpeech requires --attribute_category with one of: "
                + ", ".join(self.CATEGORY_TASK)
            )
        root = Path(self.effective_path())
        checkpoint_path = root / f"nar_{task}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing CapSpeech checkpoint: {checkpoint_path}")

        params = load_yaml_with_includes(str(root / "nar_pretrain.yaml"))
        model = CrossDiT(**params["model"]).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")["model"]
        model.load_state_dict(checkpoint, strict=True)
        model.eval()

        bigvgan_path = self.weights_dir / "nvidia--bigvgan_v2_24khz_100band_256x"
        bigvgan_source = checkpoint_path('nvidia/bigvgan_v2_24khz_100band_256x', bigvgan_path)
        vocoder = bigvgan.BigVGAN.from_pretrained(
            bigvgan_source, use_cuda_kernel=False
        )
        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(self.device)

        flan_path = self.weights_dir / "google--flan-t5-large"
        flan_source = checkpoint_path('google/flan-t5-large', flan_path)
        caption_tokenizer = AutoTokenizer.from_pretrained(flan_source)
        caption_encoder = T5EncoderModel.from_pretrained(flan_source).to(
            self.device
        ).eval()

        clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        # LAION-CLAP prints every checkpoint tensor; suppress this noisy but
        # otherwise harmless output in long resumable inference logs.
        with contextlib.redirect_stdout(io.StringIO()):
            clap_model.load_ckpt(str(root / "clap-630k-best.pt"))
        duration_path = root / "nar_duration_predictor"
        duration_tokenizer = AutoTokenizer.from_pretrained(duration_path)
        duration_model = AutoModelForSequenceClassification.from_pretrained(
            duration_path
        ).to(self.device).eval()

        phn2num = {}
        with open(root / "vocab.txt", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split(" ")
                if len(parts) >= 2:
                    phn2num[parts[1]] = int(parts[0])

        self._components = {
            "model": model,
            "vocoder": vocoder,
            "phn2num": phn2num,
            "g2p": G2p(),
            "clap": clap_model,
            "duration_tokenizer": duration_tokenizer,
            "duration_model": duration_model,
            "caption_tokenizer": caption_tokenizer,
            "caption_encoder": caption_encoder,
            "mel": MelSpec(**params["mel"]).to(self.device),
        }
        print(f"[CapSpeech] Loaded {task} for category={category}.")

    @staticmethod
    def _tag(text: str) -> str:
        if "<B_start>" in text or "<I_start>" in text:
            return text.split(">", 1)[0].strip()[1:].lower().replace("_", " ")
        return "none"

    @staticmethod
    def _bounded_duration(text: str, predicted: float) -> float:
        words = max(1, len(re.sub(r"<[^>]*>", "", text).split()))
        minimum = words / 4.0
        maximum = words / 1.5
        return min(max(predicted, minimum), maximum)

    def _encode_text(self, text: str) -> List[int]:
        from capspeech.nar.generate import encode

        phonemes = encode(text, self._components["g2p"])
        return [self._components["phn2num"][item] for item in phonemes]

    def _sample(self, x, text, prompt, clap, audio_mask, prompt_mask, steps, cfg):
        import torch
        from einops import rearrange
        from torchdiffeq import odeint

        model = self._components["model"]
        vocoder = self._components["vocoder"]
        y0 = torch.randn_like(x)
        neg_text = torch.full_like(text, -1)
        neg_clap = torch.zeros_like(clap)
        neg_prompt = torch.zeros_like(prompt)
        neg_prompt_mask = torch.zeros_like(prompt_mask)
        neg_prompt_mask[:, 0] = True

        def derivative(time, state):
            pred = model(
                x=state, cond=None, text=text, time=time, prompt=prompt,
                clap=clap, mask=audio_mask, prompt_mask=prompt_mask,
            )
            null_pred = model(
                x=state, cond=None, text=neg_text, time=time,
                prompt=neg_prompt, clap=neg_clap, mask=audio_mask,
                prompt_mask=neg_prompt_mask,
            )
            return pred + (pred - null_pred) * cfg

        times = torch.linspace(0, 1, steps, device=self.device)
        times = times - (torch.cos(torch.pi / 2 * times) - 1 + times)
        output = odeint(derivative, y0, times, method="euler")[-1]
        output = rearrange(output, "b n d -> b d n")
        waveform = vocoder(output)
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform[:, 0]
        return waveform.detach().cpu().to(torch.float32).numpy()

    def synthesize(
        self,
        text: str,
        style_prompt: str = "",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        language: str = "auto",
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        return self.synthesize_batch(
            [{"text": text, "style_prompt": style_prompt}], **kwargs
        )[0]

    def synthesize_batch(
        self, items: List[Dict[str, Any]], **kwargs
    ) -> List[Tuple[np.ndarray, int]]:
        import torch

        self.ensure_loaded()
        c = self._components
        texts = [item["text"] for item in items]
        captions = [item.get("style_prompt") or "A natural speaking voice." for item in items]
        token_lists = [self._encode_text(text) for text in texts]

        duration_strings = [
            f"{caption} <NEW_SEP> {text}"
            for caption, text in zip(captions, texts)
        ]
        duration_inputs = c["duration_tokenizer"](
            duration_strings,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=400,
        ).to(self.device)
        # torchdiffeq's solver accepts no-grad tensors but rejects tensors
        # created in inference_mode when it evaluates the ODE derivative.
        with torch.no_grad():
            predicted = c["duration_model"](**duration_inputs).logits.reshape(-1)
        durations = [
            self._bounded_duration(text, float(value))
            for text, value in zip(texts, predicted.detach().cpu())
        ]
        frame_lengths = [
            max(len(tokens) + 1, math.ceil(duration * self._sample_rate / 256))
            for tokens, duration in zip(token_lists, durations)
        ]
        max_frames = max(frame_lengths)
        max_tokens = max(len(tokens) for tokens in token_lists)

        text_tensor = torch.full(
            (len(items), max_tokens), -1, dtype=torch.long, device=self.device
        )
        for index, tokens in enumerate(token_lists):
            text_tensor[index, :len(tokens)] = torch.tensor(
                tokens, dtype=torch.long, device=self.device
            )

        caption_inputs = c["caption_tokenizer"](
            captions, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            prompt = c["caption_encoder"](**caption_inputs).last_hidden_state
        prompt_mask = caption_inputs["attention_mask"].bool()
        clap = c["clap"].get_text_embedding(
            [self._tag(text) for text in texts], use_tensor=True
        ).to(self.device)
        if clap.ndim > 2:
            clap = clap.squeeze(1)

        frame_tensor = torch.tensor(frame_lengths, device=self.device)
        audio_mask = torch.arange(max_frames, device=self.device)[None, :] < frame_tensor[:, None]
        audio = torch.zeros(
            (len(items), max_frames, 100), device=self.device
        )
        with torch.no_grad():
            waveforms = self._sample(
                audio,
                text_tensor,
                prompt,
                clap,
                audio_mask,
                prompt_mask,
                int(kwargs.get("steps", 25)),
                float(kwargs.get("cfg", 2.0)),
            )
        results = []
        for waveform, frames in zip(waveforms, frame_lengths):
            samples = min(len(waveform), frames * 256)
            results.append((np.asarray(waveform[:samples], dtype=np.float32), self._sample_rate))
        return results
