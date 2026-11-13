"""Shims for library-version mismatches in the embedding environments.

Applied once by extract_embeddings.py before any encoder is imported.
"""
import functools
import sys
import types

import soundfile as sf
import torch
import torchaudio


def patch_huggingface_hub():
    """huggingface_hub >= 0.25 removed HfFolder / ModelFilter and the use_auth_token argument;
    NeMo (TitaNet) still uses them."""
    import huggingface_hub as hfhub
    if not hasattr(hfhub, "HfFolder"):
        class _HfFolderShim:
            @staticmethod
            def get_token():
                return hfhub.get_token()
        hfhub.HfFolder = _HfFolderShim
    if not hasattr(hfhub, "ModelFilter"):
        hfhub.ModelFilter = None

    orig = hfhub.hf_hub_download

    @functools.wraps(orig)
    def hf_hub_download(*args, **kwargs):
        kwargs.pop("use_auth_token", None)
        return orig(*args, **kwargs)
    hfhub.hf_hub_download = hf_hub_download


def patch_torchaudio():
    """torchaudio 2.x dropped set_audio_backend and sox_effects; WeSpeaker still calls them.
    The replacement sox_effects only implements the "rate" effect, the one WeSpeaker uses."""
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **k: None
    if "torchaudio.sox_effects" not in sys.modules:
        mod = types.ModuleType("torchaudio.sox_effects")

        def apply_effects_tensor(waveform, sample_rate, effects):
            target_sr = sample_rate
            for effect in effects:
                if effect[0] == "rate":
                    target_sr = int(effect[1])
            if target_sr != sample_rate:
                waveform = torchaudio.transforms.Resample(sample_rate, target_sr)(waveform)
            return waveform, target_sr

        def apply_effects_file(path, effects, normalize=True):
            waveform, sr = sf.read(path)
            waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
            return apply_effects_tensor(waveform, sr, effects)

        mod.apply_effects_tensor = apply_effects_tensor
        mod.apply_effects_file = apply_effects_file
        sys.modules["torchaudio.sox_effects"] = mod
        torchaudio.sox_effects = mod


def apply():
    patch_huggingface_hub()
    patch_torchaudio()
