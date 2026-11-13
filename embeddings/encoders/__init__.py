"""Speaker-encoder registry.

Every encoder is an `Encoder(load, extract, batch=None)`:
  load(device)              -> model
  extract(model, wav_path)  -> (dim,) embedding of one file
  batch(model, items)       -> (n, dim) embeddings; `items` are (1, T) waveforms for
                               mode "signals" and file paths for mode "paths"

Loader modules are imported lazily, so an environment only needs the toolkits of the
encoders it actually runs (TitaNet lives in a separate NeMo environment).
"""
import importlib
from typing import Callable, NamedTuple, Optional


class Encoder(NamedTuple):
    load: Callable
    extract: Callable
    batch: Optional[Callable] = None
    batch_mode: Optional[str] = None      # "signals" or "paths"


# The eight encoders kept after validation (paper §3.2); all diversity results use these.
PAPER_ENCODERS = [
    "speechbrain_resnet",
    "speechbrain_ecapa",
    "wespeaker",
    "modelscope_eres2net",
    "modelscope_campplus",
    "nemo_titanet",
    "modelscope_eres2netv2",
    "wespeaker_voxblink2_resnet152_16k",
]
# Also embedded in the LibriSpeech encoder validation, then dropped (paper §3.2, Appendix A).
VALIDATION_ONLY_ENCODERS = [
    "speechbrain_xvector",
    "wavlm_base_plus_sv",
    "unispeech_sat_base_plus_sv",
]

_MODULE = {
    "speechbrain_resnet": "speechbrain",
    "speechbrain_ecapa": "speechbrain",
    "speechbrain_xvector": "speechbrain",
    "wespeaker": "wespeaker",
    "wespeaker_voxblink2_resnet152_16k": "wespeaker",
    "modelscope_eres2net": "modelscope",
    "modelscope_eres2netv2": "modelscope",
    "modelscope_campplus": "modelscope",
    "nemo_titanet": "nemo_titanet",
    "wavlm_base_plus_sv": "hf_sv",
    "unispeech_sat_base_plus_sv": "hf_sv",
}
SUPPORTED = list(_MODULE)


def get(name: str) -> Encoder:
    if name not in _MODULE:
        raise KeyError(f"unknown encoder {name!r}; choose from {', '.join(SUPPORTED)}")
    module = importlib.import_module(f"{__name__}.{_MODULE[name]}")
    return module.ENCODERS[name]
