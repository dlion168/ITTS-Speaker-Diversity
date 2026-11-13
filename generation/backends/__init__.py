"""
ITTS Diversity - Unified Instruction-to-Speech Inference Pipeline
Model backends registry.
"""

from .parler_tts_model import ParlerTTSBackend
from .moss_voicegen_model import MOSSVoiceGenBackend
from .qwen3_tts_model import Qwen3TTSVoiceDesignBackend
from .voxinstruct_model import VoxInstructBackend
from .voxcpm2_model import VoxCPM2Backend
from .capspeech_model import CapSpeechBackend

# Registry: CLI name -> (BackendClass, default_hf_model_id)
MODEL_REGISTRY = {
    "parler-mini":     (ParlerTTSBackend,           "parler-tts/parler-tts-mini-v1"),
    "parler-large":    (ParlerTTSBackend,           "parler-tts/parler-tts-large-v1"),
    "moss-voicegen":   (MOSSVoiceGenBackend,        "OpenMOSS-Team/MOSS-VoiceGenerator"),
    "qwen3-voicedesign": (Qwen3TTSVoiceDesignBackend, "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    "voxinstruct":     (VoxInstructBackend,         None),
    "voxcpm2":         (VoxCPM2Backend,             "openbmb/VoxCPM2"),
    "capspeech":       (CapSpeechBackend,            "OpenSound/CapSpeech-models"),
}

__all__ = [
    "MODEL_REGISTRY",
    "ParlerTTSBackend",
    "MOSSVoiceGenBackend",
    "Qwen3TTSVoiceDesignBackend",
    "VoxInstructBackend",
    "VoxCPM2Backend",
    "CapSpeechBackend",
]
