"""Hugging Face x-vector heads on self-supervised models (encoder validation only).

Loaded as described on the models' Hugging Face model cards.
"""
import torch

from audio import load_audio_tensor
from encoders import Encoder

CHECKPOINTS = {
    "wavlm_base_plus_sv": ("microsoft/wavlm-base-plus-sv", "WavLMForXVector"),
    "unispeech_sat_base_plus_sv": ("microsoft/unispeech-sat-base-plus-sv", "UniSpeechSatForXVector"),
}


def _loader(repo, cls_name):
    def load(device):
        import transformers
        fe = transformers.AutoFeatureExtractor.from_pretrained(repo)
        model = getattr(transformers, cls_name).from_pretrained(repo).to(device).eval()
        return fe, model
    return load


def extract(model, wav_path):
    fe, net = model
    wav = load_audio_tensor(wav_path, target_sr=16000).squeeze(0).numpy()
    inputs = fe(wav, sampling_rate=16000, return_tensors="pt").to(net.device)
    with torch.no_grad():
        emb = net(**inputs).embeddings
    return emb.squeeze(0).cpu().numpy()


ENCODERS = {name: Encoder(_loader(repo, cls), extract) for name, (repo, cls) in CHECKPOINTS.items()}
