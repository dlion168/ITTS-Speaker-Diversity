"""WeSpeaker encoders, loaded through `wespeaker.load_model(<hub name>)`.

The folder names do not match the checkpoints:
  wespeaker                          -> "english" -> voxceleb_resnet221_LM  (ResNet221, VoxCeleb, LM fine-tuned)
  wespeaker_voxblink2_resnet152_16k  -> "vblinkp" -> voxblink2_samresnet34  (SimAM-ResNet34, VoxBlink2, no fine-tuning)
The hub names resolve to those files in wespeaker/cli/hub.py at the pinned commit (envs/).
"""
import numpy as np
import torch

from audio import load_audio_tensor
from encoders import Encoder

HUB_NAMES = {
    "wespeaker": "english",
    "wespeaker_voxblink2_resnet152_16k": "vblinkp",
}


def _loader(hub_name):
    def load(device):
        import wespeaker
        model = wespeaker.load_model(hub_name)
        if hasattr(model, "set_gpu"):
            model.set_gpu(int(device.split(":")[-1]) if "cuda" in device else -1)
        elif hasattr(model, "set_device"):
            model.set_device(device)
        return model
    return load


def extract(model, wav_path):
    if hasattr(model, "extract_embedding_from_pcm"):
        pcm = load_audio_tensor(wav_path, target_sr=16000)
        if hasattr(model, "device"):
            pcm = pcm.to(model.device)
        emb = model.extract_embedding_from_pcm(pcm, 16000)
    else:
        emb = model.extract_embedding(wav_path)
    if isinstance(emb, torch.Tensor):
        return emb.squeeze().cpu().numpy()
    return np.array(emb).squeeze()


ENCODERS = {name: Encoder(_loader(hub), extract) for name, hub in HUB_NAMES.items()}
