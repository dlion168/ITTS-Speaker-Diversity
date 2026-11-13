"""ModelScope 3D-Speaker encoders: ERes2Net, ERes2NetV2, CAM++.

All three are the Mandarin (zh-cn) releases trained on about 200k Chinese speakers.
"""
import logging

import numpy as np
import torch

from encoders import Encoder

logging.getLogger("modelscope").setLevel(logging.ERROR)

MODEL_IDS = {
    "modelscope_eres2net": "iic/speech_eres2net_sv_zh-cn_16k-common",
    "modelscope_eres2netv2": "iic/speech_eres2netv2_sv_zh-cn_16k-common",
    "modelscope_campplus": "iic/speech_campplus_sv_zh-cn_16k-common",
}


def _loader(model_id):
    def load(device):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        pipe = pipeline(task=Tasks.speaker_verification, model=model_id, device=device)
        # building a pipeline resets the modelscope logger to INFO (one resample warning per file)
        logging.getLogger("modelscope").setLevel(logging.ERROR)
        return pipe
    return load


def extract_batch(model, wav_paths):
    """One embedding per WAV, bypassing the verification pipeline's pair post-processing."""
    with torch.no_grad():
        embs = model.forward(model.preprocess([str(p) for p in wav_paths]))
    if isinstance(embs, torch.Tensor):
        return embs.detach().cpu().numpy()
    return np.asarray(embs)


def extract(model, wav_path):
    return extract_batch(model, [wav_path])[0]


ENCODERS = {name: Encoder(_loader(mid), extract, extract_batch, "paths")
            for name, mid in MODEL_IDS.items()}
