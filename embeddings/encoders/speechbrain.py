"""SpeechBrain encoders: ResNet and ECAPA-TDNN (paper), x-vector (validation only).

Checkpoints are pulled from the Hugging Face hub by repo id.
"""
from audio import load_audio_tensor, pad_and_stack
from encoders import Encoder

CHECKPOINTS = {
    "speechbrain_resnet": "speechbrain/spkrec-resnet-voxceleb",
    "speechbrain_ecapa": "speechbrain/spkrec-ecapa-voxceleb",
    # x-vector of the encoder validation (XV)
    "speechbrain_xvector": "speechbrain/spkrec-xvect-voxceleb",
}


def _loader(source):
    def load(device):
        from speechbrain.inference.speaker import EncoderClassifier
        return EncoderClassifier.from_hparams(source=source, run_opts={"device": device})
    return load


def extract_single(model, wav_path):
    signal = load_audio_tensor(wav_path)
    emb = model.encode_batch(signal)
    return emb.squeeze().cpu().numpy()


def extract_batch(model, signals):
    """Zero-pads the batch and passes relative lengths, so SpeechBrain masks the padding."""
    batch, wav_lens = pad_and_stack(signals)
    emb = model.encode_batch(batch, wav_lens)
    return emb.squeeze(1).cpu().numpy()


ENCODERS = {name: Encoder(_loader(src), extract_single, extract_batch, "signals")
            for name, src in CHECKPOINTS.items()}
