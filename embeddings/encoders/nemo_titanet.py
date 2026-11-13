"""NVIDIA TitaNet-Large. Needs the separate itts_nemo environment (envs/itts_nemo.lock.txt)."""
from encoders import Encoder

CHECKPOINT = "nvidia/speakerverification_en_titanet_large"


def load(device):
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(CHECKPOINT)
    model = model.to(device)
    model.eval()
    return model


def extract(model, wav_path):
    emb = model.get_embedding(wav_path)
    return emb.squeeze().cpu().numpy()


ENCODERS = {"nemo_titanet": Encoder(load, extract)}
