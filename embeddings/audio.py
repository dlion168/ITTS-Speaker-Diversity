"""Audio helpers shared by the encoder loaders."""
import soundfile as sf
import torch
import torchaudio


def load_audio_tensor(wav_path: str, target_sr: int = 16000) -> torch.Tensor:
    """Mono float32 waveform as a (1, T) tensor at target_sr."""
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    signal = torch.from_numpy(data).unsqueeze(0)
    if sr != target_sr:
        signal = torchaudio.transforms.Resample(sr, target_sr)(signal)
    return signal


def pad_and_stack(signals):
    """Pad variable-length (1, T) signals to the same length and stack.

    Returns (batch, wav_lens) where wav_lens are relative lengths, as SpeechBrain expects."""
    max_len = max(s.shape[-1] for s in signals)
    padded = []
    lengths = []
    for s in signals:
        lengths.append(s.shape[-1])
        if s.shape[-1] < max_len:
            s = torch.nn.functional.pad(s, (0, max_len - s.shape[-1]))
        padded.append(s)
    batch = torch.cat(padded, dim=0)
    wav_lens = torch.tensor(lengths, dtype=torch.float32) / max_len
    return batch, wav_lens
