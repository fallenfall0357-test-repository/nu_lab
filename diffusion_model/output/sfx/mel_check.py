import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as F_audio

sample_rate = 16000
duration = 10
n_mels = 80
n_fft = 1024
hop_length = 512
audio_path = "raw_data.wav"
waveform, sr = torchaudio.load(audio_path)
if waveform.size(0) > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
if sr != sample_rate:
    waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
num_samples = duration * sample_rate
if waveform.size(1) < num_samples:
    waveform = F.pad(waveform, (0, num_samples - waveform.size(1)))
else:
    waveform = waveform[:, :num_samples]

mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )(waveform)
mel = torch.log1p(mel)
stats = torch.load("mel_stats.pt", map_location="cpu", weights_only=True)
global_mean = stats["mean"]
global_std = stats["std"]
mel = (mel - global_mean) / (global_std + 1e-8)

def mel_to_audio_griffin(mel_spec, sample_rate=16000, n_fft=1024, hop_length=512, n_iter=64, length=None):
    mel = torch.expm1(mel_spec.squeeze(0))
    mel = mel.clamp(min=1e-7) * 10.0
    mel_inv = torchaudio.transforms.InverseMelScale(n_stft=n_fft//2+1, n_mels=mel.shape[0], sample_rate=sample_rate).to(mel.device)
    spec = mel_inv(mel)
    if length is None:
        length = (spec.size(-1) - 1) * hop_length + n_fft
    window = torch.hann_window(n_fft).to(mel.device)
    waveform = F_audio.griffinlim(spec, window=window, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, power=1.0, n_iter=n_iter, momentum=0.99, length=length, rand_init=True)
    waveform = waveform / (waveform.abs().max() + 1e-9)
    return waveform.cpu()
mel = mel * global_std + global_mean
wav = mel_to_audio_griffin(mel, sample_rate=sample_rate, n_fft=1024, hop_length=512, n_iter=64, length=duration*sample_rate)
torchaudio.save("convert.wav", waveform, sample_rate)

