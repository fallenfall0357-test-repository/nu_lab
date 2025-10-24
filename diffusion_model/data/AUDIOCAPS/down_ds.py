import os
from datasets import load_dataset, Audio
import torchaudio
import torch
import numpy as np
from tqdm import tqdm

# ---------------- Config ----------------
SAVE_ROOT = "./audiocaps_mel"
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Mel transform ----------------
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    center=True,
    pad_mode="reflect",
    power=2.0,
).to(DEVICE)

# ---------------- Utility ----------------
def save_mel_item(split, idx, mel):
    os.makedirs(os.path.join(SAVE_ROOT, split), exist_ok=True)
    save_path = os.path.join(SAVE_ROOT, split, f"{idx:06d}.pt")
    torch.save(mel.cpu(), save_path)

# ---------------- Process one dataset ----------------
def process_split(split_name, limit=100):
    print(f"🔹 Processing split: {split_name}")
    
    # 加载数据集并指定音频解码后端
    ds = load_dataset("jp1924/AudioCaps", split=split_name, cache_dir="./cache", streaming=True)
    ds = ds.cast_column("audio", Audio(decode_with="librosa"))
    
    save_dir = os.path.join(SAVE_ROOT, split_name)
    os.makedirs(save_dir, exist_ok=True)
    
    for i, item in enumerate(tqdm(ds, desc=f"{split_name}")):
        if i >= limit:
            break
        save_path = os.path.join(save_dir, f"{i:06d}.pt")
        if os.path.exists(save_path):
            continue
            
        try:
            audio = item["audio"]
            waveform = torch.tensor(audio["array"]).float()
            sr = audio["sampling_rate"]
            
            # 重采样
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                waveform = resampler(waveform)
            
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # 单声道化
                
            waveform = waveform.to(DEVICE)
            mel = mel_transform(waveform)
            mel = torch.log(mel + 1e-6)  # log-mel

            save_mel_item(split_name, i, mel)

        except Exception as e:
            print(f"⚠️ Error processing item {i}: {e}")
            continue


if __name__ == "__main__":
    process_split("train",limit=50)
    process_split("validation",limit=10)
    process_split("test",limit=10)

