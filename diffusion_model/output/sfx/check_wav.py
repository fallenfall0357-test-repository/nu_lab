import torch
import torchaudio
import matplotlib.pyplot as plt

# ---- 1. 读取 wav 文件 ----
wav_path = "samples/sample_epoch200_0"
waveform, sample_rate = torchaudio.load(wav_path+".wav")  # waveform: [channels, time]

# 若为立体声，取平均转为单声道
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# ---- 2. 生成频谱图 ----
transform = torchaudio.transforms.Spectrogram(
    n_fft=1024, hop_length=512, power=2.0
)
spec = transform(waveform)  # shape: [1, freq, time]

# 对数幅度谱（dB尺度，更直观）
spec_db = torchaudio.transforms.AmplitudeToDB()(spec)

# ---- 3. 可视化 ----
plt.figure(figsize=(10, 4))
plt.imshow(spec_db.squeeze().numpy(), origin="lower", aspect="auto", cmap="magma")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram")
plt.xlabel("Frames")
plt.ylabel("Frequency bins")

# ---- 4. 保存 ----
plt.tight_layout()
plt.savefig(wav_path+"_spectrogram.png", dpi=300)