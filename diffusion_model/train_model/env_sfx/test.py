import os
import math
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.functional as F_audio
import yaml
import pandas as pd

from transformers import AutoTokenizer, AutoModel

# ---------------- Config ----------------
DATA_AUDIO_DIR = "../../data/MACS/audio"
DATA_ANNOTATION = "../../data/MACS/annotations/MACS.yaml"
N_MELS = 80
SAMPLE_RATE = 16000
DURATION = 4  # 秒数
BATCH_SIZE = 8
EPOCHS = 50
LR = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "../../output/sfx/weights"
SAMPLE_DIR = "../../output/sfx/samples"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# ---------------- Noise schedule ----------------
T = 1000
beta_start = 1e-4
beta_end = 2e-2
betas = torch.linspace(beta_start, beta_end, T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value=1.0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def extract(a, t, x_shape):
    a = a.to(t.device)
    out = a.gather(0, t).float()
    return out.reshape(-1, *((1,) * (len(x_shape) - 1)))

# ---------------- Dataset ----------------
class MACSDataset(Dataset):
    def __init__(self, audio_dir, annotation_file, n_mels=N_MELS, sample_rate=SAMPLE_RATE, duration=DURATION):
        self.audio_dir = audio_dir
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.duration = duration
        with open(annotation_file, 'r') as f:
            data = yaml.safe_load(f)
        # 提取文件列表
        self.files = data['files']  # list of dict
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=512
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_info = self.files[idx]
        audio_path = os.path.join(self.audio_dir, file_info['filename'])
        waveform, sr = torchaudio.load(audio_path)

        # 🔧 转单声道
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        num_samples = self.duration * self.sample_rate
        if waveform.size(1) < num_samples:
            waveform = F.pad(waveform, (0, num_samples - waveform.size(1)))
        else:
            waveform = waveform[:, :num_samples]
        mel = self.mel_transform(waveform)
        mel = torch.log1p(mel)
        
        # 随机选择一个 annotator 的 sentence
        if 'annotations' in file_info and len(file_info['annotations']) > 0:
            sentence = file_info['annotations'][torch.randint(0, len(file_info['annotations']), (1,)).item()]['sentence']
        else:
            sentence = ""  # 没有注释就空字符串
        
        return mel, sentence

# ---------------- Text Encoder ----------------
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE)
text_model.eval()

def encode_text(text_list):
    tokens = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        embeddings = text_model(**tokens).last_hidden_state.mean(dim=1)
    return embeddings  # (B, hidden_dim)

# ---------------- Time embedding ----------------
def sinusoidal_embedding(timesteps, dim):
    device = timesteps.device
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1,0,0))
    return emb

# ---------------- Model ----------------
class ResBlockCond(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim=None, c_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(t_dim, out_ch) if t_dim else None
        self.cond_proj = nn.Linear(c_dim, out_ch) if c_dim else None
        self.nin_shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x, t_emb=None, c_emb=None):
        h = self.act(self.norm1(self.conv1(x)))
        if self.time_proj is not None and t_emb is not None:
            h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        if self.cond_proj is not None and c_emb is not None:
            h = h + self.cond_proj(c_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.norm2(self.conv2(h)))
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h

def align_and_cat(dec_feat, enc_feat):
    _, _, h1, w1 = dec_feat.shape
    _, _, h2, w2 = enc_feat.shape

    if h1 > h2 or w1 > w2:
        dh = (h1 - h2) // 2
        dw = (w1 - w2) // 2
        dec_feat = dec_feat[:, :, dh:dh+h2, dw:dw+w2]
    elif h1 < h2 or w1 < w2:
        dec_feat = F.interpolate(dec_feat, size=(h2,w2),mode='bilinear',align_corners=False)
    return torch.cat([dec_feat,enc_feat],dim=1)

class SmallUNetCond(nn.Module):
    def __init__(self, in_channels=1, base_ch=128, t_dim=256, c_dim=384):
        super().__init__()
        self.t_dim, self.c_dim = t_dim, c_dim
        self.time_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.enc1 = ResBlockCond(in_channels, base_ch, t_dim, c_dim)
        self.enc2 = ResBlockCond(base_ch, base_ch*2, t_dim, c_dim)
        self.enc3 = ResBlockCond(base_ch*2, base_ch*4, t_dim, c_dim)
        self.mid = ResBlockCond(base_ch*4, base_ch*4, t_dim, c_dim)
        self.dec3 = ResBlockCond(base_ch*8, base_ch*2, t_dim, c_dim)
        self.dec2 = ResBlockCond(base_ch*4, base_ch, t_dim, c_dim)
        self.dec1 = ResBlockCond(base_ch*2, base_ch, t_dim, c_dim)
        self.out = nn.Conv2d(base_ch, in_channels, 1)
        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t, c_emb=None):

        # def center_crop(tensor, target):
        #     _, _, h, w = tensor.shape
        #     _, _, th, tw = target.shape
        #     dh = (h - th) // 2
        #     dw = (w - tw) // 2
        #     return tensor[:, :, dh:dh+th, dw:dw+tw]
        
        t_emb = sinusoidal_embedding(t, self.t_dim)
        t_emb = self.time_mlp(t_emb)
        e1 = self.enc1(x, t_emb, c_emb)
        e2 = self.enc2(self.pool(e1), t_emb, c_emb)
        e3 = self.enc3(self.pool(e2), t_emb, c_emb)
        m = self.mid(self.pool(e3), t_emb, c_emb)

        d3 = self.upsample(m)
        #d3 = center_crop(d3, e3)
        d3 = self.dec3(align_and_cat(d3,e3), t_emb, c_emb)

        d2 = self.upsample(d3)
        #d2 = center_crop(d2, e2)
        d2 = self.dec2(align_and_cat(d2,e2), t_emb, c_emb)

        d1 = self.upsample(d2)
        #d1 = center_crop(d1, e1)
        d1 = self.dec1(align_and_cat(d1,e1), t_emb, c_emb)

        return self.out(d1)

    # def forward(self, x, t, c_emb):
    #     t_emb = sinusoidal_embedding(t, self.t_dim)
    #     t_emb = self.time_mlp(t_emb)
    #     e1 = self.enc1(x, t_emb, c_emb)
    #     e2 = self.enc2(self.pool(e1), t_emb, c_emb)
    #     e3 = self.enc3(self.pool(e2), t_emb, c_emb)
    #     m = self.mid(self.pool(e3), t_emb, c_emb)
    #     d3 = self.upsample(m)
    #     d3 = self.dec3(torch.cat([d3, e3], dim=1), t_emb, c_emb)
    #     d2 = self.upsample(d3)
    #     d2 = self.dec2(torch.cat([d2, e2], dim=1), t_emb, c_emb)
    #     d1 = self.upsample(d2)
    #     d1 = self.dec1(torch.cat([d1, e1], dim=1), t_emb, c_emb)
    #     return self.out(d1)

# ---------------- Forward / Reverse ----------------
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alpha_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alpha_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

def p_sample(model, x_t, t_index, c_emb):
    t = torch.full((x_t.size(0),), t_index, dtype=torch.long, device=x_t.device)
    bet = extract(betas, t, x_t.shape)
    a_t = extract(alphas, t, x_t.shape)
    a_cum = extract(alphas_cumprod, t, x_t.shape)
    sqrt_recip_a = torch.sqrt(1.0 / a_t)
    eps_theta = model(x_t, t, c_emb)
    model_mean = (1./torch.sqrt(a_t))*(x_t - (bet/torch.sqrt(1.-a_cum))*eps_theta)
    if t_index == 0:
        return model_mean
    else:
        var = (bet*(1.-extract(alphas_cumprod_prev,t,x_t.shape))/(1.-a_cum)).to(x_t.device)
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(var) * noise

@torch.no_grad()
def p_sample_loop(model, shape, c_emb):
    device = next(model.parameters()).device
    x = torch.randn(shape, device=device)
    for i in tqdm(reversed(range(T))):
        x = p_sample(model, x, i, c_emb)
    return x

@torch.no_grad()
def mel_to_audio(mel_spec, n_fft=1024, hop_length=512, n_iter=32):
    mel_spec = torch.expm1(mel_spec.squeeze(0))
    window = torch.hann_window(n_fft).to(mel_spec.device)
    return F_audio.griffinlim(
        mel_spec,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        power=1.0,
        n_iter=n_iter,
        momentum=0.99,
        length=mel_spec.size(-1) * hop_length,
        rand_init=True,
    )

# ---------------- Training ----------------
def train():
    dataset = MACSDataset(DATA_AUDIO_DIR, DATA_ANNOTATION)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    model = SmallUNetCond(in_channels=1, base_ch=128, t_dim=256, c_dim=384).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    global_step = 0
    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for mel, text in pbar:
            mel = mel.to(DEVICE)
            c_emb = encode_text(text)
            bs = mel.size(0)
            t = torch.randint(0, T, (bs,), device=DEVICE).long()
            noise = torch.randn_like(mel)
            x_t = q_sample(mel, t, noise=noise)
            eps_theta = model(x_t, t, c_emb)
            loss = F.mse_loss(eps_theta, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1
            pbar.set_postfix(loss=loss.item())
        # 保存样本
        model.eval()
        text_sample = ["city street at night"]*4
        c_emb = encode_text(text_sample)
        samples = p_sample_loop(model, (4,1,N_MELS,mel.size(2)), c_emb)
        for i, s in enumerate(samples):
            waveform = mel_to_audio(s)
            torchaudio.save(os.path.join(SAMPLE_DIR,f"sample_epoch{epoch+1}_{i}.wav"), waveform.unsqueeze(0), SAMPLE_RATE)
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"ddpm_epoch{epoch+1}.pt"))
        model.train()

# ---------------- Sampling ----------------
@torch.no_grad()
def sample(model_path, text, n=1, mel_len=256):
    model = SmallUNetCond(in_channels=1, base_ch=128, t_dim=256, c_dim=384).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    c_emb = encode_text([text]*n)
    samples = p_sample_loop(model, (n,1,N_MELS,mel_len), c_emb)
    waveforms = []
    for s in samples:
        waveform =  mel_to_audio(s)
        waveforms.append(waveform)
    return waveforms

# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','sample'], default='train')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--text', type=str, default='a city street at night')
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        assert args.model
        waveforms = sample(args.model, args.text, n=1)
        for i,wf in enumerate(waveforms):
            torchaudio.save(os.path.join(SAMPLE_DIR,f"sample_{i}.wav"), wf.unsqueeze(0), SAMPLE_RATE)
        print(f"Saved sample audio to {SAMPLE_DIR}")
