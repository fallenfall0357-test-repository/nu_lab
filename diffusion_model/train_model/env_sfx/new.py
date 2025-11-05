# ddpm_sfx_improved.py
# Improved DDPM training script for mel-conditioned audio generation
# - cosine beta schedule
# - larger UNet with FiLM-style conditional injection
# - optional HiFi-GAN vocoder (load checkpoint if available)
# - mixed precision training support
# - configurable hyperparameters via argparse

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
# import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchaudio
import torchaudio.functional as F_audio
import yaml

from transformers import AutoTokenizer, AutoModel

# ---------------- Config (defaults, overridable by CLI) ----------------
DEFAULTS = {
    'data_audio_dir': '../../data/MACS/audio',
    'data_annotation': '../../data/MACS/annotations/MACS.yaml',
    'n_mels': 80,
    'sample_rate': 16000,
    'duration': 10,
    'batch_size': 8,
    'epochs': 200,
    'lr': 2e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'T': 1000,
    'base_ch': 256,
    't_dim': 512, #512
    'c_dim': 512, #512
    'save_dir': '../../output/sfx/weights',
    'sample_dir': '../../output/sfx/samples',
    'modelgen_dir': '../../output/sfx/test',
    'vocoder': 'griffinlim',
    'hifigan_ckpt': '',
    'grad_accum': 1,
    'mixed_precision': True,
    'use_global_norm': True,
    'stats_file': 'mel_stats.pt',
}

# ---------------- Utilities ----------------

def mkdirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

# ---------------- Noise schedule (cosine) ----------------

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, 0.999)

# helpers
ndef = None

def extract(a, t, x_shape):
    a = a.to(t.device)
    out = a.gather(0, t).float()
    return out.reshape(-1, *((1,) * (len(x_shape) - 1)))

# ---------------- Dataset ----------------
class MACSDataset(Dataset):
    def __init__(self, audio_dir, annotation_file, n_mels=80, sample_rate=16000, duration=10, n_fft=1024, hop_length=512, use_global_norm = True,stats_file="mel_stats.pt"):
        self.audio_dir = Path(audio_dir)
        with open(annotation_file, 'r') as f:
            data = yaml.safe_load(f)
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.duration = duration
        self.use_global_norm = use_global_norm

        # === 加载或计算全局统计 ===
        if use_global_norm:
            if Path(stats_file).exists():
                # print(f"[INFO] Loading global log-mel mean/std from {stats_file}")
                stats = torch.load(stats_file, map_location="cpu", weights_only=True)
                self.global_mean = stats["mean"]
                self.global_std = stats["std"]
                print(f"[INFO] Loading global log-mel mean{self.global_mean}/std{self.global_std} from {stats_file}")
            else:
                # 若未计算过则自动计算一次
                print("[INFO] Computing global log-mel mean/std...")
                all_mels = []
                for f in tqdm(data['files']):  # 可采样部分文件加快速度
                    wav, sr = torchaudio.load(self.audio_dir / f['filename'])
                    sr = sample_rate
                    mel = torchaudio.transforms.MelSpectrogram(
                        sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
                    )(wav)
                    # mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
                    mel_db = torch.log1p(mel)
                    all_mels.append(mel_db.mean(dim=-1))  # mean over time (no .numpy())

                # 拼接后计算全局统计
                all_mels = torch.cat(all_mels, dim=1)
                self.global_mean = all_mels.mean()
                self.global_std = all_mels.std()

                # 保存为 .pt 文件（torch 原生格式）
                torch.save(
                    {"mean": self.global_mean, "std": self.global_std},
                    stats_file
                )
                print(f"[INFO] Saved mel stats to {stats_file}")

        self.n_fft = n_fft
        self.hop_length = hop_length

        # Build (filename, sentence) pairs
        self.samples = []
        for file_info in data['files']:
            if 'annotations' in file_info and len(file_info['annotations']) > 0:
                for ann in file_info['annotations']:
                    self.samples.append({
                        'filename': file_info['filename'],
                        'sentence': ann['sentence']
                    })
            else:
                # still add one entry with empty text if no annotations
                self.samples.append({
                    'filename': file_info['filename'],
                    'sentence': ""
                })

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = (self.audio_dir / sample['filename']).resolve()
        waveform, sr = torchaudio.load(audio_path)
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
        # mel = mel.unsqueeze(0)

        if self.use_global_norm:
            mel = (mel - self.global_mean) / (self.global_std + 1e-8)
        # else:
        #     # per-sample normalization
        #     mean = mel.mean()
        #     std = mel.std()
        #     mel = (mel - mean) / (std + 1e-8)

        sentence = sample['sentence']
        # print(f"mean:{mel.mean().item()},std:{mel.std().item()}")
        return mel, sentence

# ---------------- Text encoder ----------------

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEFAULTS['device'])
text_model.eval()
for p in text_model.parameters():
    p.requires_grad = False

text_proj = None  # will create later to match c_dim
def encode_text(text_list, device):
    """仅编码文本"""
    tokens = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = text_model(**tokens).last_hidden_state.mean(dim=1)  # shape [B, 384]
    return out
# def encode_text(text_list, device, c_dim):
#     tokens = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True).to(device)
#     with torch.no_grad():
#         out = text_model(**tokens).last_hidden_state.mean(dim=1)
#     # project to c_dim
#     global text_proj
#     if text_proj is None or text_proj.weight.shape[0] != c_dim:
#         text_proj = nn.Linear(out.size(1), c_dim).to(device)
#     return text_proj(out)

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

# ---------------- Model (ResBlock with FiLM-like conditioning) ----------------
class ResBlockCond(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim=None, c_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(t_dim, out_ch*2) if t_dim else None  # produce scale+shift
        self.cond_proj = nn.Linear(c_dim, out_ch*2) if c_dim else None
        self.nin_shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x, t_emb=None, c_emb=None):
        h = self.conv1(x)
        h = self.norm1(h)
        if self.time_proj is not None and t_emb is not None:
            ts = self.time_proj(t_emb)
            scale_t, shift_t = ts.chunk(2, dim=1)
            scale_t = scale_t.unsqueeze(-1).unsqueeze(-1)
            shift_t = shift_t.unsqueeze(-1).unsqueeze(-1)
            # scale_t = torch.tanh(scale_t).unsqueeze(-1).unsqueeze(-1)
            # shift_t = torch.tanh(shift_t).unsqueeze(-1).unsqueeze(-1)
            h = h * (1 + scale_t) + shift_t
        if self.cond_proj is not None and c_emb is not None:
            cs = self.cond_proj(c_emb)
            scale_c, shift_c = cs.chunk(2, dim=1)
            scale_c = scale_c.unsqueeze(-1).unsqueeze(-1)
            shift_c = shift_c.unsqueeze(-1).unsqueeze(-1)
            # scale_c = torch.tanh(scale_c).unsqueeze(-1).unsqueeze(-1)
            # shift_c = torch.tanh(shift_c).unsqueeze(-1).unsqueeze(-1)
            h = h * (1 + scale_c) + shift_c
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h

def align_and_cat(dec_feat, enc_feat):
    _, _, h2, w2 = enc_feat.shape
    dec_feat = F.interpolate(dec_feat, size=(h2, w2), mode='bilinear', align_corners=False)
    return torch.cat([dec_feat, enc_feat], dim=1)

class ImprovedUNetCond(nn.Module):
    def __init__(self, in_channels=1, base_ch=256, t_dim=512, c_dim=512):
        super().__init__()
        self.t_dim, self.c_dim = t_dim, c_dim
        self.text_proj = nn.Linear(384, c_dim)
        self.time_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        # encoder
        self.enc1 = ResBlockCond(in_channels, base_ch, t_dim, c_dim)
        self.enc2 = ResBlockCond(base_ch, base_ch*2, t_dim, c_dim)
        self.enc3 = ResBlockCond(base_ch*2, base_ch*4, t_dim, c_dim)
        self.enc4 = ResBlockCond(base_ch*4, base_ch*8, t_dim, c_dim)
        self.mid = ResBlockCond(base_ch*8, base_ch*8, t_dim, c_dim)
        # decoder
        self.dec4 = ResBlockCond(base_ch*16, base_ch*4, t_dim, c_dim)
        self.dec3 = ResBlockCond(base_ch*8, base_ch*2, t_dim, c_dim)
        self.dec2 = ResBlockCond(base_ch*4, base_ch, t_dim, c_dim)
        self.dec1 = ResBlockCond(base_ch*2, base_ch, t_dim, c_dim)
        # self.out = nn.Sequential(nn.Conv2d(base_ch, in_channels, 1),nn.Tanh())
        self.out = nn.Conv2d(base_ch, in_channels, 1)
        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t, c_emb=None):
        c_emb = self.text_proj(c_emb)
        t_emb = sinusoidal_embedding(t, self.t_dim)
        t_emb = self.time_mlp(t_emb)
        e1 = self.enc1(x, t_emb, c_emb)
        e2 = self.enc2(self.pool(e1), t_emb, c_emb)
        e3 = self.enc3(self.pool(e2), t_emb, c_emb)
        e4 = self.enc4(self.pool(e3), t_emb, c_emb)
        m = self.mid(self.pool(e4), t_emb, c_emb)

        d4 = self.upsample(m)
        d4 = self.dec4(align_and_cat(d4, e4), t_emb, c_emb)

        d3 = self.upsample(d4)
        d3 = self.dec3(align_and_cat(d3, e3), t_emb, c_emb)

        d2 = self.upsample(d3)
        d2 = self.dec2(align_and_cat(d2, e2), t_emb, c_emb)

        d1 = self.upsample(d2)
        d1 = self.dec1(align_and_cat(d1, e1), t_emb, c_emb)

        out = self.out(d1)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return out

# ---------------- Forward / Reverse diffusion ----------------

def q_sample(x_start, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    sqrt_alpha_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alpha_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise


def p_sample(model, x_t, t_index, c_emb, betas, alphas, alphas_cumprod, alphas_cumprod_prev):
    t = torch.full((x_t.size(0),), t_index, dtype=torch.long, device=x_t.device)
    bet = extract(betas, t, x_t.shape)
    a_t = extract(alphas, t, x_t.shape)
    a_cum = extract(alphas_cumprod, t, x_t.shape)
    sqrt_recip_a = torch.sqrt(1.0 / a_t)
    eps_theta = model(x_t, t, c_emb)
    if t_index % 100 == 0: print(f"eps_theta in T={t_index}:{eps_theta.mean().item()},std:{eps_theta.std().item()},max:{eps_theta.max().item()},min:{eps_theta.min().item()}")
    model_mean = (1./torch.sqrt(a_t))*(x_t - (bet/torch.sqrt(1.-a_cum))*eps_theta)
    if t_index == 0:
        return model_mean
    else:
        var = (bet*(1.-extract(alphas_cumprod_prev,t,x_t.shape))/(1.-a_cum)).to(x_t.device)
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(var) * noise

@torch.no_grad()
def p_sample_loop(model, shape, c_emb, betas, alphas, alphas_cumprod, alphas_cumprod_prev, device, use_global_norm = True):
    x = torch.randn(shape, device=device)
    for i in tqdm(reversed(range(betas.shape[0])), desc='sampling'):
        x = p_sample(model, x, i, c_emb, betas, alphas, alphas_cumprod, alphas_cumprod_prev)
    # print(f"p_sample_loop x without global_norm mean:{x.mean().item()},std:{x.std().item()}")
    if use_global_norm and Path("mel_stats.pt").exists():
        stats = torch.load("mel_stats.pt", weights_only=True)
        mean, std = stats["mean"], stats["std"]
        x = x * std + mean
    print(f"p_sample_loop x_out with global_norm mean:{x.mean().item()},std:{x.std().item()}")
    return x

# ---------------- Vocoder ----------------

def mel_to_audio_griffin(mel_spec, sample_rate=16000, n_fft=1024, hop_length=512, n_iter=64, length=None):
    # mel = torch.clamp(mel_spec, min=-20.0, max=20.0)
    # print(f"mel_expm1 mean before expm1:{mel_spec.mean().item()},std:{mel_spec.std().item()}")
    mel = torch.expm1(mel_spec.squeeze(0))
    # print(f"mel_expm1 mean:{mel.mean().item()},std:{mel.std().item()}")
    mel = mel.clamp(min=1e-8) * 10.0
    mel_inv = torchaudio.transforms.InverseMelScale(n_stft=n_fft//2+1, n_mels=mel.shape[0], sample_rate=sample_rate).to(mel.device)
    spec = mel_inv(mel)
    if length is None:
        length = (spec.size(-1) - 1) * hop_length + n_fft
    window = torch.hann_window(n_fft).to(mel.device)
    waveform = F_audio.griffinlim(spec, window=window, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, power=1.0, n_iter=n_iter, momentum=0.99, length=length, rand_init=True)
    waveform = waveform / (waveform.abs().max() + 1e-9)
    return waveform.cpu()

# ---------------- Training loop ----------------

def train(args):
    device = torch.device(args.device)
    mkdirs(args.save_dir, args.sample_dir, args.modelgen_dir)

    dataset = MACSDataset(args.data_audio_dir, args.data_annotation, n_mels=args.n_mels, sample_rate=args.sample_rate, duration=args.duration, use_global_norm=args.use_global_norm,stats_file=args.stats_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # loader = DataLoader(Subset(dataset, range(1000)), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = ImprovedUNetCond(in_channels=1, base_ch=args.base_ch, t_dim=args.t_dim, c_dim=args.c_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # schedule
    beta_start = 1e-4
    beta_end = 2e-2
    betas = torch.linspace(beta_start, beta_end, 1000)
    # betas = cosine_beta_schedule(args.T)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision and device.type=='cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=args.mixed_precision and device.type=='cuda')

    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, (mel, text) in enumerate(pbar):
            mel = mel.to(device)
            c_emb = encode_text(text, device)
            bs = mel.size(0)
            t = torch.randint(0, args.T, (bs,), device=device).long()
            noise = torch.randn_like(mel)
            x_t = q_sample(mel, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            # print(f"x_t mean:{x_t.mean().item()},std:{x_t.std().item()}")
            # print(f"noise mean:{noise.mean().item()},std:{noise.std().item()}")

            model.train()
            # with torch.cuda.amp.autocast(enabled=args.mixed_precision and device.type=='cuda'):
            with torch.amp.autocast('cuda', enabled=args.mixed_precision and device.type=='cuda'):
                eps_theta = model(x_t, t, c_emb)
                loss = F.mse_loss(eps_theta, noise)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            if (i + 1) % args.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            global_step += 1
            pbar.set_postfix(loss=loss.item() * args.grad_accum)

        # save checkpoint
        if args.epochs <= 10 or (epoch + 1 == args.epochs):
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': global_step}, os.path.join(args.save_dir, f'model_epoch{epoch+1}.pt'))

        # generate samples for inspection
        model.eval()
        text_sample = ["a car is moving when it raining"] * 2
        # text_sample = ["a person whistling and singing"] * 2
        c_emb = encode_text(text_sample, device)
        samples = p_sample_loop(model, (2,1,args.n_mels, mel.size(3)), c_emb, betas, alphas, alphas_cumprod, alphas_cumprod_prev, device, use_global_norm=args.use_global_norm)
        # print(f"samples_mel mean:{samples.mean().item()},std:{samples.std().item()}")
        for idx, s in enumerate(samples):
            waveform = mel_to_audio_griffin(s, sample_rate=args.sample_rate, n_fft=1024, hop_length=512, n_iter=64, length=args.duration*args.sample_rate)
            out_path = os.path.join(args.sample_dir, f'sample_epoch{epoch+1}_{idx}.wav')
            torchaudio.save(out_path, waveform.unsqueeze(0), args.sample_rate)
        print(f'Finished epoch {epoch+1}, checkpoint and samples saved.')

# ---------------- Sampling function ----------------
@torch.no_grad()
def sample(model_path, text, args, n=1, mel_len=None):
    device = torch.device(args.device)
    model = ImprovedUNetCond(in_channels=1, base_ch=args.base_ch, t_dim=args.t_dim, c_dim=args.c_dim).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model.eval()

    betas = cosine_beta_schedule(args.T)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value=1.0)

    if mel_len is None:
        # default time frames for given duration
        mel_len = math.ceil((args.duration * args.sample_rate) / 512)

    c_emb = encode_text([text]*n, device)
    samples = p_sample_loop(model, (n,1,args.n_mels, mel_len), c_emb, betas, alphas, alphas_cumprod, alphas_cumprod_prev, device, use_global_norm=args.use_global_norm)
    waveforms = []
    for s in samples:
        wav = mel_to_audio_griffin(s, sample_rate=args.sample_rate, n_fft=1024, hop_length=512, n_iter=128, length=args.duration*args.sample_rate)
        waveforms.append(wav)
    return waveforms

# ---------------- CLI ----------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','sample'], default='train')

    # paths and data
    parser.add_argument('--data_audio_dir', type=str, default=DEFAULTS['data_audio_dir'])
    parser.add_argument('--data_annotation', type=str, default=DEFAULTS['data_annotation'])
    parser.add_argument('--save_dir', type=str, default=DEFAULTS['save_dir'])
    parser.add_argument('--sample_dir', type=str, default=DEFAULTS['sample_dir'])
    parser.add_argument('--modelgen_dir', type=str, default=DEFAULTS['modelgen_dir'])

    # model / training
    parser.add_argument('--n_mels', type=int, default=DEFAULTS['n_mels'])
    parser.add_argument('--sample_rate', type=int, default=DEFAULTS['sample_rate'])
    parser.add_argument('--duration', type=int, default=DEFAULTS['duration'])
    parser.add_argument('--batch_size', type=int, default=DEFAULTS['batch_size'])
    parser.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    parser.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    parser.add_argument('--device', type=str, default=DEFAULTS['device'])
    parser.add_argument('--T', type=int, default=DEFAULTS['T'])
    parser.add_argument('--base_ch', type=int, default=DEFAULTS['base_ch'])
    parser.add_argument('--t_dim', type=int, default=DEFAULTS['t_dim'])
    parser.add_argument('--c_dim', type=int, default=DEFAULTS['c_dim'])
    parser.add_argument('--vocoder', type=str, default=DEFAULTS['vocoder'])
    parser.add_argument('--hifigan_ckpt', type=str, default=DEFAULTS['hifigan_ckpt'])
    parser.add_argument('--grad_accum', type=int, default=DEFAULTS['grad_accum'])
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--use_global_norm', action='store_true')
    parser.add_argument('--stats_file', type=str, default=DEFAULTS['stats_file'])

    # sampling
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--text', type=str, default='a car is moving when it raining')
    parser.add_argument('--mel_len', type=int, default=None)
    args = parser.parse_args()

    # set flags
    args.mixed_precision = args.mixed_precision or DEFAULTS['mixed_precision']

    if args.mode == 'train':
        train(args)
    else:
        assert args.model, 'Please provide --model path for sampling'
        model_path = os.path.join(args.save_dir,args.model)
        waveforms = sample(model_path, args.text, args, n=1, mel_len=args.mel_len)
        mkdirs(args.modelgen_dir)
        sample_text = args.text.replace(' ', '_')
        for i, wf in enumerate(waveforms):
            torchaudio.save(os.path.join(args.modelgen_dir, f'sample_{sample_text}_{i}.wav'), wf.unsqueeze(0), args.sample_rate)
        print(f"Saved {len(waveforms)} wave(s) to {args.modelgen_dir}")
