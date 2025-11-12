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
from math import sqrt
# import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchaudio
import torchaudio.functional as F_audio
import matplotlib.pyplot as plt
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
    'epochs': 20,
    'lr': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'T': 1000,
    'base_ch': 256,
    't_dim': 256, #512
    'c_dim': 256, #512
    'save_dir': '../../output/sfx/weights',
    'sample_dir': '../../output/sfx/samples',
    'modelgen_dir': '../../output/sfx/test',
    'vocoder': 'griffinlim',
    'hifigan_ckpt': '',
    'grad_accum': 1,
    'mixed_precision': False,
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
    return torch.clip(betas, 1e-6, 0.999)

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

# text_proj = None  # will create later to match c_dim
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
    return emb 

class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation, time_emb_dim, cond_dim):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels, 2 * residual_channels, kernel_size=3,
            padding=dilation, dilation=dilation
        )
        self.time_proj = nn.Linear(time_emb_dim, 2 * residual_channels)
        self.cond_proj = nn.Linear(cond_dim, 2 * residual_channels)
        self.res_skip_proj = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, t_emb, c_emb):
        # x: [B, C, T]
        h = self.dilated_conv(x)
        t = self.time_proj(t_emb).unsqueeze(-1)
        t = F.normalize(t, dim=-1)
        c = self.cond_proj(c_emb).unsqueeze(-1)
        c = F.normalize(c, dim=-1)
        h = h + t + c

        gate, filter = h.chunk(2, dim=1)
        out = torch.tanh(filter) * torch.sigmoid(gate)

        res, skip = self.res_skip_proj(out).chunk(2, dim=1)
        x = x + res * (1 / math.sqrt(2.0))  # residual connection
        return x, skip


class TextDiffWave(nn.Module):
    def __init__(self, n_mels=80, residual_channels=64, n_res_layers=30,
                 time_emb_dim=128, text_dim=256):
        super().__init__()
        self.input_proj = nn.Conv1d(n_mels, residual_channels, 1)
        self.time_emb = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                residual_channels=residual_channels,
                dilation=2 ** (i % 10),
                time_emb_dim=time_emb_dim,
                cond_dim=text_dim
            )
            for i in range(n_res_layers)
        ])
        self.skip_proj = nn.Conv1d(residual_channels, residual_channels, 1)
        self.output_proj = nn.Conv1d(residual_channels, n_mels, 1)
        nn.init.zeros_(self.output_proj.weight)

    def forward(self, x, t, c_emb):
        # x: [B, n_mels, T]
        # t: [B]
        # c_emb: [B, cond_dim]
        x = x.squeeze(1)
        x = self.input_proj(x)
        t_emb = self.time_emb(sinusoidal_embedding(t, self.time_emb[0].in_features))

        skips = 0
        for layer in self.res_blocks:
            x, skip = layer(x, t_emb, c_emb)
            skips = skips + skip if isinstance(skips, torch.Tensor) else skip

        x = skips / math.sqrt(len(self.res_blocks))
        x = F.relu(self.skip_proj(x))
        x = self.output_proj(x)
        x = x.unsqueeze(1)
        return x
# ---------- end insertion ----------
# ---------------- Forward / Reverse diffusion ----------------

def q_sample(x_start, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    sqrt_alpha_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alpha_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

@torch.no_grad()
def p_sample(model, x_t, t_index, c_emb, betas, alphas, alphas_cumprod, alphas_cumprod_prev):
    """完全重写的采样函数，确保数值稳定性"""
    device = x_t.device
    b = x_t.shape[0]
    t = torch.full((b,), t_index, dtype=torch.long, device=device)

    # 1. 使用DDPM原始公式预测噪声
    eps_cond = model(x_t, t, c_emb)
    eps_uncond = model(x_t, t, torch.zeros_like(c_emb))
    eps_theta = eps_uncond + 1.5 * (eps_cond - eps_uncond)

    # 2. 计算关键参数（使用更稳定的表达式）
    alpha_t = alphas[t_index]
    alpha_cumprod_t = alphas_cumprod[t_index]
    alpha_cumprod_prev_t = alphas_cumprod_prev[t_index] if t_index > 0 else torch.tensor(1.0).to(device)
    beta_t = betas[t_index]

    # 3. 预测x0（DDPM公式10）
    sqrt_recip_alpha_cumprod_t = 1.0 / torch.sqrt(alpha_cumprod_t + 1e-8)
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t + 1e-8)
    pred_x0 = sqrt_recip_alpha_cumprod_t * (x_t - sqrt_one_minus_alpha_cumprod_t * eps_theta)
    
    # 温和的裁剪
    pred_x0 = torch.clamp(pred_x0, -2.0, 2.0)

    # 4. 计算后验均值和方差（DDPM公式7）
    if t_index == 0:
        # 最后一步直接返回预测的x0
        return pred_x0
    else:
        # 均值系数
        mean_coeff1 = (torch.sqrt(alpha_cumprod_prev_t) * beta_t) / (1.0 - alpha_cumprod_t + 1e-8)
        mean_coeff2 = (torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_prev_t)) / (1.0 - alpha_cumprod_t + 1e-8)
        
        posterior_mean = mean_coeff1 * pred_x0 + mean_coeff2 * x_t
        
        # 后验方差（DDPM公式7）
        posterior_variance = ((1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t + 1e-8)) * beta_t
        posterior_variance = torch.clamp(posterior_variance, min=1e-8)
        
        # 采样
        noise = torch.randn_like(x_t)
        x_prev = posterior_mean + torch.sqrt(posterior_variance) * noise
        
        # 调试信息
        if t_index % 100 == 0:
            print(f"t={t_index}: beta={beta_t.item():.6f}, "
                  f"pred_x0_range=[{pred_x0.min().item():.3f}, {pred_x0.max().item():.3f}], "
                  f"posterior_var={posterior_variance.item():.6f}")
        
        return x_prev


@torch.no_grad()
def p_sample_loop(model, shape, c_emb, betas, alphas, alphas_cumprod, alphas_cumprod_prev):
    device = c_emb.device
    x_t = torch.randn(shape, device=device)
    for i in reversed(range(len(betas))):
        x_t = p_sample(model, x_t, i, c_emb, betas, alphas, alphas_cumprod, alphas_cumprod_prev)
        if i % 100 == 0:
            print(f"[t={i}] x_t std={x_t.std().item():.3f}")
    return x_t

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
    # loader = DataLoader(Subset(dataset, range(10)), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # model = ImprovedUNetCond(in_channels=1, base_ch=args.base_ch, t_dim=args.t_dim, c_dim=args.c_dim).to(device)
    model = TextDiffWave(n_mels=args.n_mels,
                        residual_channels=128,      # 可试 64 或 128
                        n_res_layers=200,        # 与 diffwave 类似
                        # dilation_cycle=10,
                        # max_T=args.T,
                        text_dim=384).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # schedule
    # beta_start = 1e-4
    # beta_end = 0.05
    # betas = torch.linspace(beta_start, beta_end, 1000)
    betas = cosine_beta_schedule(args.T)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    p_drop = 0.1

    # scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision and device.type=='cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=args.mixed_precision and device.type=='cuda')

    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        pbar.postfix = {}
        for i, (mel, text) in enumerate(pbar):
            mel = mel.to(device)
            c_emb = encode_text(text, device)
            if torch.rand(1).item() < p_drop:
                c_train = torch.zeros_like(c_emb)
            else:
                c_train = c_emb
            bs = mel.size(0)
            t = torch.randint(0, args.T, (bs,), device=device).long()
            noise = torch.randn_like(mel)
            x_t = q_sample(mel, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            # print(f"x_t mean:{x_t.mean().item()},std:{x_t.std().item()}")
            # print(f"noise mean:{noise.mean().item()},std:{noise.std().item()}")

            model.train()
            eps_theta = model(x_t, t, c_train)
            # alpha_bar_t = alphas_cumprod[t].view(-1, 1, 1, 1)
            # pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_bar_t)
            # L_mse_eps = F.mse_loss(eps_theta, noise)
            # L_mel_recon = F.l1_loss(pred_x0, mel)
            # λ2 = 2.0
            # loss = L_mse_eps + λ2 * L_mel_recon
            loss = F.mse_loss(eps_theta, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if global_step % 20 == 0:
                with torch.no_grad():
                    eps_cond = model(x_t, t, c_emb)
                    eps_uncond = model(x_t, t, torch.zeros_like(c_emb))
                    diff = F.mse_loss(eps_cond, eps_uncond)
                    # print(f"diff:{diff.item()}")
            global_step += 1
            pbar.set_postfix(loss=loss.item(),diff=diff.item())
            if global_step % 100 == 0:
                print(f"c_emb batch mean diff: {(c_emb[0] - c_emb[1]).abs().mean().item():.6f}")
                print(f"c_emb mean={c_emb.mean().item():.4f}, std={c_emb.std().item():.4f}")
                print(f"mel mean={mel.mean().item():.4f}, std={mel.std().item():.4f}")
                print(f"x_t mean={x_t.mean().item():.4f}, std={x_t.std().item():.4f}")
                print(f"eps_theta mean={eps_theta.mean().item():.4f}, std={eps_theta.std().item():.4f}")

        # save checkpoint
        if args.epochs <= 10 or (epoch + 1 == args.epochs or epoch + 1 == args.epochs/2):
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': global_step}, os.path.join(args.save_dir, f'model_epoch{epoch+1}.pt'))

        # for i, block in enumerate([model.enc1, model.enc2, model.enc3, model.enc4, model.enc5,
        #         model.mid, model.dec5, model.dec4, model.dec3, model.dec2, model.dec1]):
        #     name = f'block_{i}'
        #     if hasattr(block, 'time_proj'):
        #         p = block.time_proj.weight
        #         print(f"{name} cond_proj weight mean={p.mean().item():.6e}, std={p.std().item():.6e}")
    
        # generate samples for inspection
        model.eval()
        text_sample = ["a car is moving when it raining"] * 2
        # text_sample = ["a person whistling and singing"] * 2
        c_emb = encode_text(text_sample, device)
        # with torch.no_grad():
        #     x = torch.randn((2,1,args.n_mels, mel.size(3)),device=device)
        #     t = torch.full((x.size(0),), betas.shape[0] - 1, dtype=torch.long, device=device)
        #     eps_cond = model(x, t, c_emb)
        #     eps_uncond = model(x, t, torch.zeros_like(c_emb))
        #     diff = F.mse_loss(eps_cond, eps_uncond)
        #     print(f"diff:{diff.item()}")
        samples = p_sample_loop(model, (2,1,args.n_mels, mel.size(3)), c_emb, betas, alphas, alphas_cumprod, alphas_cumprod_prev)
        stats = torch.load("mel_stats.pt", weights_only=True)
        mean, std = stats["mean"], stats["std"]
        samples = samples * std + mean
        print(f"p_sample_loop x_out with global_norm mean:{samples.mean().item()},std:{samples.std().item()}")
        # print(f"samples_mel mean:{samples.mean().item()},std:{samples.std().item()}")
        for idx, s in enumerate(samples):
            waveform = mel_to_audio_griffin(s, sample_rate=args.sample_rate, n_fft=1024, hop_length=512, n_iter=64, length=args.duration*args.sample_rate)
            out_path = os.path.join(args.sample_dir, f'sample_epoch{epoch+1}_{idx}.wav')
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
            fig_path = os.path.join(args.sample_dir, f'sample_epoch{epoch+1}_{idx}')
            plt.savefig(fig_path+"_spectrogram.png", dpi=300)
            torchaudio.save(out_path, waveform.unsqueeze(0), args.sample_rate)
        print(f'Finished epoch {epoch+1}, checkpoint and samples saved.')

# ---------------- Sampling function ----------------
@torch.no_grad()
def sample(model_path, text, args, n=1, mel_len=None):
    device = torch.device(args.device)
    # model = ImprovedUNetCond(in_channels=1, base_ch=args.base_ch, t_dim=args.t_dim, c_dim=args.c_dim).to(device)
    model = TextDiffWave(n_mels=args.n_mels,
                        residual_channels=64,      # 可试 64 或 128
                        residual_layers=30,        # 与 diffwave 类似
                        dilation_cycle=10,
                        text_dim=384).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model.eval()

    beta_start = 1e-4
    beta_end = 2e-2
    betas = torch.linspace(beta_start, beta_end, 500)
    # betas = cosine_beta_schedule(args.T)
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
        wav = mel_to_audio_griffin(s, sample_rate=args.sample_rate, n_fft=1024, hop_length=512, n_iter=64, length=args.duration*args.sample_rate)
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
    parser.add_argument('--use_global_norm', action='store_false')
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
