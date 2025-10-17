"""
DDPM baseline (discrete-time) - PyTorch
- Dataset: JPG images folder (64x64)
- Model: U-Net-like network that predicts noise (eps)
- Noise schedule: linear beta, T=1000
- Time embedding: sinusoidal
- Training: MSE on noise (DDPM objective)
- Sampling: standard DDPM sampling loop

Usage:
  python ddpm_fixed.py --mode train
  python ddpm_fixed.py --mode sample --model checkpoints/ddpm_epoch100.pt

This version fixes the continuous-time SDE mistakes and uses the standard DDPM discrete formulas.
"""

import os
import math
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

# ---------------- Config ----------------
DATA_DIR = "data/QM9/processed_qm9_5M.pt"
IMAGE_SIZE = 32
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 100
LR = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "output/molgraph/weights/checkpoints"
SAMPLE_DIR = "output/molgraph/samples"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# ---------------- Noise schedule (DDPM discrete) ----------------
T = 1000
beta_start = 1e-4
beta_end = 2e-2
betas = torch.linspace(beta_start, beta_end, T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value=1.0)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# helper to index tensors for a batch of indices t
def extract(a, t, x_shape):
    a = a.to(t.device)               # 确保和 t 一致
    out = a.gather(0, t).float()
    return out.reshape(-1, *((1,) * (len(x_shape) - 1)))

# # ---------------- Dataset ----------------
# class JPGImageFolder(Dataset):
#     def __init__(self, folder, image_size=IMAGE_SIZE):
#         self.files = [p for p in Path(folder).glob("**/*.jpg")] + [p for p in Path(folder).glob("**/*.png")]
#         assert len(self.files) > 0, f"No images found in {folder}"
#         self.transform = transforms.Compose([
#             transforms.Resize((image_size, image_size), interpolation=Image.LANCZOS),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5]*CHANNELS, [0.5]*CHANNELS),
#         ])
#     def __len__(self):
#         return len(self.files)
#     def __getitem__(self, idx):
#         img = Image.open(self.files[idx]).convert('RGB')
#         return self.transform(img)

# ---------------- Time embedding ----------------
def sinusoidal_embedding(timesteps, dim):
    # timesteps: (B,) values in [0, T)
    device = timesteps.device
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0,1,0,0))
    return emb

# ---------------- Model (UNet-like, eps prediction) ----------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(t_dim, out_ch) if t_dim is not None else None
        if in_ch != out_ch:
            self.nin_shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.nin_shortcut = None
    def forward(self, x, t_emb=None):
        h = self.act(self.norm1(self.conv1(x)))
        if self.time_proj is not None and t_emb is not None:
            h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.norm2(self.conv2(h)))
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h

class SmallUNet(nn.Module):
    def __init__(self, in_channels=3, base_ch=128, t_dim=256):
        super().__init__()
        self.t_dim = t_dim
        self.time_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        # encoder
        self.enc1 = ResBlock(in_channels, base_ch, t_dim)
        self.enc2 = ResBlock(base_ch, base_ch*2, t_dim)
        self.enc3 = ResBlock(base_ch*2, base_ch*4, t_dim)
        # middle
        self.mid = ResBlock(base_ch*4, base_ch*4, t_dim)
        # decoder
        self.dec3 = ResBlock(base_ch*8, base_ch*2, t_dim)
        self.dec2 = ResBlock(base_ch*4, base_ch, t_dim)
        self.dec1 = ResBlock(base_ch*2, base_ch, t_dim)
        self.out = nn.Conv2d(base_ch, in_channels, 1)
        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x, t):
        # t: (B,) in [0, T)
        t_emb = sinusoidal_embedding(t, self.t_dim)
        t_emb = self.time_mlp(t_emb)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e3 = self.enc3(self.pool(e2), t_emb)
        m = self.mid(self.pool(e3), t_emb)
        d3 = self.upsample(m)
        d3 = self.dec3(torch.cat([d3, e3], dim=1), t_emb)
        d2 = self.upsample(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), t_emb)
        d1 = self.upsample(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), t_emb)
        return self.out(d1)

# ---------------- q_sample (forward) ----------------
def q_sample(x_start, t, noise=None):
    # x_start: (B,C,H,W), t: (B,) indices 0..T-1
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alpha_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alpha_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

# ---------------- p_sample (reverse) ----------------
def p_sample(model, x_t, t_index):
    # t_index is scalar int in [0, T-1]
    t = torch.full((x_t.size(0),), t_index, dtype=torch.long, device=x_t.device)
    bet = extract(betas, t, x_t.shape)
    a_t = extract(alphas, t, x_t.shape)
    a_cum = extract(alphas_cumprod, t, x_t.shape)
    sqrt_recip_a = torch.sqrt(1.0 / a_t)
    # predict noise
    eps_theta = model(x_t, t)
    # compute model mean for p(x_{t-1} | x_t)
    model_mean = (1. / torch.sqrt(a_t)) * (x_t - (bet / torch.sqrt(1. - a_cum)) * eps_theta)
    if t_index == 0:
        return model_mean
    else:
        var = (bet * (1. - extract(alphas_cumprod_prev, t, x_t.shape)) / (1. - a_cum)).to(x_t.device)
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(var) * noise

@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device
    img = torch.randn(shape, device=device)
    for i in tqdm(reversed(range(T))):
        img = p_sample(model, img, i)
    return img

# ---------------- Training ----------------
def train():
    dataset = torch.load(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    model = SmallUNet(in_channels=CHANNELS, base_ch=128).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    global_step = 0
    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            batch = batch.to(DEVICE)
            bs = batch.size(0)
            t = torch.randint(0, T, (bs,), device=DEVICE).long()
            noise = torch.randn_like(batch)
            x_t = q_sample(batch, t, noise=noise)
            eps_theta = model(x_t, t)
            loss = F.mse_loss(eps_theta, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1
            pbar.set_postfix(loss=loss.item())

        model.eval()
        samples = p_sample_loop(model, (16, CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
        utils.save_image((samples + 1) / 2.0, os.path.join(SAMPLE_DIR, f"sample_epoch{epoch+1}.png"), nrow=4)
        model.train()

        if ((epoch+1) % (EPOCHS / 10) == 0 if EPOCHS >= 10 else True):
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"ddpm_epoch{epoch+1}.pt"))

# ---------------- Sampling helper ----------------
@torch.no_grad()
def sample_and_save(model_path, n=16):
    model = SmallUNet(in_channels=CHANNELS, base_ch=128).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    samples = p_sample_loop(model, (n, CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    utils.save_image((samples + 1)/2.0, os.path.join(SAMPLE_DIR, f"sample_from_{Path(model_path).stem}.png"), nrow=int(math.sqrt(n)))
    print(f"Saved samples to {SAMPLE_DIR}")

# ---------------- CLI ----------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','sample'], default='train')
    parser.add_argument('--model', type=str, default='')
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        assert args.model
        sample_and_save(args.model)
