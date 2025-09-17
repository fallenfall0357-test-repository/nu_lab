"""
Minimal Score-based Diffusion (VP-SDE) in PyTorch.
- Dataset: 64x64 PNG images in a folder.
- Model: simple U-Net-like network outputting score function s_theta(x_t, t).
- Forward: variance-preserving SDE (VP-SDE).
- Training: denoising score matching loss.
- Sampling: reverse SDE with Euler-Maruyama.

Note: This is an educational baseline, not optimized for FID.
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
DATA_DIR = "GALAXY"
IMAGE_SIZE = 64
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 100
LR = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "checkpoints"
SAMPLE_DIR = "samples"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Time-dependent beta schedule (VP-SDE)
def beta_t(t):
    # t in [0,1], linear schedule
    beta_min, beta_max = 0.1, 20.0
    return beta_min + (beta_max - beta_min) * t

# ---------------- Dataset ----------------
class JPGImageFolder(Dataset):
    def __init__(self, folder, image_size=IMAGE_SIZE):
        self.files = [p for p in Path(folder).glob("**/*.jpg")]
        assert len(self.files) > 0, f"No JPGs found in {folder}"
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*CHANNELS, [0.5]*CHANNELS),
        ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img)

# ---------------- Model ----------------
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
        self.t_proj = nn.Linear(t_dim, out_ch) if t_dim else None
    def forward(self, x, t_emb=None):
        h = self.act(self.norm1(self.conv1(x)))
        if self.t_proj is not None and t_emb is not None:
            h = h + self.t_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.norm2(self.conv2(h)))
        return h

class SimpleUNetScore(nn.Module):
    def __init__(self, in_channels=CHANNELS, base_ch=64, t_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        self.enc1 = Block(in_channels, base_ch, t_dim)
        self.enc2 = Block(base_ch, base_ch*2, t_dim)
        self.enc3 = Block(base_ch*2, base_ch*4, t_dim)
        self.mid = Block(base_ch*4, base_ch*4, t_dim)
        self.dec3 = Block(base_ch*8, base_ch*2, t_dim)
        self.dec2 = Block(base_ch*4, base_ch, t_dim)
        self.dec1 = Block(base_ch*2, base_ch, t_dim)
        self.final = nn.Conv2d(base_ch, in_channels, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x, t):
        t = t.view(-1,1)
        t_emb = self.time_mlp(t)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e3 = self.enc3(self.pool(e2), t_emb)
        m = self.mid(self.pool(e3), t_emb)
        d3 = self.up(m)
        d3 = self.dec3(torch.cat([d3,e3],dim=1), t_emb)
        d2 = self.up(d3)
        d2 = self.dec2(torch.cat([d2,e2],dim=1), t_emb)
        d1 = self.up(d2)
        d1 = self.dec1(torch.cat([d1,e1],dim=1), t_emb)
        return self.final(d1)  # score s_theta(x_t,t)

# ---------------- Training ----------------
def train():
    dataset = JPGImageFolder(DATA_DIR, IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    model = SimpleUNetScore().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x0 in pbar:
            x0 = x0.to(DEVICE)
            bs = x0.size(0)
            # sample random time t in [0,1]
            t = torch.rand(bs, device=DEVICE)
            # perturb x0 according to VP-SDE (closed form)
            mean = x0 * torch.exp(-0.5 * beta_t(t).view(-1,1,1,1))
            std = torch.sqrt(1 - torch.exp(-beta_t(t).view(-1,1,1,1)))
            noise = torch.randn_like(x0)
            xt = mean + std * noise
            # score model prediction
            score_pred = model(xt, t)
            # target score = -noise/std
            target = - noise / std
            loss = F.mse_loss(score_pred, target)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=loss.item())
        if epoch == EPOCHS - 1:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"score_epoch{epoch+1}.pt"))
        elif (epoch + 1) % (EPOCHS / 10) == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"score_epoch{epoch+1}.pt"))

# ---------------- Sampling ----------------
@torch.no_grad()
def sample(model_path, n=16, steps=1000):
    model = SimpleUNetScore().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    x = torch.randn(n, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    dt = 1.0/steps
    for i in tqdm(reversed(range(steps))):
        t = torch.ones(n, device=DEVICE) * (i/steps)
        score = model(x, t)
        b = beta_t(t).view(-1,1,1,1)
        drift = -0.5 * b * x - b * score
        diffusion = torch.sqrt(b)
        x = x + drift*dt + diffusion*math.sqrt(dt)*torch.randn_like(x)
    utils.save_image((x+1)/2, os.path.join(SAMPLE_DIR, f"sample_from_{Path(model_path).stem}.png"), nrow=int(math.sqrt(n)))

# ---------------- CLI ----------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','sample'], default='train')
    parser.add_argument('--model', type=str, default='')
    args = parser.parse_args()
    if args.mode=='train':
        train()
    else:
        assert args.model
        sample(args.model)
