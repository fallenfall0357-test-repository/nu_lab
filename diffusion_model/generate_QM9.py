import os
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt


import math
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import utils

# ---------------- Config ----------------
IMAGE_SIZE = 32
CHANNELS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "output/molgraph/weights/checkpoints"
WEIGHT_DIR = "output/molgraph/weights"
SAMPLE_DIR = "output/molgraph/samples"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(WEIGHT_DIR, exist_ok=True)
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

# ---------------- p_sample (reverse) ----------------
def p_sample(model, x_t, t_index):
    # t_index is scalar int in [0, T-1]
    t = torch.full((x_t.size(0),), t_index, dtype=torch.long, device=x_t.device)
    bet = extract(betas, t, x_t.shape)
    a_t = extract(alphas, t, x_t.shape)
    a_cum = extract(alphas_cumprod, t, x_t.shape)
    # sqrt_recip_a = torch.sqrt(1.0 / a_t)
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

def image_tensor_to_data(img_tensor, threshold=0.5):
    A = img_tensor[0]
    W = img_tensor[1]
    X = img_tensor[2]

    # max_nodes = A.shape[0]
    
    # ✅ 直接用对角线恢复节点数
    diag = torch.diag(X)
    exists = diag > threshold
    num_nodes = int(exists.sum().item())

    if num_nodes == 0:
        num_nodes = 1  # 避免空图崩溃

    node_feats = diag[:num_nodes].view(-1, 1)

    edge_index_list = []
    edge_attr_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if A[i, j] > threshold:
                edge_index_list.append([i, j])
                edge_attr_list.append([float(W[i, j])])

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    data = Data(
        x=node_feats,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    return data

def visualize_tensor_graphs_grid(dataset, n_rows=3, n_cols=3, save_path="output/molgraph/molecule_grid.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    for ax, data in zip(axes, dataset):
        G = nx.DiGraph()
        edge_index = data.edge_index.numpy()
        edge_attr = data.edge_attr.squeeze().numpy() if data.edge_attr is not None else None
        x = data.x.cpu().numpy()

        for i in range(x.shape[0]):
            G.add_node(i, label=str(int(round(x[i][0]))))
        for k, (src, dst) in enumerate(edge_index.T):
            G.add_edge(int(src), int(dst), weight=edge_attr[k] if edge_attr is not None else 1.0)

        pos = nx.spring_layout(G, seed=42)
        labels = {i: G.nodes[i]["label"] for i in G.nodes}
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

        nx.draw(G, pos, with_labels=False, node_color="lightblue", font_size=8, node_size=400, arrows=True, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, verticalalignment="bottom", ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", ax=ax)
        # ax.set_title(f"Sample", fontsize=10)
        ax.axis("off")

    # 关闭未使用的子图
    for ax in axes[len(dataset):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] 保存整体分子图到 {save_path}")

# ---------------- Sampling helper ----------------
@torch.no_grad()
def sample_and_save(model_path, n=9):
    model = SmallUNet(in_channels=CHANNELS, base_ch=128).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    samples = p_sample_loop(model, (n, CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    re_dataset = [image_tensor_to_data(img) for img in samples]
    visualize_tensor_graphs_grid(re_dataset, save_path="output/molgraph/samples/graph_samples/test.png")
    # utils.save_image((samples + 1)/2.0, os.path.join(SAMPLE_DIR, f"sample_from_{Path(model_path).stem}.png"), nrow=int(math.sqrt(n)))
    # print(f"Saved samples to {SAMPLE_DIR}")

# ---------------- CLI ----------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='output/molgraph/weights/ddpm_epoch100.pt')
    args = parser.parse_args()
    assert args.model
    sample_and_save(args.model)