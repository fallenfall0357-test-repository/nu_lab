import os
import math
import torch
from torch_geometric.data import InMemoryDataset, Data
from torchvision import transforms, utils

SAMPLE_DIR = "oridata"

class LocalQM9XYZDataset(InMemoryDataset):
    def __init__(self, root, processed_file=None):
        super().__init__(root)
        if processed_file:
            self.data, self.slices = torch.load(processed_file)

# 加载
load_path = "data/QM9/processed_qm9_1K.pt"
# dataset = LocalQM9XYZDataset(root="data/QM9", processed_file=load_path)
dataset = torch.load(load_path)

# 选取前 16 个样本
n = 16
samples = dataset[:n]
samples = torch.stack(samples)

# 直接保存，不做任何额外处理
utils.save_image(
    (samples + 1) / 2.0,
    os.path.join(SAMPLE_DIR, f"originQM9.png"),
    nrow=int(math.sqrt(n))
)
print(f"Saved samples to {SAMPLE_DIR}")
# for i, data in enumerate(dataset):
#         print(f"Sample {i}:")
#         print("x (atomic numbers):", data.x.view(-1).tolist())
#         print("edge_index:\n", data.edge_index)
#         print("edge_attr:\n", data.edge_attr.squeeze())
#         print("-" * 50)

print(len(dataset))
print(dataset[0])