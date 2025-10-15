import torch
from torch_geometric.data import InMemoryDataset, Data

class LocalQM9XYZDataset(InMemoryDataset):
    def __init__(self, root, processed_file=None):
        super().__init__(root)
        if processed_file:
            self.data, self.slices = torch.load(processed_file)

# 加载
load_path = "data/QM9/processed_qm9.pt"
dataset = LocalQM9XYZDataset(root="data/QM9", processed_file=load_path)

# for i, data in enumerate(dataset):
#         print(f"Sample {i}:")
#         print("x (atomic numbers):", data.x.view(-1).tolist())
#         print("edge_index:\n", data.edge_index)
#         print("edge_attr:\n", data.edge_attr.squeeze())
#         print("-" * 50)

print(len(dataset))
print(dataset[0])