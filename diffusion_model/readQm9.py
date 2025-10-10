import os
import tarfile
import torch
from torch_geometric.data import InMemoryDataset, Data
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdmolops

# -------------------------
# 路径配置
# -------------------------
# BZ2_PATH = "data/QM9/Data for 133885 GDB-9 molecules.bz2"
EXTRACT_DIR = "data/QM9/extracted"

# # -------------------------
# # 解压 bz2 压缩包
# # -------------------------
# def extract_bz2(bz2_path, extract_dir):
#     if not os.path.exists(extract_dir):
#         print(f"[*] 解压 {bz2_path} ...")
#         os.makedirs(extract_dir, exist_ok=True)
#         with tarfile.open(bz2_path, "r:bz2") as tar:
#             tar.extractall(path=extract_dir)
#     else:
#         print(f"[✓] 已找到解压目录: {extract_dir}")

# -------------------------
# 从单个 .xyz 文件读取原子和坐标
# -------------------------
def read_xyz_file(xyz_path):
    atoms = []
    coords = []
    with open(xyz_path, "r") as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        if len(parts) >= 4:
            atoms.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms, torch.tensor(coords, dtype=torch.float)

# -------------------------
# 根据空间距离重建键（简易近邻法）
# -------------------------
def build_edges_from_coords(coords, atoms, cutoff=1.8):
    edge_index = []
    edge_attr = []
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            dist = torch.norm(coords[i] - coords[j])
            if dist < cutoff:
                # 双向边
                edge_index.append([i, j])
                edge_index.append([j, i])
                # 用距离倒数作为权重
                edge_attr.append([1.0 / dist])
                edge_attr.append([1.0 / dist])
    if not edge_index:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous(), torch.tensor(edge_attr, dtype=torch.float)

# -------------------------
# 可视化图
# -------------------------
def visualize_tensor_graphs_grid(dataset, n_rows=3, n_cols=3, save_path="molgraph/molecule_grid.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    for ax, data in zip(axes, dataset):
        G = nx.DiGraph()
        edge_index = data.edge_index.numpy()
        edge_attr = data.edge_attr.squeeze().numpy() if data.edge_attr is not None else None
        x = data.x.numpy()

        for i in range(x.shape[0]):
            G.add_node(i, label=str(int(x[i][0])))
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
# def visualize_tensor_graph(data, title="Molecule Graph"):
#     G = nx.DiGraph()
#     edge_index = data.edge_index.numpy()
#     edge_attr = data.edge_attr.squeeze().numpy() if data.edge_attr is not None else None
#     x = data.x.numpy()

#     for i in range(x.shape[0]):
#         G.add_node(i, label=str(int(x[i][0])))
#     for k, (src, dst) in enumerate(edge_index.T):
#         G.add_edge(int(src), int(dst), weight=edge_attr[k] if edge_attr is not None else 1.0)

#     pos = nx.spring_layout(G, seed=42)
#     labels = {i: G.nodes[i]["label"] for i in G.nodes}
#     edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

#     plt.figure(figsize=(6, 6))
#     nx.draw(G, pos, with_labels=False, node_color="lightblue", node_size=600, arrows=True)
#     nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, verticalalignment="bottom")
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
#     plt.title(title)
#     plt.axis("off")
#     os.makedirs("molgraph", exist_ok=True)
#     plt.savefig(f"molgraph/{title}.png")
#     plt.close()

# -------------------------
# 自定义 QM9 数据集
# -------------------------
class LocalQM9XYZDataset(InMemoryDataset):
    def __init__(self, root, limit=5):
        super().__init__(root)
        # extract_bz2(BZ2_PATH, EXTRACT_DIR)
        files = sorted([os.path.join(EXTRACT_DIR, f) for f in os.listdir(EXTRACT_DIR) if f.endswith(".xyz")])
        if not files:
            raise FileNotFoundError("❌ 未在解压目录中找到 .xyz 文件")

        data_list = []
        for i, file in enumerate(files[:limit]):
            atoms, coords = read_xyz_file(file)
            atomic_numbers = [Chem.GetPeriodicTable().GetAtomicNumber(a) for a in atoms]
            x = torch.tensor([[z] for z in atomic_numbers], dtype=torch.float)
            edge_index, edge_attr = build_edges_from_coords(coords, atoms)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)
        print(f"[✓] 成功加载 {len(data_list)} 个分子样本")

# -------------------------
# 运行示例
# -------------------------
if __name__ == "__main__":
    dataset = LocalQM9XYZDataset(root="data/QM9", limit=9)
    for i, data in enumerate(dataset):
        print(f"Sample {i}:")
        print("x (atomic numbers):", data.x.view(-1).tolist())
        print("edge_index:\n", data.edge_index)
        print("edge_attr:\n", data.edge_attr.squeeze())
        print("-" * 50)

    visualize_tensor_graphs_grid(dataset)