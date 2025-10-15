import os
import tarfile
import torch
from torch_geometric.data import InMemoryDataset, Data
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdmolops
from tqdm import tqdm

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
# def read_xyz_file(xyz_path):
#     atoms = []
#     coords = []
#     with open(xyz_path, "r") as f:
#         lines = f.readlines()
#     n_atoms = int(lines[0].strip())
#     for line in lines[2:2 + n_atoms]:
#         parts = line.split()
#         if len(parts) >= 4:
#             atoms.append(parts[0])
#             coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
#     return atoms, torch.tensor(coords, dtype=torch.float)
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

            # ✅ 处理含 `*^` 的指数格式
            x = parts[1].replace("*^", "e")
            y = parts[2].replace("*^", "e")
            z = parts[3].replace("*^", "e")

            coords.append([float(x), float(y), float(z)])
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
def visualize_tensor_graphs_grid(dataset, n_rows=3, n_cols=3, save_path="output/molgraph/molecule_grid.png"):
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
def graph_to_image_tensor(data, max_nodes=64):
    """
    把 Data对象 转成 [C, H, W] 张量
    C = 3 (邻接矩阵 / 边权重 / 原子序号)
    H = W = max_nodes
    """
    num_nodes = data.x.size(0)

    # 获取邻接矩阵
    A = torch.zeros((max_nodes, max_nodes))
    W = torch.zeros((max_nodes, max_nodes))
    X = torch.zeros((max_nodes, max_nodes))

    edge_index = data.edge_index
    edge_attr = data.edge_attr

    for idx, (src, dst) in enumerate(edge_index.t()):
        if src < max_nodes and dst < max_nodes:
            A[src, dst] = 1
            if edge_attr is not None:
                W[src, dst] = edge_attr[idx].item()

    # 节点特征 (原子序号) 填充对角或整行广播
    for i in range(num_nodes):
        if i < max_nodes:
            X[i, :] = data.x[i]
            X[:, i] = data.x[i]

    # 拼成多通道
    img_tensor = torch.stack([A, W, X], dim=0)  # [3, max_nodes, max_nodes]
    return img_tensor

def image_tensor_to_data(img_tensor, threshold=0.5):
    A = img_tensor[0]
    W = img_tensor[1]
    X = img_tensor[2]

    max_nodes = A.shape[0]
    
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

class LocalQM9XYZDataset(InMemoryDataset):
    def __init__(self, root, limit=5):
        super().__init__(root)
        # extract_bz2(BZ2_PATH, EXTRACT_DIR)
        files = sorted([os.path.join(EXTRACT_DIR, f) for f in os.listdir(EXTRACT_DIR) if f.endswith(".xyz")])
        if not files:
            raise FileNotFoundError("❌ 未在解压目录中找到 .xyz 文件")

        data_list = []
        for i, file in tqdm(enumerate(files[:limit]), desc=f"load data"):
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
    dataset = LocalQM9XYZDataset(root="data/QM9", limit=1000)
    # save_path = "data/QM9/processed_qm9.pt"
    save_path = "data/QM9/processed_qm9_1K.pt"
    image_tensors = [graph_to_image_tensor(data) for data in dataset]
    torch.save(image_tensors, save_path)
    print(f"[✓] 数据集已保存到 {save_path}")

    for i, data in enumerate(image_tensors):
        print(f"Sample {i}:")
        # print("x (atomic numbers):", data.x.view(-1).tolist())
        # print("edge_index:\n", data.edge_index)
        # print("edge_attr:\n", data.edge_attr.squeeze())
        print(data.shape)

        print("-" * 50)
        if i > 2: break
    # re_dataset = [image_tensor_to_data(img) for img in image_tensors]
    # visualize_tensor_graphs_grid(re_dataset, save_path="molgraph/test.png")
