import torch
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
import networkx as nx
import matplotlib.pyplot as plt

# -------------------------
# 路径配置
# -------------------------
RAW_ZIP_PATH = "data/QM9/raw/qm9.zip"  # 手动下载的 QM9 zip 文件
EXTRACT_DIR = "data/QM9/raw"           # zip 解压路径

# -------------------------
# 将 SMILES 转成 RDKit Mol
# -------------------------
def smiles_to_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # 加氢原子
    return mol

# -------------------------
# 生成有向加权边
# -------------------------
def mol_to_directed_edges(mol):
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bt = bond.GetBondType()
        if bt == Chem.rdchem.BondType.SINGLE:
            w = 1.0
        elif bt == Chem.rdchem.BondType.DOUBLE:
            w = 2.0
        elif bt == Chem.rdchem.BondType.TRIPLE:
            w = 3.0
        elif bt == Chem.rdchem.BondType.AROMATIC:
            w = 1.5
        else:
            w = 0.0

        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append([w])
        edge_attr.append([w])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return edge_index, edge_attr

# -------------------------
# 自定义小型 QM9 数据集读取
# -------------------------
class LocalQM9Dataset(InMemoryDataset):
    def __init__(self, root, limit=None):
        super().__init__(root)
        # 假设你提前用 pickle 或其他方法保存了 SMILES 列表和属性
        # 这里我们示范用一个小列表
        smiles_list = [
            "O=C=O", "CCO", "CC(=O)O"  # 示例
        ]
        if limit:
            smiles_list = smiles_list[:limit]

        data_list = []
        for smiles in smiles_list:
            mol = smiles_to_mol(smiles)
            if mol is None:
                continue
            edge_index, edge_attr = mol_to_directed_edges(mol)

            # 随便生成一些节点特征
            x = torch.tensor([[atom.GetAtomicNum()] for atom in mol.GetAtoms()], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)

def visualize_tensor_graph(data, title="Molecule Graph (tensor view)"):
    """
    可视化 PyG 格式的图数据（使用 edge_index, edge_attr, x）
    """
    G = nx.DiGraph()  # 用有向图

    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.squeeze().cpu().numpy() if data.edge_attr is not None else None
    x = data.x.cpu().numpy()

    # 添加节点
    for i in range(x.shape[0]):
        node_label = ",".join([f"{v:.2f}" for v in x[i]])  # 展示节点特征
        G.add_node(i, label=node_label)

    # 添加边
    for k, (src, dst) in enumerate(edge_index.T):
        weight = edge_attr[k] if edge_attr is not None else 1.0
        G.add_edge(int(src), int(dst), weight=weight)

    pos = nx.spring_layout(G, seed=42)  # 自动布局
    labels = {i: G.nodes[i]["label"] for i in G.nodes}
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}

    plt.figure(figsize=(6, 6))
    nx.draw(
        G, pos,
        with_labels=True,
        labels={i: i for i in G.nodes},
        node_color="lightblue",
        node_size=800,
        arrows=True
    )
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, verticalalignment="bottom")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

    plt.title(title)
    plt.axis("off")
    plt.savefig("molgraph/" + title + ".png")
    # plt.show()

# -------------------------
# 使用示例
# -------------------------
if __name__ == "__main__":
    dataset = LocalQM9Dataset(root="data/QM9", limit=3)
    for i, data in enumerate(dataset):
        print(f"Sample {i}:")
        print("x (atom features):", data.x)
        print("edge_index:\n", data.edge_index)
        print("edge_attr:\n", data.edge_attr.squeeze())
        print("-" * 50)
        visualize_tensor_graph(data, title=f"Sample_{i}_tensor_view")