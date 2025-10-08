import torch
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem

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