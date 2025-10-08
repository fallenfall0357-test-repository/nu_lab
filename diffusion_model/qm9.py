import torch
from torch_geometric.datasets import QM9
from rdkit import Chem

# 把 PyG 自带的 QM9 样本中的 SMILES 转成 RDKit Mol
def smiles_to_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # 加氢原子
    return mol

# 根据 RDKit Mol 生成有向加权边 (edge_index, edge_attr)
def mol_to_directed_edges(mol):
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # 键类型 -> 数值权重
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

        # 有向边：i→j 和 j→i
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append([w])
        edge_attr.append([w])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return edge_index, edge_attr


# 封装一个函数，替换 QM9 的边特征
def process_qm9_to_directed(root="data/QM9", limit=None):
    dataset = QM9(root=root)
    new_data_list = []

    for idx, data in enumerate(dataset):
        smiles = data.smiles
        mol = smiles_to_mol(smiles)
        if mol is None:
            continue

        edge_index, edge_attr = mol_to_directed_edges(mol)

        # 替换 data 中的边
        data.edge_index = edge_index
        data.edge_attr = edge_attr

        new_data_list.append(data)

        if limit and idx >= limit - 1:
            break

    return new_data_list


if __name__ == "__main__":
    new_dataset = process_qm9_to_directed(limit=3)  # 先取 3 个样本看看
    for i, d in enumerate(new_dataset):
        print(f"Sample {i}:")
        print("x (atom features):", d.x.shape)
        print("edge_index:\n", d.edge_index)
        print("edge_attr (weights):\n", d.edge_attr.squeeze())
        print("-" * 50)