import torch
from torch.utils.data import DataLoader
from test import MACSDataset  # 如果类定义在 test.py 中
from tqdm import tqdm

# === 配置 ===
AUDIO_DIR = "./MACS_audio"      # 你音频所在目录
ANNOTATION_FILE = "./MACS.yaml" # 你 YAML 标注文件路径
BATCH_SIZE = 1                  # 统计时不用大batch
N_SAMPLES = None                # 如果只想看前100条就设为100

# === 创建数据集 ===
dataset = MACSDataset(AUDIO_DIR, ANNOTATION_FILE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

lengths = []
for i, (mel, _) in enumerate(tqdm(loader)):
    T = mel.shape[-1]
    lengths.append(T)
    if N_SAMPLES and i >= N_SAMPLES:
        break

print(f"样本数: {len(lengths)}")
print(f"平均帧长: {sum(lengths)/len(lengths):.2f}")
print(f"最短帧: {min(lengths)}, 最长帧: {max(lengths)}")