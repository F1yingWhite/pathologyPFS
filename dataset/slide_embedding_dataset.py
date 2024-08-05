import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image

warnings.filterwarnings("ignore")


class SlideEmbeddingDataset(data.Dataset):
    def __init__(self, excel_path="../data/merge.xlsx", embedding_dir="../data/slide_level_embedding/Path"):
        df = pd.read_excel(excel_path)
        df = df[["病理id", "PFS"]].dropna()
        self.embedding_dir = embedding_dir
        self.embedding_list = []
        for index, row in df.iterrows():
            slide_id = row["病理id"]
            pfs = row["PFS"]
            if not os.path.exists(os.path.join(embedding_dir, f"{int(slide_id)}.pt")):
                continue
            self.embedding_list.append((int(slide_id), pfs))

    def __len__(self):
        return len(self.embedding_list)

    def __getitem__(self, index):
        embedding, pfs = self.embedding_list[index]
        embedding_path = os.path.join(self.embedding_dir, f"{embedding}.pt")
        embedding = torch.load(embedding_path)
        return embedding.float(), torch.tensor(pfs).float()


if __name__ == "__main__":
    dataset = SlideEmbeddingDataset()

    # 计算训练集和测试集的大小
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 划分数据集
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"验证集大小: {len(valid_dataset)}")
