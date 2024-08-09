import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.utils.data as data

class BinaryDataset(Dataset):
    def __init__(self, excel_path="../data/merge.xlsx", embedding_dir="../data/slide224_level_embedding/Path"):
        # 读取并处理Excel文件
        df = pd.read_excel(excel_path)
        df = df[["病理id", "是否进展"]].dropna()

        # 初始化嵌入路径和嵌入列表
        self.embedding_dir = embedding_dir
        self.embedding_list = []

        # 遍历数据行并加载嵌入
        for index, row in df.iterrows():
            slide_id = row["病理id"]
            embedding_path = os.path.join(embedding_dir, f"{int(slide_id)}.pt")
            if os.path.exists(embedding_path):
                self.embedding_list.append((embedding_path, int(row["是否进展"])))

    def __len__(self):
        return len(self.embedding_list)

    def __getitem__(self, index):
        embedding_path, jinzhan = self.embedding_list[index]

        # 加载嵌入并展平
        embedding = torch.load(embedding_path).float().flatten()

        # 转换进展标签为Long类型，适合CrossEntropyLoss
        jinzhan = torch.tensor(jinzhan).long()

        return embedding, jinzhan


def get_binary_dataloader(excel_path="./data/merge.xlsx", embedding_dir="./data/slide224_level_embedding/Path", batch_size=32, split=[0.7, 0.15, 0.15], shuttle=[True, False, False]):
    tile = BinaryDataset(excel_path, embedding_dir)
    # Calculate dataset sizes
    train_size = int(split[0] * len(tile))
    valid_size = int(split[1] * len(tile))
    test_size = len(tile) - train_size - valid_size
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Split dataset
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(tile, [train_size, valid_size, test_size])

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"验证集大小: {len(valid_dataset)}")

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == "__main__":
    train_dataloader, valid_dataloader, test_dataloader = get_binary_dataloader("../data/merge.xlsx", "../data/slide224_level_embedding/Path")
    print(len(train_dataloader))
    for i, (data, label) in enumerate(train_dataloader):
        print(data.shape, label.shape)
        break
