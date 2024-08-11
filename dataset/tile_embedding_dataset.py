import os
import re
import warnings

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

warnings.filterwarnings("ignore")


class TileEmbeddingDataset(data.Dataset):
    def __init__(self, excel_path="../data/merge.xlsx", embedding_dir="../data/tile224embedding"):
        df = pd.read_excel(excel_path)
        df = df[["病理id", "PFS", "是否进展"]].dropna()
        self.embedding_dir = embedding_dir
        self.embedding_list = []
        for index, row in df.iterrows():
            slide_id = str(int(row["病理id"]))
            pfs = row["PFS"]
            # 查找嵌入目录中包含slide_id的文件
            matching_files = [f for f in os.listdir(embedding_dir) if re.search(slide_id, f)]
            for file_name in matching_files:
                embedding_path = os.path.join(embedding_dir, file_name)
                self.embedding_list.append((embedding_path, pfs, int(row["是否进展"])))

    def __len__(self):
        return len(self.embedding_list)

    def __getitem__(self, index):
        embedding_path, pfs, jinzhan = self.embedding_list[index]
        embedding = torch.load(embedding_path)
        output = embedding[:, :1536]
        position = embedding[:, 1536:]
        return output.float(), position.float(), torch.tensor(pfs).float(), torch.tensor(jinzhan).float()


def collate_fn(batch):
    embeddings, position, pfs, jinzhan = zip(*batch)
    max_embedding_len = max(embedding.size(0) for embedding in embeddings)

    padded_embeddings = torch.stack([torch.cat([embedding, torch.zeros(max_embedding_len - embedding.size(0), embedding.size(1))], dim=0) for embedding in embeddings])
    position = torch.stack([torch.cat([pos, torch.zeros(max_embedding_len - pos.size(0), pos.size(1))], dim=0) for pos in position])
    pfs = torch.tensor(pfs, dtype=torch.float)
    jinzhan = torch.tensor(jinzhan, dtype=torch.float)
    return padded_embeddings, position, pfs, jinzhan


def get_tile_embedding_dataloader(excel_path="./data/merge.xlsx", embedding_dir="./data/tile224embedding", batch_size=32, split=[0.7, 0.15, 0.15], shuttle=[True, False, False]):
    tile = TileEmbeddingDataset(excel_path, embedding_dir)
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

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == "__main__":
    dataset, _, _ = get_tile_embedding_dataloader("../data/merge.xlsx", "../data/tile224embedding", batch_size=32)
    for i, (x, y, z) in enumerate(dataset):
        print(x.shape, y.shape, z.shape)
        break
